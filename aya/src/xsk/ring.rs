//! Single-producer/single-consumer rings shared with the kernel.
//!
//! An `AF_XDP` socket exposes four rings (FILL, COMPLETION, RX, TX). Each ring is a region of
//! memory shared with the kernel containing a producer index, a consumer index, and a descriptor
//! array. One side (us or the kernel) produces into the ring and the other consumes; the
//! producer/consumer indices are free-running `u32` counters and the descriptor slot for index `i`
//! is `i & (size - 1)` (so `size` must be a power of two).
//!
//! The memory-ordering discipline mirrors `crate::maps::ring_buf` and the kernel's `xsk_queue.h`:
//! a consumer loads the producer index with [`Acquire`](Ordering::Acquire) and publishes its own
//! consumer index with [`Release`](Ordering::Release); a producer loads the consumer index with
//! `Acquire` and publishes its own producer index with `Release`. Our own index is only ever
//! written by us, so we read it back with [`Relaxed`](Ordering::Relaxed).

use std::{
    marker::PhantomData,
    os::fd::BorrowedFd,
    sync::atomic::{AtomicU32, Ordering},
};

use aya_obj::generated::{XDP_RING_NEED_WAKEUP, xdp_ring_offset};
use libc::{MAP_POPULATE, MAP_SHARED, PROT_READ, PROT_WRITE, off_t};

use super::XskError;
use crate::util::MMap;

/// A producer/consumer ring shared with the kernel.
///
/// `T` is the descriptor element: `u64` (a UMEM frame address) for the FILL and COMPLETION rings,
/// or [`xdp_desc`](aya_obj::generated::xdp_desc) for the RX and TX rings.
pub(crate) struct XskRing<T> {
    mmap: MMap,
    producer_offset: usize,
    consumer_offset: usize,
    flags_offset: usize,
    desc_offset: usize,
    /// Number of descriptor slots; always a power of two, so the slot for index `i` is
    /// `i & (size - 1)`.
    size: u32,
    _element: PhantomData<T>,
}

impl<T> XskRing<T> {
    /// Maps a ring of `size` descriptors from an `AF_XDP` socket.
    ///
    /// `page_offset` is the ring's fixed `mmap` page offset (e.g.
    /// [`XDP_PGOFF_RX_RING`](aya_obj::generated::XDP_PGOFF_RX_RING)); `offset` is the ring's layout
    /// returned by `getsockopt(XDP_MMAP_OFFSETS)`.
    pub(crate) fn map(
        fd: BorrowedFd<'_>,
        size: u32,
        page_offset: u64,
        offset: &xdp_ring_offset,
    ) -> Result<Self, XskError> {
        if !size.is_power_of_two() {
            return Err(XskError::InvalidRingSize);
        }
        let desc_len = u64::from(size)
            .checked_mul(size_of::<T>() as u64)
            .ok_or(XskError::InvalidRingOffsets)?;
        let len = offset
            .desc
            .checked_add(desc_len)
            .and_then(|len| usize::try_from(len).ok())
            .ok_or(XskError::InvalidRingOffsets)?;
        let word_fits = |field: u64| {
            field.is_multiple_of(align_of::<AtomicU32>() as u64)
                && field
                    .checked_add(size_of::<AtomicU32>() as u64)
                    .is_some_and(|end| end <= len as u64)
        };
        if !word_fits(offset.producer)
            || !word_fits(offset.consumer)
            || !word_fits(offset.flags)
            || !offset.desc.is_multiple_of(align_of::<T>() as u64)
        {
            return Err(XskError::InvalidRingOffsets);
        }
        let mmap = MMap::new(
            fd,
            len,
            PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_POPULATE,
            page_offset as off_t,
        )?;
        Ok(Self::from_mmap(
            mmap,
            offset.producer as usize,
            offset.consumer as usize,
            offset.flags as usize,
            offset.desc as usize,
            size,
        ))
    }

    const fn from_mmap(
        mmap: MMap,
        producer_offset: usize,
        consumer_offset: usize,
        flags_offset: usize,
        desc_offset: usize,
        size: u32,
    ) -> Self {
        Self {
            mmap,
            producer_offset,
            consumer_offset,
            flags_offset,
            desc_offset,
            size,
            _element: PhantomData,
        }
    }

    #[expect(
        clippy::cast_ptr_alignment,
        reason = "map validates that each word offset is aligned for AtomicU32"
    )]
    fn word(&self, offset: usize) -> &AtomicU32 {
        let ptr = self.mmap.ptr().as_ptr().cast::<u8>();
        // Safety: `map` validates that each word offset is aligned and in bounds. The mapping stays
        // alive for as long as `self`.
        unsafe { &*ptr.add(offset).cast::<AtomicU32>() }
    }

    const fn descs(&self) -> *mut T {
        let ptr = self.mmap.ptr().as_ptr().cast::<u8>();
        // Safety: `map` validates that the descriptor offset is aligned and the full descriptor
        // array is in bounds. The test constructor provides the same layout.
        unsafe { ptr.add(self.desc_offset).cast::<T>() }
    }

    fn producer(&self) -> &AtomicU32 {
        self.word(self.producer_offset)
    }

    fn consumer(&self) -> &AtomicU32 {
        self.word(self.consumer_offset)
    }

    /// Whether the kernel has signalled that it needs a wake-up (`sendto`/`recvfrom`) to make
    /// progress on this ring. Only meaningful when the socket was bound with
    /// [`XDP_USE_NEED_WAKEUP`](aya_obj::generated::XDP_USE_NEED_WAKEUP) (kernel 5.4+, where the ring
    /// `flags` word is present).
    pub(crate) fn needs_wakeup(&self) -> bool {
        let flags = self.word(self.flags_offset).load(Ordering::Relaxed);
        flags & XDP_RING_NEED_WAKEUP != 0
    }

    // --- Consumer side (we consume; the kernel produces): RX, COMPLETION ---

    /// Number of descriptors the kernel has produced that we have not yet released.
    pub(crate) fn available(&self) -> u32 {
        let producer = self.producer().load(Ordering::Acquire);
        let consumer = self.consumer().load(Ordering::Relaxed);
        producer.wrapping_sub(consumer).min(self.size)
    }

    /// Reads the descriptor `i` positions ahead of our consumer index.
    ///
    /// The caller must ensure `i < available()`.
    pub(crate) fn consumer_entry(&self, i: u32) -> T
    where
        T: Copy,
    {
        let consumer = self.consumer().load(Ordering::Relaxed);
        let idx = (consumer.wrapping_add(i) & (self.size - 1)) as usize;
        // Safety: `idx` is in bounds (`idx < size`) and the slot was produced by the kernel.
        unsafe { *self.descs().add(idx) }
    }

    /// Releases `n` consumed descriptors back to the kernel.
    pub(crate) fn release(&self, n: u32) {
        let consumer = self.consumer().load(Ordering::Relaxed);
        self.consumer()
            .store(consumer.wrapping_add(n), Ordering::Release);
    }

    // --- Producer side (we produce; the kernel consumes): FILL, TX ---

    /// Number of free slots we can produce into.
    pub(crate) fn free(&self) -> u32 {
        let producer = self.producer().load(Ordering::Relaxed);
        let consumer = self.consumer().load(Ordering::Acquire);
        // `saturating_sub` guards against a corrupt/desynced ring reporting more outstanding entries
        // than the ring holds, which would otherwise underflow (panic in debug, wrap in release).
        self.size.saturating_sub(producer.wrapping_sub(consumer))
    }

    /// Writes `value` into the slot `i` positions ahead of our producer index.
    ///
    /// The caller must ensure `i < free()`. Not visible to the kernel until [`submit`](Self::submit).
    pub(crate) fn set_producer_entry(&self, i: u32, value: T) {
        let producer = self.producer().load(Ordering::Relaxed);
        let idx = (producer.wrapping_add(i) & (self.size - 1)) as usize;
        // Safety: `idx` is in bounds (`idx < size`) and we own the producer slots until we submit
        // them.
        unsafe {
            *self.descs().add(idx) = value;
        }
    }

    /// Publishes `n` produced descriptors to the kernel.
    pub(crate) fn submit(&self, n: u32) {
        let producer = self.producer().load(Ordering::Relaxed);
        self.producer()
            .store(producer.wrapping_add(n), Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use std::os::fd::AsFd as _;

    use super::*;
    use crate::sys::TEST_MMAP_RET;

    impl<T> XskRing<T> {
        /// Builds a memory-mapped ring for tests, with the producer/consumer indices and descriptor
        /// array laid out at fixed cache-line-spaced offsets.
        fn new_test(size: u32) -> (Self, Box<[u64]>) {
            const PRODUCER_OFF: u64 = 0;
            const CONSUMER_OFF: u64 = 64;
            const FLAGS_OFF: u64 = 128;
            const DESC_OFF: u64 = 192;
            let len = DESC_OFF as usize + size as usize * size_of::<T>();
            let mut backing = vec![0u64; len.div_ceil(size_of::<u64>())].into_boxed_slice();
            TEST_MMAP_RET.with(|ret| *ret.borrow_mut() = backing.as_mut_ptr().cast());
            let file = tempfile::tempfile().unwrap();
            file.set_len(len as u64).unwrap();
            let mmap = MMap::new(file.as_fd(), len, PROT_READ | PROT_WRITE, MAP_SHARED, 0).unwrap();
            TEST_MMAP_RET.with(|ret| *ret.borrow_mut() = std::ptr::null_mut());
            (
                Self::from_mmap(
                    mmap,
                    PRODUCER_OFF as usize,
                    CONSUMER_OFF as usize,
                    FLAGS_OFF as usize,
                    DESC_OFF as usize,
                    size,
                ),
                backing,
            )
        }

        /// Writes the ring `flags` word, simulating the kernel raising/clearing need-wakeup.
        fn set_flags_for_test(&self, flags: u32) {
            self.word(self.flags_offset).store(flags, Ordering::Relaxed);
        }
    }

    #[test]
    fn produce_then_consume() {
        let (ring, _backing) = XskRing::<u64>::new_test(4);
        assert_eq!(ring.available(), 0);
        assert_eq!(ring.free(), 4);

        // Drive the ring as the producer (simulating the kernel for an RX ring).
        for i in 0..3 {
            ring.set_producer_entry(i, 100 + u64::from(i));
        }
        ring.submit(3);

        // Now consume.
        assert_eq!(ring.available(), 3);
        assert_eq!(ring.free(), 1);
        assert_eq!(ring.consumer_entry(0), 100);
        assert_eq!(ring.consumer_entry(2), 102);
        ring.release(3);
        assert_eq!(ring.available(), 0);
        assert_eq!(ring.free(), 4);
    }

    #[test]
    fn indices_wrap_around() {
        let (ring, _backing) = XskRing::<u64>::new_test(4);
        // Advance producer and consumer past `size` so both wrap the descriptor array.
        for round in 0..3u32 {
            for i in 0..4 {
                ring.set_producer_entry(i, u64::from(round) * 10 + u64::from(i));
            }
            ring.submit(4);
            assert_eq!(ring.available(), 4);
            // Slot for the first available entry is `consumer & mask`.
            assert_eq!(ring.consumer_entry(0), u64::from(round) * 10);
            assert_eq!(ring.consumer_entry(3), u64::from(round) * 10 + 3);
            ring.release(4);
            assert_eq!(ring.available(), 0);
        }
    }

    #[test]
    fn free_tracks_outstanding() {
        let (ring, _backing) = XskRing::<u64>::new_test(8);
        ring.set_producer_entry(0, 0xdead);
        ring.set_producer_entry(1, 0xbeef);
        ring.submit(2);
        // Two produced, none consumed: six slots free.
        assert_eq!(ring.free(), 6);
        assert_eq!(ring.available(), 2);
        ring.release(2);
        assert_eq!(ring.free(), 8);
    }

    #[test]
    fn needs_wakeup_reads_flags() {
        let (ring, _backing) = XskRing::<u64>::new_test(4);
        assert!(!ring.needs_wakeup());
        ring.set_flags_for_test(XDP_RING_NEED_WAKEUP);
        assert!(ring.needs_wakeup());
        ring.set_flags_for_test(0);
        assert!(!ring.needs_wakeup());
    }

    #[test]
    fn available_does_not_exceed_capacity() {
        let (ring, _backing) = XskRing::<u64>::new_test(4);
        ring.producer().store(9, Ordering::Release);
        assert_eq!(ring.available(), 4);
    }
}
