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
    os::fd::BorrowedFd,
    ptr::NonNull,
    sync::atomic::{AtomicU32, Ordering},
};

use aya_obj::generated::{XDP_RING_NEED_WAKEUP, xdp_ring_offset};
use libc::{MAP_POPULATE, MAP_SHARED, PROT_READ, PROT_WRITE, off_t};

use super::XskError;
use crate::util::MMap;

/// Owns the memory backing a ring so that the raw pointers below stay valid.
#[expect(
    dead_code,
    reason = "the backing is held only to keep the memory mapped; its bytes are accessed through the raw pointers and it is unmapped/freed on Drop"
)]
enum Backing {
    Mmap(MMap),
    #[cfg(test)]
    Heap(Box<[u64]>),
}

/// A producer/consumer ring shared with the kernel.
///
/// `T` is the descriptor element: `u64` (a UMEM frame address) for the FILL and COMPLETION rings,
/// or [`xdp_desc`](aya_obj::generated::xdp_desc) for the RX and TX rings.
pub(crate) struct XskRing<T> {
    _backing: Backing,
    producer: NonNull<AtomicU32>,
    consumer: NonNull<AtomicU32>,
    flags: NonNull<AtomicU32>,
    descs: NonNull<T>,
    /// Number of descriptor slots; always a power of two, so the slot for index `i` is
    /// `i & (size - 1)`.
    size: u32,
}

// Safety: a ring is owned by a single `XskSocket` and its methods take `&self`/`&mut self`, so
// there is no concurrent access from multiple threads on our side. The kernel accesses the shared
// memory concurrently but only through the atomic producer/consumer indices, which are read/written
// with the ordering documented above. This matches the justification for `MMap: Send`. The `T: Send`
// bound ensures the descriptor array is only sent across threads when its contents are themselves
// `Send` (the concrete `u64`/`xdp_desc` instantiations are).
unsafe impl<T: Send> Send for XskRing<T> {}

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
        let len = (offset.desc + u64::from(size) * size_of::<T>() as u64) as usize;
        let mmap = MMap::new(
            fd,
            len,
            PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_POPULATE,
            page_offset as off_t,
        )?;
        let base = mmap.ptr().as_ptr().cast::<u8>();
        // Safety: `base` is valid for `len` bytes (the mapping we just created), the offsets come
        // from the kernel and lie within it, and `Backing::Mmap` keeps the mapping alive.
        Ok(unsafe {
            Self::from_raw(
                Backing::Mmap(mmap),
                base,
                offset.producer,
                offset.consumer,
                offset.flags,
                offset.desc,
                size,
            )
        })
    }

    /// Builds a ring from a backing allocation and the byte offsets of its fields.
    ///
    /// # Safety
    ///
    /// `base` must be valid for `desc_off + size * size_of::<T>()` bytes, the producer/consumer/flags
    /// offsets must point at correctly aligned `u32`s within it, and `backing` must own that memory
    /// for the lifetime of the returned ring.
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the kernel places the producer/consumer/flags words at correctly aligned offsets (and the test backing is 8-byte aligned), as required by the safety contract"
    )]
    const unsafe fn from_raw(
        backing: Backing,
        base: *mut u8,
        producer_off: u64,
        consumer_off: u64,
        flags_off: u64,
        desc_off: u64,
        size: u32,
    ) -> Self {
        // Safety: the offsets lie within `base`'s allocation per the caller's contract.
        let producer = unsafe { base.add(producer_off as usize) }.cast::<AtomicU32>();
        let consumer = unsafe { base.add(consumer_off as usize) }.cast::<AtomicU32>();
        let flags = unsafe { base.add(flags_off as usize) }.cast::<AtomicU32>();
        let descs = unsafe { base.add(desc_off as usize) }.cast::<T>();
        Self {
            _backing: backing,
            producer: NonNull::new(producer).unwrap(),
            consumer: NonNull::new(consumer).unwrap(),
            flags: NonNull::new(flags).unwrap(),
            descs: NonNull::new(descs).unwrap(),
            size,
        }
    }

    const fn producer(&self) -> &AtomicU32 {
        // Safety: valid, aligned, and alive for as long as `self`.
        unsafe { self.producer.as_ref() }
    }

    const fn consumer(&self) -> &AtomicU32 {
        // Safety: valid, aligned, and alive for as long as `self`.
        unsafe { self.consumer.as_ref() }
    }

    /// Whether the kernel has signalled that it needs a wake-up (`sendto`/`recvfrom`) to make
    /// progress on this ring. Only meaningful when the socket was bound with
    /// [`XDP_USE_NEED_WAKEUP`](aya_obj::generated::XDP_USE_NEED_WAKEUP) (kernel 5.4+, where the ring
    /// `flags` word is present).
    pub(crate) fn needs_wakeup(&self) -> bool {
        // Safety: valid, aligned, and alive for as long as `self`.
        let flags = unsafe { self.flags.as_ref() }.load(Ordering::Relaxed);
        flags & XDP_RING_NEED_WAKEUP != 0
    }

    // --- Consumer side (we consume; the kernel produces): RX, COMPLETION ---

    /// Number of descriptors the kernel has produced that we have not yet released.
    pub(crate) fn available(&self) -> u32 {
        let producer = self.producer().load(Ordering::Acquire);
        let consumer = self.consumer().load(Ordering::Relaxed);
        producer.wrapping_sub(consumer)
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
        unsafe { *self.descs.as_ptr().add(idx) }
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
            *self.descs.as_ptr().add(idx) = value;
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
    use super::*;

    impl<T> XskRing<T> {
        /// Builds a heap-backed ring for tests, with the producer/consumer indices and descriptor
        /// array laid out at fixed (cache-line-spaced) offsets. The `u64` backing guarantees both
        /// `AtomicU32` (4-byte) and any `T` up to 8-byte alignment.
        fn new_test(size: u32) -> Self {
            const PRODUCER_OFF: u64 = 0;
            const CONSUMER_OFF: u64 = 64;
            const FLAGS_OFF: u64 = 128;
            const DESC_OFF: u64 = 192;
            let len = DESC_OFF as usize + size as usize * size_of::<T>();
            let mut backing: Box<[u64]> = vec![0u64; len.div_ceil(8)].into_boxed_slice();
            let base = backing.as_mut_ptr().cast::<u8>();
            // Safety: `base` is 8-byte aligned and valid for `len` bytes; `Backing::Heap` keeps it
            // alive.
            unsafe {
                Self::from_raw(
                    Backing::Heap(backing),
                    base,
                    PRODUCER_OFF,
                    CONSUMER_OFF,
                    FLAGS_OFF,
                    DESC_OFF,
                    size,
                )
            }
        }

        /// Writes the ring `flags` word, simulating the kernel raising/clearing need-wakeup.
        fn set_flags_for_test(&self, flags: u32) {
            // Safety: valid, aligned, alive for as long as `self`.
            unsafe { self.flags.as_ref() }.store(flags, Ordering::Relaxed);
        }
    }

    #[test]
    fn produce_then_consume() {
        let ring = XskRing::<u64>::new_test(4);
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
        let ring = XskRing::<u64>::new_test(4);
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
        let ring = XskRing::<u64>::new_test(8);
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
        let ring = XskRing::<u64>::new_test(4);
        assert!(!ring.needs_wakeup());
        ring.set_flags_for_test(XDP_RING_NEED_WAKEUP);
        assert!(ring.needs_wakeup());
        ring.set_flags_for_test(0);
        assert!(!ring.needs_wakeup());
    }
}
