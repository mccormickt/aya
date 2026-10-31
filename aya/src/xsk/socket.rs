//! The `AF_XDP` socket itself: ties a UMEM to FILL/COMPLETION/RX rings and binds to a queue.

use std::os::fd::{AsFd as _, AsRawFd, BorrowedFd, RawFd};

use aya_obj::generated::{
    XDP_COPY, XDP_PGOFF_RX_RING, XDP_PGOFF_TX_RING, XDP_RX_RING, XDP_TX_RING,
    XDP_UMEM_COMPLETION_RING, XDP_UMEM_FILL_RING, XDP_UMEM_PGOFF_COMPLETION_RING,
    XDP_UMEM_PGOFF_FILL_RING, XDP_USE_NEED_WAKEUP, XDP_ZEROCOPY, sockaddr_xdp, xdp_desc,
};
use libc::AF_XDP;

use super::{XskError, XskRing, XskUmem};
use crate::sys::{
    xsk_bind, xsk_kick_tx, xsk_mmap_offsets, xsk_setsockopt, xsk_socket, xsk_wake_rx,
};

/// Configuration for a [`XskSocket`].
///
/// Ring sizes are in number of descriptors and must each be a power of two; they are independent of
/// the UMEM's frame count.
#[derive(Clone, Copy, Debug)]
pub struct XskSocketConfig {
    /// Size of the RX ring.
    pub rx_size: u32,
    /// Size of the TX ring.
    pub tx_size: u32,
    /// Size of the FILL ring.
    pub fill_size: u32,
    /// Size of the COMPLETION ring. Required by the kernel at bind time even for RX-only sockets.
    pub completion_size: u32,
    /// Exactly one of `XDP_COPY` or `XDP_ZEROCOPY`, optionally with `XDP_USE_NEED_WAKEUP`.
    /// Shared UMEM, multi-buffer packets, and automatic mode selection are unsupported.
    pub bind_flags: u16,
}

/// A received frame and its aligned UMEM frame address.
#[derive(Clone, Copy, Debug)]
pub struct XskRxFrame<'a> {
    /// The aligned UMEM frame address that identifies ownership of this frame.
    pub frame_addr: u64,
    /// Packet bytes described by the RX descriptor.
    pub data: &'a [u8],
}

/// Result of an explicit kernel wake-up request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum XskWakeup {
    /// The ring did not request a wake-up, so no syscall was made.
    NotNeeded,
    /// The wake-up syscall completed successfully. Ring progress must still be observed separately.
    Woken,
    /// The nonblocking syscall returned a transient error; retry after checking ring progress.
    Retry,
}

impl Default for XskSocketConfig {
    fn default() -> Self {
        Self {
            rx_size: 2048,
            tx_size: 2048,
            fill_size: 2048,
            completion_size: 2048,
            bind_flags: XDP_COPY as u16,
        }
    }
}

/// An `AF_XDP` socket bound to a single netdev queue.
///
/// To receive, an XDP program must redirect packets into a [`crate::maps::XskMap`] whose entry for
/// the bound queue holds this socket's raw file descriptor. `XskSocket` implements [`AsRawFd`], so
/// register it with [`xskmap.set(queue, socket.as_raw_fd(), 0)`](crate::maps::XskMap::set).
///
/// Frame ownership is the caller's responsibility: a frame is owned by either the caller or the
/// kernel, never both. Methods that cannot verify this ownership are unsafe.
///
/// - **RX:** hand empty frames to the kernel with [`fill`](Self::fill); read received packets with
///   [`rx_peek`](Self::rx_peek) and return their descriptors with [`rx_release`](Self::rx_release).
/// - **TX:** write a frame with [`tx_frame_mut`](Self::tx_frame_mut), enqueue it with
///   [`transmit`](Self::transmit), wake the kernel with [`kick`](Self::kick), then reclaim sent
///   frames with [`complete`](Self::complete).
///
/// # Zero-copy
///
/// Aligned-chunk zero-copy works through the same rings as copy mode: bind with
/// `XDP_ZEROCOPY | XDP_USE_NEED_WAKEUP` (via [`XskSocketConfig::bind_flags`]) on a ZC-capable
/// driver — `lo` is not one, and `bind` returns `EOPNOTSUPP` on devices without ZC support. In this
/// mode the driver must be woken after refilling the FILL ring, so call [`wake_rx`](Self::wake_rx)
/// after [`fill`](Self::fill) (just as [`kick`](Self::kick) follows [`transmit`](Self::transmit)),
/// or RX silently stalls. Unaligned chunks and shared UMEM are not yet implemented.
pub struct XskSocket {
    fd: crate::MockableFd,
    umem: XskUmem,
    fill: XskRing<u64>,
    completion: XskRing<u64>,
    rx: XskRing<xdp_desc>,
    tx: XskRing<xdp_desc>,
    /// Whether the socket was bound with `XDP_USE_NEED_WAKEUP`; when set, [`kick`](Self::kick) only
    /// wakes the kernel if the TX ring requests it.
    uses_need_wakeup: bool,
}

impl XskSocket {
    /// Creates an `AF_XDP` socket over `umem`, sets up the FILL/COMPLETION/RX rings, and binds to
    /// `(ifindex, queue_id)`.
    ///
    /// `ifindex` can be obtained from an interface name with `libc::if_nametoindex`.
    pub fn new(
        umem: XskUmem,
        ifindex: u32,
        queue_id: u32,
        config: XskSocketConfig,
    ) -> Result<Self, XskError> {
        let XskSocketConfig {
            rx_size,
            tx_size,
            fill_size,
            completion_size,
            bind_flags,
        } = config;
        for size in [rx_size, tx_size, fill_size, completion_size] {
            if !size.is_power_of_two() {
                return Err(XskError::InvalidRingSize);
            }
        }
        let mode = u32::from(bind_flags) & (XDP_COPY | XDP_ZEROCOPY);
        if !matches!(mode, XDP_COPY | XDP_ZEROCOPY)
            || u32::from(bind_flags) & !(XDP_COPY | XDP_ZEROCOPY | XDP_USE_NEED_WAKEUP) != 0
        {
            return Err(XskError::InvalidBindFlags);
        }

        let fd = xsk_socket()?;
        let borrowed = fd.as_fd();

        // The kernel requires this exact setup order: register the UMEM and size every ring before
        // reading the mmap offsets and binding.
        umem.register(borrowed)?;
        xsk_setsockopt(borrowed, XDP_UMEM_FILL_RING as i32, &fill_size)?;
        xsk_setsockopt(borrowed, XDP_UMEM_COMPLETION_RING as i32, &completion_size)?;
        xsk_setsockopt(borrowed, XDP_RX_RING as i32, &rx_size)?;
        xsk_setsockopt(borrowed, XDP_TX_RING as i32, &tx_size)?;

        let (offsets, has_flags) = xsk_mmap_offsets(borrowed)?;
        let fill = XskRing::map(
            borrowed,
            fill_size,
            XDP_UMEM_PGOFF_FILL_RING,
            &offsets.fr,
            has_flags,
        )?;
        let completion = XskRing::map(
            borrowed,
            completion_size,
            XDP_UMEM_PGOFF_COMPLETION_RING,
            &offsets.cr,
            has_flags,
        )?;
        let rx = XskRing::map(
            borrowed,
            rx_size,
            u64::from(XDP_PGOFF_RX_RING),
            &offsets.rx,
            has_flags,
        )?;
        let tx = XskRing::map(
            borrowed,
            tx_size,
            u64::from(XDP_PGOFF_TX_RING),
            &offsets.tx,
            has_flags,
        )?;

        // Safety: `sockaddr_xdp` is composed of integers, so a zeroed value is valid.
        let mut addr = unsafe { std::mem::zeroed::<sockaddr_xdp>() };
        addr.sxdp_family = AF_XDP as u16;
        addr.sxdp_flags = bind_flags;
        addr.sxdp_ifindex = ifindex;
        addr.sxdp_queue_id = queue_id;
        xsk_bind(borrowed, &addr)?;

        Ok(Self {
            fd,
            umem,
            fill,
            completion,
            rx,
            tx,
            uses_need_wakeup: bind_flags & XDP_USE_NEED_WAKEUP as u16 != 0,
        })
    }

    /// The UMEM registered on this socket.
    pub const fn umem(&self) -> &XskUmem {
        &self.umem
    }

    /// Number of received packets available to read.
    pub fn rx_available(&self) -> u32 {
        self.rx.available()
    }

    /// Returns the bytes of the `index`-th currently-available received packet without consuming
    /// it, or `None` if `index >= rx_available()`.
    ///
    /// The returned slice borrows the UMEM and is valid until the frame is released and refilled.
    pub fn rx_peek(&self, index: u32) -> Option<&[u8]> {
        self.rx_frame(index).ok().flatten().map(|frame| frame.data)
    }

    /// Returns the `index`-th received packet and its aligned UMEM frame address.
    ///
    /// The address identifies the exact frame returned by the kernel; RX completion order is not
    /// required to match FILL submission order. Empty descriptors, multi-buffer descriptors, and
    /// descriptors that cross an aligned UMEM frame boundary are rejected.
    ///
    /// This method trusts the kernel to return only frames transferred through [`fill`](Self::fill).
    /// Use [`rx_frame_checked`](Self::rx_frame_checked) to check a caller-owned frame ledger before
    /// borrowing packet bytes.
    pub fn rx_frame(&self, index: u32) -> Result<Option<XskRxFrame<'_>>, XskError> {
        self.rx_frame_checked(index, |_| true)
    }

    /// Checks a received frame's identity before borrowing its packet bytes.
    ///
    /// After validating the descriptor range and options, calls `validate` with the normalized
    /// aligned frame address. Returning `false` rejects the descriptor with
    /// [`XskError::InvalidRxDescriptor`], without creating a UMEM slice or consuming the entry.
    /// The callback is not called for an unavailable or malformed descriptor.
    ///
    /// This check can enforce RX pool membership and reject frames already returned to userspace.
    /// Like [`rx_frame`](Self::rx_frame), it relies on the kernel to stop writing a returned frame.
    pub fn rx_frame_checked(
        &self,
        index: u32,
        validate: impl FnOnce(u64) -> bool,
    ) -> Result<Option<XskRxFrame<'_>>, XskError> {
        if index >= self.rx.available() {
            return Ok(None);
        }
        let desc = self.rx.consumer_entry(index);
        if desc.len == 0
            || desc.options != 0
            || !self.umem.contains_frame_range(desc.addr, desc.len)
        {
            return Err(XskError::InvalidRxDescriptor);
        }
        let frame_addr = desc.addr & !(u64::from(self.umem.frame_size()) - 1);
        if !validate(frame_addr) {
            return Err(XskError::InvalidRxDescriptor);
        }
        // Safety: the descriptor came from the kernel. `frame_bytes` validates that its range lies
        // inside one aligned UMEM frame before creating the slice.
        let data = unsafe { self.umem.frame_bytes(desc.addr, desc.len) }
            .ok_or(XskError::InvalidRxDescriptor)?;
        Ok(Some(XskRxFrame { frame_addr, data }))
    }

    /// Releases up to `n` of the oldest received descriptors back to the kernel.
    ///
    /// `n` is clamped to the number currently available, so over-releasing cannot advance the
    /// consumer past the producer. Returns the number actually released. Once all descriptors for a
    /// frame are released, its caller may reuse it for TX or return it through [`fill`](Self::fill).
    pub fn rx_release(&mut self, n: u32) -> u32 {
        let n = n.min(self.rx.available());
        self.rx.release(n);
        n
    }

    /// Submits UMEM frames to the FILL ring so the kernel can receive packets into them.
    ///
    /// Returns the number of frames actually submitted, which may be fewer than supplied if the
    /// FILL ring fills up. Only that prefix is consumed from the iterator. A full ring consumes no
    /// items. On error or iterator panic, no frames are published, including valid earlier items.
    ///
    /// # Safety
    ///
    /// Every consumed, in-range frame index must identify a distinct frame exclusively owned by
    /// the caller.
    /// A frame must not be present in the FILL or TX ring, contain an RX packet that has not been
    /// released, or await TX completion. Ownership of each successfully submitted frame transfers
    /// to the kernel and remains there until an RX descriptor returns it.
    ///
    /// ```compile_fail,E0133
    /// # use aya::xsk::{XskError, XskSocket};
    /// # fn submit(socket: &mut XskSocket) -> Result<(), XskError> {
    /// socket.fill([0])?;
    /// # Ok(())
    /// # }
    /// ```
    pub unsafe fn fill(
        &mut self,
        frame_indices: impl IntoIterator<Item = u32>,
    ) -> Result<u32, XskError> {
        let free = self.fill.free();
        let mut submitted = 0;
        for index in frame_indices.into_iter().take(free as usize) {
            let addr = self.umem.frame_addr(index)?;
            self.fill.set_producer_entry(submitted, addr);
            submitted += 1;
        }
        self.fill.submit(submitted);
        Ok(submitted)
    }

    /// Wakes the kernel to refill RX from the FILL ring.
    ///
    /// Only needed in zero-copy + `XDP_USE_NEED_WAKEUP` mode, where the driver must be woken after
    /// [`fill`](Self::fill) or RX stalls. This is a no-op unless the socket was bound with
    /// `XDP_USE_NEED_WAKEUP` and the FILL ring has flagged that it needs a wake-up; in copy mode it
    /// never issues a syscall. (Unlike [`kick`](Self::kick), which must always fire to drive
    /// copy-mode TX.)
    pub fn wake_rx(&mut self) -> Result<XskWakeup, XskError> {
        if self.uses_need_wakeup && self.fill.needs_wakeup() {
            return Ok(if xsk_wake_rx(self.fd.as_fd())? {
                XskWakeup::Woken
            } else {
                XskWakeup::Retry
            });
        }
        Ok(XskWakeup::NotNeeded)
    }

    /// Returns a mutable view of the frame at UMEM offset `addr` for `len` bytes, into which a
    /// packet to transmit can be written, or `None` if `[addr, addr + len)` is outside the UMEM.
    ///
    /// Use [`XskUmem::frame_addr`] to turn a frame index into an `addr`.
    ///
    /// # Safety
    ///
    /// The entire range must lie in one frame exclusively owned by the caller. The frame must not
    /// be present in the FILL or TX ring, contain an RX packet that has not been released, or await
    /// TX completion. No other reference may access the range while the returned slice is alive.
    ///
    /// ```compile_fail,E0133
    /// # use aya::xsk::XskSocket;
    /// # fn write(socket: &mut XskSocket) {
    /// let _ = socket.tx_frame_mut(0, 64);
    /// # }
    /// ```
    pub unsafe fn tx_frame_mut(&mut self, addr: u64, len: u32) -> Option<&mut [u8]> {
        // Safety: the caller guarantees exclusive ownership of this frame and `&mut self` prevents
        // another access through this socket while the returned slice is alive.
        unsafe { self.umem.frame_bytes_mut(addr, len) }
    }

    /// Enqueues frames for transmission on the TX ring.
    ///
    /// Each item is a `(addr, len)` pair describing a frame previously written via
    /// [`tx_frame_mut`](Self::tx_frame_mut). Returns the number of descriptors actually enqueued,
    /// which may be fewer than supplied if the TX ring fills up. Call [`kick`](Self::kick)
    /// afterwards so the kernel processes them. Only the returned prefix is consumed from the
    /// iterator. A full ring consumes no items. On error or iterator panic, nothing is published.
    ///
    /// # Safety
    ///
    /// Every consumed descriptor must identify a distinct frame exclusively owned by the caller.
    /// The described frames must not be present in the FILL or TX ring, contain an RX packet that
    /// has not been released, or await TX completion. Ownership of the first `n` frames transfers
    /// to the kernel when this method returns `Ok(n)`; the caller must not access them until matching
    /// and independently validated completion entries return ownership.
    ///
    /// ```compile_fail,E0133
    /// # use aya::xsk::{XskError, XskSocket};
    /// # fn submit(socket: &mut XskSocket) -> Result<(), XskError> {
    /// socket.transmit([(0, 64)])?;
    /// # Ok(())
    /// # }
    /// ```
    pub unsafe fn transmit(
        &mut self,
        descs: impl IntoIterator<Item = (u64, u32)>,
    ) -> Result<u32, XskError> {
        let free = self.tx.free();
        let mut submitted = 0;
        for (addr, len) in descs.into_iter().take(free as usize) {
            if len == 0 || !self.umem.contains_frame_range(addr, len) {
                return Err(XskError::InvalidTxDescriptor);
            }
            self.tx.set_producer_entry(
                submitted,
                xdp_desc {
                    addr,
                    len,
                    options: 0,
                },
            );
            submitted += 1;
        }
        self.tx.submit(submitted);
        Ok(submitted)
    }

    /// Wakes the kernel to process the TX ring.
    ///
    /// When the socket was bound with `XDP_USE_NEED_WAKEUP`, this is a no-op unless the kernel has
    /// flagged that it needs a wake-up; otherwise it always issues the wake-up syscall.
    pub fn kick(&mut self) -> Result<XskWakeup, XskError> {
        if self.uses_need_wakeup && !self.tx.needs_wakeup() {
            return Ok(XskWakeup::NotNeeded);
        }
        Ok(if xsk_kick_tx(self.fd.as_fd())? {
            XskWakeup::Woken
        } else {
            XskWakeup::Retry
        })
    }

    /// Drains the COMPLETION ring, invoking `f` with the UMEM `addr` of each transmitted frame the
    /// kernel has finished with, and returns the number of frames reclaimed.
    ///
    /// Completion addresses come from a shared kernel ring and are not validated by this method.
    /// Before treating a frame as caller-owned, the callback must verify that its address identifies
    /// a distinct, outstanding TX submission. The frame can then be reused for TX or the FILL ring.
    pub fn complete(&mut self, mut f: impl FnMut(u64)) -> u32 {
        let n = self.completion.available();
        for i in 0..n {
            f(self.completion.consumer_entry(i));
        }
        self.completion.release(n);
        n
    }
}

impl AsRawFd for XskSocket {
    fn as_raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }
}

impl std::os::fd::AsFd for XskSocket {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }
}

#[cfg(test)]
mod tests {
    use std::{os::fd::FromRawFd as _, ptr::NonNull};

    use super::*;
    use crate::{MockableFd, xsk::XskUmemConfig};

    #[repr(C, align(4096))]
    struct Memory([u8; 4 * 4096]);

    fn with_socket(test: impl FnOnce(&mut XskSocket)) {
        let mut memory = Box::new(Memory([0x5a; 4 * 4096]));
        let (fill, _fill_memory) = XskRing::new_test(4);
        let (completion, _completion_memory) = XskRing::new_test(4);
        let (rx, _rx_memory) = XskRing::new_test(4);
        let (tx, _tx_memory) = XskRing::new_test(4);
        // Safety: the initialized allocation outlives the socket and is never registered.
        let umem = unsafe {
            XskUmem::new(
                XskUmemConfig::default(),
                NonNull::from(memory.0.as_mut_slice()),
            )
            .unwrap()
        };
        let mut socket = XskSocket {
            // SAFETY: these tests use only ring memory; the mock descriptor is not closed.
            fd: unsafe { MockableFd::from_raw_fd(MockableFd::mock_signed_fd()) },
            umem,
            fill,
            completion,
            rx,
            tx,
            uses_need_wakeup: true,
        };
        test(&mut socket);
    }

    #[test]
    fn fill_publishes_only_a_valid_prefix() {
        with_socket(|socket| {
            let mut invalid = [2, 8, 3].into_iter();
            // Safety: all in-range frames are distinct and caller-owned.
            assert!(matches!(
                unsafe { socket.fill(&mut invalid) },
                Err(XskError::FrameOutOfBounds(8))
            ));
            assert_eq!(invalid.next(), Some(3));
            assert_eq!(socket.fill.available(), 0);

            let mut frames = [3, 1, 2, 0, 7].into_iter();
            // Safety: these frames are distinct and caller-owned.
            assert_eq!(unsafe { socket.fill(&mut frames) }.unwrap(), 4);
            assert_eq!(unsafe { socket.fill(&mut frames) }.unwrap(), 0);
            assert_eq!(frames.next(), Some(7));
            assert_eq!(socket.fill.consumer_entry(0), 3 * 4096);
            assert_eq!(socket.fill.consumer_entry(3), 0);
        });
    }

    #[test]
    fn transmit_validates_before_publication() {
        with_socket(|socket| {
            for invalid in [(4096, 0), (8191, 2), (u64::MAX, 1), (8 * 4096, 1)] {
                // Safety: no frame has been transferred to a kernel ring.
                assert!(matches!(
                    unsafe { socket.transmit([(256, 16), invalid]) },
                    Err(XskError::InvalidTxDescriptor)
                ));
                assert_eq!(socket.tx.available(), 0);
            }
            let mut descs = [
                (256, 16),
                (4096, 32),
                (8192, 64),
                (12288, 128),
                (16384, 256),
            ]
            .into_iter();
            // Safety: all frames are distinct and caller-owned.
            assert_eq!(unsafe { socket.transmit(&mut descs) }.unwrap(), 4);
            assert_eq!(unsafe { socket.transmit(&mut descs) }.unwrap(), 0);
            assert_eq!(descs.next(), Some((16384, 256)));
            let first = socket.tx.consumer_entry(0);
            let last = socket.tx.consumer_entry(3);
            assert_eq!((first.addr, first.len, first.options), (256, 16, 0));
            assert_eq!((last.addr, last.len, last.options), (12288, 128, 0));
        });
    }

    #[test]
    fn rx_checks_range_options_and_identity() {
        with_socket(|socket| {
            for (addr, len, options) in [
                (4095, 2, 0),
                (32768, 1, 0),
                (0, 0, 0),
                (0, 1, 1),
                (u64::MAX, 2, 0),
            ] {
                socket
                    .rx
                    .set_producer_entry(0, xdp_desc { addr, len, options });
                socket.rx.submit(1);
                let mut checked = false;
                assert!(matches!(
                    socket.rx_frame_checked(0, |_| {
                        checked = true;
                        true
                    }),
                    Err(XskError::InvalidRxDescriptor)
                ));
                assert!(!checked);
                assert_eq!(socket.rx_release(1), 1);
            }
            socket.rx.set_producer_entry(
                0,
                xdp_desc {
                    addr: 4096 + 320,
                    len: 7,
                    options: 0,
                },
            );
            socket.rx.submit(1);
            assert!(matches!(
                socket.rx_frame_checked(0, |addr| addr == 0),
                Err(XskError::InvalidRxDescriptor)
            ));
            let frame = socket
                .rx_frame_checked(0, |addr| addr == 4096)
                .unwrap()
                .unwrap();
            assert_eq!(frame.frame_addr, 4096);
            assert_eq!(frame.data, &[0x5a; 7]);
            assert_eq!(socket.rx_release(u32::MAX), 1);
            assert_eq!(socket.rx_available(), 0);
            let mut checked = false;
            assert!(
                socket
                    .rx_frame_checked(0, |_| {
                        checked = true;
                        true
                    })
                    .unwrap()
                    .is_none()
            );
            assert!(!checked);
        });
    }

    #[test]
    fn completion_drains_raw_addresses_in_order() {
        with_socket(|socket| {
            let entries = [8192, u64::MAX, 4096, 8192];
            for (i, entry) in entries.into_iter().enumerate() {
                socket.completion.set_producer_entry(i as u32, entry);
            }
            socket.completion.submit(4);
            let mut seen = Vec::new();
            assert_eq!(socket.complete(|addr| seen.push(addr)), 4);
            assert_eq!(socket.complete(|addr| seen.push(addr)), 0);
            assert_eq!(seen, entries);
        });
    }

    #[test]
    fn unsupported_bind_flags_fail_before_socket_setup() {
        for bind_flags in [
            0,
            XDP_COPY | XDP_ZEROCOPY,
            XDP_COPY | 1,
            XDP_COPY | 16,
            XDP_COPY | 32,
        ] {
            let mut memory = Box::new(Memory([0; 4 * 4096]));
            // Safety: the initialized allocation outlives the rejected socket setup.
            let umem = unsafe {
                XskUmem::new(
                    XskUmemConfig::default(),
                    NonNull::from(memory.0.as_mut_slice()),
                )
                .unwrap()
            };
            let config = XskSocketConfig {
                bind_flags: bind_flags as u16,
                ..XskSocketConfig::default()
            };
            assert!(matches!(
                XskSocket::new(umem, 0, 0, config),
                Err(XskError::InvalidBindFlags)
            ));
        }
    }
}
