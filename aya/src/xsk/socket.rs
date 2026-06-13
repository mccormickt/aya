//! The `AF_XDP` socket itself: ties a UMEM to FILL/COMPLETION/RX rings and binds to a queue.

use std::os::fd::{AsFd as _, AsRawFd, BorrowedFd, RawFd};

use aya_obj::generated::{
    XDP_PGOFF_RX_RING, XDP_PGOFF_TX_RING, XDP_RX_RING, XDP_TX_RING, XDP_UMEM_COMPLETION_RING,
    XDP_UMEM_FILL_RING, XDP_UMEM_PGOFF_COMPLETION_RING, XDP_UMEM_PGOFF_FILL_RING,
    XDP_USE_NEED_WAKEUP, sockaddr_xdp, xdp_desc,
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
    /// Bind flags (`sockaddr_xdp::sxdp_flags`): e.g. `XDP_COPY`, `XDP_ZEROCOPY`,
    /// `XDP_USE_NEED_WAKEUP`. `0` lets the kernel choose copy or zero-copy mode.
    pub bind_flags: u16,
}

impl Default for XskSocketConfig {
    fn default() -> Self {
        Self {
            rx_size: 2048,
            tx_size: 2048,
            fill_size: 2048,
            completion_size: 2048,
            bind_flags: 0,
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
/// kernel, never both.
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

        let fd = xsk_socket()?;
        let borrowed = fd.as_fd();

        // The kernel requires this exact setup order: register the UMEM and size every ring before
        // reading the mmap offsets and binding.
        umem.register(borrowed)?;
        xsk_setsockopt(borrowed, XDP_UMEM_FILL_RING as i32, &fill_size)?;
        xsk_setsockopt(borrowed, XDP_UMEM_COMPLETION_RING as i32, &completion_size)?;
        xsk_setsockopt(borrowed, XDP_RX_RING as i32, &rx_size)?;
        xsk_setsockopt(borrowed, XDP_TX_RING as i32, &tx_size)?;

        let offsets = xsk_mmap_offsets(borrowed)?;
        let fill = XskRing::map(borrowed, fill_size, XDP_UMEM_PGOFF_FILL_RING, &offsets.fr)?;
        let completion = XskRing::map(
            borrowed,
            completion_size,
            XDP_UMEM_PGOFF_COMPLETION_RING,
            &offsets.cr,
        )?;
        let rx = XskRing::map(borrowed, rx_size, u64::from(XDP_PGOFF_RX_RING), &offsets.rx)?;
        let tx = XskRing::map(borrowed, tx_size, u64::from(XDP_PGOFF_TX_RING), &offsets.tx)?;

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
        if index >= self.rx.available() {
            return None;
        }
        let desc = self.rx.consumer_entry(index);
        // Safety: `desc` was produced by the kernel for this socket's UMEM. `frame_bytes` also
        // bounds-checks `addr`/`len` against the UMEM and returns `None` on a malformed descriptor.
        unsafe { self.umem.frame_bytes(desc.addr, desc.len) }
    }

    /// Releases up to `n` of the oldest received descriptors back to the kernel.
    ///
    /// `n` is clamped to the number currently available, so over-releasing cannot advance the
    /// consumer past the producer. Returns the number actually released. The released frames are not
    /// reusable until they are returned to the kernel via [`fill`](Self::fill).
    pub fn rx_release(&mut self, n: u32) -> u32 {
        let n = n.min(self.rx.available());
        self.rx.release(n);
        n
    }

    /// Submits UMEM frames to the FILL ring so the kernel can receive packets into them.
    ///
    /// Returns the number of frames actually submitted, which may be fewer than supplied if the
    /// FILL ring fills up.
    pub fn fill(&mut self, frame_indices: impl IntoIterator<Item = u32>) -> Result<u32, XskError> {
        let free = self.fill.free();
        let mut submitted = 0;
        for index in frame_indices {
            if submitted >= free {
                break;
            }
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
    pub fn wake_rx(&mut self) -> Result<(), XskError> {
        if self.uses_need_wakeup && self.fill.needs_wakeup() {
            xsk_wake_rx(self.fd.as_fd())?;
        }
        Ok(())
    }

    /// Returns a mutable view of the frame at UMEM offset `addr` for `len` bytes, into which a
    /// packet to transmit can be written, or `None` if `[addr, addr + len)` is outside the UMEM.
    ///
    /// The frame must be one the caller currently owns (not enqueued in the FILL or TX rings). Use
    /// [`XskUmem::frame_addr`] to turn a frame index into an `addr`.
    pub fn tx_frame_mut(&mut self, addr: u64, len: u32) -> Option<&mut [u8]> {
        // Safety: `&mut self` proves no other access to the UMEM from our side, and the caller
        // guarantees this frame is not currently published to the kernel.
        unsafe { self.umem.frame_bytes_mut(addr, len) }
    }

    /// Enqueues frames for transmission on the TX ring.
    ///
    /// Each item is a `(addr, len)` pair describing a frame previously written via
    /// [`tx_frame_mut`](Self::tx_frame_mut). Returns the number of descriptors actually enqueued,
    /// which may be fewer than supplied if the TX ring fills up. Call [`kick`](Self::kick)
    /// afterwards so the kernel processes them.
    pub fn transmit(
        &mut self,
        descs: impl IntoIterator<Item = (u64, u32)>,
    ) -> Result<u32, XskError> {
        let free = self.tx.free();
        let mut submitted = 0;
        for (addr, len) in descs {
            if submitted >= free {
                break;
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
    pub fn kick(&mut self) -> Result<(), XskError> {
        if self.uses_need_wakeup && !self.tx.needs_wakeup() {
            return Ok(());
        }
        xsk_kick_tx(self.fd.as_fd())?;
        Ok(())
    }

    /// Drains the COMPLETION ring, invoking `f` with the UMEM `addr` of each transmitted frame the
    /// kernel has finished with, and returns the number of frames reclaimed.
    ///
    /// The caller regains ownership of these frames and can reuse them for TX or the FILL ring.
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
