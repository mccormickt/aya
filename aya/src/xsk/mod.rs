//! `AF_XDP` (XSK) sockets.
//!
//! `AF_XDP` is a high-performance address family that lets userspace receive and transmit raw
//! frames directly through a shared memory region (a [`XskUmem`]), bypassing the kernel network
//! stack. An XDP program redirects selected packets into a [`crate::maps::XskMap`], which delivers
//! them to the bound [`XskSocket`].
//!
//! Setup, end to end:
//!
//! 1. Allocate page-aligned memory and wrap it in a [`XskUmem`].
//! 2. Create a [`XskSocket`] with [`XskSocket::new`], binding it to an interface queue.
//! 3. Register the socket's file descriptor in a [`crate::maps::XskMap`] for the bound queue with
//!    [`XskMap::set`](crate::maps::XskMap::set).
//! 4. Load and attach an XDP program that redirects into that map.
//! 5. For RX, submit frames with [`XskSocket::fill`] and read packets with
//!    [`XskSocket::rx_peek`] / [`XskSocket::rx_release`]. For TX, write a frame with
//!    [`XskSocket::tx_frame_mut`], enqueue it with [`XskSocket::transmit`], wake the kernel with
//!    [`XskSocket::kick`], and reclaim sent frames with [`XskSocket::complete`].
//!
//! # Scope
//!
//! This implementation covers one RX/TX socket with aligned chunks and explicit copy or zero-copy
//! mode, including `XDP_USE_NEED_WAKEUP`. Zero-copy requires a compatible driver (see [`XskSocket`]
//! and [`wake_rx`](XskSocket::wake_rx)). Shared UMEM, unaligned chunks, multi-buffer packets, TX
//! metadata, and automatic mode selection are rejected.
//!
//! # Minimum kernel version
//!
//! The minimum kernel version required to use `AF_XDP` is 4.18.

mod ring;
mod socket;
mod umem;

pub(crate) use ring::XskRing;
pub use socket::{XskRxFrame, XskSocket, XskSocketConfig, XskWakeup};
pub use umem::{XskUmem, XskUmemConfig};

/// Errors that can occur while setting up or using an `AF_XDP` socket.
#[derive(Debug, thiserror::Error)]
pub enum XskError {
    /// A system call failed.
    #[error(transparent)]
    Syscall(#[from] crate::sys::SyscallError),
    /// Invalid UMEM alignment, length, frame size, headroom, or flags.
    #[error("invalid UMEM alignment, length, frame size, headroom, or flags")]
    InvalidUmem,
    /// Bind flags do not select exactly one supported copy or zero-copy mode.
    #[error("select exactly one of XDP_COPY or XDP_ZEROCOPY, optionally with XDP_USE_NEED_WAKEUP")]
    InvalidBindFlags,
    /// A ring size was not a non-zero power of two.
    #[error("ring size must be a non-zero power of two")]
    InvalidRingSize,
    /// The kernel returned ring offsets that do not fit the mapped ring.
    #[error("kernel returned invalid ring offsets")]
    InvalidRingOffsets,
    /// An RX descriptor is empty, spans UMEM frames, or uses unsupported options.
    #[error("invalid or unsupported RX descriptor")]
    InvalidRxDescriptor,
    /// A TX descriptor is empty, outside the UMEM, or spans UMEM frames.
    #[error("invalid TX descriptor")]
    InvalidTxDescriptor,
    /// A frame index was outside the UMEM.
    #[error("frame index {0} is out of bounds")]
    FrameOutOfBounds(u32),
}
