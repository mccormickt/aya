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
//! This implementation covers the RX and TX data paths in copy, zero-copy, or kernel-chosen mode,
//! including `XDP_USE_NEED_WAKEUP`. Zero-copy is supported at the protocol level for aligned chunks
//! (see [`XskSocket`] for the bind flags and the [`wake_rx`](XskSocket::wake_rx) requirement); the
//! remaining gaps are unaligned chunks and shared UMEM.
//!
//! # Minimum kernel version
//!
//! The minimum kernel version required to use `AF_XDP` is 4.18.

mod ring;
mod socket;
mod umem;

pub(crate) use ring::XskRing;
pub use socket::{XskSocket, XskSocketConfig};
pub use umem::{XskUmem, XskUmemConfig};

/// Errors that can occur while setting up or using an `AF_XDP` socket.
#[derive(Debug, thiserror::Error)]
pub enum XskError {
    /// A system call failed.
    #[error(transparent)]
    Syscall(#[from] crate::sys::SyscallError),
    /// The UMEM memory is not page-aligned, not a whole number of frames, or `frame_size` is not a
    /// power of two in `2048..=page_size`.
    #[error(
        "invalid UMEM: memory must be page-aligned and a whole number of frames, and frame_size a power of two in 2048..=page_size"
    )]
    InvalidUmem,
    /// A ring size was not a non-zero power of two.
    #[error("ring size must be a non-zero power of two")]
    InvalidRingSize,
    /// A frame index was outside the UMEM.
    #[error("frame index {0} is out of bounds")]
    FrameOutOfBounds(u32),
}
