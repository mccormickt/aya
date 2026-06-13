//! System calls for `AF_XDP` (XSK) sockets.

use std::{
    io, mem,
    os::fd::{AsRawFd as _, BorrowedFd, FromRawFd as _},
    ptr,
};

use aya_obj::generated::{XDP_MMAP_OFFSETS, sockaddr_xdp, xdp_mmap_offsets, xdp_ring_offset};
use libc::{AF_XDP, MSG_DONTWAIT, SOCK_RAW, SOL_XDP};

use super::SyscallError;

/// Creates an `AF_XDP` socket.
pub(crate) fn xsk_socket() -> Result<crate::MockableFd, SyscallError> {
    // Safety: libc wrapper.
    let fd = unsafe { libc::socket(AF_XDP, SOCK_RAW, 0) };
    if fd < 0 {
        return Err(SyscallError {
            call: "socket",
            io_error: io::Error::last_os_error(),
        });
    }
    // SAFETY: `socket` returns an owned file descriptor on success.
    Ok(unsafe { crate::MockableFd::from_raw_fd(fd) })
}

/// Calls `setsockopt(SOL_XDP, optname, value)`.
///
/// Used to register the UMEM ([`XDP_UMEM_REG`](aya_obj::generated::XDP_UMEM_REG)) and to size the
/// rings ([`XDP_UMEM_FILL_RING`](aya_obj::generated::XDP_UMEM_FILL_RING),
/// [`XDP_UMEM_COMPLETION_RING`](aya_obj::generated::XDP_UMEM_COMPLETION_RING),
/// [`XDP_RX_RING`](aya_obj::generated::XDP_RX_RING)).
pub(crate) fn xsk_setsockopt<T>(
    fd: BorrowedFd<'_>,
    optname: i32,
    value: &T,
) -> Result<(), SyscallError> {
    // Safety: libc wrapper; `value` is valid for `size_of::<T>()` bytes for the duration of the
    // call.
    let ret = unsafe {
        libc::setsockopt(
            fd.as_raw_fd(),
            SOL_XDP,
            optname,
            ptr::from_ref(value).cast(),
            size_of::<T>() as libc::socklen_t,
        )
    };
    if ret < 0 {
        return Err(SyscallError {
            call: "setsockopt",
            io_error: io::Error::last_os_error(),
        });
    }
    Ok(())
}

/// The pre-5.4 layout of `xdp_ring_offset`, before the `flags` field was added.
#[repr(C)]
#[derive(Clone, Copy)]
struct XdpRingOffsetV1 {
    producer: u64,
    consumer: u64,
    desc: u64,
}

/// The pre-5.4 layout of `xdp_mmap_offsets`, with no per-ring `flags` field.
#[repr(C)]
#[derive(Clone, Copy)]
struct XdpMmapOffsetsV1 {
    rx: XdpRingOffsetV1,
    tx: XdpRingOffsetV1,
    fr: XdpRingOffsetV1,
    cr: XdpRingOffsetV1,
}

/// Translates the pre-5.4 offsets into the current struct, leaving each ring's `flags` zeroed.
const fn mmap_offsets_from_v1(v1: &XdpMmapOffsetsV1) -> xdp_mmap_offsets {
    const fn ring(src: &XdpRingOffsetV1) -> xdp_ring_offset {
        xdp_ring_offset {
            producer: src.producer,
            consumer: src.consumer,
            desc: src.desc,
            flags: 0,
        }
    }
    xdp_mmap_offsets {
        rx: ring(&v1.rx),
        tx: ring(&v1.tx),
        fr: ring(&v1.fr),
        cr: ring(&v1.cr),
    }
}

/// Reads the kernel-assigned ring layout via `getsockopt(SOL_XDP, XDP_MMAP_OFFSETS)`.
///
/// The returned [`xdp_mmap_offsets`] describes, for each ring, where the producer/consumer indices
/// and the descriptor array live within the ring's `mmap`ed region. It is only meaningful after the
/// rings have been sized with [`xsk_setsockopt`].
///
/// Kernels before 5.4 use a smaller `xdp_ring_offset` without the `flags` field; we detect that via
/// the returned `optlen` and translate the older layout into the current struct, leaving `flags`
/// zeroed. This mirrors libbpf's `xsk_get_mmap_offsets`.
pub(crate) fn xsk_mmap_offsets(fd: BorrowedFd<'_>) -> Result<xdp_mmap_offsets, SyscallError> {
    // Safety: `xdp_mmap_offsets` is composed solely of integers, so a zeroed value is valid.
    let mut offsets = unsafe { mem::zeroed::<xdp_mmap_offsets>() };
    let mut optlen = size_of::<xdp_mmap_offsets>() as libc::socklen_t;
    // Safety: libc wrapper; `offsets` is valid for `optlen` bytes.
    let ret = unsafe {
        libc::getsockopt(
            fd.as_raw_fd(),
            SOL_XDP,
            XDP_MMAP_OFFSETS as i32,
            ptr::from_mut(&mut offsets).cast(),
            &raw mut optlen,
        )
    };
    if ret < 0 {
        return Err(SyscallError {
            call: "getsockopt",
            io_error: io::Error::last_os_error(),
        });
    }

    let optlen = optlen as usize;
    if optlen == size_of::<xdp_mmap_offsets>() {
        return Ok(offsets);
    }
    if optlen == size_of::<XdpMmapOffsetsV1>() {
        // Pre-5.4 kernel: only the first `size_of::<XdpMmapOffsetsV1>()` bytes were written, using
        // the older (smaller, flag-less) ring layout. Reinterpret those bytes and translate.
        // Safety: the kernel wrote at least `size_of::<XdpMmapOffsetsV1>()` bytes into `offsets`,
        // and `XdpMmapOffsetsV1` has the same alignment as `xdp_mmap_offsets`.
        let v1 = unsafe { ptr::from_ref(&offsets).cast::<XdpMmapOffsetsV1>().read() };
        return Ok(mmap_offsets_from_v1(&v1));
    }
    Err(SyscallError {
        call: "getsockopt",
        io_error: io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unexpected XDP_MMAP_OFFSETS size: {optlen}"),
        ),
    })
}

/// Maps the result of a driver wake-up syscall (`sendto`/`recvfrom` with a NULL buffer) to a
/// `Result`.
///
/// Mirrors libbpf: the errnos that merely mean "the wake-up was unnecessary or will be retried"
/// (`EBUSY`, `EAGAIN`, `ENOBUFS`, `ENETDOWN`) are treated as success. Must be called immediately
/// after the syscall, while `errno` still reflects it.
fn wakeup_result(ret: isize, call: &'static str) -> Result<(), SyscallError> {
    if ret < 0 {
        let io_error = io::Error::last_os_error();
        return match io_error.raw_os_error() {
            Some(libc::EBUSY | libc::EAGAIN | libc::ENOBUFS | libc::ENETDOWN) => Ok(()),
            _ => Err(SyscallError { call, io_error }),
        };
    }
    Ok(())
}

/// Wakes the kernel to process the TX ring via `sendto(fd, NULL, 0, MSG_DONTWAIT, NULL, 0)`.
///
/// A NULL buffer with zero length is a pure wake-up carrying no data. Benign errnos are treated as
/// success; see [`wakeup_result`].
pub(crate) fn xsk_kick_tx(fd: BorrowedFd<'_>) -> Result<(), SyscallError> {
    // Safety: libc wrapper; a NULL buffer with zero length is a pure wake-up with no data.
    let ret = unsafe { libc::sendto(fd.as_raw_fd(), ptr::null(), 0, MSG_DONTWAIT, ptr::null(), 0) };
    wakeup_result(ret, "sendto")
}

/// Wakes the kernel to refill RX from the FILL ring via
/// `recvfrom(fd, NULL, 0, MSG_DONTWAIT, NULL, NULL)`.
///
/// In zero-copy + `XDP_USE_NEED_WAKEUP` mode the driver must be woken after the FILL ring is
/// refilled, or RX silently stalls. The NULL/zero-length read is a pure driver wake-up: it dequeues
/// nothing (the RX ring is read via `mmap`). Benign errnos are treated as success; see
/// [`wakeup_result`].
pub(crate) fn xsk_wake_rx(fd: BorrowedFd<'_>) -> Result<(), SyscallError> {
    // Safety: libc wrapper; a NULL buffer with zero length dequeues nothing and only wakes the
    // driver.
    let ret = unsafe {
        libc::recvfrom(
            fd.as_raw_fd(),
            ptr::null_mut(),
            0,
            MSG_DONTWAIT,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    wakeup_result(ret, "recvfrom")
}

/// Binds an `AF_XDP` socket to a netdev queue via `bind(&sockaddr_xdp)`.
pub(crate) fn xsk_bind(fd: BorrowedFd<'_>, addr: &sockaddr_xdp) -> Result<(), SyscallError> {
    // Safety: libc wrapper; `addr` is valid for `size_of::<sockaddr_xdp>()` bytes.
    let ret = unsafe {
        libc::bind(
            fd.as_raw_fd(),
            ptr::from_ref(addr).cast(),
            size_of::<sockaddr_xdp>() as libc::socklen_t,
        )
    };
    if ret < 0 {
        return Err(SyscallError {
            call: "bind",
            io_error: io::Error::last_os_error(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_offsets_layout() {
        // The v1 struct must be strictly smaller than the current one, so the size returned in
        // `optlen` disambiguates the two layouts.
        assert_eq!(size_of::<XdpMmapOffsetsV1>(), 4 * 3 * size_of::<u64>());
        assert!(size_of::<xdp_mmap_offsets>() > size_of::<XdpMmapOffsetsV1>());
    }

    #[test]
    fn v1_offsets_translation() {
        let v1 = XdpMmapOffsetsV1 {
            rx: XdpRingOffsetV1 {
                producer: 1,
                consumer: 2,
                desc: 3,
            },
            tx: XdpRingOffsetV1 {
                producer: 4,
                consumer: 5,
                desc: 6,
            },
            fr: XdpRingOffsetV1 {
                producer: 7,
                consumer: 8,
                desc: 9,
            },
            cr: XdpRingOffsetV1 {
                producer: 10,
                consumer: 11,
                desc: 12,
            },
        };
        let off = mmap_offsets_from_v1(&v1);
        assert_eq!((off.rx.producer, off.rx.consumer, off.rx.desc), (1, 2, 3));
        assert_eq!((off.tx.producer, off.tx.consumer, off.tx.desc), (4, 5, 6));
        assert_eq!((off.fr.producer, off.fr.consumer, off.fr.desc), (7, 8, 9));
        assert_eq!(
            (off.cr.producer, off.cr.consumer, off.cr.desc),
            (10, 11, 12)
        );
        // The `flags` fields have no v1 counterpart and must be zeroed.
        assert_eq!(off.rx.flags, 0);
        assert_eq!(off.cr.flags, 0);
    }
}
