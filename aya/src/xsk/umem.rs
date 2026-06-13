//! `AF_XDP` UMEM: the shared memory region that holds packet frames.

use std::{os::fd::BorrowedFd, ptr::NonNull, slice};

use aya_obj::generated::{XDP_UMEM_REG, XDP_UMEM_UNALIGNED_CHUNK_FLAG, xdp_umem_reg};

use super::XskError;
use crate::{sys::xsk_setsockopt, util::page_size};

/// The minimum UMEM chunk (frame) size accepted by the kernel.
const XDP_UMEM_MIN_CHUNK_SIZE: u32 = 2048;

/// Configuration for a [`XskUmem`].
#[derive(Clone, Copy, Debug)]
pub struct XskUmemConfig {
    /// Size of each frame (chunk) in bytes. Must be a power of two in `2048..=page_size`.
    pub frame_size: u32,
    /// Per-frame headroom reserved for the application, in bytes.
    pub headroom: u32,
    /// UMEM flags (`xdp_umem_reg::flags`). `0` selects aligned-chunk mode.
    pub flags: u32,
}

impl Default for XskUmemConfig {
    fn default() -> Self {
        Self {
            frame_size: 4096,
            headroom: 0,
            flags: 0,
        }
    }
}

/// A registered UMEM region.
///
/// The UMEM is a contiguous, page-aligned chunk of memory divided into equal-sized frames. The
/// caller supplies and owns the memory; the kernel reads received packets into it and reads
/// outgoing packets from it. A [`crate::xsk::XskSocket`] registers the UMEM on its socket and binds
/// the FILL/COMPLETION rings to it.
pub struct XskUmem {
    mem: NonNull<u8>,
    len: usize,
    config: XskUmemConfig,
}

// Safety: `XskUmem` owns the only handle to `mem` from our side. The kernel writes into it
// concurrently, but only into frames we have published to the FILL ring and not yet read back, and
// the public API only ever hands out short-lived `&[u8]` slices of individual received frames.
unsafe impl Send for XskUmem {}

impl XskUmem {
    /// Creates a UMEM over the caller-provided memory `mem`.
    ///
    /// # Safety
    ///
    /// `mem` must point to memory that:
    /// - is page-aligned and a whole number of `config.frame_size` frames long;
    /// - remains valid and exclusively owned by this `XskUmem` for its entire lifetime; and
    /// - is not otherwise read or written (e.g. via an aliasing `&mut`) while the socket is bound,
    ///   since the kernel writes received packets into it asynchronously.
    pub unsafe fn new(config: XskUmemConfig, mem: NonNull<[u8]>) -> Result<Self, XskError> {
        let len = mem.len();
        let mem = mem.cast::<u8>();
        let frame_size = config.frame_size;
        if config.flags & XDP_UMEM_UNALIGNED_CHUNK_FLAG != 0 {
            // Unaligned-chunk mode packs the chunk index and intra-chunk offset into the UMEM
            // address; `frame_addr`/`frame_bytes` assume aligned chunks and do not handle it.
            return Err(XskError::InvalidUmem);
        }
        if !frame_size.is_power_of_two()
            || frame_size < XDP_UMEM_MIN_CHUNK_SIZE
            || frame_size as usize > page_size()
            || len == 0
            || !len.is_multiple_of(frame_size as usize)
            || !(mem.as_ptr() as usize).is_multiple_of(page_size())
        {
            return Err(XskError::InvalidUmem);
        }
        Ok(Self { mem, len, config })
    }

    /// Registers the UMEM on `fd` via `setsockopt(XDP_UMEM_REG)`.
    pub(crate) fn register(&self, fd: BorrowedFd<'_>) -> Result<(), XskError> {
        // Safety: `xdp_umem_reg` is composed of integers, so a zeroed value is valid. Using
        // `zeroed` rather than a struct literal keeps this robust to fields that only exist in
        // newer kernel headers (e.g. `tx_metadata_len`).
        let mut reg = unsafe { std::mem::zeroed::<xdp_umem_reg>() };
        reg.addr = self.mem.as_ptr() as u64;
        reg.len = self.len as u64;
        reg.chunk_size = self.config.frame_size;
        reg.headroom = self.config.headroom;
        reg.flags = self.config.flags;
        xsk_setsockopt(fd, XDP_UMEM_REG as i32, &reg)?;
        Ok(())
    }

    /// Number of frames in the UMEM.
    pub const fn frame_count(&self) -> u32 {
        // `new` guarantees `len` is a whole multiple of `frame_size`, so this is exact.
        (self.len / self.config.frame_size as usize) as u32
    }

    /// Size of each frame in bytes.
    pub const fn frame_size(&self) -> u32 {
        self.config.frame_size
    }

    /// The UMEM address (offset) of frame `index`, for use with the FILL and TX rings.
    ///
    /// # Errors
    ///
    /// Returns [`XskError::FrameOutOfBounds`] if `index >= frame_count()`.
    pub fn frame_addr(&self, index: u32) -> Result<u64, XskError> {
        if index >= self.frame_count() {
            return Err(XskError::FrameOutOfBounds(index));
        }
        Ok(u64::from(index) * u64::from(self.config.frame_size))
    }

    /// Returns the packet bytes at UMEM offset `addr` for `len` bytes, or `None` if `[addr, addr +
    /// len)` does not lie within the UMEM.
    ///
    /// # Safety
    ///
    /// `addr` and `len` must come from an RX descriptor the kernel produced for this UMEM; the
    /// returned slice is only valid until the frame is returned to the FILL ring.
    pub(crate) unsafe fn frame_bytes(&self, addr: u64, len: u32) -> Option<&[u8]> {
        let start = usize::try_from(addr).ok()?;
        let len = len as usize;
        if start.checked_add(len)? > self.len {
            return None;
        }
        // Safety: `[start, start + len)` lies within the UMEM (checked above) and the bytes were
        // written by the kernel.
        Some(unsafe { slice::from_raw_parts(self.mem.as_ptr().add(start), len) })
    }

    /// Returns a mutable view of the UMEM bytes at offset `addr` for `len` bytes, or `None` if
    /// `[addr, addr + len)` does not lie within the UMEM. Used to write a frame before transmitting.
    ///
    /// # Safety
    ///
    /// The frame `[addr, addr + len)` must currently be owned by us — i.e. not published to the
    /// kernel via the FILL or TX rings — since the kernel may otherwise access it concurrently.
    pub(crate) unsafe fn frame_bytes_mut(&mut self, addr: u64, len: u32) -> Option<&mut [u8]> {
        let start = usize::try_from(addr).ok()?;
        let len = len as usize;
        if start.checked_add(len)? > self.len {
            return None;
        }
        // Safety: `[start, start + len)` lies within the UMEM (checked above) and we hold `&mut
        // self`, so the slice does not alias any other access from our side.
        Some(unsafe { slice::from_raw_parts_mut(self.mem.as_ptr().add(start), len) })
    }
}
