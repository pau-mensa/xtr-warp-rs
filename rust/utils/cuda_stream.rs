//! Minimal wrapper around a libtorch C++ shim that exposes CUDAStream
//! management. tch 0.20 has no stream API, so we call into `c10::cuda`
//! directly via `shim/cuda_stream_shim.cpp`.
//!
//! Only compiled when `build.rs` was able to find libtorch CUDA headers.
//! When `cfg(xtr_has_cuda_shim)` is not set, the safe wrapper is absent
//! and Phase-3 falls back to the default-stream path.

#![cfg(xtr_has_cuda_shim)]

use std::os::raw::c_void;

unsafe extern "C" {
    fn xtr_cuda_stream_from_pool(device_index: i64, high_priority: i32) -> *mut c_void;
    fn xtr_cuda_get_current_stream(device_index: i64) -> *mut c_void;
    fn xtr_cuda_set_current_stream(stream_ptr: *mut c_void);
    fn xtr_cuda_stream_synchronize(stream_ptr: *mut c_void);
    fn xtr_cuda_stream_destroy(stream_ptr: *mut c_void);
}

/// Owned CUDAStream handle. Backed by a heap-allocated `c10::cuda::CUDAStream`
/// inside the shim. Dropping this destroys the wrapper (the underlying
/// cudaStream_t is pool-managed by libtorch and is not destroyed).
pub struct CudaStream {
    ptr: *mut c_void,
    device: i32,
}

// The underlying c10::cuda::CUDAStream is just (StreamId, DeviceIndex,
// DeviceType) — no interior mutability, immutable once constructed. Safe
// to share across threads.
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    /// Pull a stream from libtorch's per-device pool.
    pub fn from_pool(device: i32, high_priority: bool) -> Self {
        let ptr = unsafe {
            xtr_cuda_stream_from_pool(device as i64, if high_priority { 1 } else { 0 })
        };
        assert!(!ptr.is_null(), "xtr_cuda_stream_from_pool returned null");
        CudaStream { ptr, device }
    }

    /// Snapshot of the current stream on the given device.
    pub fn current(device: i32) -> Self {
        let ptr = unsafe { xtr_cuda_get_current_stream(device as i64) };
        assert!(!ptr.is_null(), "xtr_cuda_get_current_stream returned null");
        CudaStream { ptr, device }
    }

    pub fn device(&self) -> i32 {
        self.device
    }

    /// Block host until all enqueued work on this stream completes.
    pub fn synchronize(&self) {
        unsafe { xtr_cuda_stream_synchronize(self.ptr) };
    }

    /// Internal accessor for the raw pointer. Used by [`StreamGuard`].
    fn raw(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { xtr_cuda_stream_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// RAII guard that snapshots the current CUDA stream on construct and
/// restores it on drop. Pair with [`set_current_stream`] to switch streams
/// many times within a scope with only one save/restore, instead of one
/// per switch. A3's per-query round-robin uses this pattern.
pub struct SavedStream {
    prev: CudaStream,
}

impl SavedStream {
    pub fn save(device: i32) -> Self {
        SavedStream { prev: CudaStream::current(device) }
    }
}

impl Drop for SavedStream {
    fn drop(&mut self) {
        unsafe { xtr_cuda_set_current_stream(self.prev.raw()) };
    }
}

/// Set the current stream without saving the previous one. Only safe
/// inside a scope guarded by a [`SavedStream`] that will restore on drop.
pub fn set_current_stream(stream: &CudaStream) {
    unsafe { xtr_cuda_set_current_stream(stream.raw()) };
}

/// Pool of CUDAStreams for a single device, used by Pass A3 to submit
/// per-query merger work round-robin. Size is read from
/// `XTR_WARP_MERGER_STREAMS` (default 8). Set to 0 or 1 to disable
/// (fall back to default stream).
pub struct StreamPool {
    streams: Vec<CudaStream>,
}

impl StreamPool {
    pub fn for_device(device: i32) -> Self {
        let n = std::env::var("XTR_WARP_MERGER_STREAMS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(8);
        let streams = (0..n)
            .map(|_| CudaStream::from_pool(device, false))
            .collect();
        StreamPool { streams }
    }

    pub fn len(&self) -> usize {
        self.streams.len()
    }

    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }

    /// Pick a stream by index (caller does modular arithmetic).
    pub fn get(&self, idx: usize) -> Option<&CudaStream> {
        if self.streams.is_empty() {
            None
        } else {
            Some(&self.streams[idx % self.streams.len()])
        }
    }

    pub fn synchronize_all(&self) {
        for s in &self.streams {
            s.synchronize();
        }
    }
}
