// Minimal libtorch C++ shim for CUDA stream control. tch 0.20 exposes
// no stream API, so we call into c10::cuda directly. Each function
// returns/accepts an opaque `c10::cuda::CUDAStream*` owned by the Rust
// side; Rust must call `xtr_cuda_stream_destroy` to free.
//
// Used by Pass A3 of `rank_multi_shard` to submit per-query merger
// kernels across a pool of streams so the default stream's per-query
// sync (inside the merger's `unique_consecutive`) doesn't serialize
// other queries' work.

#include <c10/cuda/CUDAStream.h>
#include <cstdint>

extern "C" {

// Allocate a new CUDAStream from the pool. Pool streams are round-
// robin-reused by libtorch; allocating doesn't create a new cudaStream_t
// so this is cheap even for large N.
void* xtr_cuda_stream_from_pool(int64_t device_index, int32_t high_priority) {
    auto s = c10::cuda::getStreamFromPool(
        high_priority != 0,
        static_cast<c10::DeviceIndex>(device_index));
    return new c10::cuda::CUDAStream(s);
}

// Snapshot the current CUDA stream for `device_index`.
void* xtr_cuda_get_current_stream(int64_t device_index) {
    auto s = c10::cuda::getCurrentCUDAStream(
        static_cast<c10::DeviceIndex>(device_index));
    return new c10::cuda::CUDAStream(s);
}

// Set the current stream on the device of the passed-in stream.
void xtr_cuda_set_current_stream(void* stream_ptr) {
    auto* s = static_cast<c10::cuda::CUDAStream*>(stream_ptr);
    c10::cuda::setCurrentCUDAStream(*s);
}

// Block host until all work on this stream completes.
void xtr_cuda_stream_synchronize(void* stream_ptr) {
    auto* s = static_cast<c10::cuda::CUDAStream*>(stream_ptr);
    s->synchronize();
}

void xtr_cuda_stream_destroy(void* stream_ptr) {
    delete static_cast<c10::cuda::CUDAStream*>(stream_ptr);
}

} // extern "C"
