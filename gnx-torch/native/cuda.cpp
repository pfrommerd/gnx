#include <cstdio>
#include <exception>

#if __has_include(<ATen/cuda/CUDAConfig.h>)
#include <ATen/cuda/CUDAConfig.h>
#if !AT_ROCM_ENABLED()
#define GNX_LIBTORCH_CUDA 1
#if defined(__linux__)
#include <ATen/ops/empty.h>
#else
#include <ATen/cuda/EmptyTensor.h>
#endif
#endif
#endif

#if defined(GNX_LIBTORCH_CUDA) && defined(__linux__)
#include <dlfcn.h>
#endif

// On NixOS the NVIDIA driver is in a system profile outside the loader's
// default search path. Resolve it before libcudart's first availability probe;
// keep it loaded, as we do for XLA. CPU startup never calls this function.
extern "C" const char *gnx_torch_prepare_cuda_driver() noexcept {
#if defined(GNX_LIBTORCH_CUDA) && defined(__linux__)
  static const auto driver = [] {
    auto handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!handle)
      handle =
          dlopen("/run/opengl-driver/lib/libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    return handle;
  }();
  (void)driver;
#ifdef GNX_NVRTC_BUILTINS
  static const auto builtins = dlopen(GNX_NVRTC_BUILTINS, RTLD_NOW | RTLD_LOCAL);
  if (!builtins) return "could not load " GNX_NVRTC_BUILTINS;
#endif
  // Load Torch kernels on first GPU use. In managed CUDA 13 builds the
  // kernel DSO and XLA's PJRT plugin resolve to the same NVIDIA libraries.
  static const auto kernels =
      dlopen(GNX_TORCH_LIB_DIR "/libtorch_cuda.so", RTLD_NOW | RTLD_LOCAL);
  if (!kernels)
    return "could not load " GNX_TORCH_LIB_DIR "/libtorch_cuda.so";
#endif
  return nullptr;
}

// Probe the chosen ordinal. Outside Linux a direct kernel reference keeps
// the platform's CUDA library linked, so its ATen kernels are registered.
extern "C" bool gnx_torch_open_cuda(int ordinal, char *error,
                                    size_t capacity) noexcept {
#ifdef GNX_LIBTORCH_CUDA
  try {
#if defined(__linux__)
    (void)at::empty({1}, at::TensorOptions().dtype(at::kByte).device(
                             at::Device(at::kCUDA, ordinal)));
#else
    (void)at::detail::empty_cuda({1}, at::kByte, at::Device(at::kCUDA, ordinal),
                                 std::nullopt);
#endif
    return true;
  } catch (const std::exception &e) {
    std::snprintf(error, capacity, "%s", e.what());
  } catch (...) {
    std::snprintf(error, capacity, "failed to initialize Torch CUDA device");
  }
#else
  std::snprintf(error, capacity, "LibTorch was built without CUDA support");
#endif
  return false;
}
