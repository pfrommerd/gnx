#include <cstdio>
#include <exception>
#include <sstream>
#include <string>

#include <torch/script.h>

// Serialize a TorchScript module to memory, in the format `torch::jit::load`
// reads. The bytes are handed to `sink` before returning; nothing is retained.
extern "C" bool gnx_torch_save_module(const torch::jit::Module *module,
                                      void (*sink)(void *, const char *, size_t),
                                      void *ctx, char *error,
                                      size_t capacity) noexcept {
  try {
    std::ostringstream out(std::ios::binary);
    module->save(out);
    const std::string bytes = out.str();
    sink(ctx, bytes.data(), bytes.size());
    return true;
  } catch (const std::exception &e) {
    std::snprintf(error, capacity, "%s", e.what());
    return false;
  } catch (...) {
    std::snprintf(error, capacity, "unknown error saving a TorchScript module");
    return false;
  }
}
