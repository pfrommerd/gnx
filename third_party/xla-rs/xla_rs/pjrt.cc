// Single-device PJRT ownership and transfer semantics for the Rust bindings.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "xla/hlo/builder/xla_computation.h"
#include "gpu_client.h"
#include "xla/literal.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/shape_util.h"

extern "C" {
struct GnxReport {
  void* context;
  void (*write)(void*, const char*, size_t);
};
}

namespace {
int Report(GnxReport report, const absl::Status& status) {
  if (!status.ok()) {
    auto message = status.message();
    report.write(report.context, message.data(), message.size());
  }
  return static_cast<int>(status.code());
}
}  // namespace

extern "C" int gnx_pjrt_new(xla::PjRtClient** out, GnxReport report) {
  auto client = xla::GetXlaPjrtCpuClient(xla::CpuClientOptions{});
  if (!client.ok()) return Report(report, client.status());
  if ((*client)->device_count() != 1 || (*client)->addressable_device_count() != 1) {
    return Report(report, absl::UnimplementedError("gnx-xla requires one addressable CPU device"));
  }
  *out = client->release();
  return 0;
}

extern "C" int gnx_pjrt_cuda_new(int ordinal, xla::PjRtClient** out, GnxReport report) {
  auto client = xla_rs_gpu_client(0.75, false, ordinal);
  if (!client.ok()) return Report(report, client.status());
  auto devices = (*client)->addressable_devices();
  if (devices.size() != 1 || devices.front()->local_hardware_id().value() != ordinal)
    return Report(report, absl::InvalidArgumentError("requested CUDA device is not addressable"));
  *out = client->release();
  return 0;
}

extern "C" void gnx_pjrt_info(xla::PjRtClient* client, int* ordinal, GnxReport kind) {
  auto* device = client->addressable_devices().front();
  *ordinal = device->local_hardware_id().value();
  auto name = device->device_kind();
  kind.write(kind.context, name.data(), name.size());
}

extern "C" int gnx_pjrt_compile(xla::PjRtClient* client, const void* data,
                               int size, xla::PjRtLoadedExecutable** out,
                               GnxReport report) {
  xla::HloModuleProto proto;
  if (!proto.ParseFromArray(data, size)) {
    return Report(report, absl::InvalidArgumentError("invalid XLA HLO module"));
  }
  auto executable = client->CompileAndLoad(xla::XlaComputation(std::move(proto)), xla_rs_compile_options(client));
  if (!executable.ok()) return Report(report, executable.status());
  *out = executable->release();
  return 0;
}

extern "C" int gnx_pjrt_import(xla::PjRtClient* client, const void* data,
                              int type, const int64_t* dims, size_t rank,
                              bool require_alias, void* lease,
                              void (*release)(void*), xla::PjRtBuffer** out,
                              GnxReport report) {
  // Adopt the lease before any error return. PJRT gets a copy of the owner
  // when it aliases the bytes, not just a borrowed pointer from the caller.
  std::shared_ptr<void> owner(lease, release);
  auto memory = client->addressable_devices().front()->default_memory_space();
  if (!memory.ok()) return Report(report, memory.status());
  auto* cpu = dynamic_cast<xla::CommonPjRtClient*>(client);
  const auto primitive = static_cast<xla::PrimitiveType>(type);
  const absl::Span<const int64_t> dimensions(dims, rank);
  bool alias = false;
  if (client->platform_name() == "cpu" && cpu != nullptr) {
    auto shape = cpu->MakeDefaultShapeForMemorySpace(
        *memory, xla::ShapeUtil::MakeShape(primitive, dimensions), nullptr);
    if (!shape.ok()) return Report(report, shape.status());
    alias = cpu->BufferFromHostBufferSupportsZeroCopy(
        data, primitive, dimensions, {}, *shape, *memory, nullptr);
  }
  if (require_alias && !alias) {
    return Report(report, absl::UnimplementedError(
        "PJRT cannot alias this host allocation; use AllowCopy to transfer it"));
  }
  const auto semantics = alias ? xla::PjRtClient::HostBufferSemantics::kImmutableZeroCopy
                               : xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall;
  auto buffer = client->BufferFromHostBuffer(data, primitive, dimensions, {},
                                             semantics, [owner]() {}, *memory,
                                             /*device_layout=*/nullptr);
  if (!buffer.ok()) return Report(report, buffer.status());
  *out = buffer->release();
  return 0;
}

extern "C" bool gnx_pjrt_matches(xla::PjRtBuffer* buffer, int type,
                                const int64_t* dims, size_t rank) {
  auto shape = buffer->on_device_shape();
  return shape.IsArray() && shape.element_type() == type &&
         shape.dimensions() == absl::Span<const int64_t>(dims, rank);
}

extern "C" int gnx_pjrt_execute(xla::PjRtLoadedExecutable* executable,
                               xla::PjRtBuffer* const* inputs, size_t ninputs,
                               xla::PjRtBuffer** outputs, size_t noutputs,
                               GnxReport report) {
  xla::ExecuteOptions options;
  for (size_t i = 0; i < ninputs; ++i) options.non_donatable_input_indices.insert(i);
  std::vector<xla::PjRtBuffer*> arguments(inputs, inputs + ninputs);
  auto result = executable->Execute({arguments}, options);
  if (!result.ok()) return Report(report, result.status());
  if (result->size() != 1 || result->front().size() != noutputs) {
    return Report(report, absl::InternalError("unexpected PJRT replica or output count"));
  }
  for (size_t i = 0; i < noutputs; ++i) outputs[i] = result->front()[i].release();
  return 0;
}

extern "C" int gnx_pjrt_download(xla::PjRtBuffer* buffer, char* destination,
                                size_t size, GnxReport report) {
  auto shape = buffer->on_device_shape();
  if (!shape.IsArray()) return Report(report, absl::InvalidArgumentError("expected array buffer"));
  auto packed = xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());
  if (size != xla::ShapeUtil::ByteSizeOf(packed))
    return Report(report, absl::InvalidArgumentError("destination size does not match buffer"));
  xla::MutableBorrowingLiteral literal(destination, packed);
  return Report(report, buffer->ToLiteralSync(&literal));
}

extern "C" int gnx_pjrt_address(xla::PjRtBuffer* buffer, const void** out,
                               GnxReport report) {
  auto lease = buffer->AcquireExternalReference();
  if (!lease.ok()) return Report(report, lease.status());
  *out = (*lease)->OpaqueDeviceMemoryDataPointer();
  return 0;
}

extern "C" void gnx_pjrt_free_client(xla::PjRtClient* client) { delete client; }
extern "C" void gnx_pjrt_free_buffer(xla::PjRtBuffer* buffer) { delete buffer; }
extern "C" void gnx_pjrt_free_executable(xla::PjRtLoadedExecutable* executable) { delete executable; }
