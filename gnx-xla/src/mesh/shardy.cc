// The Rust xla-rs API does not expose the MLIR importer or Shardy passes.
// Keep this bridge to the pinned XLA build small: StableHLO in, local HLO out.
#include <cstddef>
#include <string>

#include "mlir/IR/MLIRContext.h"
#include "xla/client/executable_build_options.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/service/spmd/shardy/shardy_xla_pass.h"
#include "xla/service/spmd/spmd_partitioner.h"

namespace {

absl::StatusOr<std::string> Partition(absl::string_view mlir_text,
                                      int partitions) {
  mlir::MLIRContext context;
  auto parsed = xla::ParseMlirModuleString(mlir_text, context);
  if (!parsed.ok()) return parsed.status();
  xla::ExecutableBuildOptions options;
  options.set_num_replicas(1);
  options.set_num_partitions(partitions);
  options.set_use_spmd_partitioning(true);
  options.set_use_shardy_partitioner(true);
  xla::XlaComputation computation;
  auto imported = xla::MlirToXlaComputation(**parsed, computation, false, true,
                                            &options);
  if (!imported.ok()) return imported;
  auto config = xla::HloModule::CreateModuleConfigFromProto(
      computation.proto(), xla::GetDebugOptionsFromFlags());
  if (!config.ok()) return config.status();
  config->set_replica_count(1);
  config->set_num_partitions(partitions);
  config->set_use_spmd_partitioning(true);
  // The MLIR import stores SDY metadata in frontend attributes.
  config->mutable_debug_options().set_xla_enable_hlo_sharding_v3(false);
  auto module = xla::HloModule::CreateFromProto(computation.proto(), *config);
  if (!module.ok()) return module.status();
  auto shardy = xla::sdy::ShardyXLA().Run(module->get());
  if (!shardy.ok()) return shardy.status();
  xla::spmd::SpmdPartitionerOptions spmd_options;
  spmd_options.allow_module_signature_change = true;
  auto partitioned =
      xla::spmd::SpmdPartitioner(partitions, 1, spmd_options).Run(module->get());
  if (!partitioned.ok()) return partitioned.status();
  auto dce = xla::HloDCE().Run(module->get());
  if (!dce.ok()) return dce.status();

  // This worker runtime has no cross-host collective transport. Validate a
  // closed set of partition-independent opcodes before compiling on workers.
  for (const auto* computation : (*module)->computations()) {
    for (const auto* op : computation->instructions()) {
      switch (op->opcode()) {
        case xla::HloOpcode::kParameter:
        case xla::HloOpcode::kTuple:
        case xla::HloOpcode::kGetTupleElement:
        case xla::HloOpcode::kAdd:
        case xla::HloOpcode::kOr:
        case xla::HloOpcode::kCopy:
        case xla::HloOpcode::kConstant:
        case xla::HloOpcode::kBitcast:
          break;
        default:
          return absl::UnimplementedError(
              "mesh program requires collectives or partition-specific code: " +
              std::string(xla::HloOpcodeString(op->opcode())));
      }
    }
  }
  return (*module)->ToProto().SerializeAsString();
}

}  // namespace

extern "C" bool gnx_partition_shardy(
    const char* input, size_t len, int partitions, void* context,
    void (*output)(void*, const char*, size_t)) {
  auto result = Partition(absl::string_view(input, len), partitions);
  if (result.ok()) {
    output(context, result->data(), result->size());
  } else {
    auto status = result.status().ToString();
    output(context, status.data(), status.size());
  }
  return result.ok();
}
