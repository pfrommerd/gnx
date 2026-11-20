#pragma once

#include <memory>
#include <optional>
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"

absl::StatusOr<std::unique_ptr<xla::PjRtClient>> xla_rs_gpu_client(
    double memory_fraction, bool preallocate, std::optional<int> ordinal);
xla::CompileOptions xla_rs_compile_options(xla::PjRtClient* client);
