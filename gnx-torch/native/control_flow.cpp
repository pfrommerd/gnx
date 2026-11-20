#include <ATen/ATen.h>
#include <ATen/TracerMode.h>
#include <cstdio>
#include <limits>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <vector>

namespace {
namespace jit = torch::jit;
using State = jit::tracer::TracingState;
using Tensors = std::vector<at::Tensor>;
// tch handles point to at::Tensor objects. Callback outputs stay borrowed until
// its next invocation. Loop block parameters are borrowed only during the call.
using Emit = bool (*)(void *, size_t, size_t, const at::Tensor *const *,
                      const at::Tensor **);

jit::Value *boolean(State &s, const at::Tensor &tensor) {
  auto &g = *s.graph;
  return g.insertNode(g.create(c10::aten::Bool, {s.getValue(tensor)}))
      ->output()
      ->setType(c10::BoolType::get());
}

Tensors trace_if(State &s, jit::Node *node, const at::Tensor &pred, Emit emit,
                 void *ctx) {
  size_t sample;
  {
    at::tracer::impl::NoTracerDispatchMode guard;
    sample = pred.item<bool>() ? 0 : 1;
  }
  Tensors examples;
  for (size_t i = 0; i < 2; ++i) {
    auto block = node->addBlock();
    jit::WithInsertPoint insert(block);
    jit::tracer::WithNestedTracingFrame frame;
    std::vector<const at::Tensor *> values(node->outputs().size());
    TORCH_CHECK(emit(ctx, i, 0, nullptr, values.data()),
                "conditional branch lowering failed");
    for (auto t : values) {
      block->registerOutput(s.getValue(*t));
      if (i == sample)
        examples.push_back(*t);
    }
  }
  return examples;
}

Tensors trace_loop(State &s, jit::Node *node, const at::Tensor *const *carry,
                   Emit emit, void *ctx) {
  const auto count = node->outputs().size();
  auto block = node->addBlock();
  jit::WithInsertPoint insert(block);
  jit::tracer::WithNestedTracingFrame frame;
  block->addInput()->setType(c10::IntType::get()); // iteration counter
  Tensors examples(count);
  std::vector<const at::Tensor *> params(count);
  for (size_t j = 0; j < count; ++j) {
    // The same parent value may be both a capture and a carry or initialize
    // several carries. Each block parameter must have a distinct identity.
    at::tracer::impl::NoTracerDispatchMode guard;
    examples[j] = carry[j]->clone();
    s.setValue(examples[j], block->addInput());
    params[j] = &examples[j];
  }
  std::vector<const at::Tensor *> next(count + 1);
  TORCH_CHECK(emit(ctx, 0, count, params.data(), next.data()),
              "loop body lowering failed");
  block->registerOutput(boolean(s, *next[0]));
  for (size_t j = 0; j < count; ++j)
    block->registerOutput(s.getValue(*next[j + 1]));
  // Fixed GNX carry metadata suffices to continue tracing. Never execute a
  // runtime-controlled loop to termination on the compiler's example data.
  return examples;
}
} // namespace

extern "C" bool gnx_torch_trace_control(bool loop, const at::Tensor *predicate,
                                        size_t count,
                                        const at::Tensor *const *carry,
                                        at::Tensor **outputs, Emit emit,
                                        void *ctx, char *error,
                                        size_t capacity) noexcept {
  try {
    auto state = jit::tracer::getTracingState();
    TORCH_CHECK(state, "control flow requires an active Torch trace");
    auto &g = *state->graph;
    std::vector<jit::Value *> inputs;
    if (loop)
      inputs.push_back(g.insertConstant(std::numeric_limits<int64_t>::max()));
    inputs.push_back(boolean(*state, *predicate));
    if (loop)
      for (size_t j = 0; j < count; ++j)
        inputs.push_back(state->getValue(*carry[j]));
    auto node = g.insertNode(
        g.create(loop ? c10::prim::Loop : c10::prim::If, inputs, count));
    try {
      auto examples = loop ? trace_loop(*state, node, carry, emit, ctx)
                           : trace_if(*state, node, *predicate, emit, ctx);
      // Fresh identities prevent a forwarded/repeated result from replacing
      // the binding of the parent argument. Only the IR node runs on replay.
      at::tracer::impl::NoTracerDispatchMode guard;
      for (size_t j = 0; j < count; ++j) {
        *outputs[j] = examples[j].clone();
        state->setValue(*outputs[j], node->output(j));
      }
    } catch (...) {
      node->destroy();
      throw;
    }
    return true;
  } catch (const std::exception &e) {
    std::snprintf(error, capacity, "%s", e.what());
  } catch (...) {
    std::snprintf(error, capacity, "unknown LibTorch tracing error");
  }
  return false;
}
