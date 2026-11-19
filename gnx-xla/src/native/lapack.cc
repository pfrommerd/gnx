#include <algorithm>
#include <complex>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include <lapacke.h>

#include "xla/ffi/api/ffi.h"

namespace {
namespace ffi = xla::ffi;
using ffi::AnyBuffer;
using ffi::DataType;

enum Kernel : int64_t {
  kLu,
  kHessenberg,
  kTridiagonal,
  kSchur,
  kEig,
  kSvd,
  kPivotedQr,
  kTridiagonalSolve,
  kPivotsToPermutation,
};

template <typename T> struct Lapack;
#define GNX_LAPACK(T, R, prefix, reduce, qr)                                   \
  template <> struct Lapack<T> {                                               \
    using Real = R;                                                            \
    static constexpr auto getrf = LAPACKE_##prefix##getrf;                     \
    static constexpr auto gehrd = LAPACKE_##prefix##gehrd;                     \
    static constexpr auto trd = LAPACKE_##prefix##reduce;                      \
    static constexpr auto gees = LAPACKE_##prefix##gees;                       \
    static constexpr auto geev = LAPACKE_##prefix##geev;                       \
    static constexpr auto gesvd = LAPACKE_##prefix##gesvd;                     \
    static constexpr auto geqp3 = LAPACKE_##prefix##geqp3;                     \
    static constexpr auto gqr = LAPACKE_##prefix##qr;                          \
    static constexpr auto gtsv = LAPACKE_##prefix##gtsv;                       \
  };
GNX_LAPACK(float, float, s, sytrd, orgqr)
GNX_LAPACK(double, double, d, sytrd, orgqr)
GNX_LAPACK(std::complex<float>, float, c, hetrd, ungqr)
GNX_LAPACK(std::complex<double>, double, z, hetrd, ungqr)
#undef GNX_LAPACK

template <typename T> constexpr DataType Type();
template <> constexpr DataType Type<float>() { return DataType::F32; }
template <> constexpr DataType Type<double>() { return DataType::F64; }
template <> constexpr DataType Type<std::complex<float>>() {
  return DataType::C64;
}
template <> constexpr DataType Type<std::complex<double>>() {
  return DataType::C128;
}
template <> constexpr DataType Type<int32_t>() { return DataType::S32; }

class Frame {
public:
  Frame(ffi::RemainingArgs args, ffi::RemainingRets results)
      : args_(args), results_(results) {}

  AnyBuffer Input(size_t index) const {
    auto buffer = args_.get<AnyBuffer>(index);
    if (!buffer.has_value())
      throw std::invalid_argument("missing LAPACK input");
    return *buffer;
  }

  template <typename T> T *Output(size_t index, size_t size) const {
    auto buffer = results_.get<AnyBuffer>(index);
    if (!buffer.has_value() || (*buffer)->element_count() != size ||
        (*buffer)->element_type() != Type<T>()) {
      throw std::invalid_argument("invalid LAPACK result shape or type");
    }
    return (*buffer)->typed_data<T>();
  }

private:
  ffi::RemainingArgs args_;
  ffi::RemainingRets results_;
};

template <typename T>
const T *Input(const Frame &frame, size_t index, size_t size) {
  AnyBuffer buffer = frame.Input(index);
  if (buffer.element_count() != size || buffer.element_type() != Type<T>()) {
    throw std::invalid_argument("invalid LAPACK input shape or type");
  }
  return buffer.typed_data<T>();
}

lapack_int Size(int64_t size) {
  if (size < 0 || size > std::numeric_limits<lapack_int>::max()) {
    throw std::invalid_argument("LAPACK matrix dimension out of range");
  }
  return static_cast<lapack_int>(size);
}

size_t Batch(AnyBuffer input, size_t matrix_rank) {
  auto dims = input.dimensions();
  if (dims.size() < matrix_rank)
    throw std::invalid_argument("missing matrix dimensions");
  return std::accumulate(dims.begin(), dims.end() - matrix_rank, size_t{1},
                         std::multiplies<>());
}

void ApplyPivots(const int32_t *pivots, size_t count, int32_t *permutation,
                 size_t n) {
  std::iota(permutation, permutation + n, 0);
  for (size_t i = 0; i < count; ++i) {
    if (i < n && pivots[i] >= 0 && static_cast<size_t>(pivots[i]) < n) {
      std::swap(permutation[i], permutation[pivots[i]]);
    }
  }
}

void Permutation(const Frame &frame, lapack_int n) {
  auto input = frame.Input(0);
  const size_t batch = Batch(input, 1);
  const size_t count = Size(input.dimensions()[input.dimensions().size() - 1]);
  auto pivots = Input<int32_t>(frame, 0, batch * count);
  auto out = frame.Output<int32_t>(0, batch * n);
  for (size_t i = 0; i < batch; ++i) {
    ApplyPivots(pivots + i * count, count, out + i * n, n);
  }
}

template <typename T> void Nan(T *out, size_t size) {
  using R = typename Lapack<T>::Real;
  const R nan = std::numeric_limits<R>::quiet_NaN();
  std::fill_n(out, size, T(nan));
}

template <typename T> void SolveTridiagonal(const Frame &frame, bool perturb) {
  using R = typename Lapack<T>::Real;
  const auto d = frame.Input(1);
  const auto b = frame.Input(3);
  const size_t batch = Batch(d, 1);
  const auto dims = b.dimensions();
  if (dims.size() < 2)
    throw std::invalid_argument("tridiagonal RHS must be a matrix");
  const auto n = Size(d.dimensions()[d.dimensions().size() - 1]);
  const auto nrhs = Size(dims[dims.size() - 1]);
  const auto stride = static_cast<size_t>(n) * nrhs;
  auto below = Input<T>(frame, 0, batch * n);
  auto diag = Input<T>(frame, 1, batch * n);
  auto above = Input<T>(frame, 2, batch * n);
  auto rhs = Input<T>(frame, 3, batch * stride);
  auto out = frame.Output<T>(0, batch * stride);
  if (stride == 0)
    return;
  for (size_t i = 0; i < batch; ++i) {
    std::vector<T> dl(below + i * n + 1, below + (i + 1) * n);
    std::vector<T> d(diag + i * n, diag + (i + 1) * n);
    std::vector<T> du(above + i * n, above + (i + 1) * n - 1);
    auto x = out + i * stride;
    std::copy_n(rhs + i * stride, stride, x);
    auto status = Lapack<T>::gtsv(LAPACK_ROW_MAJOR, n, nrhs, dl.data(),
                                  d.data(), du.data(), x, nrhs);
    if (status > 0 && perturb) {
      dl.assign(below + i * n + 1, below + (i + 1) * n);
      d.assign(diag + i * n, diag + (i + 1) * n);
      du.assign(above + i * n, above + (i + 1) * n - 1);
      R scale = 1;
      for (const auto &value : d)
        scale = std::max(scale, std::abs(value));
      const R delta = scale * std::numeric_limits<R>::epsilon();
      for (auto &value : d)
        value += T(delta);
      std::copy_n(rhs + i * stride, stride, x);
      status = Lapack<T>::gtsv(LAPACK_ROW_MAJOR, n, nrhs, dl.data(), d.data(),
                               du.data(), x, nrhs);
    }
    if (status != 0)
      Nan(x, stride);
  }
}

template <typename T>
void Matrix(const Frame &frame, Kernel operation, int64_t flags) {
  using L = Lapack<T>;
  using R = typename L::Real;
  constexpr bool complex = !std::is_same_v<T, R>;
  auto a = frame.Input(0);
  auto dims = a.dimensions();
  const size_t batch = Batch(a, 2);
  const lapack_int m = Size(dims[dims.size() - 2]);
  const lapack_int n = Size(dims[dims.size() - 1]);
  const lapack_int k = std::min(m, n);
  const size_t mn = static_cast<size_t>(m) * n;
  const size_t nn = static_cast<size_t>(n) * n;
  const bool flag = flags & 1;
  const bool flag2 = flags & 2;
  const size_t off = std::max<lapack_int>(0, n - 1);
  const auto *input = Input<T>(frame, 0, batch * mn);
  if (operation == kLu) {
    auto lu = frame.Output<T>(0, batch * mn);
    auto pivots = frame.Output<int32_t>(1, batch * k);
    auto permutation = frame.Output<int32_t>(2, batch * m);
    for (size_t i = 0; i < batch; ++i) {
      if (mn != 0) {
        std::copy_n(input + i * mn, mn, lu + i * mn);
        auto p = pivots + i * k;
        auto status = L::getrf(LAPACK_ROW_MAJOR, m, n, lu + i * mn, n, p);
        if (status < 0) {
          Nan(lu + i * mn, mn);
          std::iota(p, p + k, 0);
        } else {
          for (int j = 0; j < k; ++j)
            --p[j];
        }
      }
      if (m != 0)
        ApplyPivots(k == 0 ? nullptr : pivots + i * k, k, permutation + i * m,
                    m);
    }
    return;
  }
  if (operation == kSvd) {
    const bool vectors = flag2;
    const size_t uc = flag ? m : k;
    const size_t vr = flag ? n : k;
    const char job = vectors ? (flag ? 'A' : 'S') : 'N';
    auto u = vectors ? frame.Output<T>(0, batch * m * uc) : nullptr;
    auto s = frame.Output<R>(vectors ? 1 : 0, batch * k);
    auto vt = vectors ? frame.Output<T>(2, batch * vr * n) : nullptr;
    for (size_t i = 0; i < batch; ++i) {
      if (mn == 0) {
        if (vectors && flag) {
          if (m != 0) {
            auto p = u + i * m * uc;
            std::fill_n(p, m * uc, T{});
            for (int j = 0; j < m; ++j)
              p[j * uc + j] = T(1);
          }
          if (n != 0) {
            auto p = vt + i * vr * n;
            std::fill_n(p, vr * n, T{});
            for (int j = 0; j < n; ++j)
              p[j * static_cast<size_t>(n) + j] = T(1);
          }
        }
        continue;
      }
      std::vector<T> scratch(input + i * mn, input + (i + 1) * mn);
      std::vector<R> superb(std::max(1, k));
      T dummy{};
      auto ui = vectors ? u + i * m * uc : &dummy;
      auto vti = vectors ? vt + i * vr * n : &dummy;
      auto status = L::gesvd(LAPACK_ROW_MAJOR, job, job, m, n, scratch.data(),
                             n, s + i * k, ui, vectors ? Size(uc) : 1, vti,
                             vectors ? n : 1, superb.data());
      if (status != 0) {
        Nan(s + i * k, k);
        if (vectors) {
          Nan(ui, m * uc);
          Nan(vti, vr * n);
        }
      }
    }
    return;
  }
  if (operation == kPivotedQr) {
    const lapack_int qc = flag ? m : k;
    const size_t qs = static_cast<size_t>(m) * qc;
    const size_t rs = static_cast<size_t>(qc) * n;
    auto q = frame.Output<T>(0, batch * qs);
    auto r = frame.Output<T>(1, batch * rs);
    auto p = frame.Output<int32_t>(2, batch * n);
    for (size_t i = 0; i < batch; ++i) {
      if (n)
        std::fill_n(p + i * n, n, 0);
      if (!mn) {
        if (n)
          std::iota(p + i * n, p + (i + 1) * n, 0);
        if (qs) {
          std::fill_n(q + i * qs, qs, T{});
          for (int j = 0; j < m; ++j)
            q[i * qs + j * static_cast<size_t>(qc) + j] = T(1);
        }
        continue;
      }
      std::vector<T> a(input + i * mn, input + (i + 1) * mn);
      std::vector<T> tau(k);
      auto status =
          L::geqp3(LAPACK_ROW_MAJOR, m, n, a.data(), n, p + i * n, tau.data());
      if (status == 0) {
        for (int j = 0; j < n; ++j)
          --p[i * n + j];
        for (int row = 0; row < qc; ++row) {
          for (int col = 0; col < n; ++col) {
            r[i * rs + row * static_cast<size_t>(n) + col] =
                row <= col ? a[row * static_cast<size_t>(n) + col] : T{};
          }
        }
        auto qi = q + i * qs;
        std::fill_n(qi, qs, T{});
        for (int row = 0; row < m; ++row) {
          std::copy_n(a.data() + row * static_cast<size_t>(n), k,
                      qi + row * static_cast<size_t>(qc));
        }
        status = L::gqr(LAPACK_ROW_MAJOR, m, qc, k, qi, qc, tau.data());
      }
      if (status != 0) {
        Nan(q + i * qs, qs);
        Nan(r + i * rs, rs);
      }
    }
    return;
  }
  if (m != n)
    throw std::invalid_argument("expected square LAPACK matrix");
  T *matrix = operation == kEig ? nullptr : frame.Output<T>(0, batch * nn);
  if (operation == kHessenberg || operation == kTridiagonal) {
    auto tau = frame.Output<T>(operation == kHessenberg ? 1 : 3, batch * off);
    auto diag =
        operation == kTridiagonal ? frame.Output<R>(1, batch * n) : nullptr;
    auto sub =
        operation == kTridiagonal ? frame.Output<R>(2, batch * off) : nullptr;
    if (n == 0)
      return;
    for (size_t i = 0; i < batch; ++i) {
      auto mat = matrix + i * nn;
      std::copy_n(input + i * nn, nn, mat);
      T dummy_tau{};
      R dummy_sub{};
      auto t = off ? tau + i * off : &dummy_tau;
      int status =
          operation == kHessenberg
              ? L::gehrd(LAPACK_ROW_MAJOR, n, 1, n, mat, n, t)
              : L::trd(LAPACK_ROW_MAJOR, flag ? 'L' : 'U', n, mat, n,
                       diag + i * n, off ? sub + i * off : &dummy_sub, t);
      if (status != 0) {
        Nan(mat, nn);
        Nan(t, off);
        if (diag) {
          Nan(diag + i * n, n);
          Nan(off ? sub + i * off : &dummy_sub, off);
        }
      }
    }
    return;
  }
  if (operation == kSchur) {
    auto vectors = flag ? frame.Output<T>(1, batch * nn) : nullptr;
    if (n == 0)
      return;
    for (size_t i = 0; i < batch; ++i) {
      auto mat = matrix + i * nn;
      std::copy_n(input + i * nn, nn, mat);
      std::vector<T> eig(n);
      T dummy{};
      auto v = flag ? vectors + i * nn : &dummy;
      lapack_int count;
      const char job = flag ? 'V' : 'N';
      int status;
      if constexpr (complex) {
        status = L::gees(LAPACK_ROW_MAJOR, job, 'N', nullptr, n, mat, n, &count,
                         eig.data(), v, flag ? n : 1);
      } else {
        std::vector<T> imag(n);
        status = L::gees(LAPACK_ROW_MAJOR, job, 'N', nullptr, n, mat, n, &count,
                         eig.data(), imag.data(), v, flag ? n : 1);
      }
      if (status != 0) {
        Nan(mat, nn);
        if (flag)
          Nan(v, nn);
      }
    }
    return;
  }
  if (operation == kEig) {
    if constexpr (complex) {
      auto values = frame.Output<T>(0, batch * n);
      auto left = flag ? frame.Output<T>(1, batch * nn) : nullptr;
      auto right = flag2 ? frame.Output<T>(1 + flag, batch * nn) : nullptr;
      if (n == 0)
        return;
      for (size_t i = 0; i < batch; ++i) {
        std::vector<T> a(input + i * nn, input + (i + 1) * nn);
        T dummy{};
        auto l = flag ? left + i * nn : &dummy;
        auto r = flag2 ? right + i * nn : &dummy;
        auto status = L::geev(LAPACK_ROW_MAJOR, flag ? 'V' : 'N',
                              flag2 ? 'V' : 'N', n, a.data(), n, values + i * n,
                              l, flag ? n : 1, r, flag2 ? n : 1);
        if (status != 0) {
          Nan(values + i * n, n);
          if (flag)
            Nan(l, nn);
          if (flag2)
            Nan(r, nn);
        }
      }
      return;
    }
  }
  throw std::invalid_argument("unknown LAPACK operation or unsupported dtype");
}

ffi::Error Run(int64_t operation, int64_t flags, ffi::RemainingArgs args,
               ffi::RemainingRets results) {
  try {
    const Frame frame(args, results);
    if (operation == kPivotsToPermutation) {
      Permutation(frame, Size(flags));
    } else {
      auto dispatch = [&](auto type) {
        using T = decltype(type);
        if (operation == kTridiagonalSolve)
          SolveTridiagonal<T>(frame, flags & 1);
        else
          Matrix<T>(frame, static_cast<Kernel>(operation), flags);
      };
      switch (frame.Input(0).element_type()) {
      case DataType::F32:
        dispatch(float{});
        break;
      case DataType::F64:
        dispatch(double{});
        break;
      case DataType::C64:
        dispatch(std::complex<float>{});
        break;
      case DataType::C128:
        dispatch(std::complex<double>{});
        break;
      default:
        return ffi::Error::InvalidArgument("invalid LAPACK input dtype");
      }
    }
    return ffi::Error::Success();
  } catch (const std::exception &error) {
    return ffi::Error::InvalidArgument(error.what());
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GnxLapack, Run,
                              ffi::Ffi::Bind()
                                  .Attr<int64_t>("operation")
                                  .Attr<int64_t>("flags")
                                  .RemainingArgs()
                                  .RemainingRets());
} // namespace

extern "C" XLA_FFI_Error *gnx_lapack_execute(XLA_FFI_CallFrame *frame) {
  return GnxLapack(frame);
}
