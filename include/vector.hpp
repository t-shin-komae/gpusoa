#pragma once
#include "macros.hpp"
namespace soa {
template <int N> class VectorProxy;

template <int N> class Vector {
public:
  double __storage[N];
  DEVICEHOST
  double &operator[](int i) { return __storage[i]; }
  DEVICEHOST
  const double &operator[](int i) const { return __storage[i]; }
};

template <int N> class VectorProxy {
  double *__field_ptrs[N];
  int index;

public:
  using vector_type = Vector<N>;
  DEVICEHOST
  VectorProxy(double *(&ptrs)[N], int index) {
    this->index = index;
    for (auto i = 0; i < N; i++)
      __field_ptrs[i] = ptrs[i];
  }
  DEVICEHOST
  double &operator[](int i) { return __field_ptrs[i][index]; }

  DEVICEHOST
  void operator=(const vector_type &rhs) {
    for (auto i = 0; i < N; i++)
      __field_ptrs[i][index] = rhs[i];
  }
  DEVICEHOST
  explicit operator vector_type() {
    vector_type result;
    for (auto i = 0; i < N; i++)
      result[i] = __field_ptrs[i][index];
    return result;
  }
};
template <int N> class ConstVectorProxy {
  double *__field_ptrs[N];
  int index;

public:
  using vector_type = Vector<N>;
  DEVICEHOST
  ConstVectorProxy(double *const (&ptrs)[N], int index) {
    this->index = index;
    for (auto i = 0; i < N; i++)
      __field_ptrs[i] = ptrs[i];
  }
  DEVICEHOST
  const double &operator[](int i) const { return __field_ptrs[i][index]; }

  DEVICEHOST
  explicit operator vector_type() {
    vector_type result;
    for (auto i = 0; i < N; i++)
      result[i] = __field_ptrs[i][index];
    return result;
  }
};

using Vector2 = Vector<2>;
using Vector3 = Vector<3>;
using Vector4 = Vector<4>;
using Vector5 = Vector<5>;
using Vector6 = Vector<6>;

using Vector2Proxy = VectorProxy<2>;
using Vector3Proxy = VectorProxy<3>;
using Vector4Proxy = VectorProxy<4>;
using Vector5Proxy = VectorProxy<5>;
using Vector6Proxy = VectorProxy<6>;

using ConstVector2Proxy = ConstVectorProxy<2>;
using ConstVector3Proxy = ConstVectorProxy<3>;
using ConstVector4Proxy = ConstVectorProxy<4>;
using ConstVector5Proxy = ConstVectorProxy<5>;
using ConstVector6Proxy = ConstVectorProxy<6>;

} // namespace soa
