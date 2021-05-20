#pragma once
#include "macros.hpp"
#include "vector.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>
namespace soa {
void check() {
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << err << std::endl;
    throw std::runtime_error(cudaGetErrorString(err));
  }
}
class DArray1D {
  double *q;
  int Nx;
  int count;

public:
  DArray1D(int Nx) {
    this->Nx = Nx;
    count = 1;
    std::cout << "cudaMalloc\n" << std::endl;
    cudaMalloc(&q, sizeof(double) * Nx);
    check();
  }
  ~DArray1D() {
    std::cout << "cudaFree\n" << std::endl;
    count--;
    if (count == 0) {
      cudaFree(q);
      check();
    }
  }
  DArray1D &operator=(const DArray1D &) = delete;
  DArray1D(const DArray1D &src) {
    q = src.q;
    Nx = src.Nx;
    count++;
    printf("Copy Constructor\n");
  }
  DArray1D(DArray1D &&) = delete;
  DArray1D &operator=(DArray1D &&) = delete;
  DEVICEHOST
  int size() const { return Nx; }

  DEVICE
  inline double &operator()(int i) {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx);
#endif
    return q[i];
  }
  DEVICE
  inline const double &operator()(int i) const {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx);
#endif
    return q[i];
  }
};
class DArray2D {
  double *q;
  int Nx, Ny;

public:
  DArray2D(int Nx, int Ny);
  ~DArray2D();
  DArray2D &operator=(const DArray2D &);
  DArray2D(const DArray2D &);
  DArray2D(DArray2D &&);
  DArray2D &operator=(DArray2D &&);
  DEVICE
  inline double &operator()(int i, int j) {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx && j < Ny);
#endif
    return q[i + j * Nx];
  }
  DEVICE
  inline const double &operator()(int i, int j) const {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx && j < Ny);
#endif
    return q[i + j * Nx];
  }
};
class DArray3D {
  double *q;
  int Nx, Ny, Nz;

public:
  DArray3D(int Nx, int Ny, int Nz);
  ~DArray3D();
  DArray3D &operator=(const DArray3D &);
  DArray3D(const DArray3D &);
  DArray3D(DArray3D &&);
  DArray3D &operator=(DArray3D &&);
  DEVICE
  inline double &operator()(int i, int j, int k) {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx && j < Ny && k < Nz);
#endif
    return q[i + j * Nx + k * Nx * Ny];
  }
  DEVICE
  inline const double &operator()(int i, int j, int k) const {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx && j < Ny && k < Nz);
#endif
    return q[i + j * Nx + k * Nx * Ny];
  }
};

template <int N> class DVectorArray1D {
  double *q[N];
  int Nx;

public:
  DVectorArray1D(int Nx);
  ~DVectorArray1D();
  DVectorArray1D &operator=(const DVectorArray1D &);
  DVectorArray1D(const DVectorArray1D &);
  DVectorArray1D(DVectorArray1D &&);
  DVectorArray1D &operator=(DVectorArray1D &&);

  DEVICE
  inline VectorProxy<N> operator()(int i) {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx);
#endif
    return VectorProxy<N>(q, i);
  }
  DEVICE
  inline ConstVectorProxy<N> operator()(int i) const {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx);
#endif
    return ConstVectorProxy<N>(q, i);
  }
};

template <int N> class DVectorArray2D {
  double *q[N];
  int Nx;

public:
  DVectorArray2D(int Nx);
  ~DVectorArray2D();
  DVectorArray2D &operator=(const DVectorArray2D &);
  DVectorArray2D(const DVectorArray2D &);
  DVectorArray2D(DVectorArray2D &&);
  DVectorArray2D &operator=(DVectorArray2D &&);

  DEVICE
  inline VectorProxy<N> operator()(int i, int j) {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx && j < Ny);
#endif
    return VectorProxy<N>(q, i + j * Nx);
  }
  DEVICE
  inline ConstVectorProxy<N> operator()(int i, int j) const {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx && j < Ny);
#endif
    return ConstVectorProxy<N>(q, i + j * Nx);
  }
};

} // namespace soa
