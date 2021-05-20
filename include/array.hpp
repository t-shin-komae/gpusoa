#pragma once
#include "macros.hpp"
#include "vector.hpp"
#include <functional>
#include <iostream>
#include <memory>
#include <tuple>
#include <thrust/pair.h>
#include <thrust/tuple.h>
namespace soa {
class KArray1D {
  double *D_ptr;
  int Nx;

public:
  KArray1D(double *D_ptr, int Nx);
  DEVICE
  inline double &operator()(int i) {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx);
#endif
    return D_ptr[i];
  }
  DEVICE
  inline const double &operator()(int i) const {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx);
#endif
    return D_ptr[i];
  }
  DEVICE
  int size() const { return Nx; }
};

class KArray2D {
  double *D_ptr;
  int Nx, Ny;

public:
  KArray2D(double *D_ptr, int Nx, int Ny);
  DEVICE inline double &operator()(int i, int j) {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx && j < Ny);
#endif
    return D_ptr[i + j * Nx];
  }
  DEVICE inline const double &operator()(int i, int j) const {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx && j < Ny);
#endif
    return D_ptr[i + j * Nx];
  }
  DEVICE thrust::pair<int, int> size() const { return {Nx, Ny}; }
  DEVICE int sizeX() const { return Nx; }
  DEVICE int sizeY() const { return Ny; }
};

class KArray3D {
  double *D_ptr;
  int Nx, Ny, Nz;

public:
  KArray3D(double *D_ptr, int Nx, int Ny, int Nz);
  DEVICE inline double &operator()(int i, int j, int k) {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx && j < Ny && k < Nz);
#endif
    return D_ptr[i + j * Nx + k * Nx * Ny];
  }
  DEVICE inline const double &operator()(int i, int j, int k) const {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx && j < Ny && k < Nz);
#endif
    return D_ptr[i + j * Nx + k * Nx * Ny];
  }
  DEVICE thrust::tuple<int, int, int> size() const { return {Nx, Ny, Nz}; }
  DEVICE int sizeX() const { return Nx; }
  DEVICE int sizeY() const { return Ny; }
  DEVICE int sizeZ() const { return Nz; }
};

class Array1D {
  double *H_ptr, *D_ptr;
  int Nx;

public:
  Array1D(int Nx);
  Array1D(int Nx, std::function<double(int)> f);
  ~Array1D();
  Array1D &operator=(const Array1D &) = delete;
  Array1D(const Array1D &src) = delete;
  Array1D(Array1D &&) noexcept;
  Array1D &operator=(Array1D &&) noexcept;

  void DeviceToHost();
  void HostToDevice();
  inline int size() const { return Nx; }
  inline KArray1D gpu() const { return KArray1D(D_ptr, Nx); }

  inline double &operator()(int i) {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx);
#endif
    return H_ptr[i];
  }
  inline const double &operator()(int i) const {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx);
#endif
    return H_ptr[i];
  }
};
class Array2D {
  double *H_ptr, *D_ptr;
  int Nx, Ny;

public:
  Array2D(int Nx, int Ny);
  Array2D(int Nx, int Ny, std::function<double(int, int)> f);
  ~Array2D();
  Array2D &operator=(const Array2D &) = delete;
  Array2D(const Array2D &) = delete;
  Array2D(Array2D &&) noexcept;
  Array2D &operator=(Array2D &&) noexcept;
  void DeviceToHost();
  void HostToDevice();
  inline std::pair<int, int> size() const { return {Nx, Ny}; }
  inline int sizeX() const { return Nx; }
  inline int sizeY() const { return Ny; }

  inline double &operator()(int i, int j) {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx && j < Ny);
#endif
    return H_ptr[i + j * Nx];
  }
  inline const double &operator()(int i, int j) const {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx && j < Ny);
#endif
    return H_ptr[i + j * Nx];
  }
};
class Array3D {
  double *H_ptr, *D_ptr;
  int Nx, Ny, Nz;

public:
  Array3D(int Nx, int Ny, int Nz);
  Array3D(int Nx, int Ny, int Nz, std::function<double(int, int, int)> f);
  ~Array3D();
  Array3D &operator=(const Array3D &) = delete;
  Array3D(const Array3D &) = delete;
  Array3D(Array3D &&) noexcept;
  Array3D &operator=(Array3D &&) noexcept;
  void DeviceToHost();
  void HostToDevice();
  inline std::tuple<int, int, int> size() const { return {Nx, Ny, Nz}; }
  inline int sizeX() const { return Nx; }
  inline int sizeY() const { return Ny; }
  inline int sizeZ() const { return Nz; }
  inline double &operator()(int i, int j, int k) {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx && j < Ny && k < Nz);
#endif
    return H_ptr[i + j * Nx + k * Nx * Ny];
  }
  inline const double &operator()(int i, int j, int k) const {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx && j < Ny && k < Nz);
#endif
    return H_ptr[i + j * Nx + k * Nx * Ny];
  }
};

} // namespace soa
