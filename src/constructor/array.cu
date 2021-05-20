#include "array.hpp"
#include "cuda_utils.hpp"
#include <stdexcept>

namespace soa {
KArray1D::KArray1D(double *D_ptr, int Nx) : D_ptr(D_ptr), Nx(Nx) {}

KArray2D::KArray2D(double *D_ptr, int Nx, int Ny)
    : D_ptr(D_ptr), Nx(Nx), Ny(Ny) {}
KArray3D::KArray3D(double *D_ptr, int Nx, int Ny, int Nz)
    : D_ptr(D_ptr), Nx(Nx), Ny(Ny), Nz(Nz) {}

#ifdef __CUDACC__
Array1D::Array1D(int Nx) : Nx(Nx) {
  std::cout << "cudaMalloc\n" << std::endl;
  check(cudaMallocHost(&H_ptr, sizeof(double) * Nx));
  check(cudaMalloc(&D_ptr, sizeof(double) * Nx));
}
Array1D::Array1D(int Nx, std::function<double(int)> f) : Array1D(Nx) {
  for (int i = 0; i < Nx; i++)
    H_ptr[i] = f(i);

  check(cudaMemcpy(D_ptr, H_ptr, sizeof(double) * Nx, cudaMemcpyHostToDevice));
}

Array1D::~Array1D() {
  std::cout << "cudaFree\n" << std::endl;
  check(cudaFreeHost(H_ptr));
  check(cudaFree(D_ptr));
}
Array1D::Array1D(Array1D &&rhs) noexcept
    : H_ptr(rhs.H_ptr), D_ptr(rhs.D_ptr), Nx(rhs.Nx) {
  rhs.H_ptr = nullptr;
  rhs.D_ptr = nullptr;
}
Array1D &Array1D::operator=(Array1D &&rhs) noexcept {
  if (this != &rhs) {
    H_ptr = rhs.H_ptr;
    D_ptr = rhs.D_ptr;
    rhs.H_ptr = nullptr;
    rhs.D_ptr = nullptr;
    Nx = rhs.Nx;
  }
  return *this;
}

Array2D::Array2D(int Nx, int Ny) : Nx(Nx), Ny(Ny) {
  std::cout << "cudaMalloc\n" << std::endl;
  check(cudaMallocHost(&H_ptr, sizeof(double) * Nx * Ny));
  check(cudaMalloc(&D_ptr, sizeof(double) * Nx * Ny));
}
Array2D::Array2D(int Nx, int Ny, std::function<double(int, int)> f)
    : Array2D(Nx, Ny) {
  for (int j = 0; j < Ny; j++)
    for (int i = 0; i < Nx; i++)
      H_ptr[i + j * Nx] = f(i, j);

  check(cudaMemcpy(D_ptr, H_ptr, sizeof(double) * Nx * Ny,
                   cudaMemcpyHostToDevice));
}
Array2D::~Array2D() {
  std::cout << "cudaFree\n" << std::endl;
  check(cudaFreeHost(H_ptr));
  check(cudaFree(D_ptr));
}
Array2D::Array2D(Array2D &&rhs) noexcept
    : H_ptr(rhs.H_ptr), D_ptr(rhs.D_ptr), Nx(rhs.Nx), Ny(rhs.Ny) {
  rhs.H_ptr = nullptr;
  rhs.D_ptr = nullptr;
}
Array2D &Array2D::operator=(Array2D &&rhs) noexcept {
  if (this != &rhs) {
    H_ptr = rhs.H_ptr;
    D_ptr = rhs.D_ptr;
    rhs.H_ptr = nullptr;
    rhs.D_ptr = nullptr;
    Nx = rhs.Nx;
    Ny = rhs.Ny;
  }
  return *this;
}

Array3D::Array3D(int Nx, int Ny, int Nz) : Nx(Nx), Ny(Ny), Nz(Nz) {
  std::cout << "cudaMalloc\n" << std::endl;
  check(cudaMallocHost(&H_ptr, sizeof(double) * Nx * Ny * Nz));
  check(cudaMalloc(&D_ptr, sizeof(double) * Nx * Ny * Nz));
}
Array3D::Array3D(int Nx, int Ny, int Nz, std::function<double(int, int, int)> f)
    : Array3D(Nx, Ny, Nz) {
  for (int k = 0; k < Nz; k++)
    for (int j = 0; j < Ny; j++)
      for (int i = 0; i < Nx; i++)
        H_ptr[i + j * Nx + k * Nx * Ny] = f(i, j, k);

  check(cudaMemcpy(D_ptr, H_ptr, sizeof(double) * Nx * Ny * Nz,
                   cudaMemcpyHostToDevice));
}
Array3D::~Array3D() {
  std::cout << "cudaFree\n" << std::endl;
  check(cudaFreeHost(H_ptr));
  check(cudaFree(D_ptr));
}
Array3D::Array3D(Array3D &&rhs) noexcept
    : H_ptr(rhs.H_ptr), D_ptr(rhs.D_ptr), Nx(rhs.Nx), Ny(rhs.Ny), Nz(rhs.Nz) {
  rhs.H_ptr = nullptr;
  rhs.D_ptr = nullptr;
}
Array3D &Array3D::operator=(Array3D &&rhs) noexcept {
  if (this != &rhs) {
    H_ptr = rhs.H_ptr;
    D_ptr = rhs.D_ptr;
    rhs.H_ptr = nullptr;
    rhs.D_ptr = nullptr;
    Nx = rhs.Nx;
    Ny = rhs.Ny;
    Nz = rhs.Nz;
  }
  return *this;
}
#endif
} // namespace soa
