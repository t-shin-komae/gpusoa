#include "cuda_utils.hpp"
#include "vectorarray.hpp"
namespace soa {
// ===================== Kernel Vector Construction =====================================
template <int N>
KVectorArray1D<N>::KVectorArray1D(double *const (&D_ptrs)[N], int Nx) : Nx(Nx) {
  for (int i = 0; i < N; i++)
    this->D_ptrs[i] = D_ptrs[i];
}
template <int N>
KVectorArray2D<N>::KVectorArray2D(double *const (&D_ptrs)[N], int Nx, int Ny)
    : Nx(Nx), Ny(Ny) {
  for (int i = 0; i < N; i++)
    this->D_ptrs[i] = D_ptrs[i];
}
template <int N>
KVectorArray3D<N>::KVectorArray3D(double *const (&D_ptrs)[N], int Nx, int Ny, int Nz)
    : Nx(Nx), Ny(Ny), Nz(Nz) {
  for (int i = 0; i < N; i++)
    this->D_ptrs[i] = D_ptrs[i];
}

// ================ VectorArray1D Construction and move assignment ========================

template <int N> VectorArray1D<N>::VectorArray1D(int Nx) : Nx(Nx) {
  double *H_ptr, *D_ptr;
  check(cudaMallocHost(&H_ptr, sizeof(double) * Nx * N));
  check(cudaMalloc(&D_ptr, sizeof(double) * Nx * N));
  for (int i = 0; i < N; i++) {
    H_ptrs[i] = H_ptr + Nx * i;
    D_ptrs[i] = D_ptr + Nx * i;
  }
}
template <int N>
VectorArray1D<N>::VectorArray1D(int Nx, std::function<Vector<N>(int)> f)
    : VectorArray1D(Nx) {
  for (int i = 0; i < Nx; i++)
    this->operator()(i) = f(i);

  check(cudaMemcpy(D_ptrs[0], H_ptrs[0], sizeof(double) * Nx * N,
                   cudaMemcpyHostToDevice));
}
template <int N> VectorArray1D<N>::~VectorArray1D() {
  check(cudaFreeHost(H_ptrs[0]));
  check(cudaFree(D_ptrs[0]));
}
template <int N>
VectorArray1D<N>::VectorArray1D(VectorArray1D<N> &&rhs) noexcept : Nx(rhs.Nx) {
  for (int i = 0; i < N; i++) {
    H_ptrs[i] = rhs.H_ptrs[i];
    D_ptrs[i] = rhs.D_ptrs[i];
    rhs.H_ptrs[i] = nullptr;
    rhs.D_ptrs[i] = nullptr;
  }
}
template <int N>
VectorArray1D<N> &VectorArray1D<N>::operator=(VectorArray1D<N> &&rhs) noexcept {
  if (this != &rhs) {
    for (int i = 0; i < N; i++) {
      H_ptrs[i] = rhs.H_ptrs[i];
      D_ptrs[i] = rhs.D_ptrs[i];
      rhs.H_ptrs[i] = nullptr;
      rhs.D_ptrs[i] = nullptr;
    }
    Nx = rhs.Nx;
  }
  return *this;
}
// ================ VectorArray2D Construction and move assignment ========================
template <int N> VectorArray2D<N>::VectorArray2D(int Nx, int Ny) : Nx(Nx) ,Ny(Ny){
  double *H_ptr, *D_ptr;
  check(cudaMallocHost(&H_ptr, sizeof(double) * Nx * Ny * N));
  check(cudaMalloc(&D_ptr, sizeof(double) * Nx * Ny * N));
  for (int i = 0; i < N; i++) {
    H_ptrs[i] = H_ptr + Nx * Ny * i;
    D_ptrs[i] = D_ptr + Nx * Ny * i;
  }
}
template <int N>
VectorArray2D<N>::VectorArray2D(int Nx, int Ny,
                                std::function<Vector<N>(int, int)> f)
    : VectorArray2D(Nx, Ny) {
  for (int j = 0; j < Ny; j++)
    for (int i = 0; i < Nx; i++)
      this->operator()(i, j) = f(i, j);

  check(cudaMemcpy(D_ptrs[0], H_ptrs[0], sizeof(double) * Nx * Ny * N,
                   cudaMemcpyHostToDevice));
}
template <int N> VectorArray2D<N>::~VectorArray2D() {
  check(cudaFreeHost(H_ptrs[0]));
  check(cudaFree(D_ptrs[0]));
}
template <int N>
VectorArray2D<N>::VectorArray2D(VectorArray2D<N> &&rhs) noexcept : Nx(rhs.Nx) , Ny(rhs.Ny){
  for (int i = 0; i < N; i++) {
    H_ptrs[i] = rhs.H_ptrs[i];
    D_ptrs[i] = rhs.D_ptrs[i];
    rhs.H_ptrs[i] = nullptr;
    rhs.D_ptrs[i] = nullptr;
  }
}
template <int N>
VectorArray2D<N> &VectorArray2D<N>::operator=(VectorArray2D<N> &&rhs) noexcept {
  if (this != &rhs) {
    for (int i = 0; i < N; i++) {
      H_ptrs[i] = rhs.H_ptrs[i];
      D_ptrs[i] = rhs.D_ptrs[i];
      rhs.H_ptrs[i] = nullptr;
      rhs.D_ptrs[i] = nullptr;
    }
    Nx = rhs.Nx;
    Ny = rhs.Ny;
  }
  return *this;
}

// ================ VectorArray3D Construction and move assignment ========================
template <int N> VectorArray3D<N>::VectorArray3D(int Nx, int Ny, int Nz) : Nx(Nx) ,Ny(Ny), Nz(Nz){
  double *H_ptr, *D_ptr;
  check(cudaMallocHost(&H_ptr, sizeof(double) * Nx * Ny * Nz * N));
  check(cudaMalloc(&D_ptr, sizeof(double) * Nx * Ny * Nz * N));
  for (int i = 0; i < N; i++) {
    H_ptrs[i] = H_ptr + Nx * Ny * Nz * i;
    D_ptrs[i] = D_ptr + Nx * Ny * Nz * i;
  }
}
template <int N>
VectorArray3D<N>::VectorArray3D(int Nx, int Ny, int Nz,
                                std::function<Vector<N>(int, int, int)> f)
    : VectorArray3D(Nx, Ny, Nz) {
  for (int k = 0; k < Nz; k++)
  for (int j = 0; j < Ny; j++)
    for (int i = 0; i < Nx; i++)
      this->operator()(i, j, k) = f(i, j, k);

  check(cudaMemcpy(D_ptrs[0], H_ptrs[0], sizeof(double) * Nx * Ny * Nz * N,
                   cudaMemcpyHostToDevice));
}
template <int N> VectorArray3D<N>::~VectorArray3D() {
  check(cudaFreeHost(H_ptrs[0]));
  check(cudaFree(D_ptrs[0]));
}
template <int N>
VectorArray3D<N>::VectorArray3D(VectorArray3D<N> &&rhs) noexcept
    : Nx(rhs.Nx), Ny(rhs.Ny), Nz(rhs.Nz) {
  for (int i = 0; i < N; i++) {
    H_ptrs[i] = rhs.H_ptrs[i];
    D_ptrs[i] = rhs.D_ptrs[i];
    rhs.H_ptrs[i] = nullptr;
    rhs.D_ptrs[i] = nullptr;
  }
}
template <int N>
VectorArray3D<N> &VectorArray3D<N>::operator=(VectorArray3D<N> &&rhs) noexcept {
  if (this != &rhs) {
    for (int i = 0; i < N; i++) {
      H_ptrs[i] = rhs.H_ptrs[i];
      D_ptrs[i] = rhs.D_ptrs[i];
      rhs.H_ptrs[i] = nullptr;
      rhs.D_ptrs[i] = nullptr;
    }
    Nx = rhs.Nx;
    Ny = rhs.Ny;
    Nz = rhs.Nz;
  }
  return *this;
}

template <int N> void VectorArray1D<N>::DeviceToHost() {
  check(cudaMemcpy(H_ptrs[0], D_ptrs[0], sizeof(double) * Nx * N,
                   cudaMemcpyDeviceToHost));
}
template <int N> void VectorArray1D<N>::HostToDevice() {
  check(cudaMemcpy(D_ptrs[0], H_ptrs[0], sizeof(double) * Nx * N,
                   cudaMemcpyHostToDevice));
}

template <int N> void VectorArray2D<N>::DeviceToHost() {
  check(cudaMemcpy(H_ptrs[0], D_ptrs[0], sizeof(double) * Nx * Ny * N,
                   cudaMemcpyDeviceToHost));
}
template <int N> void VectorArray2D<N>::HostToDevice() {
  check(cudaMemcpy(D_ptrs[0], H_ptrs[0], sizeof(double) * Nx * Ny * N,
                   cudaMemcpyHostToDevice));
}

template <int N> void VectorArray3D<N>::DeviceToHost() {
  check(cudaMemcpy(H_ptrs[0], D_ptrs[0], sizeof(double) * Nx * Ny * Nz * N,
                   cudaMemcpyDeviceToHost));
}
template <int N> void VectorArray3D<N>::HostToDevice() {
  check(cudaMemcpy(D_ptrs[0], H_ptrs[0], sizeof(double) * Nx * Ny * Nz * N,
                   cudaMemcpyHostToDevice));
}
} // namespace soa

#include "instanciation.ipp"
