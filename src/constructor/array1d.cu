#include "array.hpp"
#include "cuda_utils.hpp"
#include <stdexcept>

namespace soa {
KArray1D::KArray1D(double *D_ptr, int Nx) : D_ptr(D_ptr), Nx(Nx) {}
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
void Array1D::DeviceToHost() {
  check(cudaMemcpy(H_ptr, D_ptr, sizeof(double) * Nx, cudaMemcpyDeviceToHost));
}
void Array1D::HostToDevice() {
  check(cudaMemcpy(D_ptr, H_ptr, sizeof(double) * Nx, cudaMemcpyHostToDevice));
}

#endif
} // namespace soa
