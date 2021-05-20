#include "array.hpp"
#include "cuda_utils.hpp"
#include <stdexcept>

namespace soa {
void Array1D::DeviceToHost() {
  check(cudaMemcpy(H_ptr, D_ptr, sizeof(double) * Nx, cudaMemcpyDeviceToHost));
}
void Array1D::HostToDevice() {
  check(cudaMemcpy(D_ptr, H_ptr, sizeof(double) * Nx, cudaMemcpyHostToDevice));
}
void Array2D::DeviceToHost() {
  check(cudaMemcpy(H_ptr, D_ptr, sizeof(double) * Nx * Ny,
                   cudaMemcpyDeviceToHost));
}
void Array2D::HostToDevice() {
  check(cudaMemcpy(D_ptr, H_ptr, sizeof(double) * Nx * Ny,
                   cudaMemcpyHostToDevice));
}
void Array3D::DeviceToHost() {
  check(cudaMemcpy(H_ptr, D_ptr, sizeof(double) * Nx * Ny * Nz,
                   cudaMemcpyDeviceToHost));
}
void Array3D::HostToDevice() {
  check(cudaMemcpy(D_ptr, H_ptr, sizeof(double) * Nx * Ny * Nz,
                   cudaMemcpyHostToDevice));
}
} // namespace soa
