#pragma once

#include <iostream>
#include <stdexcept>

namespace soa {
inline void check() {
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << err << std::endl;
    throw std::runtime_error(cudaGetErrorString(err));
  }
}
inline void check(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << err << std::endl;
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

} // namespace soa
