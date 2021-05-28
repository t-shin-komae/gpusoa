#include "array.hpp"
#include "vectorarray.hpp"
#include <fstream>
#include <iostream>

using KRGBImage = soa::KVectorArray2D<3>;
using RGBImage = soa::VectorArray2D<3>;
using RGB = soa::Vector3;

__global__ void gaussian(KRGBImage src, KRGBImage dst) {
  auto i = threadIdx.x + blockIdx.x * blockDim.x;
  auto j = threadIdx.y + blockIdx.y * blockDim.y;
  auto Nx = src.sizeX();
  auto Ny = src.sizeY();
  if (1 < i && i < Nx - 1 && 1 < j && j < Ny - 1) {
    soa::Vector3 sum{0., 0., 0.};
    for (int k = -1; k <= 1; k++) {
      for (int l = -1; l <= 1; l++) {
        sum[0] += 1. / 9. * src(i + k, j + l)[0];
        sum[1] += 1. / 9. * src(i + k, j + l)[1];
        sum[2] += 1. / 9. * src(i + k, j + l)[2];
      }
    }
    dst(i, j) = sum;
  }
}

int main() {
  using namespace std::literals::string_literals;
  int Nx = 1024;
  int Ny = 1024;
  auto sq = [](auto x) { return x * x; };
  auto within = [=](int i, int j, int r) {
    return sq(i - Nx / 2) + sq(j - Ny / 2) < sq(r);
  };
  RGBImage image(Nx, Ny, [=](int i, int j) {
    if (within(i, j, Nx / 8)) {
      return RGB{1.0, 1.0, 1.0};
    } else if (within(i, j, Nx / 4)) {
      return RGB{0.75, 0.25, 0.25};
    } else if (within(i, j, Nx / 2)) {
      return RGB{0.5, 0.25, 0.0};
    } else {
      return RGB{0., 0., 0.};
    }
  });
  RGBImage tmp(Nx, Ny);
  auto filename = "output.ppm"s;
  std::ofstream ofs(filename);
  ofs << "P3" << std::endl;
  ofs << Nx << ' ' << Ny << std::endl;
  ofs << 255 << std::endl;
  // Apply gaussian filter 100 times
  for (int i = 0; i < 100; i++) {
    gaussian<<<dim3(Nx / 32, Ny / 32), dim3(32, 32)>>>(image.gpu(),
                                                           tmp.gpu());
    std::swap(tmp,image);
  }
  image.DeviceToHost();

  for (int i = 0; i < Nx; i++) {
    for (int j = 0; j < Ny; j++) {
      auto tmp = static_cast<soa::Vector3>(image(i, j));
      ofs << int(259.99 * tmp[0]) << ' ' << int(259.99 * tmp[1]) << ' '
          << int(259.99 * tmp[2]) << ' ' << std::endl;
    }
  }
}
