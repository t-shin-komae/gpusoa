#include "cuda_utils.hpp"
#include "vectorarray.hpp"
namespace soa {

template <int N> VectorArray1D<N>::VectorArray1D(int Nx) : Nx(Nx) {
  double *H_ptr, *D_ptr;
  std::cout << "cudaMalloc\n" << std::endl;
  check(cudaMallocHost(&H_ptr, sizeof(double) * Nx * N));
  check(cudaMalloc(&D_ptr, sizeof(double) * Nx * N));
  for (int i = 0; i < N; i++) {
    H_ptrs[i] = H_ptr + Nx * i;
    D_ptrs[i] = D_ptr + Nx * i;
  }
}
template <int N>
VectorArray1D<N>::VectorArray1D(int Nx, std::function<Vector<N>(int)> f)
    : Nx(Nx) {
  for (int i = 0; i < Nx; i++)
    this->operator()(i) = f(i);

  check(cudaMemcpy(D_ptrs[0], H_ptrs[0], sizeof(double) * Nx * N,
                   cudaMemcpyHostToDevice));
}
template <int N> VectorArray1D<N>::~VectorArray1D() {
  std::cout << "cudaFree\n" << std::endl;
  check(cudaFreeHost(H_ptrs[0]));
  check(cudaFree(D_ptrs[0]));
}

template class VectorArray1D<3>;
template class VectorArray1D<4>;
template class VectorArray1D<5>;
template class VectorArray1D<6>;
template class VectorArray1D<7>;
template class VectorArray1D<8>;
template class VectorArray1D<9>;
template class VectorArray1D<10>;
template class VectorArray1D<11>;
template class VectorArray1D<12>;
template class VectorArray1D<13>;
template class VectorArray1D<14>;
template class VectorArray1D<15>;
template class VectorArray1D<16>;
template class VectorArray1D<17>;
template class VectorArray1D<18>;
template class VectorArray1D<19>;
template class VectorArray1D<20>;
template class VectorArray1D<21>;
template class VectorArray1D<22>;
template class VectorArray1D<23>;
template class VectorArray1D<24>;
template class VectorArray1D<25>;
template class VectorArray1D<26>;
template class VectorArray1D<27>;
template class VectorArray1D<28>;
template class VectorArray1D<29>;
template class VectorArray1D<30>;
template class VectorArray1D<31>;
template class VectorArray1D<32>;
template class VectorArray1D<33>;
template class VectorArray1D<34>;
template class VectorArray1D<35>;
template class VectorArray1D<36>;
template class VectorArray1D<37>;
template class VectorArray1D<38>;
template class VectorArray1D<39>;
template class VectorArray1D<40>;
template class VectorArray1D<41>;
template class VectorArray1D<42>;
template class VectorArray1D<43>;
template class VectorArray1D<44>;
template class VectorArray1D<45>;
template class VectorArray1D<46>;
template class VectorArray1D<47>;
template class VectorArray1D<48>;
template class VectorArray1D<49>;
template class VectorArray1D<50>;
template class VectorArray1D<51>;
template class VectorArray1D<52>;
template class VectorArray1D<53>;
template class VectorArray1D<54>;
template class VectorArray1D<55>;
template class VectorArray1D<56>;
template class VectorArray1D<57>;
template class VectorArray1D<58>;
template class VectorArray1D<59>;
template class VectorArray1D<60>;
template class VectorArray1D<61>;
template class VectorArray1D<62>;
template class VectorArray1D<63>;
template class VectorArray1D<64>;
template class VectorArray1D<65>;
template class VectorArray1D<66>;
template class VectorArray1D<67>;
template class VectorArray1D<68>;
template class VectorArray1D<69>;
template class VectorArray1D<70>;
template class VectorArray1D<71>;
template class VectorArray1D<72>;
template class VectorArray1D<73>;
template class VectorArray1D<74>;
template class VectorArray1D<75>;
template class VectorArray1D<76>;
template class VectorArray1D<77>;
template class VectorArray1D<78>;
template class VectorArray1D<79>;
template class VectorArray1D<80>;
template class VectorArray1D<81>;
template class VectorArray1D<82>;
template class VectorArray1D<83>;
template class VectorArray1D<84>;
template class VectorArray1D<85>;
template class VectorArray1D<86>;
template class VectorArray1D<87>;
template class VectorArray1D<88>;
template class VectorArray1D<89>;
template class VectorArray1D<90>;
template class VectorArray1D<91>;
template class VectorArray1D<92>;
template class VectorArray1D<93>;
template class VectorArray1D<94>;
template class VectorArray1D<95>;
template class VectorArray1D<96>;
template class VectorArray1D<97>;
template class VectorArray1D<98>;
template class VectorArray1D<99>;
} // namespace soa
