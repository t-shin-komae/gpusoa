#include "macros.hpp"
#include "vector.hpp"
#include <functional>
#include <thrust/pair.h>
#include <thrust/tuple.h>
namespace soa {
template <int N> class KVectorArray1D {
  double *D_ptrs[N];
  int Nx;

public:
  KVectorArray1D(double *const (&D_ptrs)[N], int Nx);
  DEVICE inline int size() const { return Nx; }
  DEVICE inline VectorProxy<N> operator()(int i) {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx);
#endif
    return VectorProxy<N>(D_ptrs, i);
  }
  DEVICE inline ConstVectorProxy<N> operator()(int i) const {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx);
#endif
    return ConstVectorProxy<N>(D_ptrs, i);
  }
};
template <int N> class KVectorArray2D {
  double *D_ptrs[N];
  int Nx, Ny;

public:
  KVectorArray2D(double *const (&D_ptrs)[N], int Nx, int Ny);
  DEVICE inline int sizeX() const { return Nx; }
  DEVICE inline int sizeY() const { return Ny; }
  DEVICE inline thrust::pair<int, int> size() const { return {Nx, Ny}; }
  DEVICE inline VectorProxy<N> operator()(int i, int j) {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx && j< Ny);
#endif
    return VectorProxy<N>(D_ptrs, i + j * Nx);
  }
  DEVICE inline ConstVectorProxy<N> operator()(int i, int j) const {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx && j < Ny);
#endif
    return ConstVectorProxy<N>(D_ptrs, i + j * Nx);
  }
};
template <int N> class KVectorArray3D {
  double *D_ptrs[N];
  int Nx, Ny, Nz;

public:
  KVectorArray3D(double *const (&D_ptrs)[N], int Nx, int Ny, int Nz);
  DEVICE inline int sizeX() const { return Nx; }
  DEVICE inline int sizeY() const { return Ny; }
  DEVICE inline int sizeZ() const { return Nz; }
  DEVICE inline thrust::tuple<int, int, int> size() const { return {Nx, Ny, Nz}; }
  DEVICE inline VectorProxy<N> operator()(int i, int j, int k) {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx && j < Ny && k < Nz);
#endif
    return VectorProxy<N>(D_ptrs, i + j * Nx + k * Nx * Ny);
  }
  DEVICE inline ConstVectorProxy<N> operator()(int i, int j, int k) const {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx && j < Ny && k < Nz);
#endif
    return ConstVectorProxy<N>(D_ptrs, i + j * Nx + k * Nx * Ny);
  }
};

template <int N> class VectorArray1D {
  double *H_ptrs[N], *D_ptrs[N];
  int Nx;

public:
  VectorArray1D(int Nx);
  VectorArray1D(int Nx, std::function<Vector<N>(int)> f);
  ~VectorArray1D();
  VectorArray1D &operator=(const VectorArray1D &) = delete;
  VectorArray1D(const VectorArray1D &) = delete;
  VectorArray1D(VectorArray1D &&) noexcept;
  VectorArray1D &operator=(VectorArray1D &&) noexcept;
  void DeviceToHost();
  void HostToDevice();
  inline KVectorArray1D<N> gpu() const { return {D_ptrs, Nx}; }
  inline int size() const { return Nx; }

  inline VectorProxy<N> operator()(int i) {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx);
#endif
    return VectorProxy<N>(H_ptrs, i);
  }
  inline ConstVectorProxy<N> operator()(int i) const {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx);
#endif
    return ConstVectorProxy<N>(H_ptrs, i);
  }
};

template <int N> class VectorArray2D {
  double *H_ptrs[N], *D_ptrs[N];
  int Nx, Ny;

public:
  VectorArray2D(int Nx, int Ny);
  VectorArray2D(int Nx, int Ny, std::function<Vector<N>(int, int)> f);
  ~VectorArray2D();
  VectorArray2D &operator=(const VectorArray2D &) = delete;
  VectorArray2D(const VectorArray2D &) = delete;
  VectorArray2D(VectorArray2D &&) noexcept;
  VectorArray2D &operator=(VectorArray2D &&) noexcept;
  void DeviceToHost();
  void HostToDevice();
  inline KVectorArray2D<N> gpu() const { return {D_ptrs, Nx, Ny}; }
  inline int sizeX() const { return Nx; }
  inline int sizeY() const { return Ny; }
  inline std::pair<int, int> size() const { return {Nx, Ny}; }

  inline VectorProxy<N> operator()(int i, int j) {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx && j < Ny);
#endif
    return VectorProxy<N>(H_ptrs, i + j * Nx);
  }
  inline ConstVectorProxy<N> operator()(int i, int j) const {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx && j < Ny);
#endif
    return ConstVectorProxy<N>(H_ptrs, i + j * Nx);
  }
};
template <int N> class VectorArray3D {
  double *H_ptrs[N], *D_ptrs[N];
  int Nx, Ny, Nz;

public:
  VectorArray3D(int Nx, int Ny, int Nz);
  VectorArray3D(int Nx, int Ny, int Nz, std::function<Vector<N>(int, int, int)> f);
  ~VectorArray3D();
  VectorArray3D &operator=(const VectorArray3D &) = delete;
  VectorArray3D(const VectorArray3D &) = delete;
  VectorArray3D(VectorArray3D &&) noexcept;
  VectorArray3D &operator=(VectorArray3D &&) noexcept;
  void DeviceToHost();
  void HostToDevice();
  inline KVectorArray3D<N> gpu() const { return {D_ptrs, Nx, Ny, Nz}; }
  inline int sizeX() const { return Nx; }
  inline int sizeY() const { return Ny; }
  inline int sizeZ() const { return Nz; }
  inline std::tuple<int, int, int> size() const { return {Nx, Ny, Nz}; }

  inline VectorProxy<N> operator()(int i, int j, int k) {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx && j < Ny && k < Nz);
#endif
    return VectorProxy<N>(H_ptrs, i + j * Nx + k * Nx * Ny);
  }
  inline ConstVectorProxy<N> operator()(int i, int j, int k) const {
#ifdef _ARRAY_INDEX_BOUNDS_CHECK
    assert(i < Nx && j < Ny && k < Nz);
#endif
    return ConstVectorProxy<N>(H_ptrs, i + j * Nx + k * Nx * Ny);
  }
};

using Vector2Array1D = VectorArray1D<2>;
using Vector3Array1D = VectorArray1D<3>;
using Vector4Array1D = VectorArray1D<4>;
using Vector5Array1D = VectorArray1D<5>;
using Vector6Array1D = VectorArray1D<6>;
using Vector7Array1D = VectorArray1D<7>;
using Vector8Array1D = VectorArray1D<8>;
using Vector9Array1D = VectorArray1D<9>;
using Vector10Array1D = VectorArray1D<10>;
using Vector11Array1D = VectorArray1D<11>;
using Vector12Array1D = VectorArray1D<12>;
using Vector13Array1D = VectorArray1D<13>;
using Vector14Array1D = VectorArray1D<14>;
using Vector15Array1D = VectorArray1D<15>;
using Vector16Array1D = VectorArray1D<16>;
using Vector17Array1D = VectorArray1D<17>;
using Vector18Array1D = VectorArray1D<18>;
using Vector19Array1D = VectorArray1D<19>;
using Vector20Array1D = VectorArray1D<20>;
using Vector21Array1D = VectorArray1D<21>;
using Vector22Array1D = VectorArray1D<22>;
using Vector23Array1D = VectorArray1D<23>;
using Vector24Array1D = VectorArray1D<24>;
using Vector25Array1D = VectorArray1D<25>;
using Vector26Array1D = VectorArray1D<26>;
using Vector27Array1D = VectorArray1D<27>;
using Vector28Array1D = VectorArray1D<28>;
using Vector29Array1D = VectorArray1D<29>;
using Vector30Array1D = VectorArray1D<30>;
using Vector31Array1D = VectorArray1D<31>;
using Vector32Array1D = VectorArray1D<32>;
using Vector33Array1D = VectorArray1D<33>;
using Vector34Array1D = VectorArray1D<34>;
using Vector35Array1D = VectorArray1D<35>;
using Vector36Array1D = VectorArray1D<36>;
using Vector37Array1D = VectorArray1D<37>;
using Vector38Array1D = VectorArray1D<38>;
using Vector39Array1D = VectorArray1D<39>;
using Vector40Array1D = VectorArray1D<40>;
using Vector41Array1D = VectorArray1D<41>;
using Vector42Array1D = VectorArray1D<42>;
using Vector43Array1D = VectorArray1D<43>;
using Vector44Array1D = VectorArray1D<44>;
using Vector45Array1D = VectorArray1D<45>;
using Vector46Array1D = VectorArray1D<46>;
using Vector47Array1D = VectorArray1D<47>;
using Vector48Array1D = VectorArray1D<48>;
using Vector49Array1D = VectorArray1D<49>;
using Vector50Array1D = VectorArray1D<50>;
using Vector51Array1D = VectorArray1D<51>;
using Vector52Array1D = VectorArray1D<52>;
using Vector53Array1D = VectorArray1D<53>;
using Vector54Array1D = VectorArray1D<54>;
using Vector55Array1D = VectorArray1D<55>;
using Vector56Array1D = VectorArray1D<56>;
using Vector57Array1D = VectorArray1D<57>;
using Vector58Array1D = VectorArray1D<58>;
using Vector59Array1D = VectorArray1D<59>;
using Vector60Array1D = VectorArray1D<60>;
using Vector61Array1D = VectorArray1D<61>;
using Vector62Array1D = VectorArray1D<62>;
using Vector63Array1D = VectorArray1D<63>;
using Vector64Array1D = VectorArray1D<64>;
using Vector65Array1D = VectorArray1D<65>;
using Vector66Array1D = VectorArray1D<66>;
using Vector67Array1D = VectorArray1D<67>;
using Vector68Array1D = VectorArray1D<68>;
using Vector69Array1D = VectorArray1D<69>;
using Vector70Array1D = VectorArray1D<70>;
using Vector71Array1D = VectorArray1D<71>;
using Vector72Array1D = VectorArray1D<72>;
using Vector73Array1D = VectorArray1D<73>;
using Vector74Array1D = VectorArray1D<74>;
using Vector75Array1D = VectorArray1D<75>;
using Vector76Array1D = VectorArray1D<76>;
using Vector77Array1D = VectorArray1D<77>;
using Vector78Array1D = VectorArray1D<78>;
using Vector79Array1D = VectorArray1D<79>;
using Vector80Array1D = VectorArray1D<80>;
using Vector81Array1D = VectorArray1D<81>;
using Vector82Array1D = VectorArray1D<82>;
using Vector83Array1D = VectorArray1D<83>;
using Vector84Array1D = VectorArray1D<84>;
using Vector85Array1D = VectorArray1D<85>;
using Vector86Array1D = VectorArray1D<86>;
using Vector87Array1D = VectorArray1D<87>;
using Vector88Array1D = VectorArray1D<88>;
using Vector89Array1D = VectorArray1D<89>;
using Vector90Array1D = VectorArray1D<90>;
using Vector91Array1D = VectorArray1D<91>;
using Vector92Array1D = VectorArray1D<92>;
using Vector93Array1D = VectorArray1D<93>;
using Vector94Array1D = VectorArray1D<94>;
using Vector95Array1D = VectorArray1D<95>;
using Vector96Array1D = VectorArray1D<96>;
using Vector97Array1D = VectorArray1D<97>;
using Vector98Array1D = VectorArray1D<98>;
using Vector99Array1D = VectorArray1D<99>;
using Vector100Array1D = VectorArray1D<100>;

} // namespace soa
