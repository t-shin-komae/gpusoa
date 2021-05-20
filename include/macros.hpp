#pragma once

#ifdef __CUDACC__
#define DEVICEHOST __device__ __host__
#define DEVICE __device__
#define HOST __host__
#elif _OPENACC
#define DEVICEHOST #pragma acc routine seq
#define DEVICE #pragma acc routine seq nohost
#define HOST
#else
#define DEVICEHOST
#define DEVICE
#define HOST
#endif
