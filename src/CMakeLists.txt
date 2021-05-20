cuda_add_library(gpusoa STATIC
    ./constructor/array1d.cu
    ./constructor/vectorarray1d.cu
)

target_include_directories(gpusoa
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)
