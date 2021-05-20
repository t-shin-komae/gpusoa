cuda_add_library(gpusoa STATIC
    array.cu
    ./constructor/array.cu
    ./constructor/vectorarray1d.cu
)

target_include_directories(gpusoa
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)
