cuda_add_library(gpusoa STATIC
    array.cu
    vector/array.cu
)

target_include_directories(gpusoa
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)
