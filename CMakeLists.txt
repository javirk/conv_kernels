cmake_minimum_required(VERSION 3.18)
project(easy_engine LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

include(FetchContent)
FetchContent_Declare(
  cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
  GIT_TAG        v3.5.1
)
set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_TOOLS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(cutlass)

add_executable(engine main.cu)
target_include_directories(engine PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/cutlass/include/)
target_include_directories(engine PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/cutlass/tools/util/include)
target_include_directories(engine PRIVATE ${CMAKE_SOURCE_DIR})

add_executable(benchmark_sweep benchmark_sweep.cu)
target_include_directories(benchmark_sweep PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/cutlass/include/)
target_include_directories(benchmark_sweep PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/cutlass/tools/util/include)
target_include_directories(benchmark_sweep PRIVATE ${CMAKE_SOURCE_DIR})

add_executable(test_conv_kernel test_conv_kernel.cu)
target_include_directories(test_conv_kernel PRIVATE ${CMAKE_SOURCE_DIR})

add_executable(cutlass_example cutlass_example.cu)
target_link_libraries(cutlass_example PRIVATE CUTLASS)
