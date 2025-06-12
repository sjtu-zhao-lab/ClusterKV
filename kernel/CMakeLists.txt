cmake_minimum_required(VERSION 3.26.4)  # Specify your minimum CMake version

set(CMAKE_C_COMPILER "/usr/bin/gcc-11")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

# ------------- configure rapids-cmake --------------#
include(${CMAKE_SOURCE_DIR}/cmake/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

project(_clusterkv_kernels LANGUAGES CUDA CXX)  # Replace with your project's name

# ------------- configure raft -----------------#
rapids_cpm_init()
include(${CMAKE_SOURCE_DIR}/cmake/get_raft.cmake)

# Check: https://stackoverflow.com/questions/68401650/how-can-i-make-a-pytorch-extension-with-cmake
# Fix linking error: https://github.com/pytorch/pytorch/issues/108041
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

# Try combine pybind
# Check: https://qiita.com/syoyo/items/c3e8e6e5c3e2d69c2325
add_subdirectory(${CMAKE_SOURCE_DIR}/../3rdparty/pybind/ ${CMAKE_BINARY_DIR}/pybind11)

file(GLOB SOURCES "cluster/src/*.cu")
pybind11_add_module(_clusterkv_knl MODULE ${PYTORCH_CPP_SOURCES} ${SOURCES})

# target_compile_definitions(_clusterkv_knl PRIVATE -DBSK_TORCH_CHECK) # Enable Torch Tensor Dimension Check
target_include_directories(_clusterkv_knl PRIVATE ${CMAKE_SOURCE_DIR}/cluster/include)
target_include_directories(_clusterkv_knl PRIVATE ${CMAKE_SOURCE_DIR}/../3rdparty/flashinfer/include)
target_include_directories(_clusterkv_knl PRIVATE ${CMAKE_SOURCE_DIR}/../3rdparty/pybind/include)
target_compile_options(_clusterkv_knl PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
target_link_libraries(_clusterkv_knl PRIVATE ${TORCH_LIBRARIES} raft::raft Python::Python pybind11::module ${TORCH_PYTHON_LIBRARY})

file(GLOB SOURCESQ "quest/src/*.cu")
pybind11_add_module(_quest_knl MODULE ${PYTORCH_CPP_SOURCES} ${SOURCESQ})

# target_compile_definitions(_quest_knl PRIVATE -DBSK_TORCH_CHECK) # Enable Torch Tensor Dimension Check
target_include_directories(_quest_knl PRIVATE ${CMAKE_SOURCE_DIR}/quest/include)
target_include_directories(_quest_knl PRIVATE ${CMAKE_SOURCE_DIR}/../3rdparty/flashinfer/include)
target_include_directories(_quest_knl PRIVATE ${CMAKE_SOURCE_DIR}/../3rdparty/pybind/include)
target_compile_options(_quest_knl PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
target_link_libraries(_quest_knl PRIVATE ${TORCH_LIBRARIES} raft::raft Python::Python pybind11::module ${TORCH_PYTHON_LIBRARY})