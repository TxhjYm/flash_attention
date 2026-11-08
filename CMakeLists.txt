cmake_minimum_required(VERSION 3.18)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING
        "CUDA architectures to compile for" FORCE)
endif()

project(flash_attention VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(FLASH_ATTENTION_LINEINFO "Include CUDA line information for profiling" OFF)

find_package(CUDAToolkit REQUIRED)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/CMakeLists.txt")
    add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
else()
    find_package(GTest CONFIG REQUIRED)
endif()

file(GLOB SOURCES_GPU CONFIGURE_DEPENDS src/*.cu)
file(GLOB SOURCES_TEST CONFIGURE_DEPENDS tests/*.cu)

enable_testing()

add_executable(flash_attention_tests
    ${SOURCES_TEST}
    ${SOURCES_GPU}
)

target_include_directories(flash_attention_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(flash_attention_tests PRIVATE
    CUDA::cudart
    CUDA::cublas
    GTest::gtest_main
)

if(FLASH_ATTENTION_LINEINFO)
    target_compile_options(flash_attention_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )
endif()

include(GoogleTest)
gtest_discover_tests(flash_attention_tests)
