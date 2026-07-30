cmake_minimum_required(VERSION 3.18)  # find_library(... REQUIRED) needs 3.18
project(stockvision_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra)
endif()

# ONNX Runtime: pass -DONNXRUNTIME_ROOT=/path or set the environment variable.
if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
endif()
if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
          HINTS ${ONNXRUNTIME_ROOT}/include ${ONNXRUNTIME_ROOT}/include/onnxruntime REQUIRED)
find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_ROOT}/lib REQUIRED)
find_package(Threads REQUIRED)

add_executable(stock_server
    src/main.cpp
    src/csv_loader.cpp
    src/features.cpp
    src/predictor.cpp
    src/server.cpp
)
target_include_directories(stock_server PRIVATE include ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(stock_server PRIVATE ${ONNXRUNTIME_LIB} Threads::Threads)

if(WIN32)
    target_link_libraries(stock_server PRIVATE ws2_32)
endif()

# The CSV and feature tests need no ONNX Runtime, so they build on their own.
option(STOCKVISION_BUILD_TESTS "Build C++ unit tests" OFF)
if(STOCKVISION_BUILD_TESTS)
    enable_testing()
    foreach(name csv_loader features)
        add_executable(test_${name} tests/test_${name}.cpp src/csv_loader.cpp src/features.cpp)
        target_include_directories(test_${name} PRIVATE include)
        add_test(NAME ${name} COMMAND test_${name})
    endforeach()
endif()
