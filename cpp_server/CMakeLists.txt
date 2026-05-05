cmake_minimum_required(VERSION 3.16)
project(stockvision_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# ONNX Runtime: pass -DONNXRUNTIME_ROOT=/path or set the env var.
if(NOT DEFINED ONNXRUNTIME_ROOT)
    if(DEFINED ENV{ONNXRUNTIME_ROOT})
        set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
    endif()
endif()

if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR
        "ONNXRUNTIME_ROOT is not set. Pass -DONNXRUNTIME_ROOT=/path/to/onnxruntime "
        "or export the environment variable.")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT}/include ${ONNXRUNTIME_ROOT}/include/onnxruntime)
find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_ROOT}/lib)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "Could not locate ONNX Runtime headers or libraries under ${ONNXRUNTIME_ROOT}")
endif()

find_package(Threads REQUIRED)

set(SOURCES
    src/main.cpp
    src/config.cpp
    src/csv_loader.cpp
    src/data_repository.cpp
    src/feature_engineer.cpp
    src/predictor.cpp
    src/http_server.cpp
)

add_executable(stock_server ${SOURCES})
target_include_directories(stock_server PRIVATE include ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(stock_server PRIVATE ${ONNXRUNTIME_LIB} Threads::Threads)

if(WIN32)
    target_link_libraries(stock_server PRIVATE ws2_32)
endif()

option(STOCKVISION_BUILD_TESTS "Build C++ unit tests" OFF)
if(STOCKVISION_BUILD_TESTS)
    enable_testing()
    add_executable(test_csv_loader tests/test_csv_loader.cpp src/csv_loader.cpp)
    target_include_directories(test_csv_loader PRIVATE include)
    add_test(NAME csv_loader COMMAND test_csv_loader)

    add_executable(test_features tests/test_features.cpp src/feature_engineer.cpp src/csv_loader.cpp)
    target_include_directories(test_features PRIVATE include)
    add_test(NAME features COMMAND test_features)
endif()
