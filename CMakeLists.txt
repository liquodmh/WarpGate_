cmake_minimum_required(VERSION 3.24)
project(warpgate VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(WARPGATE_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

add_library(warpgate_core
    src/cpu_exact_backend.cpp
    src/latency_model.cpp
    src/scheduler.cpp
)

target_include_directories(warpgate_core PUBLIC include)

target_compile_options(warpgate_core PRIVATE -Wall -Wextra -Wpedantic)
if(WARPGATE_WARNINGS_AS_ERRORS)
  target_compile_options(warpgate_core PRIVATE -Werror)
endif()

add_executable(warpgate_bench apps/warpgate_bench.cpp)
target_link_libraries(warpgate_bench PRIVATE warpgate_core)
target_compile_options(warpgate_bench PRIVATE -Wall -Wextra -Wpedantic)

add_executable(warpgate_tests tests/warpgate_tests.cpp)
target_link_libraries(warpgate_tests PRIVATE warpgate_core)
target_compile_options(warpgate_tests PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_test(NAME warpgate_tests COMMAND warpgate_tests)
