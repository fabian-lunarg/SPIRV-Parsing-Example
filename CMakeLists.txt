cmake_minimum_required(VERSION 3.22)
project(spirv-parsing-examples)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_subdirectory(bda_address)
add_subdirectory(vertex_input_position)