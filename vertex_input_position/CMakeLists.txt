add_executable(vertex_input_position)

target_sources(vertex_input_position PRIVATE
    vertex_input_position.cpp
)

target_include_directories(vertex_input_position PRIVATE
    ${CMAKE_SOURCE_DIR}/spirv-headers)