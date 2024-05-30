add_executable(bda_address)

target_sources(bda_address PRIVATE
    bda_address.cpp
)

target_include_directories(bda_address PRIVATE
    ${CMAKE_SOURCE_DIR}/spirv-headers)