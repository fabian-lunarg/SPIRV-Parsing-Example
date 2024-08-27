#include <iostream>
#include <filesystem>

#include <chrono>
#include <optional>

#include "spirv_parsing_util.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage:\n\t" << argv[0] << " input.spv\n";
        return EXIT_FAILURE;
    } else if (!std::filesystem::exists(argv[1])) {
        std::cout << "ERROR: " << argv[1] << " Does not exists\n";
        return EXIT_FAILURE;
    }

    FILE* fp = fopen(argv[1], "rb");
    if (!fp) {
        std::cout << "ERROR: Unable to open the input file " << argv[1] << "\n";
        return EXIT_FAILURE;
    }

    std::vector<uint32_t> spirv;
    const int buf_size = 1024;
    uint32_t buf[buf_size];
    while (size_t len = fread(buf, sizeof(uint32_t), buf_size, fp)) {
        spirv.insert(spirv.end(), buf, buf + len);
    }
    fclose(fp);

    auto start_time = std::chrono::high_resolution_clock::now();

    SpirVParsingUtil parsing_util;
    parsing_util.ParseBufferReferences(spirv.data(), spirv.size() * sizeof(uint32_t));

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    std::cout << "Time = " << duration.count() << " ms\n";

    return 0;
}