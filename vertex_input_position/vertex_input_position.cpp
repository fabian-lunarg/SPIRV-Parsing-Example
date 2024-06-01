#include <iostream>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include <chrono>

#include "helper.h"
#include "spirv.hpp"

// Represents a single Spv::Op instruction
class Instruction {
  public:
    Instruction(std::vector<uint32_t>::const_iterator it) {
        words_.emplace_back(*it++);
        words_.reserve(Length());
        for (uint32_t i = 1; i < Length(); i++) {
            words_.emplace_back(*it++);
        }

        const bool has_result = OpcodeHasResult(Opcode());
        if (OpcodeHasType(Opcode())) {
            type_id_index_ = 1;
            operand_index_++;
            if (has_result) {
                result_id_index_ = 2;
                operand_index_++;
            }
        } else if (has_result) {
            result_id_index_ = 1;
            operand_index_++;
        }
    }

    // The word used to define the Instruction
    uint32_t Word(uint32_t index) const { return words_[index]; }
    // Skips pass any optional Result or Result Type word
    uint32_t Operand(uint32_t index) const { return words_[operand_index_ + index]; }

    uint32_t Length() const { return words_[0] >> 16; }

    uint32_t Opcode() const { return words_[0] & 0x0ffffu; }

    // operand id, return 0 if no result
    uint32_t ResultId() const { return (result_id_index_ == 0) ? 0 : words_[result_id_index_]; }
    // operand id, return 0 if no type
    uint32_t TypeId() const { return (type_id_index_ == 0) ? 0 : words_[type_id_index_]; }

  private:
    // Store minimal extra data
    uint32_t result_id_index_ = 0;
    uint32_t type_id_index_ = 0;
    uint32_t operand_index_ = 1;

    std::vector<uint32_t> words_;
};

// This is the LUT for hoping around instruction from the result ID
std::unordered_map<uint32_t, const Instruction*> definitions;
const Instruction* FindDef(uint32_t id) {
    auto it = definitions.find(id);
    if (it == definitions.end()) return nullptr;
    return it->second;
}

// < Variable ID, Location > (only for Input locations)
std::unordered_map<uint32_t, uint32_t> variable_to_location_map;
// OpStore < pointer, object > operands
std::unordered_map<uint32_t, uint32_t> store_map;

void Search(uint32_t id) {
    const Instruction* insn = FindDef(id);
    while (insn) {
        switch (insn->Opcode()) {
            case spv::OpLoad: {
                auto it = variable_to_location_map.find(insn->Operand(0));
                if (it != variable_to_location_map.end()) {
                    printf("Position is stored using Input Location %u (OpLoad %%%u)\n", it->second, insn->ResultId());
                    return;
                }
                it = store_map.find(insn->Operand(0));
                if (it != store_map.end()) {
                    insn = FindDef(it->second);
                    break;
                }
                return;
            }
            case spv::OpCompositeExtract:
                insn = FindDef(insn->Operand(0));
                break;
            case spv::OpVectorTimesScalar:
            case spv::OpMatrixTimesScalar:
            case spv::OpVectorTimesMatrix:
            case spv::OpMatrixTimesVector:
            case spv::OpMatrixTimesMatrix:
                Search(insn->Operand(0));
                Search(insn->Operand(1));
                return;
            case spv::OpCompositeConstruct:
                for (uint32_t i = 3; i < insn->Length(); i++) {
                    Search(insn->Word(i));
                }
                return;
            case spv::OpConstant:
            case spv::OpConstantNull:
                return;
            default:
                printf("Unsupported instruction %s\n", string_SpvOpcode(insn->Opcode()));
                return;
        }
    }
}

void Parse(const std::vector<uint32_t>& spirv) {
    std::vector<uint32_t>::const_iterator it = spirv.cbegin();
    it += 5;  // skip first 5 word of header

    bool has_vertex_entry_point = false;
    std::vector<Instruction> instructions;
    // First build up instructions object to make it easier to work with the SPIR-V
    while (it != spirv.cend()) {
        Instruction insn = instructions.emplace_back((it));
        it += insn.Length();

        if (insn.Opcode() == spv::OpEntryPoint && insn.Operand(0) == spv::ExecutionModelVertex) {
            has_vertex_entry_point = true;
        }
    }
    if (!has_vertex_entry_point) {
        printf("Not a vertex shader, so no Position builtin to find\n");
        return;
    }
    instructions.shrink_to_fit();

    // There are VU to make sure the Position BuiltIn is only used once
    uint32_t position_var = 0;
    uint32_t position_member_index = 0;

    // Now we can walk the SPIR-V one more time to find what we need
    for (const Instruction& insn : instructions) {
        // because it is SSA, we can build this up as we are looping in this pass
        const uint32_t result_id = insn.ResultId();
        if (result_id != 0) {
            definitions[result_id] = &insn;
        }

        const uint32_t opcode = insn.Opcode();

        // First find the Position builtin
        if (opcode == spv::OpDecorate) {
            if (insn.Operand(1) == spv::DecorationBuiltIn && insn.Operand(2) == spv::BuiltInPosition) {
                position_var = insn.Operand(0);
            }
            if (insn.Operand(1) == spv::DecorationLocation) {
                variable_to_location_map[insn.Operand(0)] = insn.Operand(2);
            }
        } else if (opcode == spv::OpMemberDecorate) {
            if (insn.Operand(2) == spv::DecorationBuiltIn && insn.Operand(3) == spv::BuiltInPosition) {
                position_var = insn.Operand(0);  // actually OpTypeStruct, resolve below
                position_member_index = insn.Operand(1);
            }
        }

        // Find the variable it is tied to if Position is in a block
        if (opcode == spv::OpVariable && insn.Operand(0) == spv::StorageClassOutput) {
            const Instruction* pointer_type = FindDef(insn.TypeId());
            if (pointer_type && pointer_type->Opcode() == spv::OpTypePointer) {
                if (pointer_type->Operand(1) == position_var) {
                    position_var = insn.ResultId();
                }
            }

            // Remove any output Location we added before
            auto it = variable_to_location_map.find(insn.ResultId());
            if (it != variable_to_location_map.end()) {
                variable_to_location_map.erase(it);
            }
        }

        if (opcode != spv::OpStore) {
            continue;
        }
        store_map[insn.Operand(0)] = insn.Operand(1);

        // Check if OpStore is writing to Position or not
        if (insn.Operand(0) != position_var) {
            // if in a block, will have an access chain
            const Instruction* access_chain = FindDef(insn.Operand(0));
            if (!access_chain || access_chain->Opcode() != spv::OpAccessChain || access_chain->Operand(0) != position_var) {
                continue;
            }
        }

        // We have spotted where the Position was written,
        // now work backward to see if we can find any Input Locations that was involved
        Search(insn.Operand(1));
    }
}

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

    std::vector<uint32_t> spirv_data;
    const int buf_size = 1024;
    uint32_t buf[buf_size];
    while (size_t len = fread(buf, sizeof(uint32_t), buf_size, fp)) {
        spirv_data.insert(spirv_data.end(), buf, buf + len);
    }
    fclose(fp);

    auto start_time = std::chrono::high_resolution_clock::now();

    Parse(spirv_data);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    std::cout << "Time = " << duration.count() << " ms\n";

    return 0;
}