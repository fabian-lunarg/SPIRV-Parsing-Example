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

const Instruction* FindVariableStoring(std::vector<const Instruction*>& store_instructions, uint32_t variable_id) {
    for (const Instruction* store_insn : store_instructions) {
        if (store_insn->Operand(0) == variable_id) {
            // Note: This will find the first store, there could be multiple
            return FindDef(store_insn->Operand(1));
        }
    }
    return nullptr;
}

void PrintVariableDecorations(std::vector<const Instruction*>& decorations_instructions, const Instruction& variable_insn) {
    const uint32_t variable_id = variable_insn.ResultId();
    const uint32_t storage_class = variable_insn.Operand(0);
    if (storage_class == spv::StorageClassPushConstant) {
        printf("Address from Push Constant block\n");
        return;
    }

    uint32_t set = 0;
    uint32_t binding = 0;
    for (const Instruction* insn : decorations_instructions) {
        if (insn->Operand(0) != variable_id) continue;
        if (insn->Operand(1) == spv::DecorationDescriptorSet) {
            set = insn->Operand(2);
        } else if (insn->Operand(1) == spv::DecorationBinding) {
            binding = insn->Operand(2);
        }
    }

    if (storage_class == spv::StorageClassStorageBuffer || storage_class == spv::StorageClassUniform) {
        printf("Address from Descriptor in Set %u, Binding %u\n", set, binding);
    } else {
        printf("Storage class %u not handled\n", storage_class);
    }
}

void Parse(const std::vector<uint32_t>& spirv) {
    std::vector<uint32_t>::const_iterator it = spirv.cbegin();
    it += 5;  // skip first 5 word of header

    std::vector<Instruction> instructions;
    // First build up instructions object to make it easier to work with the SPIR-V
    while (it != spirv.cend()) {
        Instruction insn = instructions.emplace_back((it));
        it += insn.Length();
    }
    instructions.shrink_to_fit();

    std::vector<const Instruction*> store_instructions;
    std::vector<const Instruction*> decorations_instructions;

    auto track_back_instruction = [&store_instructions, &decorations_instructions](const Instruction* object_insn) {
        // We are where a buffer-reference was accessed, now walk back to find where it came from
        while (object_insn) {
            switch (object_insn->Opcode()) {
                case spv::OpConvertUToPtr:
                case spv::OpCopyLogical:
                case spv::OpLoad:
                case spv::OpAccessChain:
                    object_insn = FindDef(object_insn->Operand(0));
                    break;
                case spv::OpVariable: {
                    const uint32_t storage_class = object_insn->Operand(0);
                    if (storage_class == spv::StorageClassFunction) {
                        // When casting to a struct, can get a 2nd function variable, just keep following
                        object_insn = FindVariableStoring(store_instructions, object_insn->ResultId());
                    } else {
                        PrintVariableDecorations(decorations_instructions, *object_insn);
                        object_insn = nullptr;
                    }
                    break;
                }
                default:
                    printf("Failed to track back the Function Variable OpStore, hit a %s\n",
                           string_SpvOpcode(object_insn->Opcode()));
                    object_insn = nullptr;
                    break;
            }
        }
    };

    // Now we can walk the SPIR-V one more time to find what we need
    for (const Instruction& insn : instructions) {
        // because it is SSA, we can build this up as we are looping in this pass
        const uint32_t result_id = insn.ResultId();
        if (result_id != 0) {
            definitions[result_id] = &insn;
        }

        const uint32_t opcode = insn.Opcode();

        if (opcode == spv::OpStore) {
            store_instructions.push_back(&insn);
        } else if (opcode == spv::OpDecorate) {
            decorations_instructions.push_back(&insn);
        }

        // There is always a load that does the dereferencing
        if (opcode != spv::OpLoad) {
            continue;
        }

        // Confirms the load is used for a buffer device address
        const Instruction* type_pointer_insn = FindDef(insn.TypeId());
        if (!type_pointer_insn || type_pointer_insn->Opcode() != spv::OpTypePointer ||
            type_pointer_insn->Operand(0) != spv::StorageClassPhysicalStorageBuffer) {
            continue;
        }

        const Instruction* load_pointer_insn = FindDef(insn.Operand(0));
        if (load_pointer_insn && load_pointer_insn->Opcode() == spv::OpVariable &&
            load_pointer_insn->Operand(0) == spv::StorageClassFunction) {
            const Instruction* object_insn = FindVariableStoring(store_instructions, load_pointer_insn->ResultId());
            if (!object_insn) continue;

            track_back_instruction(object_insn);
        } else if (load_pointer_insn && load_pointer_insn->Opcode() == spv::OpAccessChain) {
            track_back_instruction(load_pointer_insn);
        }
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