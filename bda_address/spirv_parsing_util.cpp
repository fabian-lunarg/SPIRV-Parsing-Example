/*
** Copyright (c) 2024 LunarG, Inc.
**
** Permission is hereby granted, free of charge, to any person obtaining a
** copy of this software and associated documentation files (the "Software"),
** to deal in the Software without restriction, including without limitation
** the rights to use, copy, modify, merge, publish, distribute, sublicense,
** and/or sell copies of the Software, and to permit persons to whom the
** Software is furnished to do so, subject to the following conditions:
**
** The above copyright notice and this permission notice shall be included in
** all copies or substantial portions of the Software.
**
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
** LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
** FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
** DEALINGS IN THE SOFTWARE.
*/

#define SPV_ENABLE_UTILITY_CODE

#include "spirv_parsing_util.h"
#include "helper.h"
#include "spirv_reflect.h"
#include <functional>
#include <optional>
#include <cassert>
#include <deque>

// used to enable type as key for std::set/map
bool operator<(const SpirVParsingUtil::BufferReferenceInfo& lhs, const SpirVParsingUtil::BufferReferenceInfo& rhs)
{
    return std::make_tuple(lhs.source, lhs.set, lhs.binding, lhs.buffer_offset, lhs.array_stride) <
           std::make_tuple(rhs.source, rhs.set, rhs.binding, rhs.buffer_offset, rhs.array_stride);
}

// Instruction represents a single Spv::Op instruction.
class SpirVParsingUtil::Instruction
{
  public:
    explicit Instruction(const uint32_t* spirv)
    {
        assert(spirv != nullptr);

        words_ = spirv;

        const bool has_result = OpcodeHasResult(opcode());
        if (OpcodeHasType(opcode()))
        {
            type_id_index_ = 1;
            operand_index_++;
            if (has_result)
            {
                result_id_index_ = 2;
                operand_index_++;
            }
        }
        else if (has_result)
        {
            result_id_index_ = 1;
            operand_index_++;
        }
    }

    //! the word used to define the Instruction
    [[nodiscard]] uint32_t word(uint32_t index) const { return words_[index]; }

    //! skips pass any optional Result or Result Type word
    [[nodiscard]] uint32_t operand(uint32_t index) const { return words_[operand_index_ + index]; }

    //! number of words used as operands
    [[nodiscard]] uint32_t num_operands() const { return length() - operand_index_; }

    //! length of instruction in words
    [[nodiscard]] uint32_t length() const { return words_[0] >> 16; }

    //! the instruction's op-code
    [[nodiscard]] uint32_t opcode() const { return words_[0] & 0x0ffffu; }

    //! operand id, return 0 if no result
    [[nodiscard]] uint32_t resultId() const { return (result_id_index_ == 0) ? 0 : words_[result_id_index_]; }

    //! operand id, return 0 if no type
    [[nodiscard]] uint32_t typeId() const { return (type_id_index_ == 0) ? 0 : words_[type_id_index_]; }

    //! constant values can safely be returned as uint32_t
    [[nodiscard]] uint32_t constant_value() const
    {
        assert(opcode() == spv::OpConstant);
        return words_[3];
    }

  private:
    // store minimal extra data
    uint32_t result_id_index_ = 0;
    uint32_t type_id_index_   = 0;
    uint32_t operand_index_   = 1;

    const uint32_t* words_ = nullptr;
};

//// This is the LUT for hoping around instruction from the result ID
// std::unordered_map<uint32_t, const SpirVInstruction*> definitions;

const SpirVParsingUtil::Instruction* SpirVParsingUtil::FindDef(uint32_t id)
{
    auto it = definitions_.find(id);
    if (it == definitions_.end())
    {
        return nullptr;
    }
    return it->second;
}

const SpirVParsingUtil::Instruction* SpirVParsingUtil::FindVariableStoring(uint32_t variable_id)
{
    for (const Instruction* store_insn : store_instructions_)
    {
        if (store_insn->operand(0) == variable_id)
        {
            // Note: This will find the first store, there could be multiple
            return FindDef(store_insn->operand(1));
        }
    }
    return nullptr;
}

bool SpirVParsingUtil::GetVariableDecorations(const Instruction*   variable_insn,
                                              BufferReferenceInfo& buffer_reference_info)
{
    const uint32_t variable_id   = variable_insn->resultId();
    const uint32_t storage_class = variable_insn->operand(0);

    switch (storage_class)
    {

        case spv::StorageClassUniform:
            buffer_reference_info.source = BufferReferenceLocation::UNIFORM_BUFFER;
            break;

        case spv::StorageClassStorageBuffer:
            buffer_reference_info.source = BufferReferenceLocation::STORAGE_BUFFER;
            break;

        case spv::StorageClassShaderRecordBufferKHR:
            buffer_reference_info.source = BufferReferenceLocation::SHADER_RECORD_BUFFER;
            break;

        case spv::StorageClassPushConstant:
            buffer_reference_info.source = BufferReferenceLocation::PUSH_CONSTANT_BLOCK;
            return true;

        default:
            printf("Storage class %u not handled\n", storage_class);
            return false;
    }

    for (const Instruction* insn : decorations_instructions_)
    {
        if (insn->operand(0) != variable_id)
        {
            continue;
        }
        if (insn->operand(1) == spv::DecorationDescriptorSet)
        {
            buffer_reference_info.set = insn->operand(2);
        }
        else if (insn->operand(1) == spv::DecorationBinding)
        {
            buffer_reference_info.binding = insn->operand(2);
        }
    }
    return true;
}

bool SpirVParsingUtil::ParseBufferReferences(const uint32_t* const spirv_code, size_t spirv_num_bytes)
{
    if (spirv_code == nullptr)
    {
        return false;
    }

    definitions_.clear();
    store_instructions_.clear();
    decorations_instructions_.clear();
    buffer_reference_map_.clear();

    // use in combination with spirv-reflect
    std::optional<SpvReflectShaderModule> spv_shader_module;

    // spirv-header is 5 d-words
    constexpr uint32_t spirv_header_size = 5;
    const uint32_t*    spirv_ptr         = spirv_code;
    const uint32_t*    spirv_end         = spirv_code + (spirv_num_bytes / sizeof(uint32_t));

    // skip header
    spirv_ptr += spirv_header_size;

    std::vector<Instruction> instructions;

    bool found_buffer_ref = false;

    // build up instructions object to make it easier to work with the SPIR-V
    // also checks for required capability
    while (spirv_ptr < spirv_end)
    {
        Instruction& insn = instructions.emplace_back(spirv_ptr);
        spirv_ptr += insn.length();
        assert(insn.length() > 0);

        if (insn.opcode() == spv::OpCapability && insn.word(1) == spv::CapabilityPhysicalStorageBufferAddresses)
        {
            found_buffer_ref = true;
        }

        // arrived at 'OpFunction' -> we have seen all metadata incl. capabilities
        if (insn.opcode() == spv::OpFunction)
        {
            // CapabilityPhysicalStorageBufferAddresses not found
            if (!found_buffer_ref)
            {
                return true;
            }
        }
    }
    if (spirv_ptr != spirv_end)
    {
        printf("warning: error during SpirV-parsing, mismatching instruction-lengths");
        return false;
    }
    instructions.shrink_to_fit();

    if (spv_shader_module == std::nullopt)
    {
        // spirv-reflect parsing only on-demand
        spv_shader_module = SpvReflectShaderModule();
        spvReflectCreateShaderModule(spirv_num_bytes, spirv_code, &spv_shader_module.value());
    }

    {
        // define a function to walk blocks breadth-first and check for buffer-references
        auto check_buffer_references =
            [this](const SpvReflectTypeDescription* type, BufferReferenceLocation source, uint32_t set, uint32_t binding)
        {
            // bfs: (type, offset)
            std::deque<std::pair<const SpvReflectTypeDescription*, uint32_t>> queue = {{type, 0}};

            while(!queue.empty())
            {
                auto [td, offset] = queue.front();
                queue.pop_front();

                if(td)
                {
                    if(td->storage_class == spv::StorageClassPhysicalStorageBuffer)
                    {
                        BufferReferenceInfo ref_info;
                        ref_info.source = source;
                        ref_info.set = set;
                        ref_info.binding = binding;
                        ref_info.buffer_offset = offset;

                        if (td->op == SpvOpTypeArray || td->op == SpvOpTypeRuntimeArray)
                        {
                            ref_info.array_stride = td->traits.array.stride;
                        }

                        // insert into map
                        buffer_reference_map_[ref_info] = {td->struct_member_name};
                    }

                    for (uint32_t j = 0; j < td->member_count; ++j)
                    {
                        auto* member = td->members + j;
                        queue.emplace_back(member, offset);

                        uint32_t num_scalar_bytes = member->traits.numeric.scalar.width / 8;

                        if (member->op == SpvOpTypeVector)
                        {
                            num_scalar_bytes *= member->traits.numeric.vector.component_count;
                        }
                        else if (member->op == SpvOpTypeMatrix)
                        {
                            num_scalar_bytes *= member->traits.numeric.matrix.column_count;
                            num_scalar_bytes *= member->traits.numeric.matrix.row_count;
                            num_scalar_bytes = std::max(num_scalar_bytes, member->traits.numeric.matrix.stride);
                        }
                        else if (member->op == SpvOpTypePointer || member->op == SpvOpTypeForwardPointer)
                        {
                            num_scalar_bytes = sizeof(uint64_t);
                        }
                        else if (member->op == SpvOpTypeArray || member->op == SpvOpTypeRuntimeArray)
                        {
                            num_scalar_bytes = std::max(num_scalar_bytes, member->traits.array.stride);
                            for(uint32_t d = 0; d < member->traits.array.dims_count; ++d)
                            {
                                num_scalar_bytes *=
                                    member->traits.array.dims[d] == spv::OpTypeRuntimeArray ? 1 : member->traits.array.dims[d];
                            }
                        }
                        offset += num_scalar_bytes;
                    }
                }
            }
        };

        // check descriptor sets
        uint32_t num_descriptor_set;
        spvReflectEnumerateDescriptorSets(&*spv_shader_module, &num_descriptor_set, nullptr);
        std::vector<SpvReflectDescriptorSet*> descriptor_sets(num_descriptor_set);
        spvReflectEnumerateDescriptorSets(&*spv_shader_module, &num_descriptor_set, descriptor_sets.data());

        for(const auto& descriptor_set : descriptor_sets)
        {
            for(uint32_t i = 0;i < descriptor_set->binding_count; ++i)
            {
                auto *binding = descriptor_set->bindings[i];
                BufferReferenceLocation source = BufferReferenceLocation::INVALID;;

                switch (binding->descriptor_type)
                {
                    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
                        source = BufferReferenceLocation::UNIFORM_BUFFER;
                        break;
                    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                        source = BufferReferenceLocation::STORAGE_BUFFER;
                        break;
                    default:
                        break;
                }
                check_buffer_references(binding->type_description, source, descriptor_set->set, binding->binding);
            }
        }

        // check push-constants
        uint32_t num_push_constant_blocks;
        spvReflectEnumeratePushConstantBlocks(&*spv_shader_module, &num_push_constant_blocks, nullptr);
        std::vector<SpvReflectBlockVariable*> push_constant_blocks(num_push_constant_blocks);
        spvReflectEnumeratePushConstantBlocks(&*spv_shader_module, &num_push_constant_blocks, push_constant_blocks.data());

        for(const auto& block : push_constant_blocks)
        {
            check_buffer_references(block->type_description, BufferReferenceLocation::PUSH_CONSTANT_BLOCK, 0, 0);
        }
    }

    auto track_back_instruction = [this, &spv_shader_module, spirv_code, spirv_num_bytes](
                                      const Instruction* object_insn) {
        // keep track of access-chain
        std::vector<uint32_t> access_indices;

        // We are where a buffer-reference was accessed, now walk back to find where it came from
        while (object_insn)
        {
            switch (object_insn->opcode())
            {
                case spv::OpConvertUToPtr:
                case spv::OpCopyLogical:
                case spv::OpLoad:
                    object_insn = FindDef(object_insn->operand(0));
                    break;
                case spv::OpAccessChain:
                {
                    std::vector<uint32_t> indices;
                    for (uint32_t i = 1; i < object_insn->num_operands(); ++i)
                    {
                        if (auto ins = FindDef(object_insn->operand(i)))
                        {
                            if (ins->opcode() == spv::OpConstant)
                            {
                                // store access-chain index
                                indices.push_back(ins->constant_value());
                            }
                        }
                    }
                    // insert new indices in front
                    access_indices.insert(access_indices.begin(), indices.begin(), indices.end());

                    // continue with base object
                    object_insn = FindDef(object_insn->operand(0));
                    break;
                }
                case spv::OpVariable:
                {
                    const uint32_t storage_class = object_insn->operand(0);
                    if (storage_class == spv::StorageClassFunction)
                    {
                        // When casting to a struct, can get a 2nd function variable, just keep following
                        object_insn = FindVariableStoring(object_insn->resultId());
                    }
                    else
                    {
                        BufferReferenceInfo buffer_reference_info = {};

                        if (GetVariableDecorations(object_insn, buffer_reference_info))
                        {
                            SpvReflectResult                 spv_result;
                            const SpvReflectTypeDescription* td = nullptr;

                            // access-chain starts with descriptor-binding root
                            std::string root_name;

                            if (buffer_reference_info.source == BufferReferenceLocation::PUSH_CONSTANT_BLOCK)
                            {
                                const SpvReflectBlockVariable* block = spvReflectGetEntryPointPushConstantBlock(
                                    &spv_shader_module.value(), spv_shader_module->entry_point_name, &spv_result);
                                td = block->type_description;
                            }
                            else
                            {
                                const SpvReflectDescriptorBinding* spv_descriptor_binding =
                                    spvReflectGetDescriptorBinding(&spv_shader_module.value(),
                                                                   buffer_reference_info.binding,
                                                                   buffer_reference_info.set,
                                                                   &spv_result);
                                td        = spv_descriptor_binding->type_description;
                                root_name = spv_descriptor_binding->name;
                            }

                            if (root_name.empty())
                            {
                                // e.g. push-constant-block or anonymous uniform-block
                                // store typename instead
                                root_name = td->type_name ? "(" + std::string(td->type_name) + ")" : "";
                            }
                            std::vector<std::string> access_chain_names = { root_name };

                            // follow access-chain
                            for (uint32_t idx : access_indices)
                            {
                                if (idx < td->member_count)
                                {
                                    if (td->op == SpvOpTypeArray || td->op == SpvOpTypeRuntimeArray)
                                    {
                                        buffer_reference_info.array_stride = td->traits.array.stride;
                                    }

                                    // offset calculation
                                    for (uint32_t m = 0; m < idx; ++m)
                                    {
                                        uint32_t    num_scalar_bytes = 0;
                                        const auto& member           = td->members[m];
                                        num_scalar_bytes             = member.traits.numeric.scalar.width / 8;

                                        if (member.op == SpvOpTypeVector)
                                        {
                                            num_scalar_bytes *= member.traits.numeric.vector.component_count;
                                        }
                                        else if (member.op == SpvOpTypeMatrix)
                                        {
                                            num_scalar_bytes *= member.traits.numeric.matrix.column_count;
                                            num_scalar_bytes *= member.traits.numeric.matrix.row_count;
                                            num_scalar_bytes =
                                                std::max(num_scalar_bytes, member.traits.numeric.matrix.stride);
                                        }
                                        else if (member.op == SpvOpTypePointer || member.op == SpvOpTypeForwardPointer)
                                        {
                                            num_scalar_bytes = sizeof(uint64_t);
                                        }
                                        else if (member.op == SpvOpTypeArray || member.op == SpvOpTypeRuntimeArray)
                                        {
                                            num_scalar_bytes = std::max(num_scalar_bytes, member.traits.array.stride);
                                            assert(false); // not handled
                                        }
                                        buffer_reference_info.buffer_offset += num_scalar_bytes;
                                    }

                                    td = td->members + idx;
                                    access_chain_names.emplace_back(td->struct_member_name ? td->struct_member_name
                                                                                           : "unknown");
                                }
                                else
                                {
                                    printf("warning: Access-chain index is out-of-bounds for op: %s\n",
                                                         string_SpvOpcode(td->op));
                                    return;
                                }
                            }
                            access_indices.clear();

                            if (td->op == SpvOpTypeRuntimeArray)
                            {
                                buffer_reference_info.array_stride = td->traits.array.stride;
                            }

                            // buffer-references traced back to either pointer-type, uin64_t or arrays of those
                            if (td->op == SpvOpTypePointer || td->op == SpvOpTypeForwardPointer ||
                                (td->op == SpvOpTypeInt && td->traits.numeric.scalar.width == 64) ||
                                td->op == SpvOpTypeRuntimeArray)
                            {
                                buffer_reference_map_[buffer_reference_info] = access_chain_names;
                            }
                            else
                            {
                                printf("warning: Traced back a potential buffer-reference, but type does not match: %s\n",
                                       string_SpvOpcode(td->op));
                            }
                        }
                        object_insn = nullptr;
                    }
                    break;
                }
                default:
                    printf("warning: Failed to track back the Function Variable OpStore, hit a %s\n",
                           string_SpvOpcode(object_insn->opcode()));
                    object_insn = nullptr;
                    break;
            }
        }
    };

    // Now we can walk the SPIR-V one more time to find what we need
    for (const Instruction& insn : instructions)
    {
        // because it is SSA, we can build this up as we are looping in this pass
        const uint32_t result_id = insn.resultId();
        if (result_id != 0)
        {
            definitions_[result_id] = &insn;
        }

        const uint32_t opcode = insn.opcode();

        if (opcode == spv::OpStore)
        {
            store_instructions_.push_back(&insn);
        }
        else if (opcode == spv::OpDecorate)
        {
            decorations_instructions_.push_back(&insn);
        }

        // There is always a load that does the dereferencing
        if (opcode != spv::OpLoad)
        {
            continue;
        }

        // Confirms the load is used for a buffer device address
        const Instruction* type_pointer_insn = FindDef(insn.typeId());
        if (!type_pointer_insn || type_pointer_insn->opcode() != spv::OpTypePointer ||
            type_pointer_insn->operand(0) != spv::StorageClassPhysicalStorageBuffer)
        {
            continue;
        }

        const Instruction* load_pointer_insn = FindDef(insn.operand(0));

        if (load_pointer_insn && load_pointer_insn->opcode() == spv::OpVariable &&
            load_pointer_insn->operand(0) == spv::StorageClassFunction)
        {
            const Instruction* object_insn = FindVariableStoring(load_pointer_insn->resultId());
            if (!object_insn)
            {
                continue;
            }

            track_back_instruction(object_insn);
        }
        else if (load_pointer_insn && load_pointer_insn->opcode() == spv::OpAccessChain)
        {
            track_back_instruction(load_pointer_insn);
        }
    }

    for (const auto& [buffer_reference_info, chain_names] : buffer_reference_map_)
    {
        std::string name;
        for (const auto& sn : chain_names)
        {
            name += sn + " -> ";
        }
        name = name.substr(0, name.size() - 4);

        char buf[128];
        switch (buffer_reference_info.source)
        {
            case BufferReferenceLocation::PUSH_CONSTANT_BLOCK:
                snprintf(buf, sizeof(buf), "push-constant-block");
                break;

            case BufferReferenceLocation::UNIFORM_BUFFER:
            case BufferReferenceLocation::STORAGE_BUFFER:
                snprintf(
                    buf, sizeof(buf), "set: %u, binding: %u", buffer_reference_info.set, buffer_reference_info.binding);
                break;
            default:
                break;
        }

        printf("buffer-reference: %s (%s, buffer-offset: %u, array-stride: %u)\n",
                           name.c_str(),
                           buf,
                           buffer_reference_info.buffer_offset,
                           buffer_reference_info.array_stride);
    }
    // cleanup spirv-module
    if (spv_shader_module != std::nullopt)
    {
        spvReflectDestroyShaderModule(&spv_shader_module.value());
    }

    // successfully parsed
    return true;
}

std::vector<SpirVParsingUtil::BufferReferenceInfo> SpirVParsingUtil::GetBufferReferenceInfos() const
{
    std::vector<BufferReferenceInfo> ret;
    ret.reserve(buffer_reference_map_.size());
    for (const auto& [buffer_ref_info, chain_names] : buffer_reference_map_)
    {
        ret.push_back(buffer_ref_info);
    }
    return ret;
}
