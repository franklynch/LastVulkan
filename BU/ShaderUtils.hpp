#pragma once

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ShaderUtils
{

    std::vector<uint32_t> readSpirvFile(const std::string& filename);

    vk::raii::ShaderModule createShaderModule(
        const vk::raii::Device& device,
        const std::string& filename);
}
