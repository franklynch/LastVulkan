#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <vector>
#include <string>

std::vector<char> readFile(const std::string& filename);

vk::raii::ShaderModule createShaderModule(
    const vk::raii::Device& device,
    const std::vector<char>& code);