#include "ShaderUtils.hpp"

#include <fstream>
#include <stdexcept>

namespace ShaderUtils
{

    std::vector<uint32_t> readSpirvFile(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open())
        {
            throw std::runtime_error("failed to open SPIR-V file: " + filename);
        }

        const std::streamsize fileSize = file.tellg();

        if (fileSize <= 0 || fileSize % sizeof(uint32_t) != 0)
        {
            throw std::runtime_error("invalid SPIR-V file size: " + filename);
        }

        std::vector<uint32_t> buffer(
            static_cast<size_t>(fileSize) / sizeof(uint32_t));

        file.seekg(0);
        file.read(
            reinterpret_cast<char*>(buffer.data()),
            fileSize);

        if (!file)
        {
            throw std::runtime_error("failed to read SPIR-V file: " + filename);
        }

        return buffer;
    }

    vk::raii::ShaderModule createShaderModule(
        const vk::raii::Device& device,
        const std::string& filename)
    {
        const std::vector<uint32_t> code = readSpirvFile(filename);

        vk::ShaderModuleCreateInfo createInfo{};
        createInfo
            .setCodeSize(code.size() * sizeof(uint32_t))
            .setPCode(code.data());

        return vk::raii::ShaderModule(device, createInfo);
    }
}
