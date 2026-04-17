#pragma once

#include <memory>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include "VulkanContext.hpp"

class BufferUtils
{
public:
    explicit BufferUtils(VulkanContext& vkContext);

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const;

    void createBuffer(vk::DeviceSize size,
        vk::BufferUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        vk::raii::Buffer& buffer,
        vk::raii::DeviceMemory& bufferMemory) const;

    void copyBuffer(vk::raii::Buffer& srcBuffer,
        vk::raii::Buffer& dstBuffer,
        vk::DeviceSize size) const;

    std::unique_ptr<vk::raii::CommandBuffer> beginSingleTimeCommands() const;
    void endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer) const;

private:
    VulkanContext& vkContext;
};