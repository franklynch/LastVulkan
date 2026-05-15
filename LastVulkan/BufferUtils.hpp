#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include "VulkanContext.hpp"
#include "GpuResources.hpp"

class BufferUtils
{
public:
    explicit BufferUtils(VulkanContext& vkContext);

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const;

    void createBuffer(
        vk::DeviceSize size,
        vk::BufferUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        GpuBuffer& outBuffer);

    void copyBuffer(
        VkBuffer srcBuffer,
        VkBuffer dstBuffer,
        vk::DeviceSize size);

    vk::raii::CommandBuffer beginSingleTimeCommands() const;
    void endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer) const;

    void destroyBuffer(GpuBuffer& buffer);

private:
    VulkanContext& vkContext;
};