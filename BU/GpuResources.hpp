#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#   include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include <vk_mem_alloc.h>

struct GpuBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void* mapped = nullptr;
};

struct GpuImage
{
    vk::raii::Image image{ nullptr };
    vk::raii::DeviceMemory memory{ nullptr };
};