#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#   include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

struct GpuBuffer
{
    vk::raii::Buffer buffer{ nullptr };
    vk::raii::DeviceMemory memory{ nullptr };
    void* mapped = nullptr;
};

struct GpuImage
{
    vk::raii::Image image{ nullptr };
    vk::raii::DeviceMemory memory{ nullptr };
};