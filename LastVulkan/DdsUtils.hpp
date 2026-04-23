#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#   include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

struct DdsSubresource
{
    uint32_t mipLevel = 0;
    uint32_t arrayLayer = 0; // cubemap face 0..5
    uint32_t width = 0;
    uint32_t height = 0;

    size_t rowPitch = 0;
    size_t slicePitch = 0;

    std::vector<uint8_t> pixels;
};

struct DdsCubemapData
{
    vk::Format format = vk::Format::eUndefined;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipLevels = 0;
    uint32_t arrayLayers = 0; // should be 6
    bool isCompressed = false;

    std::vector<DdsSubresource> subresources;
};

namespace DdsUtils
{
    DdsCubemapData loadCubemapDDS(const std::string& path);
}