#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include "VulkanContext.hpp"
#include "BufferUtils.hpp"

class ImageUtils
{
public:
    ImageUtils(VulkanContext& vkContext, BufferUtils& bufferUtils);

    void createImage(
        uint32_t width,
        uint32_t height,
        uint32_t mipLevels,
        vk::SampleCountFlagBits numSamples,
        vk::Format format,
        vk::ImageTiling tiling,
        vk::ImageUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        vk::raii::Image& image,
        vk::raii::DeviceMemory& imageMemory) const;

    [[nodiscard]] vk::raii::ImageView createImageView(
        const vk::raii::Image& image,
        vk::Format format,
        vk::ImageAspectFlags aspectFlags,
        uint32_t mipLevels) const;

    void transitionImageLayout(
        const vk::raii::Image& image,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        uint32_t mipLevels) const;

    void copyBufferToImage(
        const vk::raii::Buffer& buffer,
        vk::raii::Image& image,
        uint32_t width,
        uint32_t height) const;

    void generateMipmaps(
        vk::raii::Image& image,
        vk::Format format,
        int32_t width,
        int32_t height,
        uint32_t mipLevels) const;

private:
    VulkanContext& vkContext;
    BufferUtils& bufferUtils;
};