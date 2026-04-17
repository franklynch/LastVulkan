#pragma once

#include <string>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include "VulkanContext.hpp"
#include "BufferUtils.hpp"
#include "ImageUtils.hpp"

class Texture2D
{
public:
    Texture2D(VulkanContext& vkContext,
        BufferUtils& bufferUtils,
        ImageUtils& imageUtils,
        const std::string& path);

    Texture2D(
        VulkanContext& vkContext,
        BufferUtils& bufferUtils,
        ImageUtils& imageUtils,
        const unsigned char* pixelData,
        uint32_t width,
        uint32_t height,
        uint32_t channels,
        const std::string& debugName = "<memory>");


    Texture2D(const Texture2D&) = delete;
    Texture2D& operator=(const Texture2D&) = delete;

    [[nodiscard]] const vk::raii::ImageView& getImageView() const { return imageView; }
    [[nodiscard]] const vk::raii::Sampler& getSampler() const { return sampler; }
    [[nodiscard]] uint32_t getMipLevels() const { return mipLevels; }

    [[nodiscard]] const std::string& getSourcePath() const { return sourcePath; }

private:
    void loadFromFile(const std::string& path);

    void loadFromMemory(
        const unsigned char* pixelData,
        uint32_t width,
        uint32_t height,
        uint32_t channels);

    void createImageView();
    void createSampler();

    void generateMipmaps(vk::raii::Image& image,
        vk::Format imageFormat,
        int32_t texWidth,
        int32_t texHeight,
        uint32_t mipLevels);

private:
    VulkanContext& vkContext;
    BufferUtils& bufferUtils;
    ImageUtils& imageUtils;

    std::string sourcePath;

    uint32_t mipLevels = 0;
    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory imageMemory = nullptr;
    vk::raii::ImageView imageView = nullptr;
    vk::raii::Sampler sampler = nullptr;


};