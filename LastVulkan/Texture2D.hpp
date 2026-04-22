#pragma once

#include <string>
#include <vector>
#include <cstddef>

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
    struct SamplerOptions
    {
        vk::Filter magFilter = vk::Filter::eLinear;
        vk::Filter minFilter = vk::Filter::eLinear;
        vk::SamplerMipmapMode mipmapMode = vk::SamplerMipmapMode::eLinear;

        vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat;
        vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat;
        vk::SamplerAddressMode addressModeW = vk::SamplerAddressMode::eRepeat;

        bool enableAnisotropy = true;
        float maxLod = -1.0f; // negative = use mipLevels
    };

    struct UploadDesc
    {
        const void* data = nullptr;
        uint32_t width = 0;
        uint32_t height = 0;

        vk::Format format = vk::Format::eR8G8B8A8Unorm;
        size_t dataSizeBytes = 0;
        bool generateMips = true;
    };

public:
    Texture2D(
        VulkanContext& vkContext,
        BufferUtils& bufferUtils,
        ImageUtils& imageUtils,
        const std::string& path);

    // Convenience byte upload path kept for existing renderer code
    Texture2D(
        VulkanContext& vkContext,
        BufferUtils& bufferUtils,
        ImageUtils& imageUtils,
        const unsigned char* pixelData,
        uint32_t width,
        uint32_t height,
        uint32_t channels,
        const std::string& debugName = "<memory>",
        vk::Format format = vk::Format::eR8G8B8A8Srgb,
        SamplerOptions samplerOptions = {});

    // Raw upload path for future LUT/HDR work
    Texture2D(
        VulkanContext& vkContext,
        BufferUtils& bufferUtils,
        ImageUtils& imageUtils,
        const UploadDesc& desc,
        const std::string& debugName = "<memory>",
        SamplerOptions samplerOptions = {});

    Texture2D(const Texture2D&) = delete;
    Texture2D& operator=(const Texture2D&) = delete;

    [[nodiscard]] const vk::raii::ImageView& getImageView() const { return imageView; }
    [[nodiscard]] const vk::raii::Sampler& getSampler() const { return sampler; }
    [[nodiscard]] uint32_t getMipLevels() const { return mipLevels; }
    [[nodiscard]] const std::string& getSourcePath() const { return sourcePath; }
    [[nodiscard]] vk::Format getFormat() const { return imageFormat; }

private:
    void loadFromFile(const std::string& path);
    void loadFromMemory(const UploadDesc& desc);

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

    vk::Format imageFormat = vk::Format::eR8G8B8A8Srgb;
    SamplerOptions samplerOptions{};
};