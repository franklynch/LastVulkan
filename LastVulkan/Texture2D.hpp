#pragma once

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <string>

class VulkanContext;
class BufferUtils;
class ImageUtils;

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
        float maxAnisotropy = 16.0f;

        float minLod = 0.0f;
        float maxLod = 1.0f;
    };

    Texture2D(
        VulkanContext& context,
        BufferUtils& bufferUtils,
        ImageUtils& imageUtils,
        const unsigned char* pixels,
        uint32_t width,
        uint32_t height,
        uint32_t channels,
        const std::string& name,
        vk::Format format);

    Texture2D(
        VulkanContext& context,
        BufferUtils& bufferUtils,
        ImageUtils& imageUtils,
        const unsigned char* pixels,
        uint32_t width,
        uint32_t height,
        uint32_t channels,
        const std::string& name,
        vk::Format format,
        const SamplerOptions& samplerOptions);

    Texture2D(
        VulkanContext& context,
        BufferUtils& bufferUtils,
        ImageUtils& imageUtils,
        const std::string& path,
        vk::Format format);

    Texture2D(
        VulkanContext& context,
        BufferUtils& bufferUtils,
        ImageUtils& imageUtils,
        const std::string& path,
        vk::Format format,
        const SamplerOptions& samplerOptions);

    ~Texture2D() = default;

    Texture2D(const Texture2D&) = delete;
    Texture2D& operator=(const Texture2D&) = delete;

    Texture2D(Texture2D&&) noexcept = default;
    Texture2D& operator=(Texture2D&&) noexcept = default;

    vk::ImageView getImageView() const { return *imageView; }
    vk::Sampler getSampler() const { return *sampler; }

    uint32_t getWidth() const { return width; }
    uint32_t getHeight() const { return height; }
    uint32_t getMipLevels() const { return mipLevels; }

    vk::Format getFormat() const { return format; }

    const std::string& getSourcePath() const { return sourcePath; }

private:
    void createFromPixels(
        const unsigned char* pixels,
        uint32_t width,
        uint32_t height,
        uint32_t channels,
        const std::string& name,
        vk::Format format,
        const SamplerOptions& samplerOptions);

private:
    VulkanContext& vkContext;
    BufferUtils& bufferUtils;
    ImageUtils& imageUtils;

    vk::raii::Image image{ nullptr };
    vk::raii::DeviceMemory memory{ nullptr };
    vk::raii::ImageView imageView{ nullptr };
    vk::raii::Sampler sampler{ nullptr };

    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipLevels = 1;

    vk::Format format = vk::Format::eUndefined;
    std::string sourcePath = "<generated>";
};