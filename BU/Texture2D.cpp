#include "Texture2D.hpp"

#include "BufferUtils.hpp"
#include "ImageUtils.hpp"
#include "VulkanContext.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>


#include "stb_image.h"

Texture2D::Texture2D(
    VulkanContext& context,
    BufferUtils& bufferUtils_,
    ImageUtils& imageUtils_,
    const unsigned char* pixels,
    uint32_t width_,
    uint32_t height_,
    uint32_t channels,
    const std::string& name,
    vk::Format format_)
    : Texture2D(
        context,
        bufferUtils_,
        imageUtils_,
        pixels,
        width_,
        height_,
        channels,
        name,
        format_,
        SamplerOptions{})
{
}

Texture2D::Texture2D(
    VulkanContext& context,
    BufferUtils& bufferUtils_,
    ImageUtils& imageUtils_,
    const unsigned char* pixels,
    uint32_t width_,
    uint32_t height_,
    uint32_t channels,
    const std::string& name,
    vk::Format format_,
    const SamplerOptions& samplerOptions)
    : vkContext(context)
    , bufferUtils(bufferUtils_)
    , imageUtils(imageUtils_)
{
    createFromPixels(
        pixels,
        width_,
        height_,
        channels,
        name,
        format_,
        samplerOptions);
}

Texture2D::Texture2D(
    VulkanContext& context,
    BufferUtils& bufferUtils_,
    ImageUtils& imageUtils_,
    const std::string& path,
    vk::Format format_)
    : Texture2D(
        context,
        bufferUtils_,
        imageUtils_,
        path,
        format_,
        SamplerOptions{})
{
}

Texture2D::Texture2D(
    VulkanContext& context,
    BufferUtils& bufferUtils_,
    ImageUtils& imageUtils_,
    const std::string& path,
    vk::Format format_,
    const SamplerOptions& samplerOptions)
    : vkContext(context)
    , bufferUtils(bufferUtils_)
    , imageUtils(imageUtils_)
{
    int texWidth = 0;
    int texHeight = 0;
    int texChannels = 0;

    stbi_uc* pixels = stbi_load(
        path.c_str(),
        &texWidth,
        &texHeight,
        &texChannels,
        STBI_rgb_alpha);

    if (!pixels)
    {
        throw std::runtime_error("Failed to load texture: " + path);
    }

    createFromPixels(
        pixels,
        static_cast<uint32_t>(texWidth),
        static_cast<uint32_t>(texHeight),
        4,
        path,
        format_,
        samplerOptions);

    stbi_image_free(pixels);
}

void Texture2D::createFromPixels(
    const unsigned char* pixels,
    uint32_t width_,
    uint32_t height_,
    uint32_t channels,
    const std::string& name,
    vk::Format format_,
    const SamplerOptions& samplerOptions)
{
    if (!pixels)
    {
        throw std::runtime_error("Texture2D: pixels pointer is null");
    }

    if (width_ == 0 || height_ == 0)
    {
        throw std::runtime_error("Texture2D: invalid texture dimensions");
    }

    width = width_;
    height = height_;
    format = format_;
    sourcePath = name;
    mipLevels = 1;

    std::vector<unsigned char> rgbaPixels;
    const unsigned char* uploadPixels = pixels;

    if (channels == 4)
    {
        uploadPixels = pixels;
    }
    else if (channels == 3)
    {
        rgbaPixels.resize(static_cast<size_t>(width) * height * 4);

        for (uint32_t i = 0; i < width * height; ++i)
        {
            rgbaPixels[i * 4 + 0] = pixels[i * 3 + 0];
            rgbaPixels[i * 4 + 1] = pixels[i * 3 + 1];
            rgbaPixels[i * 4 + 2] = pixels[i * 3 + 2];
            rgbaPixels[i * 4 + 3] = 255;
        }

        uploadPixels = rgbaPixels.data();
    }
    else if (channels == 1)
    {
        rgbaPixels.resize(static_cast<size_t>(width) * height * 4);

        for (uint32_t i = 0; i < width * height; ++i)
        {
            const unsigned char v = pixels[i];

            rgbaPixels[i * 4 + 0] = v;
            rgbaPixels[i * 4 + 1] = v;
            rgbaPixels[i * 4 + 2] = v;
            rgbaPixels[i * 4 + 3] = 255;
        }

        uploadPixels = rgbaPixels.data();
    }
    else
    {
        throw std::runtime_error("Texture2D: unsupported channel count");
    }

    const vk::DeviceSize imageSize =
        static_cast<vk::DeviceSize>(width) *
        static_cast<vk::DeviceSize>(height) *
        4;

    vk::raii::Buffer stagingBuffer{ nullptr };
    vk::raii::DeviceMemory stagingBufferMemory{ nullptr };

    bufferUtils.createBuffer(
        imageSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer,
        stagingBufferMemory);

    void* mapped = stagingBufferMemory.mapMemory(0, imageSize);
    std::memcpy(mapped, uploadPixels, static_cast<size_t>(imageSize));
    stagingBufferMemory.unmapMemory();

    imageUtils.createImage(
        width,
        height,
        mipLevels,
        vk::SampleCountFlagBits::e1,
        format,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst |
        vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        image,
        memory);

    imageUtils.transitionImageLayout(
        image,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal,
        mipLevels);

    imageUtils.copyBufferToImage(
        stagingBuffer,
        image,
        width,
        height);

    imageUtils.transitionImageLayout(
        image,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        mipLevels);

    imageView = imageUtils.createImageView(
        image,
        format,
        vk::ImageAspectFlagBits::eColor,
        mipLevels);

    auto& device = vkContext.getDevice();

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo
        .setMagFilter(samplerOptions.magFilter)
        .setMinFilter(samplerOptions.minFilter)
        .setAddressModeU(samplerOptions.addressModeU)
        .setAddressModeV(samplerOptions.addressModeV)
        .setAddressModeW(samplerOptions.addressModeW)
        .setAnisotropyEnable(samplerOptions.enableAnisotropy ? VK_TRUE : VK_FALSE)
        .setMaxAnisotropy(samplerOptions.maxAnisotropy)
        .setBorderColor(vk::BorderColor::eIntOpaqueBlack)
        .setUnnormalizedCoordinates(VK_FALSE)
        .setCompareEnable(VK_FALSE)
        .setCompareOp(vk::CompareOp::eAlways)
        .setMipmapMode(samplerOptions.mipmapMode)
        .setMipLodBias(0.0f)
        .setMinLod(samplerOptions.minLod)
        .setMaxLod(samplerOptions.maxLod);

    sampler = vk::raii::Sampler(device, samplerInfo);
}