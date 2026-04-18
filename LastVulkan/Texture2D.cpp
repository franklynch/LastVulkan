#include "Texture2D.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>


#include <stb_image.h>

Texture2D::Texture2D(VulkanContext& vkContext,
    BufferUtils& bufferUtils,
    ImageUtils& imageUtils,
    const std::string& path)
    : vkContext(vkContext)
    , bufferUtils(bufferUtils)
    , imageUtils(imageUtils)
    , sourcePath(path)
{
    loadFromFile(path);
    createImageView();
    createSampler();
}

Texture2D::Texture2D(
    VulkanContext& vkContext,
    BufferUtils& bufferUtils,
    ImageUtils& imageUtils,
    const unsigned char* pixelData,
    uint32_t width,
    uint32_t height,
    uint32_t channels,
    const std::string& debugName,
    vk::Format format)
    : vkContext(vkContext)
    , bufferUtils(bufferUtils)
    , imageUtils(imageUtils)
    , sourcePath(debugName)
    , imageFormat(format)
{
    loadFromMemory(pixelData, width, height, channels, format);
    createImageView();
    createSampler();
}



void Texture2D::loadFromFile(const std::string& path)
{
    int texWidth = 0;
    int texHeight = 0;
    int texChannels = 0;

    imageFormat = vk::Format::eR8G8B8A8Srgb;

    stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!pixels)
    {
        throw std::runtime_error("failed to load texture image: " + path);
    }

    vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(texWidth) *
        static_cast<vk::DeviceSize>(texHeight) * 4;

    mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

    vk::raii::Buffer stagingBuffer({ });
    vk::raii::DeviceMemory stagingBufferMemory({ });

    bufferUtils.createBuffer(
        imageSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer,
        stagingBufferMemory
    );

    void* data = stagingBufferMemory.mapMemory(0, imageSize);
    std::memcpy(data, pixels, static_cast<size_t>(imageSize));
    stagingBufferMemory.unmapMemory();

    stbi_image_free(pixels);

    imageUtils.createImage(
        static_cast<uint32_t>(texWidth),
        static_cast<uint32_t>(texHeight),
        mipLevels,
        vk::SampleCountFlagBits::e1,
        imageFormat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferSrc |
        vk::ImageUsageFlagBits::eTransferDst |
        vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        image,
        imageMemory
    );

    imageUtils.transitionImageLayout(
        image,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal,
        mipLevels
    );

    imageUtils.copyBufferToImage(
        stagingBuffer,
        image,
        static_cast<uint32_t>(texWidth),
        static_cast<uint32_t>(texHeight)
    );

    generateMipmaps(image, imageFormat, texWidth, texHeight, mipLevels);
}

void Texture2D::loadFromMemory(
    const unsigned char* pixelData,
    uint32_t width,
    uint32_t height,
    uint32_t channels,
    vk::Format format)
{
    if (!pixelData)
        throw std::runtime_error("Texture2D: pixelData is null");

    if (width == 0 || height == 0)
        throw std::runtime_error("Texture2D: invalid dimensions");

    if (channels != 3 && channels != 4)
        throw std::runtime_error("Texture2D: only RGB8 and RGBA8 supported");

    std::vector<unsigned char> rgbaPixels;
    const unsigned char* uploadPixels = pixelData;

    if (channels == 3)
    {
        rgbaPixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 4);

        for (size_t i = 0; i < static_cast<size_t>(width) * static_cast<size_t>(height); ++i)
        {
            rgbaPixels[i * 4 + 0] = pixelData[i * 3 + 0];
            rgbaPixels[i * 4 + 1] = pixelData[i * 3 + 1];
            rgbaPixels[i * 4 + 2] = pixelData[i * 3 + 2];
            rgbaPixels[i * 4 + 3] = 255;
        }

        uploadPixels = rgbaPixels.data();
    }

    vk::DeviceSize imageSize =
        static_cast<vk::DeviceSize>(width) *
        static_cast<vk::DeviceSize>(height) * 4;

    mipLevels = static_cast<uint32_t>(
        std::floor(std::log2(std::max(width, height)))) + 1;

    vk::raii::Buffer stagingBuffer({});
    vk::raii::DeviceMemory stagingBufferMemory({});

    bufferUtils.createBuffer(
        imageSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer,
        stagingBufferMemory
    );

    void* data = stagingBufferMemory.mapMemory(0, imageSize);
    std::memcpy(data, uploadPixels, static_cast<size_t>(imageSize));
    stagingBufferMemory.unmapMemory();

    imageUtils.createImage(
        width,
        height,
        mipLevels,
        vk::SampleCountFlagBits::e1,
        format,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferSrc |
        vk::ImageUsageFlagBits::eTransferDst |
        vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        image,
        imageMemory
    );

    imageUtils.transitionImageLayout(
        image,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal,
        mipLevels
    );

    imageUtils.copyBufferToImage(
        stagingBuffer,
        image,
        width,
        height
    );

    generateMipmaps(
        image,
        format,
        static_cast<int32_t>(width),
        static_cast<int32_t>(height),
        mipLevels
    );
}

void Texture2D::createImageView()
{
        imageView = imageUtils.createImageView(
            image,
            imageFormat,
            vk::ImageAspectFlagBits::eColor,
            mipLevels);
}

void Texture2D::createSampler()
{
    auto& physicalDevice = vkContext.getPhysicalDevice();
    auto& device = vkContext.getDevice();

    vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo
        .setMagFilter(vk::Filter::eLinear)
        .setMinFilter(vk::Filter::eLinear)
        .setMipmapMode(vk::SamplerMipmapMode::eLinear)
        .setAddressModeU(vk::SamplerAddressMode::eRepeat)
        .setAddressModeV(vk::SamplerAddressMode::eRepeat)
        .setAddressModeW(vk::SamplerAddressMode::eRepeat)
        .setMipLodBias(0.0f)
        .setAnisotropyEnable(VK_TRUE)
        .setMaxAnisotropy(properties.limits.maxSamplerAnisotropy)
        .setCompareEnable(VK_FALSE)
        .setCompareOp(vk::CompareOp::eAlways)
        .setMinLod(0.0f)
        .setMaxLod(static_cast<float>(mipLevels))
        .setBorderColor(vk::BorderColor::eIntOpaqueBlack)
        .setUnnormalizedCoordinates(VK_FALSE);

    sampler = vk::raii::Sampler(device, samplerInfo);
}

void Texture2D::generateMipmaps(vk::raii::Image& image,
    vk::Format imageFormat,
    int32_t texWidth,
    int32_t texHeight,
    uint32_t mipLevels)
{
    auto& physicalDevice = vkContext.getPhysicalDevice();

    vk::FormatProperties formatProperties = physicalDevice.getFormatProperties(imageFormat);

    if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear))
    {
        throw std::runtime_error("texture image format does not support linear blitting!");
    }

    auto commandBuffer = bufferUtils.beginSingleTimeCommands();

    vk::ImageSubresourceRange subresourceRange{};
    subresourceRange
        .setAspectMask(vk::ImageAspectFlagBits::eColor)
        .setBaseArrayLayer(0)
        .setLayerCount(1)
        .setLevelCount(1);

    vk::ImageMemoryBarrier barrier{};
    barrier
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*image)
        .setSubresourceRange(subresourceRange);

    int32_t mipWidth = texWidth;
    int32_t mipHeight = texHeight;

    for (uint32_t i = 1; i < mipLevels; i++)
    {
        barrier
            .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
            .setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
            .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(vk::AccessFlagBits::eTransferRead);

        barrier.subresourceRange.setBaseMipLevel(i - 1);

        commandBuffer->pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eTransfer,
            {},
            nullptr,
            nullptr,
            barrier
        );

        std::array<vk::Offset3D, 2> srcOffsets = {
            vk::Offset3D{0, 0, 0},
            vk::Offset3D{mipWidth, mipHeight, 1}
        };

        std::array<vk::Offset3D, 2> dstOffsets = {
            vk::Offset3D{0, 0, 0},
            vk::Offset3D{
                mipWidth > 1 ? mipWidth / 2 : 1,
                mipHeight > 1 ? mipHeight / 2 : 1,
                1
            }
        };

        vk::ImageSubresourceLayers srcSubresource{};
        srcSubresource
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setMipLevel(i - 1)
            .setBaseArrayLayer(0)
            .setLayerCount(1);

        vk::ImageSubresourceLayers dstSubresource{};
        dstSubresource
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setMipLevel(i)
            .setBaseArrayLayer(0)
            .setLayerCount(1);

        vk::ImageBlit blit{};
        blit
            .setSrcSubresource(srcSubresource)
            .setSrcOffsets(srcOffsets)
            .setDstSubresource(dstSubresource)
            .setDstOffsets(dstOffsets);

        commandBuffer->blitImage(
            *image,
            vk::ImageLayout::eTransferSrcOptimal,
            *image,
            vk::ImageLayout::eTransferDstOptimal,
            blit,
            vk::Filter::eLinear
        );

        barrier
            .setOldLayout(vk::ImageLayout::eTransferSrcOptimal)
            .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
            .setSrcAccessMask(vk::AccessFlagBits::eTransferRead)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

        commandBuffer->pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eFragmentShader,
            {},
            nullptr,
            nullptr,
            barrier
        );

        if (mipWidth > 1) mipWidth /= 2;
        if (mipHeight > 1) mipHeight /= 2;
    }

    barrier.subresourceRange.setBaseMipLevel(mipLevels - 1);
    barrier
        .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
        .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

    commandBuffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader,
        {},
        nullptr,
        nullptr,
        barrier
    );

    bufferUtils.endSingleTimeCommands(*commandBuffer);
}

