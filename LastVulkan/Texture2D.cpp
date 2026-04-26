#include "Texture2D.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <stb_image.h>
Texture2D::Texture2D(
    VulkanContext& vkContext,
    BufferUtils& bufferUtils,
    ImageUtils& imageUtils,
    const std::string& path,
    vk::Format format)
    : vkContext(vkContext)
    , bufferUtils(bufferUtils)
    , imageUtils(imageUtils)
    , sourcePath(path)
    , imageFormat(format) // THIS is "passing it through"
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
    vk::Format format,
    SamplerOptions samplerOptions)
    : vkContext(vkContext)
    , bufferUtils(bufferUtils)
    , imageUtils(imageUtils)
    , sourcePath(debugName)
    , imageFormat(format)
    , samplerOptions(samplerOptions)
{
    if (!pixelData)
    {
        throw std::runtime_error("Texture2D: pixelData is null");
    }

    if (width == 0 || height == 0)
    {
        throw std::runtime_error("Texture2D: invalid dimensions");
    }

    if (channels != 3 && channels != 4)
    {
        throw std::runtime_error("Texture2D: only RGB8 and RGBA8 convenience uploads are supported");
    }

    std::vector<unsigned char> expandedRgbaPixels;
    const unsigned char* resolvedPixelData = pixelData;

    if (channels == 3)
    {
        const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
        expandedRgbaPixels.resize(pixelCount * 4);

        for (size_t i = 0; i < pixelCount; ++i)
        {
            expandedRgbaPixels[i * 4 + 0] = pixelData[i * 3 + 0];
            expandedRgbaPixels[i * 4 + 1] = pixelData[i * 3 + 1];
            expandedRgbaPixels[i * 4 + 2] = pixelData[i * 3 + 2];
            expandedRgbaPixels[i * 4 + 3] = 255;
        }

        resolvedPixelData = expandedRgbaPixels.data();
    }

    UploadDesc uploadDesc{};
    uploadDesc.data = resolvedPixelData;
    uploadDesc.width = width;
    uploadDesc.height = height;
    uploadDesc.format = format;
    uploadDesc.dataSizeBytes =
        static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
    uploadDesc.generateMips = true;

    loadFromMemory(uploadDesc);
    createImageView();
    createSampler();
}

Texture2D::Texture2D(
    VulkanContext& vkContext,
    BufferUtils& bufferUtils,
    ImageUtils& imageUtils,
    const UploadDesc& desc,
    const std::string& debugName,
    SamplerOptions samplerOptions)
    : vkContext(vkContext)
    , bufferUtils(bufferUtils)
    , imageUtils(imageUtils)
    , sourcePath(debugName)
    , imageFormat(desc.format)
    , samplerOptions(samplerOptions)
{
    loadFromMemory(desc);
    createImageView();
    createSampler();
}

void Texture2D::loadFromFile(const std::string& path)
{
    int texWidth = 0;
    int texHeight = 0;
    int texChannels = 0;

    

    stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!pixels)
    {
        throw std::runtime_error("failed to load texture image: " + path);
    }

    UploadDesc desc{};
    desc.data = pixels;
    desc.width = static_cast<uint32_t>(texWidth);
    desc.height = static_cast<uint32_t>(texHeight);
    desc.format = imageFormat;
    desc.dataSizeBytes = static_cast<size_t>(texWidth) * static_cast<size_t>(texHeight) * 4;
    desc.generateMips = true;

    loadFromMemory(desc);

    stbi_image_free(pixels);
}

void Texture2D::loadFromMemory(const UploadDesc& desc)
{
    if (!desc.data)
    {
        throw std::runtime_error("Texture2D: uploadDesc.data is null");
    }

    if (desc.width == 0 || desc.height == 0)
    {
        throw std::runtime_error("Texture2D: invalid dimensions");
    }

    if (desc.dataSizeBytes == 0)
    {
        throw std::runtime_error("Texture2D: uploadDesc.dataSizeBytes must be non-zero");
    }

    imageFormat = desc.format;

    if (desc.generateMips)
    {
        mipLevels = static_cast<uint32_t>(
            std::floor(std::log2(std::max(desc.width, desc.height)))
            ) + 1;
    }
    else
    {
        mipLevels = 1;
    }

    const vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(desc.dataSizeBytes);

    vk::raii::Buffer stagingBuffer{ nullptr };
    vk::raii::DeviceMemory stagingBufferMemory{ nullptr };

    bufferUtils.createBuffer(
        imageSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer,
        stagingBufferMemory
    );

    void* mappedData = stagingBufferMemory.mapMemory(0, imageSize);
    std::memcpy(mappedData, desc.data, desc.dataSizeBytes);
    stagingBufferMemory.unmapMemory();

    vk::ImageUsageFlags usage =
        vk::ImageUsageFlagBits::eTransferDst |
        vk::ImageUsageFlagBits::eSampled;

    if (desc.generateMips)
    {
        usage |= vk::ImageUsageFlagBits::eTransferSrc;
    }

    imageUtils.createImage(
        desc.width,
        desc.height,
        mipLevels,
        vk::SampleCountFlagBits::e1,
        imageFormat,
        vk::ImageTiling::eOptimal,
        usage,
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
        desc.width,
        desc.height
    );

    if (desc.generateMips)
    {
        generateMipmaps(
            image,
            imageFormat,
            static_cast<int32_t>(desc.width),
            static_cast<int32_t>(desc.height),
            mipLevels
        );
    }
    else
    {
        imageUtils.transitionImageLayout(
            image,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            1
        );
    }
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

    float resolvedMaxLod =
        (samplerOptions.maxLod >= 0.0f)
        ? samplerOptions.maxLod
        : static_cast<float>(mipLevels);

    float resolvedAnisotropy =
        samplerOptions.enableAnisotropy
        ? properties.limits.maxSamplerAnisotropy
        : 1.0f;

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo
        .setMagFilter(samplerOptions.magFilter)
        .setMinFilter(samplerOptions.minFilter)
        .setMipmapMode(samplerOptions.mipmapMode)
        .setAddressModeU(samplerOptions.addressModeU)
        .setAddressModeV(samplerOptions.addressModeV)
        .setAddressModeW(samplerOptions.addressModeW)
        .setMipLodBias(0.0f)
        .setAnisotropyEnable(samplerOptions.enableAnisotropy ? VK_TRUE : VK_FALSE)
        .setMaxAnisotropy(resolvedAnisotropy)
        .setCompareEnable(VK_FALSE)
        .setCompareOp(vk::CompareOp::eAlways)
        .setMinLod(0.0f)
        .setMaxLod(resolvedMaxLod)
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

        commandBuffer.pipelineBarrier(
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
                1}
        };

        vk::ImageBlit blit{};
        blit
            .setSrcOffsets(srcOffsets)
            .setSrcSubresource(vk::ImageSubresourceLayers{
                vk::ImageAspectFlagBits::eColor,
                i - 1,
                0,
                1
                })
            .setDstOffsets(dstOffsets)
            .setDstSubresource(vk::ImageSubresourceLayers{
                vk::ImageAspectFlagBits::eColor,
                i,
                0,
                1
                });

        commandBuffer.blitImage(
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

        commandBuffer.pipelineBarrier(
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

    barrier
        .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
        .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

    barrier.subresourceRange.setBaseMipLevel(mipLevels - 1);

    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader,
        {},
        nullptr,
        nullptr,
        barrier
    );

    bufferUtils.endSingleTimeCommands(commandBuffer);
}