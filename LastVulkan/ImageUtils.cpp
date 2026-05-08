#include "ImageUtils.hpp"

#include <array>
#include <stdexcept>

ImageUtils::ImageUtils(VulkanContext& vkContext, BufferUtils& bufferUtils)
    : vkContext(vkContext)
    , bufferUtils(bufferUtils)
{
}

namespace
{
    void pipelineImageBarrier2(
        vk::raii::CommandBuffer& commandBuffer,
        vk::Image image,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        vk::PipelineStageFlags2 srcStage,
        vk::AccessFlags2 srcAccess,
        vk::PipelineStageFlags2 dstStage,
        vk::AccessFlags2 dstAccess,
        vk::ImageSubresourceRange subresourceRange)
    {
        vk::ImageMemoryBarrier2 barrier{};
        barrier
            .setSrcStageMask(srcStage)
            .setSrcAccessMask(srcAccess)
            .setDstStageMask(dstStage)
            .setDstAccessMask(dstAccess)
            .setOldLayout(oldLayout)
            .setNewLayout(newLayout)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(image)
            .setSubresourceRange(subresourceRange);

        vk::DependencyInfo dependencyInfo{};
        dependencyInfo.setImageMemoryBarriers(barrier);

        commandBuffer.pipelineBarrier2(dependencyInfo);
    }
}


void ImageUtils::createImage(
    uint32_t width,
    uint32_t height,
    uint32_t mipLevels,
    vk::SampleCountFlagBits numSamples,
    vk::Format format,
    vk::ImageTiling tiling,
    vk::ImageUsageFlags usage,
    vk::MemoryPropertyFlags properties,
    vk::raii::Image& image,
    vk::raii::DeviceMemory& imageMemory) const
{
    auto& device = vkContext.getDevice();

    vk::ImageCreateInfo imageInfo{};
    imageInfo
        .setImageType(vk::ImageType::e2D)
        .setExtent(vk::Extent3D{ width, height, 1 })
        .setMipLevels(mipLevels)
        .setArrayLayers(1)
        .setFormat(format)
        .setTiling(tiling)
        .setInitialLayout(vk::ImageLayout::eUndefined)
        .setUsage(usage)
        .setSamples(numSamples)
        .setSharingMode(vk::SharingMode::eExclusive);

    image = vk::raii::Image(device, imageInfo);

    vk::MemoryRequirements memRequirements = image.getMemoryRequirements();

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo
        .setAllocationSize(memRequirements.size)
        .setMemoryTypeIndex(
            bufferUtils.findMemoryType(memRequirements.memoryTypeBits, properties));

    imageMemory = vk::raii::DeviceMemory(device, allocInfo);
    image.bindMemory(*imageMemory, 0);
}

vk::raii::ImageView ImageUtils::createImageView(
    const vk::raii::Image& image,
    vk::Format format,
    vk::ImageAspectFlags aspectFlags,
    uint32_t mipLevels) const
{
    auto& device = vkContext.getDevice();

    vk::ImageViewCreateInfo viewInfo{};
    viewInfo
        .setImage(*image)
        .setViewType(vk::ImageViewType::e2D)
        .setFormat(format)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(aspectFlags)
            .setBaseMipLevel(0)
            .setLevelCount(mipLevels)
            .setBaseArrayLayer(0)
            .setLayerCount(1));

    return vk::raii::ImageView(device, viewInfo);
}

void ImageUtils::transitionImageLayout(
    const vk::raii::Image& image,
    vk::ImageLayout oldLayout,
    vk::ImageLayout newLayout,
    uint32_t mipLevels) const
{
    auto commandBuffer = bufferUtils.beginSingleTimeCommands();

    vk::PipelineStageFlags2 sourceStage{};
    vk::PipelineStageFlags2 destinationStage{};
    vk::AccessFlags2 srcAccess{};
    vk::AccessFlags2 dstAccess{};

    if (oldLayout == vk::ImageLayout::eUndefined &&
        newLayout == vk::ImageLayout::eTransferDstOptimal)
    {
        srcAccess = vk::AccessFlagBits2::eNone;
        dstAccess = vk::AccessFlagBits2::eTransferWrite;

        sourceStage = vk::PipelineStageFlagBits2::eNone;
        destinationStage = vk::PipelineStageFlagBits2::eTransfer;
    }
    else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
        newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        srcAccess = vk::AccessFlagBits2::eTransferWrite;
        dstAccess = vk::AccessFlagBits2::eShaderSampledRead;

        sourceStage = vk::PipelineStageFlagBits2::eTransfer;
        destinationStage = vk::PipelineStageFlagBits2::eFragmentShader;
    }
    else
    {
        throw std::invalid_argument("unsupported layout transition!");
    }

    vk::ImageMemoryBarrier2 barrier{};
    barrier
        .setSrcStageMask(sourceStage)
        .setSrcAccessMask(srcAccess)
        .setDstStageMask(destinationStage)
        .setDstAccessMask(dstAccess)
        .setOldLayout(oldLayout)
        .setNewLayout(newLayout)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*image)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(mipLevels)
            .setBaseArrayLayer(0)
            .setLayerCount(1));

    vk::DependencyInfo dependencyInfo{};
    dependencyInfo.setImageMemoryBarriers(barrier);

    commandBuffer.pipelineBarrier2(dependencyInfo);

    bufferUtils.endSingleTimeCommands(commandBuffer);
}

void ImageUtils::copyBufferToImage(
    const vk::raii::Buffer& buffer,
    vk::raii::Image& image,
    uint32_t width,
    uint32_t height) const
{
    auto commandBuffer = bufferUtils.beginSingleTimeCommands();

    vk::BufferImageCopy region{};
    region
        .setBufferOffset(0)
        .setBufferRowLength(0)
        .setBufferImageHeight(0)
        .setImageSubresource(
            vk::ImageSubresourceLayers{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setMipLevel(0)
            .setBaseArrayLayer(0)
            .setLayerCount(1))
        .setImageOffset(vk::Offset3D{ 0, 0, 0 })
        .setImageExtent(vk::Extent3D{ width, height, 1 });

    commandBuffer.copyBufferToImage(
        *buffer,
        *image,
        vk::ImageLayout::eTransferDstOptimal,
        region);

    bufferUtils.endSingleTimeCommands(commandBuffer);
}

void ImageUtils::generateMipmaps(
    vk::raii::Image& image,
    vk::Format format,
    int32_t width,
    int32_t height,
    uint32_t mipLevels) const
{
    auto& physicalDevice = vkContext.getPhysicalDevice();

    vk::FormatProperties formatProperties =
        physicalDevice.getFormatProperties(format);

    if (!(formatProperties.optimalTilingFeatures &
        vk::FormatFeatureFlagBits::eSampledImageFilterLinear))
    {
        throw std::runtime_error(
            "texture image format does not support linear blitting!");
    }

    auto commandBuffer = bufferUtils.beginSingleTimeCommands();

    int32_t mipWidth = width;
    int32_t mipHeight = height;

    for (uint32_t i = 1; i < mipLevels; i++)
    {
        pipelineImageBarrier2(
            commandBuffer,
            *image,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eTransferSrcOptimal,
            vk::PipelineStageFlagBits2::eTransfer,
            vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eTransfer,
            vk::AccessFlagBits2::eTransferRead,
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(i - 1)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(1));

        std::array<vk::Offset3D, 2> srcOffsets = {
            vk::Offset3D{ 0, 0, 0 },
            vk::Offset3D{ mipWidth, mipHeight, 1 }
        };

        std::array<vk::Offset3D, 2> dstOffsets = {
            vk::Offset3D{ 0, 0, 0 },
            vk::Offset3D{
                mipWidth > 1 ? mipWidth / 2 : 1,
                mipHeight > 1 ? mipHeight / 2 : 1,
                1
            }
        };

        vk::ImageBlit blit{};
        blit
            .setSrcOffsets(srcOffsets)
            .setSrcSubresource(
                vk::ImageSubresourceLayers{}
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setMipLevel(i - 1)
                .setBaseArrayLayer(0)
                .setLayerCount(1))
            .setDstOffsets(dstOffsets)
            .setDstSubresource(
                vk::ImageSubresourceLayers{}
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setMipLevel(i)
                .setBaseArrayLayer(0)
                .setLayerCount(1));

        commandBuffer.blitImage(
            *image,
            vk::ImageLayout::eTransferSrcOptimal,
            *image,
            vk::ImageLayout::eTransferDstOptimal,
            blit,
            vk::Filter::eLinear);

        pipelineImageBarrier2(
            commandBuffer,
            *image,
            vk::ImageLayout::eTransferSrcOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eTransfer,
            vk::AccessFlagBits2::eTransferRead,
            vk::PipelineStageFlagBits2::eFragmentShader,
            vk::AccessFlagBits2::eShaderSampledRead,
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(i - 1)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(1));

        if (mipWidth > 1)
        {
            mipWidth /= 2;
        }

        if (mipHeight > 1)
        {
            mipHeight /= 2;
        }
    }

    pipelineImageBarrier2(
        commandBuffer,
        *image,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::AccessFlagBits2::eShaderSampledRead,
        vk::ImageSubresourceRange{}
        .setAspectMask(vk::ImageAspectFlagBits::eColor)
        .setBaseMipLevel(mipLevels - 1)
        .setLevelCount(1)
        .setBaseArrayLayer(0)
        .setLayerCount(1));

    bufferUtils.endSingleTimeCommands(commandBuffer);
}