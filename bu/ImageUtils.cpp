#include "ImageUtils.hpp"

#include <stdexcept>

ImageUtils::ImageUtils(VulkanContext& vkContext, BufferUtils& bufferUtils)
    : vkContext(vkContext)
    , bufferUtils(bufferUtils)
{
}

void ImageUtils::createImage(uint32_t width,
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
        .setMemoryTypeIndex(bufferUtils.findMemoryType(memRequirements.memoryTypeBits, properties));

    imageMemory = vk::raii::DeviceMemory(device, allocInfo);
    image.bindMemory(imageMemory, 0);
}

vk::raii::ImageView ImageUtils::createImageView(const vk::raii::Image& image,
    vk::Format format,
    vk::ImageAspectFlags aspectFlags,
    uint32_t mipLevels) const
{
    auto& device = vkContext.getDevice();

    vk::ImageSubresourceRange range{};
    range
        .setAspectMask(aspectFlags)
        .setBaseMipLevel(0)
        .setLevelCount(mipLevels)
        .setBaseArrayLayer(0)
        .setLayerCount(1);

    vk::ImageViewCreateInfo viewInfo{};
    viewInfo
        .setImage(*image)
        .setViewType(vk::ImageViewType::e2D)
        .setFormat(format)
        .setSubresourceRange(range);

    return vk::raii::ImageView(device, viewInfo);
}

void ImageUtils::transitionImageLayout(const vk::raii::Image& image,
    vk::ImageLayout oldLayout,
    vk::ImageLayout newLayout,
    uint32_t mipLevels) const
{
    auto commandBuffer = bufferUtils.beginSingleTimeCommands();

    vk::ImageSubresourceRange range{};
    range
        .setAspectMask(vk::ImageAspectFlagBits::eColor)
        .setBaseMipLevel(0)
        .setLevelCount(mipLevels)
        .setBaseArrayLayer(0)
        .setLayerCount(1);

    vk::ImageMemoryBarrier barrier{};
    barrier
        .setOldLayout(oldLayout)
        .setNewLayout(newLayout)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*image)
        .setSubresourceRange(range);

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined &&
        newLayout == vk::ImageLayout::eTransferDstOptimal)
    {
        barrier
            .setSrcAccessMask({})
            .setDstAccessMask(vk::AccessFlagBits::eTransferWrite);

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    }
    else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
        newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        barrier
            .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    }
    else
    {
        throw std::invalid_argument("unsupported layout transition!");
    }

    commandBuffer->pipelineBarrier(
        sourceStage,
        destinationStage,
        {},
        nullptr,
        nullptr,
        barrier);

    bufferUtils.endSingleTimeCommands(*commandBuffer);
}

void ImageUtils::copyBufferToImage(const vk::raii::Buffer& buffer,
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
            vk::ImageSubresourceLayers(
                vk::ImageAspectFlagBits::eColor,
                0,
                0,
                1))
        .setImageOffset(vk::Offset3D{ 0, 0, 0 })
        .setImageExtent(vk::Extent3D{ width, height, 1 });

    commandBuffer->copyBufferToImage(
        *buffer,
        *image,
        vk::ImageLayout::eTransferDstOptimal,
        region);

    bufferUtils.endSingleTimeCommands(*commandBuffer);
}