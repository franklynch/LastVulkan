#include "BufferUtils.hpp"

#include <stdexcept>
#include <utility>

BufferUtils::BufferUtils(VulkanContext& vkContext)
    : vkContext(vkContext)
{
}

uint32_t BufferUtils::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const
{
    vk::PhysicalDeviceMemoryProperties memProperties =
        vkContext.getPhysicalDevice().getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void BufferUtils::createBuffer(
    vk::DeviceSize size,
    vk::BufferUsageFlags usage,
    vk::MemoryPropertyFlags properties,
    GpuBuffer& outBuffer)
{
    outBuffer.buffer = VK_NULL_HANDLE;
    outBuffer.allocation = VK_NULL_HANDLE;
    outBuffer.mapped = nullptr;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = static_cast<VkDeviceSize>(size);
    bufferInfo.usage = static_cast<VkBufferUsageFlags>(usage);
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

    if (properties & vk::MemoryPropertyFlagBits::eHostVisible)
    {
        allocInfo.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }
    else
    {
        allocInfo.preferredFlags =
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }

    VmaAllocationInfo allocationInfo{};

    VkResult result = vmaCreateBuffer(
        vkContext.getAllocator(),
        &bufferInfo,
        &allocInfo,
        &outBuffer.buffer,
        &outBuffer.allocation,
        &allocationInfo);

    if (result != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create VMA buffer");
    }

    outBuffer.mapped = allocationInfo.pMappedData;
}

void BufferUtils::destroyBuffer(GpuBuffer& buffer)
{
    if (buffer.buffer != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(
            vkContext.getAllocator(),
            buffer.buffer,
            buffer.allocation);

        buffer.buffer = VK_NULL_HANDLE;
        buffer.allocation = VK_NULL_HANDLE;
        buffer.mapped = nullptr;
    }
}

vk::raii::CommandBuffer BufferUtils::beginSingleTimeCommands() const
{
    auto& device = vkContext.getDevice();

    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo
        .setCommandPool(*vkContext.getCommandPool())
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(1);

    vk::raii::CommandBuffers commandBuffers(device, allocInfo);
    vk::raii::CommandBuffer commandBuffer = std::move(commandBuffers.front());

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    commandBuffer.begin(beginInfo);
    return commandBuffer;
}

void BufferUtils::endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer) const
{
    auto& queue = vkContext.getQueue();

    commandBuffer.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(*commandBuffer);

    queue.submit(submitInfo, nullptr);
    queue.waitIdle();
}

void BufferUtils::copyBuffer(VkBuffer srcBuffer,
    VkBuffer dstBuffer,
    vk::DeviceSize size)
{
    auto commandBuffer = beginSingleTimeCommands();

    vk::BufferCopy copyRegion{};
    copyRegion
        .setSrcOffset(0)
        .setDstOffset(0)
        .setSize(size);

    commandBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);

    endSingleTimeCommands(commandBuffer);
}