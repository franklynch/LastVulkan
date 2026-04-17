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

void BufferUtils::createBuffer(vk::DeviceSize size,
    vk::BufferUsageFlags usage,
    vk::MemoryPropertyFlags properties,
    vk::raii::Buffer& buffer,
    vk::raii::DeviceMemory& bufferMemory) const
{
    auto& device = vkContext.getDevice();

    vk::BufferCreateInfo bufferInfo{};
    bufferInfo
        .setSize(size)
        .setUsage(usage)
        .setSharingMode(vk::SharingMode::eExclusive);

    buffer = vk::raii::Buffer(device, bufferInfo);

    vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo
        .setAllocationSize(memRequirements.size)
        .setMemoryTypeIndex(findMemoryType(memRequirements.memoryTypeBits, properties));

    bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
    buffer.bindMemory(bufferMemory, 0);
}

std::unique_ptr<vk::raii::CommandBuffer> BufferUtils::beginSingleTimeCommands() const
{
    auto& device = vkContext.getDevice();

    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo
        .setCommandPool(*vkContext.getCommandPool())
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(1);

    auto commandBuffers = vk::raii::CommandBuffers(device, allocInfo);

    std::unique_ptr<vk::raii::CommandBuffer> commandBuffer =
        std::make_unique<vk::raii::CommandBuffer>(std::move(commandBuffers.front()));

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    commandBuffer->begin(beginInfo);
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

void BufferUtils::copyBuffer(vk::raii::Buffer& srcBuffer,
    vk::raii::Buffer& dstBuffer,
    vk::DeviceSize size) const
{
    auto commandBuffer = beginSingleTimeCommands();

    vk::BufferCopy copyRegion{};
    copyRegion
        .setSrcOffset(0)
        .setDstOffset(0)
        .setSize(size);

    commandBuffer->copyBuffer(*srcBuffer, *dstBuffer, copyRegion);

    endSingleTimeCommands(*commandBuffer);
}