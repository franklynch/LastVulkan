#include "FrameResources.hpp"

#include <stdexcept>

FrameResources::FrameResources(VulkanContext& vkContext)
    : vkContext(vkContext)
{
}

void FrameResources::createCommandBuffers(uint32_t maxFramesInFlight)
{
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo
        .setCommandPool(*vkContext.getCommandPool())
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(maxFramesInFlight);

    m_commandBuffers = vkContext.getDevice().allocateCommandBuffers(allocInfo);
}

void FrameResources::createSyncObjects(
    uint32_t maxFramesInFlight,
    uint32_t swapchainImageCount)
{
    vk::SemaphoreCreateInfo semaphoreInfo{};

    vk::FenceCreateInfo fenceInfo{};
    fenceInfo.setFlags(vk::FenceCreateFlagBits::eSignaled);

    m_presentCompleteSemaphores.clear();
    m_inFlightFences.clear();

    m_presentCompleteSemaphores.reserve(maxFramesInFlight);
    m_inFlightFences.reserve(maxFramesInFlight);

    for (uint32_t i = 0; i < maxFramesInFlight; ++i)
    {
        m_presentCompleteSemaphores.emplace_back(
            vkContext.getDevice(),
            semaphoreInfo);

        m_inFlightFences.emplace_back(
            vkContext.getDevice(),
            fenceInfo);
    }

    recreateSwapchainDependent(swapchainImageCount);
}

void FrameResources::recreateSwapchainDependent(uint32_t swapchainImageCount)
{
    vk::SemaphoreCreateInfo semaphoreInfo{};

    m_renderFinishedSemaphores.clear();
    m_renderFinishedSemaphores.reserve(swapchainImageCount);

    for (uint32_t i = 0; i < swapchainImageCount; ++i)
    {
        m_renderFinishedSemaphores.emplace_back(
            vkContext.getDevice(),
            semaphoreInfo);
    }

    m_imagesInFlight.assign(swapchainImageCount, vk::Fence{});
    m_swapchainImageInitialized.assign(swapchainImageCount, false);
}

void FrameResources::cleanupSwapchainDependent()
{
    m_renderFinishedSemaphores.clear();
    m_imagesInFlight.clear();
    m_swapchainImageInitialized.clear();
}