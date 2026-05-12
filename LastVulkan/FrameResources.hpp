#pragma once

#include <vector>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#   include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include "VulkanContext.hpp"

class FrameResources
{
public:
    FrameResources(VulkanContext& vkContext);

    void cleanup();

    void createCommandBuffers(uint32_t maxFramesInFlight);
    void createSyncObjects(uint32_t maxFramesInFlight, uint32_t swapchainImageCount);

    void recreateSwapchainDependent(uint32_t swapchainImageCount);
    void cleanupSwapchainDependent();

    uint32_t currentFrameIndex() const { return m_frameIndex; }
    void advanceFrame(uint32_t maxFramesInFlight)
    {
        m_frameIndex = (m_frameIndex + 1) % maxFramesInFlight;
    }

    vk::raii::CommandBuffer& commandBuffer(uint32_t frameIndex)
    {
        return m_commandBuffers[frameIndex];
    }

    vk::raii::Semaphore& presentCompleteSemaphore(uint32_t frameIndex)
    {
        return m_presentCompleteSemaphores[frameIndex];
    }

    vk::raii::Semaphore& renderFinishedSemaphore(uint32_t imageIndex)
    {
        return m_renderFinishedSemaphores[imageIndex];
    }

    vk::raii::Fence& inFlightFence(uint32_t frameIndex)
    {
        return m_inFlightFences[frameIndex];
    }

    vk::Fence& imageInFlight(uint32_t imageIndex)
    {
        return m_imagesInFlight[imageIndex];
    }

    bool imageInitialized(uint32_t imageIndex) const
    {
        return m_swapchainImageInitialized[imageIndex];
    }

    void markImageInitialized(uint32_t imageIndex)
    {
        m_swapchainImageInitialized[imageIndex] = true;
    }

    uint32_t imageStateCount() const
    {
        return static_cast<uint32_t>(m_swapchainImageInitialized.size());
    }

private:
    VulkanContext& vkContext;

    std::vector<vk::raii::CommandBuffer> m_commandBuffers;

    std::vector<vk::raii::Semaphore> m_presentCompleteSemaphores;
    std::vector<vk::raii::Semaphore> m_renderFinishedSemaphores;
    std::vector<vk::raii::Fence> m_inFlightFences;

    std::vector<vk::Fence> m_imagesInFlight;
    std::vector<bool> m_swapchainImageInitialized;

    uint32_t m_frameIndex = 0;
};