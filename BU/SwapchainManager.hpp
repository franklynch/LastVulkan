#pragma once

#include <vector>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#   include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include "Window.hpp"
#include "VulkanContext.hpp"

class SwapchainManager
{
public:
    SwapchainManager(Window& window, VulkanContext& vkContext);

    void create();
    void cleanup();

    vk::SwapchainKHR get() const { return *swapChain; }

    const vk::raii::SwapchainKHR& handle() const { return swapChain; }
    const std::vector<vk::Image>& images() const { return swapChainImages; }
    const std::vector<vk::raii::ImageView>& imageViews() const { return swapChainImageViews; }

    vk::SurfaceFormatKHR surfaceFormat() const { return swapChainSurfaceFormat; }
    vk::Format format() const { return swapChainSurfaceFormat.format; }
    vk::Extent2D extent() const { return swapChainExtent; }

    uint32_t imageCount() const
    {
        return static_cast<uint32_t>(swapChainImages.size());
    }

private:
    static uint32_t chooseMinImageCount(
        vk::SurfaceCapabilitiesKHR const& surfaceCapabilities);

    static vk::SurfaceFormatKHR chooseSurfaceFormat(
        std::vector<vk::SurfaceFormatKHR> const& availableFormats);

    static vk::PresentModeKHR choosePresentMode(
        std::vector<vk::PresentModeKHR> const& availablePresentModes);

    vk::Extent2D chooseExtent(
        vk::SurfaceCapabilitiesKHR const& capabilities) const;

    void createSwapchain();
    void createImageViews();

private:
    Window& window;
    VulkanContext& vkContext;

    vk::raii::SwapchainKHR swapChain = nullptr;
    std::vector<vk::Image> swapChainImages;
    vk::SurfaceFormatKHR swapChainSurfaceFormat{};
    vk::Extent2D swapChainExtent{};
    std::vector<vk::raii::ImageView> swapChainImageViews;
};