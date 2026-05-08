#include "SwapchainManager.hpp"



SwapchainManager::SwapchainManager(Window& window, VulkanContext& vkContext)
    : window(window),
    vkContext(vkContext)
{
}

void SwapchainManager::create()
{
    createSwapchain();
    createImageViews();
}

void SwapchainManager::cleanup()
{
    swapChainImageViews.clear();
    swapChainImages.clear();
    swapChain = nullptr;
}

uint32_t SwapchainManager::chooseMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities)
{
    auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
    if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount))
    {
        minImageCount = surfaceCapabilities.maxImageCount;
    }
    return minImageCount;
}

vk::SurfaceFormatKHR SwapchainManager::chooseSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const& availableFormats)
{
    assert(!availableFormats.empty());
    const auto formatIt = std::ranges::find_if(
        availableFormats,
        [](const auto& format) { return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear; });
    return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
}

vk::PresentModeKHR SwapchainManager::choosePresentMode(std::vector<vk::PresentModeKHR> const& availablePresentModes)
{
    assert(std::ranges::any_of(availablePresentModes, [](auto presentMode) { return presentMode == vk::PresentModeKHR::eFifo; }));
    return std::ranges::any_of(availablePresentModes,
        [](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; }) ?
        vk::PresentModeKHR::eMailbox :
        vk::PresentModeKHR::eFifo;
}

vk::Extent2D SwapchainManager::chooseExtent(vk::SurfaceCapabilitiesKHR const& capabilities) const
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
    {
        return capabilities.currentExtent;
    }
    int width, height;
    window.getFramebufferSize(width, height);

    return {
        std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height) };
}

void SwapchainManager::createSwapchain()
{
    auto& physicalDevice = vkContext.getPhysicalDevice();
    auto& surface = vkContext.getSurface();
    auto& device = vkContext.getDevice();

    
    vk::SurfaceCapabilitiesKHR surfaceCapabilities =
        physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    
        swapChainExtent = chooseExtent(surfaceCapabilities);
        uint32_t minImageCount = chooseMinImageCount(surfaceCapabilities);

    std::vector<vk::SurfaceFormatKHR> availableFormats =
            physicalDevice.getSurfaceFormatsKHR(*surface);

    swapChainSurfaceFormat = chooseSurfaceFormat(availableFormats);

    std::vector<vk::PresentModeKHR> availablePresentModes =
        physicalDevice.getSurfacePresentModesKHR(*surface);

    vk::PresentModeKHR presentMode =
        choosePresentMode(availablePresentModes);

    vk::SwapchainCreateInfoKHR swapChainCreateInfo{};
    swapChainCreateInfo
        .setSurface(*surface)
        .setMinImageCount(minImageCount)
        .setImageFormat(swapChainSurfaceFormat.format)
        .setImageColorSpace(swapChainSurfaceFormat.colorSpace)
        .setImageExtent(swapChainExtent)
        .setImageArrayLayers(1)
        .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
        .setImageSharingMode(vk::SharingMode::eExclusive)
        .setPreTransform(surfaceCapabilities.currentTransform)
        .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
        .setPresentMode(presentMode)
        .setClipped(VK_TRUE);

    swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
    swapChainImages = swapChain.getImages();
    
}

void SwapchainManager::createImageViews()
{
    assert(swapChainImageViews.empty());

    auto& device = vkContext.getDevice();

    vk::ImageSubresourceRange range{};
    range
        .setAspectMask(vk::ImageAspectFlagBits::eColor)
        .setBaseMipLevel(0)
        .setLevelCount(1)
        .setBaseArrayLayer(0)
        .setLayerCount(1);

    vk::ImageViewCreateInfo imageViewCreateInfo{};
    imageViewCreateInfo
        .setViewType(vk::ImageViewType::e2D)
        .setFormat(swapChainSurfaceFormat.format)
        .setSubresourceRange(range);

    for (const auto& image : swapChainImages)
    {
        imageViewCreateInfo.setImage(image);
        swapChainImageViews.emplace_back(device, imageViewCreateInfo);
    }
}