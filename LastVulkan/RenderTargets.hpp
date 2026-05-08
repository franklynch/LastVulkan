#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#   include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include "VulkanContext.hpp"
#include "ImageUtils.hpp"

class RenderTargets
{
public:
    RenderTargets(VulkanContext& vkContext, ImageUtils& imageUtils);

    void create(vk::Extent2D extent, vk::Format colorFormat);

    void cleanup();

    vk::Format depthFormat() const { return m_depthFormat; }
    vk::ImageAspectFlags depthAspect() const { return m_depthAspect; }

    const vk::raii::ImageView& colorView() const { return m_colorImageView; }
    const vk::raii::ImageView& depthView() const { return m_depthImageView; }

    const vk::raii::Image& colorImage() const { return m_colorImage; }
    const vk::raii::Image& depthImage() const { return m_depthImage; }

private:
    void createColorResources(vk::Extent2D extent, vk::Format colorFormat);
    void createDepthResources(vk::Extent2D extent);

    vk::Format findDepthFormat() const;
    static bool hasStencilComponent(vk::Format format);

private:
    VulkanContext& vkContext;
    ImageUtils& imageUtils;

    vk::raii::Image m_colorImage = nullptr;
    vk::raii::DeviceMemory m_colorImageMemory = nullptr;
    vk::raii::ImageView m_colorImageView = nullptr;

    vk::raii::Image m_depthImage = nullptr;
    vk::raii::DeviceMemory m_depthImageMemory = nullptr;
    vk::raii::ImageView m_depthImageView = nullptr;

    vk::Format m_depthFormat = vk::Format::eUndefined;
    vk::ImageAspectFlags m_depthAspect = vk::ImageAspectFlagBits::eDepth;

    vk::Format findSupportedFormat(
        const std::vector<vk::Format>& candidates,
        vk::ImageTiling tiling,
        vk::FormatFeatureFlags features) const;
};