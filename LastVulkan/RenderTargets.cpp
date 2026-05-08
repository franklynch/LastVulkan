#include "RenderTargets.hpp"


#include <stdexcept>

RenderTargets::RenderTargets(VulkanContext& vkContext, ImageUtils& imageUtils)
    : vkContext(vkContext),
    imageUtils(imageUtils)
{
}

void RenderTargets::create(
    vk::Extent2D extent,
    vk::Format colorFormat)
{
    m_depthFormat = findDepthFormat();

    m_depthAspect = hasStencilComponent(m_depthFormat)
        ? (vk::ImageAspectFlagBits::eDepth |
            vk::ImageAspectFlagBits::eStencil)
        : vk::ImageAspectFlagBits::eDepth;

    createColorResources(extent, colorFormat);
    createDepthResources(extent);
}

void RenderTargets::cleanup()
{
    m_colorImageView = nullptr;
    m_colorImage = nullptr;
    m_colorImageMemory = nullptr;

    m_depthImageView = nullptr;
    m_depthImage = nullptr;
    m_depthImageMemory = nullptr;
}

void RenderTargets::createColorResources(vk::Extent2D extent, vk::Format colorFormat)
{
    

    imageUtils.createImage(
        extent.width,
        extent.height,
        1,
        vkContext.getMsaaSamples(),
        colorFormat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransientAttachment |
        vk::ImageUsageFlagBits::eColorAttachment,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        m_colorImage,
        m_colorImageMemory
    );

    m_colorImageView = imageUtils.createImageView(m_colorImage, colorFormat, vk::ImageAspectFlagBits::eColor, 1);
}

void RenderTargets::createDepthResources(vk::Extent2D extent)
{
    imageUtils.createImage(
        extent.width,
        extent.height,
        1,
        vkContext.getMsaaSamples(),
        m_depthFormat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        m_depthImage,
        m_depthImageMemory
    );

    m_depthImageView = imageUtils.createImageView(
        m_depthImage,
        m_depthFormat,
        m_depthAspect,
        1
    );
}

vk::Format RenderTargets::findDepthFormat() const
{
    return findSupportedFormat(
        {
            vk::Format::eD32Sfloat,
            vk::Format::eD32SfloatS8Uint,
            vk::Format::eD24UnormS8Uint
        },
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment);
}

bool RenderTargets::hasStencilComponent(vk::Format format)
{
    return format == vk::Format::eD32SfloatS8Uint ||
        format == vk::Format::eD24UnormS8Uint;
}

vk::Format RenderTargets::findSupportedFormat(
    const std::vector<vk::Format>& candidates,
    vk::ImageTiling tiling,
    vk::FormatFeatureFlags features) const
{
    auto& physicalDevice = vkContext.getPhysicalDevice();

    auto formatIt = std::ranges::find_if(candidates, [&](auto const format) {
        vk::FormatProperties props = physicalDevice.getFormatProperties(format);

        return (((tiling == vk::ImageTiling::eLinear) &&
            ((props.linearTilingFeatures & features) == features)) ||
            ((tiling == vk::ImageTiling::eOptimal) &&
                ((props.optimalTilingFeatures & features) == features)));
        });

    if (formatIt == candidates.end())
    {
        throw std::runtime_error("failed to find supported format!");
    }

    return *formatIt;
}
