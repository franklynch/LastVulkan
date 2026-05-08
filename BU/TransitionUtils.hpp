#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#   include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

namespace TransitionUtils
{
    void transitionImageLayout(
        vk::CommandBuffer cmd,
        vk::Image image,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        vk::AccessFlags srcAccessMask,
        vk::AccessFlags dstAccessMask,
        vk::PipelineStageFlags srcStage,
        vk::PipelineStageFlags dstStage,
        vk::ImageAspectFlags aspectMask);

    void transitionToColorAttachment(
        vk::CommandBuffer cmd,
        vk::Image image,
        vk::ImageLayout oldLayout);

    void transitionToShaderReadOnly(
        vk::CommandBuffer cmd,
        vk::Image image,
        vk::ImageLayout oldLayout);

    void transitionToDepthAttachment(
        vk::CommandBuffer cmd,
        vk::Image image,
        vk::ImageAspectFlags aspectMask);

    void transitionToPresent(
        vk::CommandBuffer cmd,
        vk::Image image);
}