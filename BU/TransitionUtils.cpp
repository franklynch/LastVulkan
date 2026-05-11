#include "TransitionUtils.hpp"

namespace TransitionUtils
{
    void transitionImage(
        vk::CommandBuffer cmd,
        vk::Image image,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        vk::PipelineStageFlags2 srcStage,
        vk::AccessFlags2 srcAccess,
        vk::PipelineStageFlags2 dstStage,
        vk::AccessFlags2 dstAccess,
        vk::ImageAspectFlags aspectMask);
    
    
    void transitionImageLayout(
        vk::CommandBuffer cmd,
        vk::Image image,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        vk::AccessFlags srcAccessMask,
        vk::AccessFlags dstAccessMask,
        vk::PipelineStageFlags srcStage,
        vk::PipelineStageFlags dstStage,
        vk::ImageAspectFlags aspectMask)
    {
        vk::ImageMemoryBarrier barrier{};
        barrier
            .setOldLayout(oldLayout)
            .setNewLayout(newLayout)
            .setSrcAccessMask(srcAccessMask)
            .setDstAccessMask(dstAccessMask)
            .setImage(image)
            .setSubresourceRange(
                vk::ImageSubresourceRange{}
                .setAspectMask(aspectMask)
                .setBaseMipLevel(0)
                .setLevelCount(1)
                .setBaseArrayLayer(0)
                .setLayerCount(1));

        cmd.pipelineBarrier(
            srcStage,
            dstStage,
            {},
            nullptr,
            nullptr,
            barrier);
    }

    void TransitionUtils::transitionToColorAttachment(
        vk::CommandBuffer cmd,
        vk::Image image,
        vk::ImageLayout oldLayout)
    {
        transitionImage(
            cmd,
            image,
            oldLayout,
            vk::ImageLayout::eColorAttachmentOptimal,
            oldLayout == vk::ImageLayout::eUndefined
            ? vk::PipelineStageFlagBits2::eNone
            : vk::PipelineStageFlagBits2::eAllCommands,
            oldLayout == vk::ImageLayout::eUndefined
            ? vk::AccessFlagBits2::eNone
            : vk::AccessFlagBits2::eMemoryWrite,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::ImageAspectFlagBits::eColor);
    }


    void TransitionUtils::transitionToShaderReadOnly(
        vk::CommandBuffer cmd,
        vk::Image image,
        vk::ImageLayout oldLayout)
    {
        transitionImage(
            cmd,
            image,
            oldLayout,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eFragmentShader,
            vk::AccessFlagBits2::eShaderSampledRead,
            vk::ImageAspectFlagBits::eColor);
    }

    void TransitionUtils::transitionToDepthAttachment(
        vk::CommandBuffer cmd,
        vk::Image image,
        vk::ImageAspectFlags aspectMask)
    {
        transitionImage(
            cmd,
            image,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthAttachmentOptimal,
            vk::PipelineStageFlagBits2::eNone,
            vk::AccessFlagBits2::eNone,
            vk::PipelineStageFlagBits2::eEarlyFragmentTests |
            vk::PipelineStageFlagBits2::eLateFragmentTests,
            vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            aspectMask);
    }

    void TransitionUtils::transitionToPresent(
        vk::CommandBuffer cmd,
        vk::Image image)
    {
        transitionImage(
            cmd,
            image,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eNone,
            vk::AccessFlagBits2::eNone,
            vk::ImageAspectFlagBits::eColor);
    }

    void TransitionUtils::transitionImage(
        vk::CommandBuffer cmd,
        vk::Image image,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        vk::PipelineStageFlags2 srcStage,
        vk::AccessFlags2 srcAccess,
        vk::PipelineStageFlags2 dstStage,
        vk::AccessFlags2 dstAccess,
        vk::ImageAspectFlags aspectMask)
    {
        vk::ImageMemoryBarrier2 barrier{};
        barrier
            .setSrcStageMask(srcStage)
            .setSrcAccessMask(srcAccess)
            .setDstStageMask(dstStage)
            .setDstAccessMask(dstAccess)
            .setOldLayout(oldLayout)
            .setNewLayout(newLayout)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(image)
            .setSubresourceRange(
                vk::ImageSubresourceRange{}
                .setAspectMask(aspectMask)
                .setBaseMipLevel(0)
                .setLevelCount(1)
                .setBaseArrayLayer(0)
                .setLayerCount(1));

        vk::DependencyInfo dependencyInfo{};
        dependencyInfo.setImageMemoryBarriers(barrier);

        cmd.pipelineBarrier2(dependencyInfo);
    }
}