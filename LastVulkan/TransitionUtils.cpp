#include "TransitionUtils.hpp"

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

    void transitionToColorAttachment(
        vk::CommandBuffer cmd,
        vk::Image image,
        vk::ImageLayout oldLayout)
    {
        transitionImageLayout(
            cmd,
            image,
            oldLayout,
            vk::ImageLayout::eColorAttachmentOptimal,
            {},
            vk::AccessFlagBits::eColorAttachmentWrite,
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            vk::ImageAspectFlagBits::eColor);
    }

    void transitionToShaderReadOnly(
        vk::CommandBuffer cmd,
        vk::Image image,
        vk::ImageLayout oldLayout)
    {
        transitionImageLayout(
            cmd,
            image,
            oldLayout,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::AccessFlagBits::eColorAttachmentWrite,
            vk::AccessFlagBits::eShaderRead,
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            vk::PipelineStageFlagBits::eFragmentShader,
            vk::ImageAspectFlagBits::eColor);
    }

    void transitionToDepthAttachment(
        vk::CommandBuffer cmd,
        vk::Image image,
        vk::ImageAspectFlags aspectMask)
    {
        transitionImageLayout(
            cmd,
            image,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthAttachmentOptimal,
            {},
            vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eEarlyFragmentTests,
            aspectMask);
    }

    void transitionToPresent(
        vk::CommandBuffer cmd,
        vk::Image image)
    {
        transitionImageLayout(
            cmd,
            image,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits::eColorAttachmentWrite,
            {},
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            vk::PipelineStageFlagBits::eBottomOfPipe,
            vk::ImageAspectFlagBits::eColor);
    }
}