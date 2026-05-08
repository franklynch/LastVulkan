#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#   include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include "VulkanContext.hpp"
#include "RenderTargets.hpp"
#include "PostProcessRenderer.hpp"

class ScenePipelines
{
public:
    ScenePipelines(VulkanContext& vkContext);

    void create(
        vk::Extent2D extent,
        vk::Format hdrColorFormat,
        vk::Format depthFormat,
        vk::DescriptorSetLayout frameLayout,
        vk::DescriptorSetLayout materialLayout,
        vk::DescriptorSetLayout iblLayout,
        bool wireframeSupported);

    void createSkybox(
        vk::Extent2D extent,
        vk::Format hdrColorFormat,
        vk::Format depthFormat,
        vk::DescriptorSetLayout frameLayout,
        vk::DescriptorSetLayout iblLayout);

    void cleanup();

    vk::PipelineLayout layout() const { return *m_pipelineLayout; }
    vk::PipelineLayout skyboxLayout() const { return *m_skyboxPipelineLayout; }

    vk::Pipeline solid(bool doubleSided) const
    {
        return doubleSided ? *m_solidDoubleSidedPipeline : *m_solidPipeline;
    }

    vk::Pipeline transparent(bool doubleSided) const
    {
        return doubleSided ? *m_transparentDoubleSidedPipeline : *m_transparentPipeline;
    }

    vk::Pipeline wireframe(bool doubleSided) const
    {
        if (doubleSided && m_wireframeDoubleSidedPipeline != nullptr)
        {
            return *m_wireframeDoubleSidedPipeline;
        }

        if (!doubleSided && m_wireframePipeline != nullptr)
        {
            return *m_wireframePipeline;
        }

        return solid(doubleSided);
    }

    vk::Pipeline skybox() const { return *m_skyboxPipeline; }

   

private:
    VulkanContext& vkContext;

    vk::raii::PipelineLayout m_pipelineLayout = nullptr;
    vk::raii::PipelineLayout m_skyboxPipelineLayout = nullptr;

    vk::raii::Pipeline m_solidPipeline = nullptr;
    vk::raii::Pipeline m_solidDoubleSidedPipeline = nullptr;

    vk::raii::Pipeline m_wireframePipeline = nullptr;
    vk::raii::Pipeline m_wireframeDoubleSidedPipeline = nullptr;

    vk::raii::Pipeline m_transparentPipeline = nullptr;
    vk::raii::Pipeline m_transparentDoubleSidedPipeline = nullptr;

    vk::raii::Pipeline m_skyboxPipeline = nullptr;
};