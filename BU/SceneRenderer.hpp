#pragma once

#include <memory>
#include <vector>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#   include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include "VulkanContext.hpp"
#include "Scene.hpp"
#include "Camera.hpp"
#include "Material.hpp"

class SceneRenderer
{
public:
    struct SceneRenderContext
    {
        vk::PipelineLayout pipelineLayout{};

        vk::Pipeline solidPipeline{};
        vk::Pipeline solidDoubleSidedPipeline{};
        vk::Pipeline wireframePipeline{};
        vk::Pipeline wireframeDoubleSidedPipeline{};

        vk::DescriptorSet frameDescriptorSet{};
        vk::DescriptorSet iblDescriptorSet{};

        const std::vector<vk::raii::DescriptorSet>* materialDescriptorSets = nullptr;

        bool wireframeEnabled = false;
        bool animateModel = false;
        float currentAnimationAngle = 0.0f;

        vk::Pipeline transparentPipeline{};
        vk::Pipeline transparentDoubleSidedPipeline{};

        vk::Pipeline skyboxPipeline{};
        vk::PipelineLayout skyboxPipelineLayout{};

        



    };

    explicit SceneRenderer(VulkanContext& vkContext);

    void renderOpaque(
        vk::raii::CommandBuffer& commandBuffer,
        const Scene& scene,
        const SceneRenderContext& context);

    void renderTransparent(
        vk::raii::CommandBuffer& commandBuffer,
        const Scene& scene,
        const Camera& camera,
        const SceneRenderContext& context);

    void renderSkybox(
        vk::raii::CommandBuffer& commandBuffer,
        const SceneRenderContext& context);

private:
    VulkanContext& vkContext;
};