#pragma once

#include "EnvironmentResources.hpp"
#include "VulkanContext.hpp"
#include "BufferUtils.hpp"

#include <glm/glm.hpp>
#include <array>

class EnvironmentRenderer
{
public:
    EnvironmentRenderer(VulkanContext& vkContext, BufferUtils& bufferUtils);

    void init(
        EnvironmentResources& environment,
        const vk::raii::Sampler& hdrSampler,
        const vk::raii::ImageView& hdrView);

private:
    void createResources(EnvironmentResources& environment);
    void createFaceViews(EnvironmentResources& environment);
    void createDescriptorResources();
    void updateDescriptorSet(EnvironmentResources& environment, const vk::raii::Sampler& hdrSampler, const vk::raii::ImageView& hdrView);

    void createPipeline();
    void render(EnvironmentResources& environment);

private:
    VulkanContext& vkContext;
    BufferUtils& bufferUtils;

    uint32_t runtimeEnvironmentCubeSize = 512;

    vk::raii::DescriptorSetLayout equirectToCubeDescriptorSetLayout{ nullptr };
    vk::raii::DescriptorPool equirectToCubeDescriptorPool{ nullptr };
    vk::raii::DescriptorSet equirectToCubeDescriptorSet{ nullptr };

    vk::raii::PipelineLayout equirectToCubePipelineLayout{ nullptr };
    vk::raii::Pipeline equirectToCubePipeline{ nullptr };

  

};