#pragma once

#include "EnvironmentResources.hpp"
#include "VulkanContext.hpp"
#include "BufferUtils.hpp"

#include <glm/glm.hpp>
#include <array>

class IrradianceRenderer
{
public:
    IrradianceRenderer(VulkanContext& vkContext, BufferUtils& bufferUtils);

    void init(EnvironmentResources& environment);

private:
    void createResources(EnvironmentResources& environment);
    void createFaceViews(EnvironmentResources& environment);

    void createDescriptorResources();
    void updateDescriptorSet(EnvironmentResources& environment);
    void createPipeline();
    void render(EnvironmentResources& environment);

private:
    VulkanContext& vkContext;
    BufferUtils& bufferUtils;

    uint32_t runtimeIrradianceCubeSize = 64;

    vk::raii::DescriptorSetLayout irradianceDescriptorSetLayout{ nullptr };
    vk::raii::DescriptorPool irradianceDescriptorPool{ nullptr };
    vk::raii::DescriptorSet irradianceDescriptorSet{ nullptr };

    vk::raii::PipelineLayout irradiancePipelineLayout{ nullptr };
    vk::raii::Pipeline irradiancePipeline{ nullptr };


};