#pragma once

#include "EnvironmentResources.hpp"
#include "VulkanContext.hpp"
#include "BufferUtils.hpp"

#include <array>
#include <glm/glm.hpp>

class PrefilterRenderer
{
public:
    PrefilterRenderer(VulkanContext& vkContext, BufferUtils& bufferUtils);

    void init(EnvironmentResources& environment);

    [[nodiscard]] uint32_t getDebugRuntimePrefilteredMipLevels() const
    {
        return runtimePrefilteredMipLevels;
    }

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

    uint32_t runtimePrefilteredCubeSize = 256;
    uint32_t runtimePrefilteredMipLevels = 7;

    vk::raii::DescriptorSetLayout prefilterDescriptorSetLayout{ nullptr };
    vk::raii::DescriptorPool prefilterDescriptorPool{ nullptr };
    vk::raii::DescriptorSet prefilterDescriptorSet{ nullptr };

    vk::raii::PipelineLayout prefilterPipelineLayout{ nullptr };
    vk::raii::Pipeline prefilterPipeline{ nullptr };



};