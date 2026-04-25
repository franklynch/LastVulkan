#pragma once

#include "EnvironmentResources.hpp"
#include "VulkanContext.hpp"
#include "BufferUtils.hpp"

class BrdfLutRenderer
{
public:
    BrdfLutRenderer(VulkanContext& vkContext, BufferUtils& bufferUtils);

    void init(EnvironmentResources& environment);

private:
    void createResources(EnvironmentResources& environment);
    void createPipeline(EnvironmentResources& environment);
    void render(EnvironmentResources& environment);

private:
    VulkanContext& vkContext;
    BufferUtils& bufferUtils;
};