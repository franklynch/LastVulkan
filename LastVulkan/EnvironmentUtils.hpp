#pragma once

#include "EnvironmentResources.hpp"
#include "VulkanContext.hpp"
#include "BufferUtils.hpp"

void createCubemapResource(
    VulkanContext& vkContext,
    BufferUtils& bufferUtils,
    Cubemap& cubemap,
    uint32_t size,
    uint32_t mipLevels,
    vk::Format format,
    vk::ImageUsageFlags usage);