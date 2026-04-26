#pragma once

#include "EnvironmentResources.hpp"
#include "VulkanContext.hpp"
#include "BufferUtils.hpp"

#include <array>
#include <glm/glm.hpp>

void createCubemapResource(
    VulkanContext& vkContext,
    BufferUtils& bufferUtils,
    Cubemap& cubemap,
    uint32_t size,
    uint32_t mipLevels,
    vk::Format format,
    vk::ImageUsageFlags usage);

std::array<glm::mat4, 6> getCubemapCaptureViews();
glm::mat4 getCubemapCaptureProjection();