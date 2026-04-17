#pragma once

#include "Texture2D.hpp"

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include <glm/glm.hpp>

struct MaterialImageWrite
{
    vk::DescriptorImageInfo imageInfo;
    vk::WriteDescriptorSet write;
};

class Material
{
public:
    explicit Material(Texture2D& texture);

    const Texture2D& getTexture() const { return texture; }

    [[nodiscard]] vk::DescriptorImageInfo getImageInfo() const;
    [[nodiscard]] MaterialImageWrite makeImageWrite(
        vk::DescriptorSet dstSet,
        uint32_t binding) const;

    [[nodiscard]] const glm::vec4& getBaseColorFactor() const { return baseColorFactor; }
    void setBaseColorFactor(const glm::vec4& value) { baseColorFactor = value; }

private:
    Texture2D& texture;
    glm::vec4 baseColorFactor{ 1.0f, 1.0f, 1.0f, 1.0f };
};