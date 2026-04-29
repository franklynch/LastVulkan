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


    explicit Material(
        Texture2D& baseColorTexture,
        Texture2D* defaultNormal,
        Texture2D* defaultMetallicRoughness
    );

    const Texture2D& getTexture() const { return texture; }

    const Texture2D* getNormalTexture() const { return normalTexture; }
    void setNormalTexture(Texture2D* texture, bool provided = true);

    const Texture2D* getMetallicRoughnessTexture() const { return metallicRoughnessTexture; }
    void setMetallicRoughnessTexture(Texture2D* texture, bool provided = true);

    float getNormalScale() const { return normalScale; }
    void setNormalScale(float value) { normalScale = value; }

    bool hasNormalTexture() const { return normalTexture != nullptr; }
    bool hasMetallicRoughnessTexture() const { return metallicRoughnessTexture != nullptr; }



    [[nodiscard]] vk::DescriptorImageInfo getImageInfo() const;
    [[nodiscard]] MaterialImageWrite makeImageWrite(
        vk::DescriptorSet dstSet,
        uint32_t binding) const;

    [[nodiscard]] vk::DescriptorImageInfo getNormalImageInfo() const;
    [[nodiscard]] MaterialImageWrite makeNormalImageWrite(
        vk::DescriptorSet dstSet,
        uint32_t binding) const;

    [[nodiscard]] vk::DescriptorImageInfo getMetallicRoughnessImageInfo() const;
    [[nodiscard]] MaterialImageWrite makeMetallicRoughnessImageWrite(
        vk::DescriptorSet dstSet,
        uint32_t binding) const;

    [[nodiscard]] const glm::vec4& getBaseColorFactor() const { return baseColorFactor; }
    void setBaseColorFactor(const glm::vec4& value) { baseColorFactor = value; }

    const std::string& getName() const { return name; }
    void setName(const std::string& value) { name = value; }

    bool isDoubleSided() const { return doubleSided; }
    void setDoubleSided(bool value) { doubleSided = value; }

    float getMetallicFactor() const { return metallicFactor; }
    void setMetallicFactor(float value) { metallicFactor = value; }

    float getRoughnessFactor() const { return roughnessFactor; }
    void setRoughnessFactor(float value) { roughnessFactor = value; }

    const std::string& getAlphaMode() const { return alphaMode; }
    void setAlphaMode(const std::string& value) { alphaMode = value; }

    float getAlphaCutoff() const { return alphaCutoff; }
    void setAlphaCutoff(float value) { alphaCutoff = value; }

    bool hasProvidedNormalTexture() const { return normalTextureProvided; }
    bool hasProvidedMetallicRoughnessTexture() const { return metallicRoughnessTextureProvided; }

    void setOcclusionTexture(Texture2D* texture, bool hasRealTexture)
    {
        occlusionTexture = texture;
        hasOcclusionTexture = hasRealTexture;
    }

    const Texture2D* getOcclusionTexture() const
    {
        return occlusionTexture;
    }

    bool hasRealOcclusionTexture() const
    {
        return hasOcclusionTexture;
    }

    void setOcclusionStrength(float strength)
    {
        occlusionStrength = strength;
    }

    float getOcclusionStrength() const
    {
        return occlusionStrength;
    }

    MaterialImageWrite makeOcclusionImageWrite(
        vk::DescriptorSet descriptorSet,
        uint32_t binding) const;

    void setEmissiveTexture(Texture2D* texture, bool hasRealTexture)
    {
        emissiveTexture = texture;
        hasEmissiveTexture = hasRealTexture;
    }

    const Texture2D* getEmissiveTexture() const
    {
        return emissiveTexture;
    }

    bool hasRealEmissiveTexture() const
    {
        return hasEmissiveTexture;
    }

    void setEmissiveFactor(const glm::vec3& factor)
    {
        emissiveFactor = factor;
    }

    const glm::vec3& getEmissiveFactor() const
    {
        return emissiveFactor;
    }

private:
    Texture2D& texture;
    Texture2D* normalTexture = nullptr;
    Texture2D* metallicRoughnessTexture = nullptr;

    glm::vec4 baseColorFactor{ 1.0f, 1.0f, 1.0f, 1.0f };
    std::string name;
    bool doubleSided = false;

    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    float normalScale = 1.0f;

    std::string alphaMode = "OPAQUE";
    float alphaCutoff = 0.5f;

    bool normalTextureProvided = false;
    bool metallicRoughnessTextureProvided = false;

    Texture2D* occlusionTexture = nullptr;
    bool hasOcclusionTexture = false;
    float occlusionStrength = 1.0f;

    Texture2D* emissiveTexture = nullptr;
    bool hasEmissiveTexture = false;
    glm::vec3 emissiveFactor{ 0.0f };
};