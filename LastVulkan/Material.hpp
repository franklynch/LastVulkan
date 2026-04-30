#pragma once

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <string>

class Texture2D;

struct MaterialImageWrite
{
    vk::DescriptorImageInfo imageInfo{};
    vk::WriteDescriptorSet write{};

    MaterialImageWrite() = default;

    MaterialImageWrite(const MaterialImageWrite& other)
        : imageInfo(other.imageInfo)
        , write(other.write)
    {
        write.setImageInfo(imageInfo);
    }

    MaterialImageWrite& operator=(const MaterialImageWrite& other)
    {
        if (this != &other)
        {
            imageInfo = other.imageInfo;
            write = other.write;
            write.setImageInfo(imageInfo);
        }

        return *this;
    }

    MaterialImageWrite(MaterialImageWrite&& other) noexcept
        : imageInfo(other.imageInfo)
        , write(other.write)
    {
        write.setImageInfo(imageInfo);
    }

    MaterialImageWrite& operator=(MaterialImageWrite&& other) noexcept
    {
        if (this != &other)
        {
            imageInfo = other.imageInfo;
            write = other.write;
            write.setImageInfo(imageInfo);
        }

        return *this;
    }
};

class Material
{
public:
    Material(
        Texture2D& baseColor,
        Texture2D* normal,
        Texture2D* metallicRoughness);

    void setName(const std::string& value);
    const std::string& getName() const;

    void setDoubleSided(bool value);
    bool isDoubleSided() const;

    void setAlphaMode(const std::string& value);
    const std::string& getAlphaMode() const;

    void setAlphaCutoff(float value);
    float getAlphaCutoff() const;

    void setBaseColorFactor(const glm::vec4& value);
    const glm::vec4& getBaseColorFactor() const;

    void setMetallicFactor(float value);
    float getMetallicFactor() const;

    void setRoughnessFactor(float value);
    float getRoughnessFactor() const;

    void setNormalScale(float value);
    float getNormalScale() const;

    void setOcclusionStrength(float value);
    float getOcclusionStrength() const;

    void setEmissiveFactor(const glm::vec3& value);
    const glm::vec3& getEmissiveFactor() const;

    void setNormalTexture(Texture2D* texture, bool hasRealTexture);
    void setMetallicRoughnessTexture(Texture2D* texture, bool hasRealTexture);
    void setOcclusionTexture(Texture2D* texture, bool hasRealTexture);
    void setEmissiveTexture(Texture2D* texture, bool hasRealTexture);

    Texture2D* getBaseColorTexture() const;
    Texture2D* getNormalTexture() const;
    Texture2D* getMetallicRoughnessTexture() const;
    Texture2D* getOcclusionTexture() const;
    Texture2D* getEmissiveTexture() const;

    Texture2D& getTexture() const;

    bool hasRealNormalTexture() const;
    bool hasRealMetallicRoughnessTexture() const;
    bool hasRealOcclusionTexture() const;
    bool hasRealEmissiveTexture() const;

    vk::DescriptorImageInfo getImageInfo() const;
    vk::DescriptorImageInfo getNormalImageInfo() const;
    vk::DescriptorImageInfo getMetallicRoughnessImageInfo() const;
    vk::DescriptorImageInfo getOcclusionImageInfo() const;
    vk::DescriptorImageInfo getEmissiveImageInfo() const;

    MaterialImageWrite makeImageWrite(
        vk::DescriptorSet descriptorSet,
        uint32_t binding) const;

    MaterialImageWrite makeNormalImageWrite(
        vk::DescriptorSet descriptorSet,
        uint32_t binding) const;

    MaterialImageWrite makeMetallicRoughnessImageWrite(
        vk::DescriptorSet descriptorSet,
        uint32_t binding) const;

    MaterialImageWrite makeOcclusionImageWrite(
        vk::DescriptorSet descriptorSet,
        uint32_t binding) const;

    MaterialImageWrite makeEmissiveImageWrite(
        vk::DescriptorSet descriptorSet,
        uint32_t binding) const;

private:
    vk::DescriptorImageInfo makeDescriptorImageInfo(
        const Texture2D* texture) const;

    MaterialImageWrite makeImageWriteForTexture(
        vk::DescriptorSet descriptorSet,
        uint32_t binding,
        const Texture2D* texture) const;

private:
    Texture2D* baseColorTexture = nullptr;
    Texture2D* normalTexture = nullptr;
    Texture2D* metallicRoughnessTexture = nullptr;
    Texture2D* occlusionTexture = nullptr;
    Texture2D* emissiveTexture = nullptr;

    bool normalTextureProvided = false;
    bool metallicRoughnessTextureProvided = false;
    bool occlusionTextureProvided = false;
    bool emissiveTextureProvided = false;

    glm::vec4 baseColorFactor{ 1.0f };
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    float normalScale = 1.0f;
    float occlusionStrength = 1.0f;
    glm::vec3 emissiveFactor{ 0.0f };

    std::string name;
    std::string alphaMode = "OPAQUE";
    float alphaCutoff = 0.5f;
    bool doubleSided = false;
};