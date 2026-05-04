#include "Material.hpp"

#include "Texture2D.hpp"

#include <stdexcept>

Material::Material(
    Texture2D& baseColor,
    Texture2D* normal,
    Texture2D* metallicRoughness)
    : baseColorTexture(&baseColor)
    , normalTexture(normal)
    , metallicRoughnessTexture(metallicRoughness)
{
}

void Material::setName(const std::string& value)
{
    name = value;
}

const std::string& Material::getName() const
{
    return name;
}

void Material::setDoubleSided(bool value)
{
    doubleSided = value;
}

bool Material::isDoubleSided() const
{
    return doubleSided;
}

void Material::setAlphaMode(const std::string& value)
{
    alphaMode = value;
}

const std::string& Material::getAlphaMode() const
{
    return alphaMode;
}

void Material::setAlphaCutoff(float value)
{
    alphaCutoff = value;
}

float Material::getAlphaCutoff() const
{
    return alphaCutoff;
}

void Material::setBaseColorFactor(const glm::vec4& value)
{
    baseColorFactor = value;
}

const glm::vec4& Material::getBaseColorFactor() const
{
    return baseColorFactor;
}

void Material::setMetallicFactor(float value)
{
    metallicFactor = value;
}

float Material::getMetallicFactor() const
{
    return metallicFactor;
}

void Material::setRoughnessFactor(float value)
{
    roughnessFactor = value;
}

float Material::getRoughnessFactor() const
{
    return roughnessFactor;
}

void Material::setNormalScale(float value)
{
    normalScale = value;
}

float Material::getNormalScale() const
{
    return normalScale;
}

void Material::setOcclusionStrength(float value)
{
    occlusionStrength = value;
}

float Material::getOcclusionStrength() const
{
    return occlusionStrength;
}

void Material::setEmissiveFactor(const glm::vec3& value)
{
    emissiveFactor = value;
}

const glm::vec3& Material::getEmissiveFactor() const
{
    return emissiveFactor;
}

void Material::setNormalTexture(Texture2D* texture, bool hasRealTexture)
{
    normalTexture = texture;
    normalTextureProvided = hasRealTexture;
}

void Material::setMetallicRoughnessTexture(Texture2D* texture, bool hasRealTexture)
{
    metallicRoughnessTexture = texture;
    metallicRoughnessTextureProvided = hasRealTexture;
}

void Material::setOcclusionTexture(Texture2D* texture, bool hasRealTexture)
{
    occlusionTexture = texture;
    occlusionTextureProvided = hasRealTexture;
}

void Material::setEmissiveTexture(Texture2D* texture, bool hasRealTexture)
{
    emissiveTexture = texture;
    emissiveTextureProvided = hasRealTexture;
}

Texture2D* Material::getBaseColorTexture() const
{
    return baseColorTexture;
}

Texture2D* Material::getNormalTexture() const
{
    return normalTexture;
}

Texture2D* Material::getMetallicRoughnessTexture() const
{
    return metallicRoughnessTexture;
}

Texture2D* Material::getOcclusionTexture() const
{
    return occlusionTexture;
}

Texture2D* Material::getEmissiveTexture() const
{
    return emissiveTexture;
}

Texture2D& Material::getTexture() const
{
    if (!baseColorTexture)
    {
        throw std::runtime_error("Material base color texture is null");
    }

    return *baseColorTexture;
}

bool Material::hasRealNormalTexture() const
{
    return normalTextureProvided;
}

bool Material::hasRealMetallicRoughnessTexture() const
{
    return metallicRoughnessTextureProvided;
}

bool Material::hasRealOcclusionTexture() const
{
    return occlusionTextureProvided;
}

bool Material::hasRealEmissiveTexture() const
{
    return emissiveTextureProvided;
}

vk::DescriptorImageInfo Material::makeDescriptorImageInfo(
    const Texture2D* texture) const
{
    if (!texture)
    {
        throw std::runtime_error("Material texture is null");
    }

    vk::DescriptorImageInfo imageInfo{};
    imageInfo
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setImageView(texture->getImageView())
        .setSampler(texture->getSampler());

    return imageInfo;
}

vk::DescriptorImageInfo Material::getImageInfo() const
{
    return makeDescriptorImageInfo(baseColorTexture);
}

vk::DescriptorImageInfo Material::getNormalImageInfo() const
{
    return makeDescriptorImageInfo(normalTexture);
}

vk::DescriptorImageInfo Material::getMetallicRoughnessImageInfo() const
{
    return makeDescriptorImageInfo(metallicRoughnessTexture);
}

vk::DescriptorImageInfo Material::getOcclusionImageInfo() const
{
    return makeDescriptorImageInfo(occlusionTexture);
}

vk::DescriptorImageInfo Material::getEmissiveImageInfo() const
{
    return makeDescriptorImageInfo(emissiveTexture);
}

MaterialImageWrite Material::makeImageWriteForTexture(
    vk::DescriptorSet descriptorSet,
    uint32_t binding,
    const Texture2D* texture) const
{
    MaterialImageWrite result{};
    result.imageInfo = makeDescriptorImageInfo(texture);

    result.write
        .setDstSet(descriptorSet)
        .setDstBinding(binding)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(result.imageInfo);

    return result;
}

MaterialImageWrite Material::makeImageWrite(
    vk::DescriptorSet descriptorSet,
    uint32_t binding) const
{
    return makeImageWriteForTexture(
        descriptorSet,
        binding,
        baseColorTexture);
}

MaterialImageWrite Material::makeNormalImageWrite(
    vk::DescriptorSet descriptorSet,
    uint32_t binding) const
{
    return makeImageWriteForTexture(
        descriptorSet,
        binding,
        normalTexture);
}

MaterialImageWrite Material::makeMetallicRoughnessImageWrite(
    vk::DescriptorSet descriptorSet,
    uint32_t binding) const
{
    return makeImageWriteForTexture(
        descriptorSet,
        binding,
        metallicRoughnessTexture);
}

MaterialImageWrite Material::makeOcclusionImageWrite(
    vk::DescriptorSet descriptorSet,
    uint32_t binding) const
{
    return makeImageWriteForTexture(
        descriptorSet,
        binding,
        occlusionTexture);
}

MaterialImageWrite Material::makeEmissiveImageWrite(
    vk::DescriptorSet descriptorSet,
    uint32_t binding) const
{
    return makeImageWriteForTexture(
        descriptorSet,
        binding,
        emissiveTexture);
}