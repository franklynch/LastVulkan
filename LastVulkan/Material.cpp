#include "Material.hpp"
#include <stdexcept>


Material::Material(
    Texture2D& baseColorTexture,
    Texture2D* defaultNormal,
    Texture2D* defaultMetallicRoughness)
    : texture(baseColorTexture),
    normalTexture(defaultNormal),
    metallicRoughnessTexture(defaultMetallicRoughness),
    normalTextureProvided(false),
    metallicRoughnessTextureProvided(false)
{
    if (!defaultNormal || !defaultMetallicRoughness)
    {
        throw std::runtime_error("Material created without default textures");
    }
}

vk::DescriptorImageInfo Material::getImageInfo() const
{
    vk::DescriptorImageInfo imageInfo{};
    imageInfo
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setImageView(texture.getImageView())
        .setSampler(texture.getSampler());

    return imageInfo;
}

MaterialImageWrite Material::makeImageWrite(vk::DescriptorSet dstSet, uint32_t binding) const
{
    MaterialImageWrite result{};
    result.imageInfo = getImageInfo();

    result.write
        .setDstSet(dstSet)
        .setDstBinding(binding)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(result.imageInfo);

    return result;
}

vk::DescriptorImageInfo Material::getNormalImageInfo() const
{
    const Texture2D* tex = normalTexture;

    if (!tex)
    {
        std::cerr << "Material normal texture missing for material: "
            << name << std::endl;

        assert(tex && "Material normal texture should never be null");
    }

    vk::DescriptorImageInfo imageInfo{};
    imageInfo
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setImageView(tex->getImageView())
        .setSampler(tex->getSampler());

    return imageInfo;
}

MaterialImageWrite Material::makeNormalImageWrite(vk::DescriptorSet dstSet, uint32_t binding) const
{
    MaterialImageWrite result{};
    result.imageInfo = getNormalImageInfo();

    result.write
        .setDstSet(dstSet)
        .setDstBinding(binding)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(result.imageInfo);

    return result;
}

vk::DescriptorImageInfo Material::getMetallicRoughnessImageInfo() const
{
    const Texture2D* tex = metallicRoughnessTexture;

    if (!tex)
    {
        throw std::runtime_error("Material metallic-roughness texture was not assigned");
    }

    vk::DescriptorImageInfo imageInfo{};
    imageInfo
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setImageView(tex->getImageView())
        .setSampler(tex->getSampler());

    return imageInfo;
}

MaterialImageWrite Material::makeMetallicRoughnessImageWrite(vk::DescriptorSet dstSet, uint32_t binding) const
{
    MaterialImageWrite result{};
    result.imageInfo = getMetallicRoughnessImageInfo();

    result.write
        .setDstSet(dstSet)
        .setDstBinding(binding)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(result.imageInfo);

    return result;
}

void Material::setNormalTexture(Texture2D* texture, bool provided)
{
    normalTexture = texture;
    normalTextureProvided = provided;
}

void Material::setMetallicRoughnessTexture(Texture2D* texture, bool provided)
{
    metallicRoughnessTexture = texture;
    metallicRoughnessTextureProvided = provided;
}

MaterialImageWrite Material::makeOcclusionImageWrite(
    vk::DescriptorSet descriptorSet,
    uint32_t binding) const
{
    MaterialImageWrite result{};

    if (!occlusionTexture)
    {
        throw std::runtime_error("Material occlusion texture is null");
    }

    result.imageInfo
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setImageView(*occlusionTexture->getImageView())
        .setSampler(*occlusionTexture->getSampler());

    result.write
        .setDstSet(descriptorSet)
        .setDstBinding(binding)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setImageInfo(result.imageInfo);

    return result;
}