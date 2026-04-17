#include "Material.hpp"

Material::Material(Texture2D& texture)
    : texture(texture)
{
}

vk::DescriptorImageInfo Material::getImageInfo() const
{
    vk::DescriptorImageInfo imageInfo{};
    imageInfo
        .setSampler(*texture.getSampler())
        .setImageView(*texture.getImageView())
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

    return imageInfo;
}

MaterialImageWrite Material::makeImageWrite(vk::DescriptorSet dstSet,
    uint32_t binding) const
{
    MaterialImageWrite result{};

    result.imageInfo
        .setSampler(*texture.getSampler())
        .setImageView(*texture.getImageView())
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

    result.write
        .setDstSet(dstSet)
        .setDstBinding(binding)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(result.imageInfo);

    return result;
}