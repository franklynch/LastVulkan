#include "Material.hpp"

Material::Material(Texture2D& texture)
    : texture(texture)
{
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