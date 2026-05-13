#include "MaterialSystem.hpp"

#include <stdexcept>

MaterialSystem::MaterialSystem(
    VulkanContext& vkContext,
    BufferUtils& bufferUtils,
    ImageUtils& imageUtils)
    : vkContext(vkContext),
    bufferUtils(bufferUtils),
    imageUtils(imageUtils)
{
}

void MaterialSystem::clear()
{
    m_materials.clear();

    m_textures.clear();
    m_normalTextures.clear();
    m_metallicRoughnessTextures.clear();
    m_aoTextures.clear();
    m_emissiveTextures.clear();

    m_defaultNormalTexture.reset();
    m_defaultMetallicRoughnessTexture.reset();
    m_defaultAoTexture.reset();
    m_defaultEmissiveTexture.reset();
}

void MaterialSystem::createDefaultTextures()
{
    const unsigned char fallbackWhitePixel[4] = { 255, 255, 255, 255 };
    const unsigned char flatNormalPixels[4] = { 128, 128, 255, 255 };
    const unsigned char defaultMRPixels[4] = { 255, 255, 255, 255 };
    const unsigned char fallbackAoPixel[4] = { 255, 255, 255, 255 };
    const unsigned char fallbackEmissivePixel[4] = { 0, 0, 0, 255 };

    m_textures.push_back(std::make_unique<Texture2D>(
        vkContext, bufferUtils, imageUtils,
        fallbackWhitePixel, 1, 1, 4,
        "<fallback-white-base-color>",
        vk::Format::eR8G8B8A8Srgb));

    m_defaultNormalTexture = std::make_unique<Texture2D>(
        vkContext, bufferUtils, imageUtils,
        flatNormalPixels, 1, 1, 4,
        "Default Flat Normal",
        vk::Format::eR8G8B8A8Unorm);

    m_defaultMetallicRoughnessTexture = std::make_unique<Texture2D>(
        vkContext, bufferUtils, imageUtils,
        defaultMRPixels, 1, 1, 4,
        "Default MetallicRoughness",
        vk::Format::eR8G8B8A8Unorm);

    m_defaultAoTexture = std::make_unique<Texture2D>(
        vkContext, bufferUtils, imageUtils,
        fallbackAoPixel, 1, 1, 4,
        "<fallback-ao>",
        vk::Format::eR8G8B8A8Unorm);

    m_defaultEmissiveTexture = std::make_unique<Texture2D>(
        vkContext, bufferUtils, imageUtils,
        fallbackEmissivePixel, 1, 1, 4,
        "<fallback-emissive>",
        vk::Format::eR8G8B8A8Srgb);

}

Texture2D& MaterialSystem::defaultTexture()
{
    if (m_textures.empty() || !m_textures[0])
    {
        throw std::runtime_error("default base color texture is not available");
    }

    return *m_textures[0];
}

Texture2D& MaterialSystem::defaultNormalTexture()
{
    if (!m_defaultNormalTexture)
    {
        throw std::runtime_error("default normal texture is not available");
    }

    return *m_defaultNormalTexture;
}

Texture2D& MaterialSystem::defaultMetallicRoughnessTexture()
{
    if (!m_defaultMetallicRoughnessTexture)
    {
        throw std::runtime_error("default metallic roughness texture is not available");
    }

    return *m_defaultMetallicRoughnessTexture;
}

Texture2D& MaterialSystem::defaultAoTexture()
{
    if (!m_defaultAoTexture)
    {
        throw std::runtime_error("default AO texture is not available");
    }

    return *m_defaultAoTexture;
}

Texture2D& MaterialSystem::defaultEmissiveTexture()
{
    if (!m_defaultEmissiveTexture)
    {
        throw std::runtime_error("default emissive texture is not available");
    }

    return *m_defaultEmissiveTexture;
}

Material& MaterialSystem::defaultMaterial()
{
    if (m_materials.empty() || !m_materials[0])
    {
        throw std::runtime_error("default material is not available");
    }

    return *m_materials[0];
}

int MaterialSystem::getMaterialIndex(const Material& material) const
{
    for (size_t i = 0; i < m_materials.size(); ++i)
    {
        if (m_materials[i].get() == &material)
        {
            return static_cast<int>(i);
        }
    }

    return -1;
}