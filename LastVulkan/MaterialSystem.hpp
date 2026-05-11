#pragma once

#include <memory>
#include <vector>

#include "Texture2D.hpp"
#include "Material.hpp"
#include "VulkanContext.hpp"
#include "BufferUtils.hpp"
#include "ImageUtils.hpp"

class MaterialSystem
{
public:
    MaterialSystem(
        VulkanContext& vkContext,
        BufferUtils& bufferUtils,
        ImageUtils& imageUtils);

    void createDefaultTextures();
    void clear();

    Texture2D& defaultTexture();
    Texture2D& defaultNormalTexture();
    Texture2D& defaultMetallicRoughnessTexture();
    Texture2D& defaultAoTexture();
    Texture2D& defaultEmissiveTexture();

    Material& defaultMaterial();

    int getMaterialIndex(const Material& material) const;

    std::vector<std::unique_ptr<Texture2D>>& baseColorTextures() { return m_textures; }
    std::vector<std::unique_ptr<Texture2D>>& normalTextures() { return m_normalTextures; }
    std::vector<std::unique_ptr<Texture2D>>& metallicRoughnessTextures() { return m_metallicRoughnessTextures; }
    std::vector<std::unique_ptr<Texture2D>>& aoTextures() { return m_aoTextures; }
    std::vector<std::unique_ptr<Texture2D>>& emissiveTextures() { return m_emissiveTextures; }

    std::vector<std::unique_ptr<Material>>& materials() { return m_materials; }
    const std::vector<std::unique_ptr<Material>>& materials() const { return m_materials; }

    Texture2D& getDefaultTexture();
    Material& getDefaultMaterial();

 

private:
    VulkanContext& vkContext;
    BufferUtils& bufferUtils;
    ImageUtils& imageUtils;

    std::vector<std::unique_ptr<Texture2D>> m_textures;
    std::vector<std::unique_ptr<Texture2D>> m_normalTextures;
    std::vector<std::unique_ptr<Texture2D>> m_metallicRoughnessTextures;
    std::vector<std::unique_ptr<Texture2D>> m_aoTextures;
    std::vector<std::unique_ptr<Texture2D>> m_emissiveTextures;

    std::unique_ptr<Texture2D> m_defaultNormalTexture;
    std::unique_ptr<Texture2D> m_defaultMetallicRoughnessTexture;
    std::unique_ptr<Texture2D> m_defaultAoTexture;
    std::unique_ptr<Texture2D> m_defaultEmissiveTexture;

    std::vector<std::unique_ptr<Material>> m_materials;
};