
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "VulkanContext.hpp"
#include "BufferUtils.hpp"
#include "ImageUtils.hpp"
#include "GltfLoader.hpp"
#include "Scene.hpp"
#include "GpuMesh.hpp"
#include "Texture2D.hpp"
#include "Material.hpp"
#include "Camera.hpp"



class GltfSceneLoader
{
public:
    struct TextureUploadMaps
    {
        std::vector<int> baseColor;
        std::vector<int> normal;
        std::vector<int> metallicRoughness;
        std::vector<int> occlusion;
        std::vector<int> emissive;
    };

    struct LoadContext
    {
        Scene& scene;
        std::vector<std::unique_ptr<GpuMesh>>& gpuMeshes;

        std::vector<std::unique_ptr<Texture2D>>& baseColorTextures;
        std::vector<std::unique_ptr<Texture2D>>& normalTextures;
        std::vector<std::unique_ptr<Texture2D>>& metallicRoughnessTextures;
        std::vector<std::unique_ptr<Texture2D>>& aoTextures;
        std::vector<std::unique_ptr<Texture2D>>& emissiveTextures;

        std::vector<std::unique_ptr<Material>>& materials;

        Texture2D& defaultTexture;
        Texture2D& defaultNormalTexture;
        Texture2D& defaultMetallicRoughnessTexture;
        Texture2D& defaultAoTexture;
        Texture2D& defaultEmissiveTexture;

        Camera& camera;
    };

public:
    GltfSceneLoader(
        VulkanContext& vkContext,
        BufferUtils& bufferUtils,
        ImageUtils& imageUtils);

    

    void load(
        const std::string& path,
        LoadContext& context);

    GltfSceneData loadGltfFile(
        const std::string& path) const;

    TextureUploadMaps uploadTextures(
        const GltfSceneData& imported,
        std::vector<std::unique_ptr<Texture2D>>& textures,
        std::vector<std::unique_ptr<Texture2D>>& normalTextures,
        std::vector<std::unique_ptr<Texture2D>>& metallicRoughnessTextures,
        std::vector<std::unique_ptr<Texture2D>>& aoTextures,
        std::vector<std::unique_ptr<Texture2D>>& emissiveTextures);

    void createMaterials(
        const GltfSceneData& imported,
        const TextureUploadMaps& textureMaps,
        std::vector<std::unique_ptr<Material>>& materials,
        std::vector<std::unique_ptr<Texture2D>>& textures,
        std::vector<std::unique_ptr<Texture2D>>& normalTextures,
        std::vector<std::unique_ptr<Texture2D>>& metallicRoughnessTextures,
        std::vector<std::unique_ptr<Texture2D>>& aoTextures,
        std::vector<std::unique_ptr<Texture2D>>& emissiveTextures,
        Texture2D& defaultTexture,
        Texture2D& defaultNormalTexture,
        Texture2D& defaultMetallicRoughnessTexture,
        Texture2D& defaultAoTexture,
        Texture2D& defaultEmissiveTexture);

    struct LoadedSceneInfo
    {
        glm::vec3 minBounds{};
        glm::vec3 maxBounds{};
    };

    LoadedSceneInfo createRenderables(
        const GltfSceneData& imported,
        Scene& scene,
        std::vector<std::unique_ptr<GpuMesh>>& gpuMeshes,
        std::vector<std::unique_ptr<Material>>& materials);

   

private:
    VulkanContext& vkContext;
    BufferUtils& bufferUtils;
    ImageUtils& imageUtils;


};