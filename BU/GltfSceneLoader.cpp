#include "GltfSceneLoader.hpp"

#include <iostream>
#include <stdexcept>

GltfSceneLoader::GltfSceneLoader(
    VulkanContext& vkContext,
    BufferUtils& bufferUtils,
    ImageUtils& imageUtils)
    : vkContext(vkContext),
    bufferUtils(bufferUtils),
    imageUtils(imageUtils)
{
}

GltfSceneData GltfSceneLoader::loadGltfFile(
    const std::string& path) const
{
    if (path.empty())
    {
        throw std::runtime_error("glTF path is empty");
    }

    GltfLoader loader;
    return loader.load(path);
}

GltfSceneLoader::TextureUploadMaps
GltfSceneLoader::uploadTextures(
    const GltfSceneData& imported,
    std::vector<std::unique_ptr<Texture2D>>& textures,
    std::vector<std::unique_ptr<Texture2D>>& normalTextures,
    std::vector<std::unique_ptr<Texture2D>>& metallicRoughnessTextures,
    std::vector<std::unique_ptr<Texture2D>>& aoTextures,
    std::vector<std::unique_ptr<Texture2D>>& emissiveTextures)
{
    
    std::vector<bool> imageUsedAsBaseColor(imported.images.size(), false);
    std::vector<bool> imageUsedAsNormal(imported.images.size(), false);
    std::vector<bool> imageUsedAsMR(imported.images.size(), false);
    std::vector<bool> imageUsedAsAO(imported.images.size(), false);
    std::vector<bool> imageUsedAsEmissive(imported.images.size(), false);
    
    
    TextureUploadMaps textureMaps{};

    textureMaps.baseColor.resize(imported.images.size(), -1);
    textureMaps.normal.resize(imported.images.size(), -1);
    textureMaps.metallicRoughness.resize(imported.images.size(), -1);
    textureMaps.occlusion.resize(imported.images.size(), -1);
    textureMaps.emissive.resize(imported.images.size(), -1);

    for (const auto& importedMaterial : imported.materials)
    {
        if (importedMaterial.baseColorImageIndex >= 0 &&
            importedMaterial.baseColorImageIndex < static_cast<int>(imageUsedAsBaseColor.size()))
        {
            imageUsedAsBaseColor[importedMaterial.baseColorImageIndex] = true;
        }

        if (importedMaterial.normalImageIndex >= 0 &&
            importedMaterial.normalImageIndex < static_cast<int>(imageUsedAsNormal.size()))
        {
            imageUsedAsNormal[importedMaterial.normalImageIndex] = true;
        }

        if (importedMaterial.metallicRoughnessImageIndex >= 0 &&
            importedMaterial.metallicRoughnessImageIndex < static_cast<int>(imageUsedAsMR.size()))
        {
            imageUsedAsMR[importedMaterial.metallicRoughnessImageIndex] = true;
        }

        if (importedMaterial.occlusionImageIndex >= 0 &&
            importedMaterial.occlusionImageIndex < static_cast<int>(imageUsedAsAO.size()))
        {
            imageUsedAsAO[importedMaterial.occlusionImageIndex] = true;
        }

        if (importedMaterial.emissiveImageIndex >= 0 &&
            importedMaterial.emissiveImageIndex < static_cast<int>(imageUsedAsEmissive.size()))
        {
            imageUsedAsEmissive[importedMaterial.emissiveImageIndex] = true;
        }
    }

    for (size_t i = 0; i < imported.images.size(); ++i)
    {
        if (!imageUsedAsBaseColor[i])
            continue;

        const auto& image = imported.images[i];

        if (image.pixels.empty() || image.width <= 0 || image.height <= 0)
            continue;

        if (image.channels != 3 && image.channels != 4)
        {
            std::cout << "Skipping glTF baseColor image " << i
                << " (" << image.name << ") because channels = "
                << image.channels << " (expected 3 or 4)\n";
            continue;
        }

        textures.push_back(std::make_unique<Texture2D>(
            vkContext,
            bufferUtils,
            imageUtils,
            image.pixels.data(),
            static_cast<uint32_t>(image.width),
            static_cast<uint32_t>(image.height),
            static_cast<uint32_t>(image.channels),
            image.name.empty()
            ? ("glTF baseColor image " + std::to_string(i))
            : image.name,
            vk::Format::eR8G8B8A8Srgb));

        textureMaps.baseColor[i] = static_cast<int>(textures.size()) - 1;
    }

    for (size_t i = 0; i < imported.images.size(); ++i)
    {
        if (!imageUsedAsNormal[i])
            continue;

        const auto& image = imported.images[i];

        if (image.pixels.empty() || image.width <= 0 || image.height <= 0)
            continue;

        if (image.channels != 3 && image.channels != 4)
        {
            std::cout << "Skipping glTF normal image " << i
                << " (" << image.name << ") because channels = "
                << image.channels << " (expected 3 or 4)\n";
            continue;
        }

        normalTextures.push_back(std::make_unique<Texture2D>(
            vkContext,
            bufferUtils,
            imageUtils,
            image.pixels.data(),
            static_cast<uint32_t>(image.width),
            static_cast<uint32_t>(image.height),
            static_cast<uint32_t>(image.channels),
            image.name.empty()
            ? ("glTF normal image " + std::to_string(i))
            : image.name,
            vk::Format::eR8G8B8A8Unorm));

        textureMaps.normal[i] = static_cast<int>(normalTextures.size()) - 1;
    }

    for (size_t i = 0; i < imported.images.size(); ++i)
    {
        if (!imageUsedAsMR[i])
            continue;

        const auto& image = imported.images[i];

        if (image.pixels.empty() || image.width <= 0 || image.height <= 0)
            continue;

        if (image.channels != 3 && image.channels != 4)
        {
            std::cout << "Skipping glTF metallicRoughness image " << i
                << " (" << image.name << ") because channels = "
                << image.channels << " (expected 3 or 4)\n";
            continue;
        }

        metallicRoughnessTextures.push_back(std::make_unique<Texture2D>(
            vkContext,
            bufferUtils,
            imageUtils,
            image.pixels.data(),
            static_cast<uint32_t>(image.width),
            static_cast<uint32_t>(image.height),
            static_cast<uint32_t>(image.channels),
            image.name.empty()
            ? ("glTF metallicRoughness image " + std::to_string(i))
            : image.name,
            vk::Format::eR8G8B8A8Unorm));

        textureMaps.metallicRoughness[i] =
            static_cast<int>(metallicRoughnessTextures.size()) - 1;
    }

    for (size_t i = 0; i < imported.images.size(); ++i)
    {
        if (!imageUsedAsAO[i])
            continue;

        const auto& image = imported.images[i];

        if (image.pixels.empty() || image.width <= 0 || image.height <= 0)
            continue;

        if (image.channels != 3 && image.channels != 4)
        {
            std::cout << "Skipping glTF AO image " << i
                << " (" << image.name << ") because channels = "
                << image.channels << " (expected 3 or 4)\n";
            continue;
        }

        aoTextures.push_back(std::make_unique<Texture2D>(
            vkContext,
            bufferUtils,
            imageUtils,
            image.pixels.data(),
            static_cast<uint32_t>(image.width),
            static_cast<uint32_t>(image.height),
            static_cast<uint32_t>(image.channels),
            image.name.empty()
            ? ("glTF AO image " + std::to_string(i))
            : image.name,
            vk::Format::eR8G8B8A8Unorm));

        textureMaps.occlusion[i] = static_cast<int>(aoTextures.size()) - 1;
    }

    for (size_t i = 0; i < imported.images.size(); ++i)
    {
        if (!imageUsedAsEmissive[i])
            continue;

        const auto& image = imported.images[i];

        if (image.pixels.empty() || image.width <= 0 || image.height <= 0)
            continue;

        if (image.channels != 3 && image.channels != 4)
        {
            std::cout << "Skipping glTF emissive image " << i
                << " (" << image.name << ") because channels = "
                << image.channels << " (expected 3 or 4)\n";
            continue;
        }

        emissiveTextures.push_back(std::make_unique<Texture2D>(
            vkContext,
            bufferUtils,
            imageUtils,
            image.pixels.data(),
            static_cast<uint32_t>(image.width),
            static_cast<uint32_t>(image.height),
            static_cast<uint32_t>(image.channels),
            image.name.empty()
            ? ("glTF emissive image " + std::to_string(i))
            : image.name,
            vk::Format::eR8G8B8A8Srgb));

        textureMaps.emissive[i] = static_cast<int>(emissiveTextures.size()) - 1;
    }

    

    return textureMaps;
}

void GltfSceneLoader::createMaterials(
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
    Texture2D& defaultEmissiveTexture)
{
    // Slot 0 = default fallback material
    auto fallbackMaterial = std::make_unique<Material>(
        defaultTexture,
        &defaultNormalTexture,
        &defaultMetallicRoughnessTexture
    );

    fallbackMaterial->setName("Default fallback material");
    fallbackMaterial->setOcclusionTexture(&defaultAoTexture, false);
    fallbackMaterial->setOcclusionStrength(0.0f);
    fallbackMaterial->setEmissiveTexture(&defaultEmissiveTexture, false);
    fallbackMaterial->setEmissiveFactor(glm::vec3(0.0f));

    materials.push_back(std::move(fallbackMaterial));

    for (const auto& importedMaterial : imported.materials)
    {
        Texture2D* assignedBaseColorTexture = &defaultTexture;

        if (importedMaterial.baseColorImageIndex >= 0 &&
            importedMaterial.baseColorImageIndex < static_cast<int>(textureMaps.baseColor.size()))
        {
            const int textureIndex =
                textureMaps.baseColor[importedMaterial.baseColorImageIndex];

            if (textureIndex >= 0 &&
                textureIndex < static_cast<int>(textures.size()))
            {
                assignedBaseColorTexture = textures[textureIndex].get();
            }
        }

        auto material = std::make_unique<Material>(
            *assignedBaseColorTexture,
            &defaultNormalTexture,
            &defaultMetallicRoughnessTexture
        );

        material->setBaseColorFactor(importedMaterial.baseColorFactor);
        material->setName(importedMaterial.name);
        material->setDoubleSided(importedMaterial.doubleSided);
        material->setMetallicFactor(importedMaterial.metallicFactor);
        material->setRoughnessFactor(importedMaterial.roughnessFactor);
        material->setNormalScale(importedMaterial.normalScale);
        material->setAlphaMode(importedMaterial.alphaMode);
        material->setAlphaCutoff(importedMaterial.alphaCutoff);

        Texture2D* assignedNormalTexture = &defaultNormalTexture;
        bool hasRealNormalTexture = false;

        if (importedMaterial.normalImageIndex >= 0 &&
            importedMaterial.normalImageIndex < static_cast<int>(textureMaps.normal.size()))
        {
            const int textureIndex =
                textureMaps.normal[importedMaterial.normalImageIndex];

            if (textureIndex >= 0 &&
                textureIndex < static_cast<int>(normalTextures.size()))
            {
                assignedNormalTexture = normalTextures[textureIndex].get();
                hasRealNormalTexture = true;
            }
        }

        Texture2D* assignedMRTexture = &defaultMetallicRoughnessTexture;
        bool hasRealMRTexture = false;

        if (importedMaterial.metallicRoughnessImageIndex >= 0 &&
            importedMaterial.metallicRoughnessImageIndex <
            static_cast<int>(textureMaps.metallicRoughness.size()))
        {
            const int textureIndex =
                textureMaps.metallicRoughness[importedMaterial.metallicRoughnessImageIndex];

            if (textureIndex >= 0 &&
                textureIndex < static_cast<int>(metallicRoughnessTextures.size()))
            {
                assignedMRTexture = metallicRoughnessTextures[textureIndex].get();
                hasRealMRTexture = true;
            }
        }

        Texture2D* assignedAOTexture = &defaultAoTexture;
        bool hasRealAOTexture = false;

        if (importedMaterial.occlusionImageIndex >= 0 &&
            importedMaterial.occlusionImageIndex <
            static_cast<int>(textureMaps.occlusion.size()))
        {
            const int textureIndex =
                textureMaps.occlusion[importedMaterial.occlusionImageIndex];

            if (textureIndex >= 0 &&
                textureIndex < static_cast<int>(aoTextures.size()))
            {
                assignedAOTexture = aoTextures[textureIndex].get();
                hasRealAOTexture = true;
            }
        }

        Texture2D* assignedEmissiveTexture = &defaultEmissiveTexture;
        bool hasRealEmissiveTexture = false;

        if (importedMaterial.emissiveImageIndex >= 0 &&
            importedMaterial.emissiveImageIndex <
            static_cast<int>(textureMaps.emissive.size()))
        {
            const int textureIndex =
                textureMaps.emissive[importedMaterial.emissiveImageIndex];

            if (textureIndex >= 0 &&
                textureIndex < static_cast<int>(emissiveTextures.size()))
            {
                assignedEmissiveTexture = emissiveTextures[textureIndex].get();
                hasRealEmissiveTexture = true;
            }
        }

        material->setNormalTexture(assignedNormalTexture, hasRealNormalTexture);
        material->setMetallicRoughnessTexture(assignedMRTexture, hasRealMRTexture);
        material->setOcclusionTexture(assignedAOTexture, hasRealAOTexture);
        material->setOcclusionStrength(importedMaterial.occlusionStrength);
        material->setEmissiveTexture(assignedEmissiveTexture, hasRealEmissiveTexture);
        material->setEmissiveFactor(importedMaterial.emissiveFactor);

        materials.push_back(std::move(material));
    }
}

GltfSceneLoader::LoadedSceneInfo

GltfSceneLoader::createRenderables(
    const GltfSceneData& imported,
    Scene& scene,
    std::vector<std::unique_ptr<GpuMesh>>& gpuMeshes,
    std::vector<std::unique_ptr<Material>>& materials)

{
    if (imported.renderables.empty())
    {
        throw std::runtime_error("glTF import produced no renderables");
    }

    glm::vec3 minBounds(FLT_MAX);
    glm::vec3 maxBounds(-FLT_MAX);

    // Compute bounds
    for (const auto& importedRenderable : imported.renderables)
    {
        const glm::mat4 worldMatrix =
            importedRenderable.transform.toMatrix();

        for (const auto& vertex : importedRenderable.mesh.vertices)
        {
            glm::vec3 worldPos =
                glm::vec3(worldMatrix * glm::vec4(vertex.pos, 1.0f));

            minBounds = glm::min(minBounds, worldPos);
            maxBounds = glm::max(maxBounds, worldPos);
        }
    }

    glm::vec3 modelCenter = (minBounds + maxBounds) * 0.5f;
    glm::mat4 modelRootMatrix =
        glm::translate(glm::mat4(1.0f), -modelCenter);

    

    // Create renderables
    for (size_t i = 0; i < imported.renderables.size(); ++i)
    {
        const auto& importedRenderable = imported.renderables[i];

        gpuMeshes.push_back(std::make_unique<GpuMesh>(
            vkContext,
            bufferUtils,
            importedRenderable.mesh.vertices,
            importedRenderable.mesh.indices
        ));

        const int importedMaterialIndex =
            importedRenderable.materialIndex;

        const int rendererMaterialIndex =
            importedMaterialIndex >= 0 &&
            importedMaterialIndex + 1 < static_cast<int>(materials.size())
            ? importedMaterialIndex + 1
            : 0;

        Material& assignedMaterial =
            *materials[rendererMaterialIndex];

        Renderable& renderable = scene.addRenderable(
            *gpuMeshes.back(),
            assignedMaterial,
            "glTF " + std::to_string(i)
        );

        renderable.setMaterialIndex(
            static_cast<uint32_t>(rendererMaterialIndex));

        glm::mat4 originalMatrix =
            importedRenderable.transform.toMatrix();

        glm::mat4 finalMatrix =
            modelRootMatrix * originalMatrix;

        Transform& t = renderable.getTransform();
        t.useMatrixOverride = true;
        t.matrixOverride = finalMatrix;
    }

    return LoadedSceneInfo{
    .minBounds = minBounds,
    .maxBounds = maxBounds
    };


}



void GltfSceneLoader::load(
    const std::string& path,
    LoadContext& context)
{
    GltfSceneData imported =
        loadGltfFile(path);

    TextureUploadMaps textureMaps =
        uploadTextures(
            imported,
            context.baseColorTextures,
            context.normalTextures,
            context.metallicRoughnessTextures,
            context.aoTextures,
            context.emissiveTextures);

    createMaterials(
        imported,
        textureMaps,
        context.materials,
        context.baseColorTextures,
        context.normalTextures,
        context.metallicRoughnessTextures,
        context.aoTextures,
        context.emissiveTextures,
        context.defaultTexture,
        context.defaultNormalTexture,
        context.defaultMetallicRoughnessTexture,
        context.defaultAoTexture,
        context.defaultEmissiveTexture);

    auto loadedSceneInfo =
        createRenderables(
            imported,
            context.scene,
            context.gpuMeshes,
            context.materials);

    context.camera.frameBounds(
        loadedSceneInfo.minBounds,
        loadedSceneInfo.maxBounds);
}