#pragma once

#include <string>
#include <vector>

#include <glm/vec4.hpp>

#include "MeshData.hpp"
#include "Transform.hpp"

struct GltfRenderableData
{
    MeshData mesh;
    Transform transform;
    int materialIndex = -1;
};

struct GltfImageData
{
    std::string name;
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<unsigned char> pixels;
};

struct GltfMaterialData
{
    std::string name;
    glm::vec4 baseColorFactor{ 1.0f, 1.0f, 1.0f, 1.0f };
    int baseColorImageIndex = -1;

    std::string alphaMode = "OPAQUE";
    float alphaCutoff = 0.5f;
    bool doubleSided = false;

    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
};

struct GltfSceneData
{
    std::vector<GltfRenderableData> renderables;
    std::vector<GltfMaterialData> materials;
    std::vector<GltfImageData> images;
};

class GltfLoader
{
public:
    GltfSceneData load(const std::string& path);
};