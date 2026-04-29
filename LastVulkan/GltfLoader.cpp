#include "GltfLoader.hpp"

#include <tiny_gltf.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include <stdexcept>
#include <iostream>
#include <functional>

namespace
{
    glm::mat4 getNodeLocalMatrix(const tinygltf::Node& node)
    {
        glm::mat4 matrix(1.0f);

        if (node.matrix.size() == 16)
        {
            matrix = glm::mat4(
                static_cast<float>(node.matrix[0]), static_cast<float>(node.matrix[1]),
                static_cast<float>(node.matrix[2]), static_cast<float>(node.matrix[3]),
                static_cast<float>(node.matrix[4]), static_cast<float>(node.matrix[5]),
                static_cast<float>(node.matrix[6]), static_cast<float>(node.matrix[7]),
                static_cast<float>(node.matrix[8]), static_cast<float>(node.matrix[9]),
                static_cast<float>(node.matrix[10]), static_cast<float>(node.matrix[11]),
                static_cast<float>(node.matrix[12]), static_cast<float>(node.matrix[13]),
                static_cast<float>(node.matrix[14]), static_cast<float>(node.matrix[15]));
        }
        else
        {
            if (node.translation.size() == 3)
            {
                matrix = glm::translate(
                    matrix,
                    glm::vec3(
                        static_cast<float>(node.translation[0]),
                        static_cast<float>(node.translation[1]),
                        static_cast<float>(node.translation[2])));
            }

            if (node.rotation.size() == 4)
            {
                glm::quat q(
                    static_cast<float>(node.rotation[3]), // w
                    static_cast<float>(node.rotation[0]), // x
                    static_cast<float>(node.rotation[1]), // y
                    static_cast<float>(node.rotation[2])  // z
                );
                matrix *= glm::mat4_cast(q);
            }

            if (node.scale.size() == 3)
            {
                matrix = glm::scale(
                    matrix,
                    glm::vec3(
                        static_cast<float>(node.scale[0]),
                        static_cast<float>(node.scale[1]),
                        static_cast<float>(node.scale[2])));
            }
        }

        return matrix;
    }

    Transform transformFromMatrix(const glm::mat4& worldMatrix)
    {
        Transform transform{};

        glm::vec3 skew;
        glm::vec4 perspective;
        glm::quat rotationQuat;

        glm::decompose(
            worldMatrix,
            transform.scale,
            rotationQuat,
            transform.position,
            skew,
            perspective);

        transform.rotation = glm::degrees(glm::eulerAngles(rotationQuat));
        return transform;
    }

    const unsigned char* getAccessorElementPtr(
        const tinygltf::Model& model,
        const tinygltf::Accessor& accessor,
        size_t elementIndex)
    {
        if (accessor.bufferView < 0)
            throw std::runtime_error("glTF accessor has no bufferView");

        const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
        const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

        const size_t stride = accessor.ByteStride(bufferView);
        const size_t offset =
            static_cast<size_t>(bufferView.byteOffset) +
            static_cast<size_t>(accessor.byteOffset) +
            elementIndex * stride;

        if (offset >= buffer.data.size())
            throw std::runtime_error("glTF accessor read out of bounds");

        return buffer.data.data() + offset;
    }

    void requireAccessor(
        const tinygltf::Accessor& accessor,
        int expectedComponentType,
        int expectedType,
        const char* semanticName)
    {
        if (accessor.componentType != expectedComponentType ||
            accessor.type != expectedType)
        {
            throw std::runtime_error(
                std::string("Unsupported glTF accessor format for ") + semanticName);
        }
    }

    glm::vec3 readVec3Float(
        const tinygltf::Model& model,
        const tinygltf::Accessor& accessor,
        size_t index)
    {
        const unsigned char* ptr = getAccessorElementPtr(model, accessor, index);
        const float* f = reinterpret_cast<const float*>(ptr);

        return glm::vec3(f[0], f[1], f[2]);
    }

    glm::vec4 readVec4Float(
        const tinygltf::Model& model,
        const tinygltf::Accessor& accessor,
        size_t index)
    {
        const unsigned char* ptr = getAccessorElementPtr(model, accessor, index);
        const float* f = reinterpret_cast<const float*>(ptr);

        return glm::vec4(f[0], f[1], f[2], f[3]);
    }

    glm::vec2 readTexCoord(
        const tinygltf::Model& model,
        const tinygltf::Accessor& accessor,
        size_t index)
    {
        const unsigned char* ptr = getAccessorElementPtr(model, accessor, index);

        if (accessor.type != TINYGLTF_TYPE_VEC2)
            throw std::runtime_error("TEXCOORD_0 must be VEC2");

        switch (accessor.componentType)
        {
        case TINYGLTF_COMPONENT_TYPE_FLOAT:
        {
            const float* f = reinterpret_cast<const float*>(ptr);
            return glm::vec2(f[0], f[1]);
        }

        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        {
            const uint8_t* u = reinterpret_cast<const uint8_t*>(ptr);
            return glm::vec2(
                static_cast<float>(u[0]) / 255.0f,
                static_cast<float>(u[1]) / 255.0f);
        }

        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        {
            const uint16_t* u = reinterpret_cast<const uint16_t*>(ptr);
            return glm::vec2(
                static_cast<float>(u[0]) / 65535.0f,
                static_cast<float>(u[1]) / 65535.0f);
        }

        default:
            throw std::runtime_error("Unsupported TEXCOORD_0 component type");
        }
    }

    uint32_t readIndex(
        const tinygltf::Model& model,
        const tinygltf::Accessor& accessor,
        size_t index)
    {
        if (accessor.bufferView < 0)
            throw std::runtime_error("glTF index accessor has no bufferView");

        const tinygltf::BufferView& bufferView =
            model.bufferViews[accessor.bufferView];

        const tinygltf::Buffer& buffer =
            model.buffers[bufferView.buffer];

        if (index >= accessor.count)
            throw std::runtime_error("glTF index accessor index out of range");

        size_t componentSize = 0;

        switch (accessor.componentType)
        {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            componentSize = sizeof(uint8_t);
            break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            componentSize = sizeof(uint16_t);
            break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            componentSize = sizeof(uint32_t);
            break;
        default:
            throw std::runtime_error("Unsupported glTF index component type");
        }

        const size_t stride = accessor.ByteStride(bufferView);

        const size_t offset =
            static_cast<size_t>(bufferView.byteOffset) +
            static_cast<size_t>(accessor.byteOffset) +
            index * stride;

        if (offset + componentSize > buffer.data.size())
            throw std::runtime_error("glTF index accessor read out of bounds");

        const unsigned char* ptr = buffer.data.data() + offset;

        switch (accessor.componentType)
        {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            return static_cast<uint32_t>(*reinterpret_cast<const uint8_t*>(ptr));

        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            return static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(ptr));

        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            return *reinterpret_cast<const uint32_t*>(ptr);
        }

        throw std::runtime_error("Unreachable glTF index read path");
    }
}

GltfSceneData GltfLoader::load(const std::string& path)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;

    std::string err;
    std::string warn;

    bool ret = false;

    if (path.ends_with(".glb"))
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, path);
    else
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, path);

    if (!warn.empty()) std::cout << "glTF warning: " << warn << std::endl;
    if (!err.empty()) std::cerr << "glTF error: " << err << std::endl;

    if (!ret)
        throw std::runtime_error("Failed to load glTF: " + path);

    GltfSceneData result;

    result.images.reserve(model.images.size());

    for (const auto& gltfImage : model.images)
    {
        GltfImageData imageData{};
        imageData.name = gltfImage.name;
        imageData.width = gltfImage.width;
        imageData.height = gltfImage.height;
        imageData.channels = gltfImage.component;
        imageData.pixels = gltfImage.image;

        result.images.push_back(std::move(imageData));
    }

    result.materials.reserve(model.materials.size());

    for (const auto& gltfMaterial : model.materials)
    {
        GltfMaterialData materialData{};

        materialData.name = gltfMaterial.name;
        materialData.alphaMode = gltfMaterial.alphaMode.empty() ? "OPAQUE" : gltfMaterial.alphaMode;
        materialData.alphaCutoff = static_cast<float>(gltfMaterial.alphaCutoff);
        materialData.doubleSided = gltfMaterial.doubleSided;

        const auto& pbr = gltfMaterial.pbrMetallicRoughness;

        if (pbr.baseColorFactor.size() == 4)
        {
            materialData.baseColorFactor = glm::vec4(
                static_cast<float>(pbr.baseColorFactor[0]),
                static_cast<float>(pbr.baseColorFactor[1]),
                static_cast<float>(pbr.baseColorFactor[2]),
                static_cast<float>(pbr.baseColorFactor[3]));
        }

        materialData.metallicFactor = static_cast<float>(pbr.metallicFactor);
        materialData.roughnessFactor = static_cast<float>(pbr.roughnessFactor);

        if (pbr.baseColorTexture.index >= 0)
        {
            int textureIndex = pbr.baseColorTexture.index;
            if (textureIndex >= 0 && textureIndex < static_cast<int>(model.textures.size()))
            {
                const tinygltf::Texture& gltfTexture = model.textures[textureIndex];
                materialData.baseColorImageIndex = gltfTexture.source;
            }
        }

        if (pbr.metallicRoughnessTexture.index >= 0)
        {
            int textureIndex = pbr.metallicRoughnessTexture.index;
            if (textureIndex >= 0 && textureIndex < static_cast<int>(model.textures.size()))
            {
                const tinygltf::Texture& gltfTexture = model.textures[textureIndex];
                materialData.metallicRoughnessImageIndex = gltfTexture.source;
            }
        }

        if (gltfMaterial.normalTexture.index >= 0)
        {
            int textureIndex = gltfMaterial.normalTexture.index;
            if (textureIndex >= 0 && textureIndex < static_cast<int>(model.textures.size()))
            {
                const tinygltf::Texture& gltfTexture = model.textures[textureIndex];
                materialData.normalImageIndex = gltfTexture.source;
            }

            materialData.normalScale = static_cast<float>(gltfMaterial.normalTexture.scale);
        }

        if (gltfMaterial.occlusionTexture.index >= 0)
        {
            int textureIndex = gltfMaterial.occlusionTexture.index;

            if (textureIndex >= 0 && textureIndex < static_cast<int>(model.textures.size()))
            {
                const tinygltf::Texture& gltfTexture = model.textures[textureIndex];
                materialData.occlusionImageIndex = gltfTexture.source;
            }

            materialData.occlusionStrength =
                static_cast<float>(gltfMaterial.occlusionTexture.strength);
        }

        if (gltfMaterial.emissiveFactor.size() == 3)
        {
            materialData.emissiveFactor = glm::vec3(
                static_cast<float>(gltfMaterial.emissiveFactor[0]),
                static_cast<float>(gltfMaterial.emissiveFactor[1]),
                static_cast<float>(gltfMaterial.emissiveFactor[2]));
        }

        if (gltfMaterial.emissiveTexture.index >= 0)
        {
            int textureIndex = gltfMaterial.emissiveTexture.index;

            if (textureIndex >= 0 && textureIndex < static_cast<int>(model.textures.size()))
            {
                const tinygltf::Texture& gltfTexture = model.textures[textureIndex];
                materialData.emissiveImageIndex = gltfTexture.source;
            }
        }
  

        result.materials.push_back(materialData);
    }

    if (model.scenes.empty())
        return result;

    int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
    const tinygltf::Scene& scene = model.scenes[sceneIndex];

    std::function<void(int, const glm::mat4&)> visitNode;
    visitNode = [&](int nodeIndex, const glm::mat4& parentMatrix)
        {
            const tinygltf::Node& node = model.nodes[nodeIndex];
            glm::mat4 localMatrix = getNodeLocalMatrix(node);
            glm::mat4 worldMatrix = parentMatrix * localMatrix;

            if (node.mesh >= 0)
            {
                const tinygltf::Mesh& mesh = model.meshes[node.mesh];

                for (const auto& primitive : mesh.primitives)
                {
                    if (!primitive.attributes.contains("POSITION"))
                        continue;

                    if (primitive.mode != TINYGLTF_MODE_TRIANGLES)
                    {
                        std::cout << "Skipping non-triangle glTF primitive\n";
                        continue;
                    }

                    MeshData meshData;

                    const tinygltf::Accessor& positionAccessor =
                        model.accessors[primitive.attributes.at("POSITION")];

                    requireAccessor(
                        positionAccessor,
                        TINYGLTF_COMPONENT_TYPE_FLOAT,
                        TINYGLTF_TYPE_VEC3,
                        "POSITION");

                    const tinygltf::Accessor* normalAccessor = nullptr;
                    const tinygltf::Accessor* texCoordAccessor = nullptr;
                    const tinygltf::Accessor* tangentAccessor = nullptr;

                    if (primitive.attributes.contains("NORMAL"))
                    {
                        normalAccessor = &model.accessors[primitive.attributes.at("NORMAL")];

                        requireAccessor(
                            *normalAccessor,
                            TINYGLTF_COMPONENT_TYPE_FLOAT,
                            TINYGLTF_TYPE_VEC3,
                            "NORMAL");
                    }

                    

                    if (primitive.attributes.contains("TEXCOORD_0"))
                    {
                        texCoordAccessor = &model.accessors[primitive.attributes.at("TEXCOORD_0")];

                        if (texCoordAccessor->type != TINYGLTF_TYPE_VEC2)
                            throw std::runtime_error("TEXCOORD_0 must be VEC2");
                    }

                    if (primitive.attributes.contains("TANGENT"))
                    {
                        tangentAccessor = &model.accessors[primitive.attributes.at("TANGENT")];

                        requireAccessor(
                            *tangentAccessor,
                            TINYGLTF_COMPONENT_TYPE_FLOAT,
                            TINYGLTF_TYPE_VEC4,
                            "TANGENT");
                    }

                    const size_t vertexCount = positionAccessor.count;

                    meshData.vertices.resize(vertexCount);

                    for (size_t i = 0; i < vertexCount; ++i)
                    {
                        Vertex v{};

                        v.pos = readVec3Float(model, positionAccessor, i);

                        v.normal = normalAccessor
                            ? readVec3Float(model, *normalAccessor, i)
                            : glm::vec3(0.0f, 0.0f, 1.0f);

                        v.texCoord = texCoordAccessor
                            ? readTexCoord(model, *texCoordAccessor, i)
                            : glm::vec2(0.0f);

                        v.tangent = tangentAccessor
                            ? readVec4Float(model, *tangentAccessor, i)
                            : glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);

                        meshData.vertices[i] = v;
                    }

                    if (primitive.indices >= 0)
                    {
                        const tinygltf::Accessor& indexAccessor =
                            model.accessors[primitive.indices];

                        if (indexAccessor.type != TINYGLTF_TYPE_SCALAR)
                            throw std::runtime_error("glTF index accessor must be SCALAR");

                        meshData.indices.resize(indexAccessor.count);

                        for (size_t i = 0; i < indexAccessor.count; ++i)
                        {
                            meshData.indices[i] = readIndex(model, indexAccessor, i);
                        }
                    }
                    else
                    {
                        meshData.indices.resize(vertexCount);

                        for (uint32_t i = 0; i < static_cast<uint32_t>(vertexCount); ++i)
                        {
                            meshData.indices[i] = i;
                        }
                    }

                    GltfRenderableData data{};
                    data.mesh = std::move(meshData);
                    data.transform = transformFromMatrix(worldMatrix);
                    data.materialIndex = primitive.material;

                    result.renderables.push_back(std::move(data));
                }
            }

            for (int childIndex : node.children)
            {
                visitNode(childIndex, worldMatrix);
            }
        };

    for (int nodeIndex : scene.nodes)
    {
        visitNode(nodeIndex, glm::mat4(1.0f));
    }

    return result;
}