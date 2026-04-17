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

    // Add glTF material extraction here
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

        if (pbr.baseColorTexture.index >= 0)
        {
            int textureIndex = pbr.baseColorTexture.index;
            if (textureIndex >= 0 && textureIndex < static_cast<int>(model.textures.size()))
            {
                const tinygltf::Texture& gltfTexture = model.textures[textureIndex];
                materialData.baseColorImageIndex = gltfTexture.source;
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

                    MeshData meshData;

                    const float* positions = nullptr;
                    const float* normals = nullptr;
                    const float* texcoords = nullptr;
                    size_t vertexCount = 0;

                    // POSITION
                    {
                        const auto& accessor =
                            model.accessors[primitive.attributes.at("POSITION")];
                        const auto& bufferView =
                            model.bufferViews[accessor.bufferView];
                        const auto& buffer =
                            model.buffers[bufferView.buffer];

                        positions = reinterpret_cast<const float*>(
                            buffer.data.data() + bufferView.byteOffset + accessor.byteOffset);

                        vertexCount = accessor.count;
                    }

                    // NORMAL (optional)
                    if (primitive.attributes.contains("NORMAL"))
                    {
                        const auto& accessor =
                            model.accessors[primitive.attributes.at("NORMAL")];
                        const auto& bufferView =
                            model.bufferViews[accessor.bufferView];
                        const auto& buffer =
                            model.buffers[bufferView.buffer];

                        normals = reinterpret_cast<const float*>(
                            buffer.data.data() + bufferView.byteOffset + accessor.byteOffset);
                    }

                    // TEXCOORD_0 (optional)
                    if (primitive.attributes.contains("TEXCOORD_0"))
                    {
                        const auto& accessor =
                            model.accessors[primitive.attributes.at("TEXCOORD_0")];
                        const auto& bufferView =
                            model.bufferViews[accessor.bufferView];
                        const auto& buffer =
                            model.buffers[bufferView.buffer];

                        texcoords = reinterpret_cast<const float*>(
                            buffer.data.data() + bufferView.byteOffset + accessor.byteOffset);
                    }

                    meshData.vertices.resize(vertexCount);

                    for (size_t i = 0; i < vertexCount; ++i)
                    {
                        Vertex v{};

                        v.pos = {
                            positions[i * 3 + 0],
                            positions[i * 3 + 1],
                            positions[i * 3 + 2]
                        };

                        v.normal = normals
                            ? glm::vec3(
                                normals[i * 3 + 0],
                                normals[i * 3 + 1],
                                normals[i * 3 + 2])
                            : glm::vec3(0.0f, 0.0f, 1.0f);

                        v.texCoord = texcoords
                            ? glm::vec2(
                                texcoords[i * 2 + 0],
                                1.0f - texcoords[i * 2 + 1])
                            : glm::vec2(0.0f);

                        meshData.vertices[i] = v;
                    }

                    // INDICES
                    if (primitive.indices >= 0)
                    {
                        const auto& accessor = model.accessors[primitive.indices];
                        const auto& bufferView = model.bufferViews[accessor.bufferView];
                        const auto& buffer = model.buffers[bufferView.buffer];

                        const unsigned char* dataPtr =
                            buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;

                        meshData.indices.resize(accessor.count);

                        for (size_t i = 0; i < accessor.count; ++i)
                        {
                            switch (accessor.componentType)
                            {
                            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                                meshData.indices[i] =
                                    reinterpret_cast<const uint8_t*>(dataPtr)[i];
                                break;

                            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                                meshData.indices[i] =
                                    reinterpret_cast<const uint16_t*>(dataPtr)[i];
                                break;

                            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                                meshData.indices[i] =
                                    reinterpret_cast<const uint32_t*>(dataPtr)[i];
                                break;

                            default:
                                throw std::runtime_error("Unsupported index component type");
                            }
                        }
                    }
                    else
                    {
                        meshData.indices.resize(vertexCount);
                        for (uint32_t i = 0; i < vertexCount; ++i)
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