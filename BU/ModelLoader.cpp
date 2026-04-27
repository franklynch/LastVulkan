#include "ModelLoader.hpp"

#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <cmath>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

static void computeTangents(std::vector<Vertex>& vertices,
    const std::vector<uint32_t>& indices)
{
    std::vector<glm::vec3> tangentAccum(vertices.size(), glm::vec3(0.0f));
    std::vector<glm::vec3> bitangentAccum(vertices.size(), glm::vec3(0.0f));

    for (size_t i = 0; i + 2 < indices.size(); i += 3)
    {
        uint32_t i0 = indices[i + 0];
        uint32_t i1 = indices[i + 1];
        uint32_t i2 = indices[i + 2];

        Vertex& v0 = vertices[i0];
        Vertex& v1 = vertices[i1];
        Vertex& v2 = vertices[i2];

        glm::vec3 edge1 = v1.pos - v0.pos;
        glm::vec3 edge2 = v2.pos - v0.pos;

        glm::vec2 uv1 = v1.texCoord - v0.texCoord;
        glm::vec2 uv2 = v2.texCoord - v0.texCoord;

        float denom = uv1.x * uv2.y - uv2.x * uv1.y;

        if (std::abs(denom) < 1e-6f)
            continue;

        float r = 1.0f / denom;

        glm::vec3 tangent =
            (edge1 * uv2.y - edge2 * uv1.y) * r;

        glm::vec3 bitangent =
            (edge2 * uv1.x - edge1 * uv2.x) * r;

        tangentAccum[i0] += tangent;
        tangentAccum[i1] += tangent;
        tangentAccum[i2] += tangent;

        bitangentAccum[i0] += bitangent;
        bitangentAccum[i1] += bitangent;
        bitangentAccum[i2] += bitangent;
    }

    for (size_t i = 0; i < vertices.size(); ++i)
    {
        glm::vec3 n = glm::normalize(vertices[i].normal);
        glm::vec3 t = tangentAccum[i];

        if (glm::length(t) < 1e-6f)
        {
            vertices[i].tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
            continue;
        }

        t = glm::normalize(t - n * glm::dot(n, t));

        float handedness =
            glm::dot(glm::cross(n, t), bitangentAccum[i]) < 0.0f
            ? -1.0f
            : 1.0f;

        vertices[i].normal = glm::vec3(n);
        vertices[i].tangent = glm::vec4(t, handedness);
    }
}

MeshData ModelLoader::loadObj(const std::string& path) const
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warning;
    std::string error;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warning, &error, path.c_str()))
    {
        throw std::runtime_error(warning + error);
    }

    MeshData meshData;
    std::unordered_map<Vertex, uint32_t> uniqueVertices{};

    for (const auto& shape : shapes)
    {
        for (const auto& index : shape.mesh.indices)
        {
            Vertex vertex{};

            vertex.pos = {
                attrib.vertices[3 * index.vertex_index + 0],
                attrib.vertices[3 * index.vertex_index + 1],
                attrib.vertices[3 * index.vertex_index + 2]
            };

            if (index.normal_index >= 0)
            {
                vertex.normal = {
                    attrib.normals[3 * index.normal_index + 0],
                    attrib.normals[3 * index.normal_index + 1],
                    attrib.normals[3 * index.normal_index + 2]
                };
            }
            else
            {
                vertex.normal = { 0.0f, 0.0f, 1.0f };
            }

            if (index.texcoord_index >= 0)
            {
                vertex.texCoord = {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                };
            }
            else
            {
                vertex.texCoord = { 0.0f, 0.0f };
            }


            // -Fix


            vertex.tangent = { 1.0f, 0.0f, 0.0f, 1.0f };

            if (uniqueVertices.count(vertex) == 0)
            {
                uniqueVertices[vertex] = static_cast<uint32_t>(meshData.vertices.size());
                meshData.vertices.push_back(vertex);
            }

            meshData.indices.push_back(uniqueVertices[vertex]);
        }
    }

    computeTangents(meshData.vertices, meshData.indices);

    return meshData;
}

std::string ModelLoader::getFileExtension(const std::string& path)
{
    auto dot = path.find_last_of('.');
    if (dot == std::string::npos)
        return "";

    std::string ext = path.substr(dot + 1);

    std::transform(ext.begin(), ext.end(), ext.begin(),
        [](unsigned char c) { return std::tolower(c); });

    return ext;
}

MeshData ModelLoader::load(const std::string& path) const
{
    std::string ext = getFileExtension(path);

    if (ext == "obj")
    {
        return loadObj(path);
    }

    throw std::runtime_error("Unsupported model format: " + ext);
}