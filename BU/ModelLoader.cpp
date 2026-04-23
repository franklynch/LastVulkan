#include "ModelLoader.hpp"

#include <stdexcept>
#include <algorithm>
#include <unordered_map>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

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

            vertex.tangent = { 1.0f, 0.0f, 0.0f, 1.0f };

            if (uniqueVertices.count(vertex) == 0)
            {
                uniqueVertices[vertex] = static_cast<uint32_t>(meshData.vertices.size());
                meshData.vertices.push_back(vertex);
            }

            meshData.indices.push_back(uniqueVertices[vertex]);
        }
    }

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