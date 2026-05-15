#include "GpuMesh.hpp"
#include <cstring>

GpuMesh::GpuMesh(VulkanContext& vkContext,
    BufferUtils& bufferUtils,
    const std::vector<Vertex>& vertices,
    const std::vector<uint32_t>& indices)
    : vkContext(vkContext)
    , bufferUtils(bufferUtils)
{
    if (vertices.empty())
    {
        throw std::runtime_error("GpuMesh: vertex data is empty");
    }

    if (indices.empty())
    {
        throw std::runtime_error("GpuMesh: index data is empty");
    }


    createVertexBuffer(vertices);
    createIndexBuffer(indices);
    indexCount = static_cast<uint32_t>(indices.size());
    vertexCount = static_cast<uint32_t>(vertices.size());
}

void GpuMesh::createVertexBuffer(const std::vector<Vertex>& vertices)
{
    if (vertices.empty())
    {
        throw std::runtime_error("GpuMesh::createVertexBuffer: vertices empty");
    }
    
    vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

    GpuBuffer stagingBuffer;

    bufferUtils.createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer);


    std::memcpy(
        stagingBuffer.mapped,
        vertices.data(),
        static_cast<size_t>(bufferSize));

    bufferUtils.createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        vertexBuffer);
        
    
    bufferUtils.copyBuffer(stagingBuffer.buffer, vertexBuffer.buffer, bufferSize);

    bufferUtils.destroyBuffer(stagingBuffer);
}

void GpuMesh::createIndexBuffer(const std::vector<uint32_t>& indices)
{
    if (indices.empty())
    {
        throw std::runtime_error("GpuMesh::createIndexBuffer: indices empty");
    }
    
    vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    GpuBuffer stagingBuffer;

    bufferUtils.createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer);

    
    std::memcpy(
        stagingBuffer.mapped,
        indices.data(),
        static_cast<size_t>(bufferSize));
        
        

    bufferUtils.createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        indexBuffer);

    bufferUtils.copyBuffer(stagingBuffer.buffer, indexBuffer.buffer, bufferSize);

    bufferUtils.destroyBuffer(stagingBuffer);
}
