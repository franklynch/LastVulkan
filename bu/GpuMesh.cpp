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

    vk::raii::Buffer stagingBuffer{ nullptr };
    vk::raii::DeviceMemory stagingMemory{ nullptr };

    bufferUtils.createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer,
        stagingMemory);

    void* data = stagingMemory.mapMemory(0, bufferSize);
    std::memcpy(data, vertices.data(), static_cast<size_t>(bufferSize));
    stagingMemory.unmapMemory();

    bufferUtils.createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        vertexBuffer,
        vertexBufferMemory);

    bufferUtils.copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
}

void GpuMesh::createIndexBuffer(const std::vector<uint32_t>& indices)
{
    if (indices.empty())
    {
        throw std::runtime_error("GpuMesh::createIndexBuffer: indices empty");
    }
    
    vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    vk::raii::Buffer stagingBuffer{ nullptr };
    vk::raii::DeviceMemory stagingMemory{ nullptr };

    bufferUtils.createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer,
        stagingMemory);

    void* data = stagingMemory.mapMemory(0, bufferSize);
    std::memcpy(data, indices.data(), static_cast<size_t>(bufferSize));
    stagingMemory.unmapMemory();

    bufferUtils.createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        indexBuffer,
        indexBufferMemory);

    bufferUtils.copyBuffer(stagingBuffer, indexBuffer, bufferSize);
}
