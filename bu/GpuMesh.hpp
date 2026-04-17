#pragma once

#include "RendererTypes.hpp"
#include "VulkanContext.hpp"
#include "BufferUtils.hpp"

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

class GpuMesh
{
public:
    GpuMesh(VulkanContext& vkContext,
        BufferUtils& bufferUtils,
        const std::vector<Vertex>& vertices,
        const std::vector<uint32_t>& indices);

    GpuMesh(const GpuMesh&) = delete;
    GpuMesh& operator=(const GpuMesh&) = delete;

    [[nodiscard]] const vk::raii::Buffer& getVertexBuffer() const { return vertexBuffer; }
    [[nodiscard]] const vk::raii::Buffer& getIndexBuffer() const { return indexBuffer; }
    [[nodiscard]] uint32_t getVertexCount() const { return vertexCount; }
    [[nodiscard]] uint32_t getIndexCount() const { return indexCount; }

private:
    void createVertexBuffer(const std::vector<Vertex>& vertices);
    void createIndexBuffer(const std::vector<uint32_t>& indices);

private:
    VulkanContext& vkContext;
    BufferUtils& bufferUtils;

    vk::raii::Buffer vertexBuffer = nullptr;
    vk::raii::DeviceMemory vertexBufferMemory = nullptr;

    vk::raii::Buffer indexBuffer = nullptr;
    vk::raii::DeviceMemory indexBufferMemory = nullptr;

    uint32_t indexCount = 0;
	uint32_t vertexCount = 0;
};