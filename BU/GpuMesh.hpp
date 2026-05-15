#pragma once

#include "RendererTypes.hpp"
#include "VulkanContext.hpp"
#include "BufferUtils.hpp"

#include "GpuResources.hpp"

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

    vk::Buffer getVertexBuffer() const
    {
        return vertexBuffer.buffer;
    }

    vk::Buffer getIndexBuffer() const
    {
        return indexBuffer.buffer;
    }

    [[nodiscard]] uint32_t getVertexCount() const { return vertexCount; }
    [[nodiscard]] uint32_t getIndexCount() const { return indexCount; }

private:
    void createVertexBuffer(const std::vector<Vertex>& vertices);
    void createIndexBuffer(const std::vector<uint32_t>& indices);

private:
    VulkanContext& vkContext;
    BufferUtils& bufferUtils;

    GpuBuffer vertexBuffer;
    GpuBuffer indexBuffer;

    uint32_t indexCount = 0;
	uint32_t vertexCount = 0;
};