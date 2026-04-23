#pragma once

#include <cstdint>
#include <vector>

#include "RendererTypes.hpp"

struct MeshData
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    [[nodiscard]] bool empty() const
    {
        return vertices.empty() || indices.empty();
    }
};