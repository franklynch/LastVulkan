#pragma once

#include <functional>
#include <string>

#include "GpuMesh.hpp"
#include "Material.hpp"
#include "Transform.hpp"

class Renderable
{
public:
    Renderable(GpuMesh& mesh, Material& material, const std::string& name = "Renderable")
        : mesh(mesh)
        , material(material)
        , name(name)
    {
    }

    [[nodiscard]] GpuMesh& getMesh() const { return mesh.get(); }
    [[nodiscard]] Material& getMaterial() const { return material.get(); }

    [[nodiscard]] Transform& getTransform() { return transform; }
    [[nodiscard]] const Transform& getTransform() const { return transform; }

    [[nodiscard]] const std::string& getName() const { return name; }
    void setName(const std::string& value) { name = value; }

private:
    std::reference_wrapper<GpuMesh> mesh;
    std::reference_wrapper<Material> material;
    Transform transform;
    std::string name;
};