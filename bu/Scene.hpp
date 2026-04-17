#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "Renderable.hpp"

class Scene
{
public:
    [[nodiscard]] std::vector<Renderable>& getRenderables()
    {
        return renderables;
    }

    [[nodiscard]] const std::vector<Renderable>& getRenderables() const
    {
        return renderables;
    }

    void clear()
    {
        renderables.clear();
    }

    template <typename... Args>
    Renderable& emplaceRenderable(Args&&... args)
    {
        renderables.emplace_back(std::forward<Args>(args)...);
        return renderables.back();
    }

    Renderable& addRenderable(GpuMesh& mesh, Material& material, const std::string& name = "Renderable")
    {
        renderables.emplace_back(mesh, material, name);
        return renderables.back();
    }

    Renderable* duplicateRenderable(size_t index, const std::string& newName)
    {
        Renderable* source = getRenderable(index);
        if (!source)
        {
            return nullptr;
        }

        renderables.emplace_back(source->getMesh(), source->getMaterial(), newName);
        renderables.back().getTransform() = source->getTransform();
        return &renderables.back();
    }

    bool removeRenderable(size_t index)
    {
        if (index >= renderables.size())
        {
            return false;
        }

        renderables.erase(renderables.begin() + static_cast<std::ptrdiff_t>(index));
        return true;
    }

    [[nodiscard]] Renderable* getRenderable(size_t index)
    {
        if (index >= renderables.size())
        {
            return nullptr;
        }

        return &renderables[index];
    }

    [[nodiscard]] const Renderable* getRenderable(size_t index) const
    {
        if (index >= renderables.size())
        {
            return nullptr;
        }

        return &renderables[index];
    }

    [[nodiscard]] Renderable* getSelectedRenderable(int selectedIndex)
    {
        if (selectedIndex < 0)
        {
            return nullptr;
        }

        return getRenderable(static_cast<size_t>(selectedIndex));
    }

    [[nodiscard]] const Renderable* getSelectedRenderable(int selectedIndex) const
    {
        if (selectedIndex < 0)
        {
            return nullptr;
        }

        return getRenderable(static_cast<size_t>(selectedIndex));
    }

    [[nodiscard]] bool empty() const
    {
        return renderables.empty();
    }

    [[nodiscard]] size_t size() const
    {
        return renderables.size();
    }

private:
    std::vector<Renderable> renderables;
};

