#pragma once

#include <string>
#include "MeshData.hpp"

class ModelLoader
{
public:
    [[nodiscard]] MeshData load(const std::string& path) const;

private:
    [[nodiscard]] MeshData loadObj(const std::string& path) const;

    static std::string getFileExtension(const std::string& path);
};