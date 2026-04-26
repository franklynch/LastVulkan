#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct Transform
{
    glm::vec3 position{ 0.0f, 0.0f, 0.0f };
    glm::vec3 rotation{ 0.0f, 0.0f, 0.0f }; // radians
    glm::vec3 scale{ 1.0f, 1.0f, 1.0f };

    // NEW
    bool useMatrixOverride{ false };
    glm::mat4 matrixOverride{ 1.0f };

    [[nodiscard]] glm::mat4 toMatrix() const
    {
        if (useMatrixOverride)
        {
            return matrixOverride;
        }

        glm::mat4 model{ 1.0f };

        model = glm::translate(model, position);
        model = glm::rotate(model, rotation.x, glm::vec3(1, 0, 0));
        model = glm::rotate(model, rotation.y, glm::vec3(0, 1, 0));
        model = glm::rotate(model, rotation.z, glm::vec3(0, 0, 1));
        model = glm::scale(model, scale);

        return model;
    }
};