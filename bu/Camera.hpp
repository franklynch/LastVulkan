#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera
{
public:
    Camera();

    void setPosition(const glm::vec3& pos);
    void setTarget(const glm::vec3& tgt);

    [[nodiscard]] const glm::vec3& getPosition() const { return position; }
    [[nodiscard]] const glm::vec3& getTarget() const { return target; }

    void offsetPosition(const glm::vec3& delta) { position += delta; }
    void offsetTarget(const glm::vec3& delta) { target += delta; }

    [[nodiscard]] glm::mat4 getViewMatrix() const;
    [[nodiscard]] glm::mat4 getProjectionMatrix(float aspect) const;

    void setFov(float degrees);
    void setNearFar(float nearPlane, float farPlane);

    void setOrbit(float radius, float yaw, float pitch);

private:
    glm::vec3 position{ 2.0f, 2.0f, 2.0f };
    glm::vec3 target{ 0.0f, 0.0f, 0.0f };
    glm::vec3 up{ 0.0f, 0.0f, 1.0f };

    float fov = 45.0f;
    float nearPlane = 0.1f;
    float farPlane = 10.0f;
};