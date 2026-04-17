#include "Camera.hpp"

#include <cmath>

Camera::Camera() {}

void Camera::setPosition(const glm::vec3& pos)
{
    position = pos;
}

void Camera::setTarget(const glm::vec3& tgt)
{
    target = tgt;
}

glm::mat4 Camera::getViewMatrix() const
{
    return glm::lookAt(position, target, up);
}

glm::mat4 Camera::getProjectionMatrix(float aspect) const
{
    glm::mat4 proj = glm::perspective(
        glm::radians(fov),
        aspect,
        nearPlane,
        farPlane);

    proj[1][1] *= -1.0f;
    return proj;
}

void Camera::setFov(float degrees)
{
    fov = degrees;
}

void Camera::setNearFar(float nearP, float farP)
{
    nearPlane = nearP;
    farPlane = farP;
}

void Camera::setOrbit(float radius, float yaw, float pitch)
{
    glm::vec3 offset;
    offset.x = radius * std::cos(pitch) * std::cos(yaw);
    offset.y = radius * std::cos(pitch) * std::sin(yaw);
    offset.z = radius * std::sin(pitch);

    position = target + offset;
}