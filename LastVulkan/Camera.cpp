#include "Camera.hpp"
#include <algorithm>

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

void Camera::frameBounds(const glm::vec3& minBounds, const glm::vec3& maxBounds)
{
    glm::vec3 center = (minBounds + maxBounds) * 0.5f;
    glm::vec3 size = maxBounds - minBounds;

    float radius = glm::length(size) * 0.5f;

    if (radius < 0.001f)
        radius = 1.0f;

    target = center;

    float distance = radius / std::tan(glm::radians(fov) * 0.5f);
    distance *= 1.5f; // padding

    setOrbit(
        distance,
        glm::radians(270.0f),
        glm::radians(25.0f)
    );

    nearPlane = std::max(0.01f, distance - radius * 4.0f);
    farPlane = distance + radius * 6.0f;
}

void Camera::setOrbit(float radius, float yaw, float pitch)
{
    pitch = glm::clamp(
        pitch,
        glm::radians(-89.0f),
        glm::radians(89.0f)
    );

    glm::vec3 offset;
    offset.x = radius * std::cos(pitch) * std::cos(yaw);
    offset.y = radius * std::cos(pitch) * std::sin(yaw);
    offset.z = radius * std::sin(pitch);

    position = target + offset;
}

