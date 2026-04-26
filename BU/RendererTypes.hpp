#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <functional>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

inline constexpr uint32_t WIDTH = 800;
inline constexpr uint32_t HEIGHT = 600;

inline const std::string MODEL_PATH = "models/viking_room.obj";
inline const std::string TEXTURE_PATH = "models/BoxTextured/glTF/CesiumLogoFlat.png";




inline constexpr int MAX_FRAMES_IN_FLIGHT = 2;

struct Vertex
{
    glm::vec3 pos{ 0.0f };
    glm::vec3 normal{ 0.0f, 0.0f, 1.0f };
    glm::vec2 texCoord{ 0.0f };
    glm::vec4 tangent{ 1.0f, 0.0f, 0.0f, 1.0f };

    static vk::VertexInputBindingDescription getBindingDescription()
    {
        return vk::VertexInputBindingDescription(
            0,
            sizeof(Vertex),
            vk::VertexInputRate::eVertex
        );
    }

    static std::array<vk::VertexInputAttributeDescription, 4> getAttributeDescriptions()
    {
        return {
            vk::VertexInputAttributeDescription(
                0,
                0,
                vk::Format::eR32G32B32Sfloat,
                offsetof(Vertex, pos)
            ),
            vk::VertexInputAttributeDescription(
                1,
                0,
                vk::Format::eR32G32B32Sfloat,
                offsetof(Vertex, normal)
            ),
            vk::VertexInputAttributeDescription(
                2,
                0,
                vk::Format::eR32G32Sfloat,
                offsetof(Vertex, texCoord)
            ),
            vk::VertexInputAttributeDescription(
                3,
                0,
                vk::Format::eR32G32B32A32Sfloat,
                offsetof(Vertex, tangent)
            )
        };
    }

    bool operator==(const Vertex& other) const
    {
        return pos == other.pos &&
            normal == other.normal &&
            texCoord == other.texCoord &&
            tangent == other.tangent;
    }
};

namespace std
{
    template <>
    struct hash<Vertex>
    {
        size_t operator()(Vertex const& vertex) const noexcept
        {
            size_t h1 = hash<glm::vec3>()(vertex.pos);
            size_t h2 = hash<glm::vec3>()(vertex.normal);
            size_t h3 = hash<glm::vec2>()(vertex.texCoord);
            size_t h4 = hash<glm::vec4>()(vertex.tangent);

            return (((h1 ^ (h2 << 1)) >> 1) ^ (h3 << 1)) ^ (h4 << 2);
        }
    };
}

struct UniformBufferObject
{
    glm::mat4 view{ 1.0f };
    glm::mat4 proj{ 1.0f };

    glm::mat4 invView;
    glm::mat4 invProj;

    glm::vec4 lightDirection{ -0.5f, -1.0f, -0.3f, 0.0f };
    glm::vec4 lightColor{ 1.0f, 1.0f, 1.0f, 1.0f };
    glm::vec4 ambientColor{ 0.15f, 0.15f, 0.15f, 1.0f };
    glm::vec4 cameraPosition{ 0.0f, 0.0f, 0.0f, 1.0f };

    glm::vec4 environmentParams0; // x=skyboxExposure, y=skyboxLod, z=iblIntensity, w=showSkybox
    glm::vec4 environmentParams1; // x=diffuseIBLIntensity, y=specularIBLIntensity, z=debugReflectionOnly, w=enableIBL
    glm::vec4 postProcessParams; // x=postExposure, y=toneMappingEnabled, z=gammaEnabled, w=unusedglm::vec4 environmentControlParams; // x=rotateSkybox, y=rotateIBLLighting, z/w unused
    glm::vec4 environmentControlParams; // x=rotateSkybox, y=rotateIBLLighting, z/w unused  

};

struct PushConstantData
{
    glm::mat4 model{ 1.0f };
    glm::mat4 normalMatrix{ 1.0f };
    glm::vec4 baseColorFactor{ 1.0f, 1.0f, 1.0f, 1.0f };
    glm::vec4 materialParams{ 1.0f, 1.0f, 1.0f, 0.0f }; // x=metallic, y=roughness, z=normalScale, w=alphaCutoff
    glm::vec4 alphaModeParams{ 0.0f, 0.0f, 0.0f, 0.0f }; // x = isMask, y = isBlend (future), z/w unused
};

struct EquirectToCubePushConstants
{
    glm::mat4 viewProj{ 1.0f };
};

struct PrefilterPushConstants
{
    glm::mat4 viewProj{ 1.0f };      // 64 bytes
    glm::vec4 params{ 0.0f };        // x = roughness, y/z/w unused
};