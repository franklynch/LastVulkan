#pragma once

#include <string>
#include <functional>

#include "Scene.hpp"
#include "Camera.hpp"
#include "Material.hpp"
#include "Texture2D.hpp"
#include "VulkanContext.hpp"
#include "RendererTypes.hpp"
#include "EditorUiState.hpp"
#include "MeshData.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace EditorPanels
{
      
    
    
    void drawRendererPanel(
        VulkanContext& vkContext,
        vk::Extent2D swapChainExtent,
        size_t swapChainImageCount,
        const Texture2D* texture,
        vk::SampleCountFlagBits msaaSamples,
        size_t gpuMeshCount,
        uint32_t totalVertexCount,
        uint32_t totalIndexCount,
        float frameTimeMs,
        float fps,
        const Scene& scene);

    

    void drawAnimationPanel(
        bool& animateModel,
        float& rotationSpeed);

    void drawScenePanel(
        Scene& scene,
        EditorUiState& uiState,
        GpuMesh& gpuMesh,
        Material& material,
        Camera& camera,
        std::function<void()> resetDefaultSceneLayoutFn,
        std::function<void()> focusSelectedRenderableFn);

    void drawSelectedMaterialPanel(
        const Renderable* selectedRenderable,
        Material* selectedMaterial,
        const Texture2D* selectedTexture,
        int selectedMaterialIndex);

  /*  void drawLightingPanel(
        glm::vec3& lightDirection,
        glm::vec3& lightColor,
        float& lightIntensity,
        glm::vec3& ambientColor,
        float& ambientIntensity);; */

    void drawLookDevPanel(
        glm::vec3& lightDirection,
        glm::vec3& lightColor,
        float& lightIntensity,
        glm::vec3& ambientColor,
        float& ambientIntensity,
        int& debugViewMode,
        bool& showSkybox,
        bool& enableIBL,
        bool& debugReflectionOnly,
        bool& debugSkyboxFaces,
        float& skyboxExposure,
        float& skyboxLod,
        float& iblIntensity,
        float& diffuseIBLIntensity,
        float& specularIBLIntensity,

        bool& toneMappingEnabled,
        bool& gammaEnabled,
        float& postExposure,

        bool& bloomEnabled,
        float& bloomThreshold,
        float& bloomKnee,
        float& bloomStrength,

        float& bloomIntensity,
        float& bloomUpsampleRadius,

        float& environmentRotationDegrees,
        bool& rotateSkybox,
        bool& rotateIBLLighting,

        bool& debugForceSpecularMip,
        float& debugSpecularMip,
        float& roughnessMipScale,
        float& roughnessMipBias,
        float maxPrefilterMip,

        const std::function<void()>& onResetEnvironment,
        const std::function<void()>& onResetIblCalibration);

    void drawCameraPanel(
        Camera& camera,
        float& cameraRadius,
        float& cameraYaw,
        float& cameraPitch,
        float& cameraFov,
        float& cameraNear,
        float& cameraFar,
        float& mouseOrbitSensitivity,
        float& mousePanSensitivity,
        float& mouseZoomSensitivity,
        float minCameraRadius,
        float maxCameraRadius);

    void drawAssetInspectionPanel(
        const Scene& scene,
        const EditorUiState& uiState,
        size_t gpuMeshCount,
        size_t textureCount,
        size_t materialCount,
        int selectedMaterialIndex,
        const Material* selectedMaterial,
        const Texture2D* selectedTexture);

   /* void drawLightingPanel(
        glm::vec3& lightDirection,
        glm::vec3& lightColor,
        glm::vec3& ambientColor); */

    void drawEnvironmentPanel(
        bool& showSkybox,
        bool& enableIBL,
        bool& debugReflectionOnly,
        float& skyboxExposure,
        float& skyboxLod,
        float& iblIntensity,
        float& diffuseIBLIntensity,
        float& specularIBLIntensity,
        float& environmentRotationDegrees,
        bool& rotateSkybox,
        bool& rotateIBLLighting,
        const std::function<void()>& onResetEnvironment
    );

    void drawPostProcessPanel(
        bool& toneMappingEnabled,
        bool& gammaEnabled,
        float& postExposure);

    void drawVerificationPanel(
        const Scene& scene,
        const EditorUiState& uiState,
        const std::string& currentModelPath,
        const Renderable* selectedRenderable,
        const Material* selectedMaterial,
        const Texture2D* baseColorTexture,
        const Texture2D* normalTexture,
        const Texture2D* metallicRoughnessTexture,
        const glm::vec3& lightDirection,
        const glm::vec3& lightColor,
        const glm::vec3& ambientColor);

    void drawSelectedMaterialPanel(
        const Renderable* selectedRenderable,
        Material* selectedMaterial,
        const Texture2D* selectedTexture,
        int selectedMaterialIndex);

    void drawDebugPanel(EditorUiState& uiState, bool wireframeSupported);

    void drawUboInspector(const UniformBufferObject& ubo);

    
}