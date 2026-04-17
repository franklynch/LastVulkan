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


    void drawDebugPanel(EditorUiState& uiState, bool wireframeSupported);

    
}