#include "EditorPanels.hpp"

#include <cstdio>
#include <string>

#include "imgui.h"

namespace EditorPanels
{
    void EditorPanels::drawRendererPanel(
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
        const Scene& scene)
    {
        vk::PhysicalDeviceProperties props = vkContext.getPhysicalDevice().getProperties();

        ImGui::Text("Frame time: %.2f ms", frameTimeMs);
        ImGui::Text("FPS: %.1f", fps);
        ImGui::Text("Renderables: %u", static_cast<uint32_t>(scene.size()));

        ImGui::Separator();

        if (ImGui::CollapsingHeader("Renderer", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Text("GPU: %s", props.deviceName);
            ImGui::Text("API: %u.%u.%u",
                VK_API_VERSION_MAJOR(props.apiVersion),
                VK_API_VERSION_MINOR(props.apiVersion),
                VK_API_VERSION_PATCH(props.apiVersion));

            ImGui::Separator();
            ImGui::Text("Swapchain extent: %u x %u", swapChainExtent.width, swapChainExtent.height);
            ImGui::Text("Swapchain images: %u", static_cast<uint32_t>(swapChainImageCount));
            ImGui::Text("Texture: %s", texture ? texture->getSourcePath().c_str() : "<none>");
            ImGui::Text("MSAA samples: %d", static_cast<int>(msaaSamples));
            ImGui::Text("Mip levels: %u", texture ? texture->getMipLevels() : 0);
            ImGui::Text("GPU meshes: %u", static_cast<uint32_t>(gpuMeshCount));
            ImGui::Text("Vertices: %u", totalVertexCount);
            ImGui::Text("Indices: %u", totalIndexCount);
        }
    }


    void drawAnimationPanel(
        bool& animateModel,
        float& rotationSpeed)
    {
        if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen))
        {
            bool stopRotation = !animateModel;
            if (ImGui::Checkbox("Stop rotation", &stopRotation))
            {
                animateModel = !stopRotation;
            }

            ImGui::SliderFloat("Rotation speed", &rotationSpeed, 0.0f, 360.0f, "%.1f deg/s");
        }
    }

    void EditorPanels::drawScenePanel(
        Scene& scene,
        EditorUiState& uiState,
        GpuMesh& gpuMesh,
        Material& material,
        Camera& camera,
        std::function<void()> resetDefaultSceneLayoutFn,
        std::function<void()> focusSelectedRenderableFn)
    {
        if (!ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen))
        {
            return;
        }

        auto& renderables = scene.getRenderables();

        ImGui::Text("Renderable count: %u", static_cast<uint32_t>(scene.size()));

        if (ImGui::Button("Add Renderable"))
        {
            Renderable& created = scene.addRenderable(
                gpuMesh,
                material,
                "Renderable " + std::to_string(scene.size()));

            created.getTransform().position = camera.getTarget();
            created.getTransform().rotation = { 0.0f, 0.0f, 0.0f };
            created.getTransform().scale = { 1.0f, 1.0f, 1.0f };

            uiState.selectedRenderableIndex = static_cast<int>(scene.size()) - 1;
        }

        ImGui::SameLine();

        bool hasSelection =
            uiState.selectedRenderableIndex >= 0 &&
            uiState.selectedRenderableIndex < static_cast<int>(scene.size());

        if (!hasSelection)
        {
            ImGui::BeginDisabled();
        }

        if (ImGui::Button("Duplicate Selected"))
        {
            Renderable* duplicated = scene.duplicateRenderable(
                static_cast<size_t>(uiState.selectedRenderableIndex),
                "Renderable " + std::to_string(scene.size()));

            if (duplicated)
            {
                duplicated->getTransform().position.x += 0.5f;
                duplicated->getTransform().position.y += 0.5f;
                uiState.selectedRenderableIndex = static_cast<int>(scene.size()) - 1;
            }
        }

        ImGui::SameLine();

        if (ImGui::Button("Delete Selected"))
        {
            if (scene.removeRenderable(static_cast<size_t>(uiState.selectedRenderableIndex)))
            {
                if (scene.empty())
                {
                    uiState.selectedRenderableIndex = -1;
                }
                else if (uiState.selectedRenderableIndex >= static_cast<int>(scene.size()))
                {
                    uiState.selectedRenderableIndex = static_cast<int>(scene.size()) - 1;
                }
            }
        }

        ImGui::SameLine();

        if (ImGui::Button("Focus Selected"))
        {
            focusSelectedRenderableFn();
        }

        if (!hasSelection)
        {
            ImGui::EndDisabled();
        }

        if (ImGui::Button("Reset Scene Layout"))
        {
            resetDefaultSceneLayoutFn();
            uiState.selectedRenderableIndex = scene.empty() ? -1 : 0;
        }

        ImGui::Separator();

        for (size_t i = 0; i < renderables.size(); ++i)
        {
            bool selected = (uiState.selectedRenderableIndex == static_cast<int>(i));
            if (ImGui::Selectable(renderables[i].getName().c_str(), selected))
            {
                uiState.selectedRenderableIndex = static_cast<int>(i);
            }
        }

        Renderable* selectedRenderable = scene.getSelectedRenderable(uiState.selectedRenderableIndex);

        if (selectedRenderable)
        {
            auto& transform = selectedRenderable->getTransform();

            ImGui::Separator();
            ImGui::Text("Selected: %s", selectedRenderable->getName().c_str());

            char nameBuffer[128]{};
            std::snprintf(nameBuffer, sizeof(nameBuffer), "%s", selectedRenderable->getName().c_str());

            if (ImGui::InputText("Name", nameBuffer, sizeof(nameBuffer)))
            {
                selectedRenderable->setName(nameBuffer);
            }

            ImGui::DragFloat3("Position", &transform.position.x, 0.01f);
            ImGui::DragFloat3("Rotation", &transform.rotation.x, 0.01f);
            ImGui::DragFloat3("Scale", &transform.scale.x, 0.01f, 0.01f, 100.0f);

            if (ImGui::Button("Reset Selected Transform"))
            {
                transform.position = { 0.0f, 0.0f, 0.0f };
                transform.rotation = { 0.0f, 0.0f, 0.0f };
                transform.scale = { 1.0f, 1.0f, 1.0f };
            }
        }
    }

    void drawCameraPanel(
        Camera&,
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
        float maxCameraRadius)
    {
        if (!ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
        {
            return;
        }

        ImGui::DragFloat("Radius", &cameraRadius, 0.05f, minCameraRadius, maxCameraRadius);
        ImGui::DragFloat("Yaw", &cameraYaw, 0.01f);
        ImGui::DragFloat("Pitch", &cameraPitch, 0.01f, -1.55f, 1.55f);
        ImGui::DragFloat("FOV", &cameraFov, 0.1f, 10.0f, 120.0f);
        ImGui::DragFloat("Near", &cameraNear, 0.01f, 0.01f, 10.0f);
        ImGui::DragFloat("Far", &cameraFar, 0.1f, 1.0f, 500.0f);

        ImGui::DragFloat("Orbit sensitivity", &mouseOrbitSensitivity, 0.001f, 0.001f, 0.1f);
        ImGui::DragFloat("Pan sensitivity", &mousePanSensitivity, 0.0005f, 0.0001f, 0.1f);
        ImGui::DragFloat("Zoom sensitivity", &mouseZoomSensitivity, 0.01f, 0.01f, 10.0f);

        if (ImGui::Button("Reset camera"))
        {
            cameraRadius = 3.0f;
            cameraYaw = 0.0f;
            cameraPitch = 0.5f;
            cameraFov = 45.0f;
            cameraNear = 0.1f;
            cameraFar = 10.0f;
            mouseOrbitSensitivity = 0.01f;
            mousePanSensitivity = 0.005f;
            mouseZoomSensitivity = 0.5f;
        }
    }

    
    void EditorPanels::drawAssetInspectionPanel(
        const Scene& scene,
        const EditorUiState& uiState,
        size_t gpuMeshCount,
        size_t textureCount,
        size_t materialCount,
        int selectedMaterialIndex,
        const Material* selectedMaterial,
        const Texture2D* selectedTexture)
    {
        if (!ImGui::CollapsingHeader("Assets", ImGuiTreeNodeFlags_DefaultOpen))
        {
            return;
        }

        ImGui::Text("Scene renderables: %u", static_cast<uint32_t>(scene.size()));
        ImGui::Text("GPU meshes: %u", static_cast<uint32_t>(gpuMeshCount));
        ImGui::Text("Textures: %u", static_cast<uint32_t>(textureCount));
        ImGui::Text("Materials: %u", static_cast<uint32_t>(materialCount));

        const Renderable* selected = scene.getSelectedRenderable(uiState.selectedRenderableIndex);

        ImGui::Separator();

        if (!selected)
        {
            ImGui::TextUnformatted("No renderable selected.");
            return;
        }

        ImGui::Text("Selected renderable: %s", selected->getName().c_str());
        ImGui::Text("Selected material index: %d", selectedMaterialIndex);

        if (selectedMaterial)
        {
            glm::vec4 color = selectedMaterial->getBaseColorFactor();
            ImGui::Text("Base color factor: %.3f %.3f %.3f %.3f",
                color.r, color.g, color.b, color.a);
        }

        if (selectedTexture)
        {
            ImGui::Text("Texture source: %s", selectedTexture->getSourcePath().c_str());
            ImGui::Text("Texture mip levels: %u", selectedTexture->getMipLevels());
        }
        else
        {
            ImGui::TextUnformatted("No texture bound.");
        }
    }
    
    void EditorPanels::drawDebugPanel(EditorUiState& uiState, bool wireframeSupported)
    {
        if (!ImGui::CollapsingHeader("Debug"))
        {
            return;
        }

        ImGui::Checkbox("Show ImGui demo window", &uiState.showDemoWindow);

        if (!wireframeSupported)
        {
            ImGui::BeginDisabled();
        }

        ImGui::Checkbox("Wireframe", &uiState.wireframeRequested);

        if (!wireframeSupported)
        {
            ImGui::EndDisabled();
            ImGui::TextUnformatted("Wireframe is not supported on this GPU.");
        }
        else
        {
            ImGui::TextWrapped("Wireframe mode uses a separate graphics pipeline.");
        }
    }
}