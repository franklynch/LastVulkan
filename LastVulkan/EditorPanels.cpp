#include "EditorPanels.hpp"

#include <cstdio>
#include <string>
#include <functional>

#include "imgui.h"

#include <cfloat>

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

    void EditorPanels::drawSelectedMaterialPanel(
        const Renderable* selectedRenderable,
        Material* selectedMaterial,
        const Texture2D* selectedTexture,
        int selectedMaterialIndex)
    {
        if (!ImGui::CollapsingHeader("Selected Material", ImGuiTreeNodeFlags_DefaultOpen))
        {
            return;
        }

        if (!selectedRenderable || !selectedMaterial)
        {
            ImGui::TextUnformatted("No renderable selected.");
            return;
        }

        float metallic = selectedMaterial->getMetallicFactor();
        if (ImGui::SliderFloat("Metallic", &metallic, 0.0f, 1.0f))
        {
            selectedMaterial->setMetallicFactor(metallic);
        }

        float roughness = selectedMaterial->getRoughnessFactor();
        if (ImGui::SliderFloat("Roughness", &roughness, 0.0f, 1.0f))
        {
            selectedMaterial->setRoughnessFactor(roughness);
        }


        ImGui::Text("Renderable: %s", selectedRenderable->getName().c_str());
        ImGui::Text("Material index: %d", selectedMaterialIndex);

        ImGui::Text("Material name: %s",
            selectedMaterial->getName().empty() ? "<unnamed>" : selectedMaterial->getName().c_str());

        ImGui::Text("Double sided: %s",
            selectedMaterial->isDoubleSided() ? "true" : "false");

        ImGui::Text("Has normal map: %s", selectedMaterial->hasNormalTexture() ? "true" : "false");

        ImGui::Text("Has metallic-roughness map: %s",
            selectedMaterial->hasMetallicRoughnessTexture() ? "true" : "false");

        ImGui::Text("Alpha mode: %s", selectedMaterial->getAlphaMode().c_str());
        ImGui::Text("Alpha cutoff: %.3f", selectedMaterial->getAlphaCutoff());

      

        float normalScale = selectedMaterial->getNormalScale();
        if (ImGui::SliderFloat("Normal Scale", &normalScale, 0.0f, 2.0f))
        {
            selectedMaterial->setNormalScale(normalScale);
        }

        if (selectedMaterial && selectedMaterial->getNormalTexture())
        {
            ImGui::Text("Normal Texture: %s",
                selectedMaterial->getNormalTexture()->getSourcePath().c_str());
        }

        //THis the working example 
        glm::vec4 color = selectedMaterial->getBaseColorFactor();
        if (ImGui::ColorEdit4("Base Color Factor", &color.x))
        {
            selectedMaterial->setBaseColorFactor(color);
        }

        if (selectedTexture)
        {
            ImGui::Text("Texture: %s", selectedTexture->getSourcePath().c_str());
            ImGui::Text("Mip levels: %u", selectedTexture->getMipLevels());
        }
        else
        {
            ImGui::TextUnformatted("No texture bound.");
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
        
        ImGui::Separator();

        if (ImGui::CollapsingHeader("Sensitivity", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::DragFloat("Orbit sensitivity", &mouseOrbitSensitivity, 0.001f, 0.001f, 0.1f);
            ImGui::DragFloat("Pan sensitivity", &mousePanSensitivity, 0.0005f, 0.0001f, 0.1f);
            ImGui::DragFloat("Zoom sensitivity", &mouseZoomSensitivity, 0.01f, 0.01f, 10.0f);
        }

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

    void EditorPanels::drawLightingPanel(
        glm::vec3& lightDirection,  
        glm::vec3& lightColor,
        glm::vec3& ambientColor)
    {
        if (!ImGui::CollapsingHeader("Lighting", ImGuiTreeNodeFlags_DefaultOpen))
        {
            return;
        }

        ImGui::DragFloat3("Light Direction", &lightDirection.x, 0.01f);
        ImGui::ColorEdit3("Light Color", &lightColor.x);
        ImGui::ColorEdit3("Ambient Color", &ambientColor.x);

        if (ImGui::Button("Reset Lighting"))
        {
            lightDirection = glm::normalize(glm::vec3(-0.5f, -1.0f, -0.3f));
            lightColor = glm::vec3(1.0f, 1.0f, 1.0f);
            ambientColor = glm::vec3(0.15f, 0.15f, 0.15f);
        }
    }

    void EditorPanels::drawEnvironmentPanel(
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
    )

    {
        if (ImGui::CollapsingHeader("Environment / IBL", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Checkbox("Show Skybox", &showSkybox);
            ImGui::Checkbox("Enable IBL", &enableIBL);
            ImGui::Checkbox("Reflection Debug", &debugReflectionOnly);

            ImGui::SliderFloat("Skybox Exposure", &skyboxExposure, 0.0f, 8.0f, "%.2f");
            ImGui::SliderFloat("Skybox LOD", &skyboxLod, 0.0f, 8.0f, "%.2f");

            ImGui::Separator();

            ImGui::SliderFloat("IBL Intensity", &iblIntensity, 0.0f, 4.0f, "%.2f");
            ImGui::SliderFloat("Diffuse IBL", &diffuseIBLIntensity, 0.0f, 4.0f, "%.2f");
            ImGui::SliderFloat("Specular IBL", &specularIBLIntensity, 0.0f, 4.0f, "%.2f");

            ImGui::Separator();

            ImGui::SliderFloat("Environment Rotation", &environmentRotationDegrees, -180.0f, 180.0f, "%.1f deg");
            ImGui::Checkbox("Rotate Skybox", &rotateSkybox);
            ImGui::Checkbox("Rotate IBL Lighting", &rotateIBLLighting);

            if (ImGui::Button("Reset Environment"))
            {
                onResetEnvironment();
            }

            
        }
    }

    void EditorPanels::drawPostProcessPanel(
        bool& toneMappingEnabled,
        bool& gammaEnabled,
        float& postExposure)
    {
        if (ImGui::CollapsingHeader("Post Process", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Checkbox("Tone Mapping", &toneMappingEnabled);
            ImGui::Checkbox("Gamma Correction", &gammaEnabled);
            ImGui::SliderFloat("Post Exposure", &postExposure, 0.0f, 4.0f, "%.2f");
        }
    }

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
        const glm::vec3& ambientColor)
    {
        if (!ImGui::CollapsingHeader("Verification", ImGuiTreeNodeFlags_DefaultOpen))
            return;

        ImGui::Separator();
        ImGui::Text("Model");
        ImGui::TextWrapped("%s", currentModelPath.c_str());

        ImGui::Separator();
        ImGui::Text("Scene");
        ImGui::Text("Renderables: %u", static_cast<uint32_t>(scene.size()));

        ImGui::Separator();
        ImGui::Text("Selection");



        ImGui::Text("Model: %s", currentModelPath.c_str());


        if (selectedRenderable)
        {
            ImGui::Text("Renderable: %s", selectedRenderable->getName().c_str());

            const Transform& transform = selectedRenderable->getTransform();

            ImGui::Text("Position: %.2f %.2f %.2f",
                transform.position.x,
                transform.position.y,
                transform.position.z);

            ImGui::Text("Rotation: %.2f %.2f %.2f",
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z);

            ImGui::Text("Scale: %.2f %.2f %.2f",
                transform.scale.x,
                transform.scale.y,
                transform.scale.z);
        }
        else
        {
            ImGui::Text("Renderable: <none>");
        }

        if (selectedMaterial)
        {
            ImGui::Text("Material: %s", selectedMaterial->getName().c_str());
        }
        else
        {
            ImGui::Text("Material: <none>");
        }

        ImGui::Separator();
        ImGui::Text("Textures");

        ImGui::Text("Base Color: %s",
            baseColorTexture ? baseColorTexture->getSourcePath().c_str() : "<none>");

        ImGui::Text("Normal: %s",
            normalTexture ? normalTexture->getSourcePath().c_str() : "<none>");

        ImGui::Text("Metal/Rough: %s",
            metallicRoughnessTexture ? metallicRoughnessTexture->getSourcePath().c_str() : "<none>");

        ImGui::Separator();
        ImGui::Text("Material Params");

        if (selectedMaterial)
        {
            auto baseColor = selectedMaterial->getBaseColorFactor();

            ImGui::Text("BaseColorFactor: %.2f %.2f %.2f %.2f",
                baseColor.r, baseColor.g, baseColor.b, baseColor.a);

            ImGui::Text("Metallic: %.2f", selectedMaterial->getMetallicFactor());
            ImGui::Text("Roughness: %.2f", selectedMaterial->getRoughnessFactor());
            ImGui::Text("Normal Scale: %.2f", selectedMaterial->getNormalScale());
            ImGui::Text("Double Sided: %s", selectedMaterial->isDoubleSided() ? "Yes" : "No");
        }

        ImGui::Separator();
        ImGui::Text("Lighting");

        ImGui::Text("Light Dir: %.2f %.2f %.2f",
            lightDirection.x, lightDirection.y, lightDirection.z);

        ImGui::Text("Light Color: %.2f %.2f %.2f",
            lightColor.r, lightColor.g, lightColor.b);

        ImGui::Text("Ambient: %.2f %.2f %.2f",
            ambientColor.r, ambientColor.g, ambientColor.b);

        
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

            ImGui::Text("Material name: %s",
                selectedMaterial->getName().empty() ? "<unnamed>" : selectedMaterial->getName().c_str());

            ImGui::Text("Double sided: %s",
                selectedMaterial->isDoubleSided() ? "true" : "false");
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