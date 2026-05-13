#pragma once



#include <memory>
#include <string>
#include <array>

#include <vector>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include <glm/glm.hpp>

#include "Window.hpp"
#include "VulkanContext.hpp"

#include "SwapchainManager.hpp"

#include "RendererTypes.hpp"
#include "BufferUtils.hpp"
#include "ImageUtils.hpp"
#include "MeshData.hpp"



#include "GpuMesh.hpp"

#include "Renderable.hpp"
#include "Camera.hpp"
#include "Scene.hpp"


#include "InputState.hpp"


#include "EnvironmentSystem.hpp"  
#include "SceneRenderer.hpp"
#include "DescriptorManager.hpp"
#include "MaterialSystem.hpp"
#include "EditorUiSystem.hpp"






#include "PostProcessRenderer.hpp"
#include "RenderTargets.hpp"
#include "FrameResources.hpp"
#include "ScenePipelines.hpp"
#include "GltfSceneLoader.hpp"





class Renderer
{
public:
    Renderer(Window& window, VulkanContext& vkContext);

    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void init();

    void cleanupSwapChain();
    void recreateSwapChain();
  
    void drawFrame();

    void createUniformBuffers();
   
    void resetDefaultSceneLayout();
    
    void updateCameraControls(
        const InputState& input);

    void updateUniformBuffer(uint32_t currentFrame);

    void recordCommandBuffer(uint32_t imageIndex);

  



private:
    Window&                 window;
    VulkanContext&          vkContext;  
    
    BufferUtils             bufferUtils;
    ImageUtils              imageUtils;
    GltfSceneLoader         gltfSceneLoader;
    SwapchainManager        swapchain;
    RenderTargets           renderTargets;
    FrameResources          frameResources;
    ScenePipelines          scenePipelines;
    EnvironmentSystem       environmentSystem;
    SceneRenderer           sceneRenderer;
    DescriptorManager       descriptorManager;
    MaterialSystem          materialSystem;
    EditorUiSystem          editorUi;
    

    
    Camera                  camera;
    UniformBufferObject     lastUbo;
    Scene                   scene;


    void clearSceneResources();
    
    void setupCameraDefaults();
    
    std::string currentModelPath;


    struct AcquiredImage
    {
        uint32_t imageIndex = 0;
        bool valid = false;
    };


    AcquiredImage acquireSwapchainImage(uint32_t currentFrame);

    void updateFrameTiming();
    void updateEditorUiFrame();

    void waitForSwapchainImageFence(
        uint32_t imageIndex,
        uint32_t currentFrame);

    void submitCommandBuffer(
        uint32_t currentFrame,
        uint32_t imageIndex,
        vk::raii::CommandBuffer& commandBuffer);

    bool presentFrame(uint32_t imageIndex);

    void recordScenePass(
        vk::raii::CommandBuffer& commandBuffer,
        uint32_t imageIndex);

    SceneRenderer::SceneRenderContext buildSceneRenderContext() const;

    void recordBloomPass(
        vk::raii::CommandBuffer& commandBuffer);

    void recordFinalCompositePass(
        vk::raii::CommandBuffer& commandBuffer,
        uint32_t imageIndex);

    


    bool isWireframeSupported() const;
    

    std::unique_ptr<PostProcessRenderer> postProcessRenderer;


    std::vector<std::unique_ptr<GpuMesh>> gpuMeshes;


    

   

    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;


    
    
    
    void resetEnvironmentSettings();


   


    struct IblCalibrationPreset
    {
        float lightIntensity = 3.0f;
        float skyboxExposure = 0.3f;
        float iblIntensity = 1.0f;
        float diffuseIBLIntensity = 0.2f;
        float specularIBLIntensity = 1.2f;
        float postExposure = 1.0f;
    };

    IblCalibrationPreset defaultIblCalibrationPreset{
        .lightIntensity = 3.0f,
        .skyboxExposure = 1.0f,
        .iblIntensity = 1.0f,
        .diffuseIBLIntensity = 1.0f,
        .specularIBLIntensity = 1.0f,
        .postExposure = 1.0f
    };

    void applyIblCalibrationPreset(const IblCalibrationPreset& preset);
    void resetIblEnergyCalibration();

    bool debugForceSpecularMip = false;
    float debugSpecularMip = 0.0f;

    float roughnessMipScale = 1.0f;
    float roughnessMipBias = 0.0f;


    // --- Environment / IBL controls ---
    bool showSkybox = true;
    bool enableIBL = false;
    bool debugReflectionOnly = false;
    float skyboxExposure = 0.3f;
    float skyboxLod = 0.0f;
    float iblIntensity = 1.0f;
    float environmentRotationDegrees = 0.0f;
    bool rotateSkybox = true;
    bool rotateIBLLighting = true;
    float diffuseIBLIntensity = 0.2f;
    float specularIBLIntensity = 1.2f;

  
    bool debugSkyboxFaces = false;

    float rotationSpeed = 10.0f;


    float cameraRadius = 3.0f;
    float cameraYaw = glm::radians(270.0f);
    float cameraPitch = glm::radians(85.0f); // or slightly less, e.g. 75–80
    float cameraFov = 45.0f;
    float cameraNear = 0.1f;
    float cameraFar = 10.0f;

    float mouseOrbitSensitivity = 0.01f;
    float mousePanSensitivity = 0.005f;
    float mouseZoomSensitivity = 0.5f;
    float minCameraRadius = 0.5f;
    float maxCameraRadius = 50.0f;

    float currentAnimationAngle = 0.0f;
    bool animateModel = true;

    glm::vec4 clearColor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

    glm::vec3 lightDirection = glm::normalize(glm::vec3(-0.5f, -1.0f, -0.3f));




    glm::vec3 lightColor{ 1.0f };
    float lightIntensity = 3.0f;

    glm::vec3 ambientColor{ 1.0f };
    float ambientIntensity = 0.0f;

    float frameTimeMs = 0.0f;
    float fps = 0.0f;


};