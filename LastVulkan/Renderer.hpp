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

#include "GltfLoader.hpp"
#include "Texture2D.hpp"
#include "GpuMesh.hpp"
#include "Material.hpp"
#include "Renderable.hpp"
#include "Camera.hpp"
#include "Scene.hpp"
#include "EditorUiState.hpp"



#include "EnvironmentSystem.hpp"  
#include "SceneRenderer.hpp"
#include "DescriptorManager.hpp"





#include "EnvironmentUtils.hpp"
#include "BrdfLutRenderer.hpp"
#include "EnvironmentRenderer.hpp"
#include "IrradianceRenderer.hpp"
#include "PrefilterRenderer.hpp"
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
    
    
    
    
    void initImGui();
    void shutdownImGui();
    void beginImGuiFrame();
    void buildImGui();
    void renderImGui(vk::CommandBuffer commandBuffer);
    void buildOverlay();

    void focusSelectedRenderable();
    void resetDefaultSceneLayout();
    void cleanupDescriptorResources();
    void updateCameraControls();

    

    void updateUniformBuffer(uint32_t currentFrame);


    void recordCommandBuffer(uint32_t imageIndex);

    Texture2D& getDefaultTexture();
    Material& getDefaultMaterial();



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
    

    EditorUiState           uiState;
    Camera                  camera;
    UniformBufferObject     lastUbo;
    Scene                   scene;


    void clearSceneResources();
    void createDefaultMaterialTextures();
    void setupCameraDefaults();

    vk::DescriptorSetLayout externalBloomBlurDescriptorSetLayout{};
    vk::Sampler externalPostProcessSampler{};
        
    bool imguiInitialized = false;
    vk::raii::DescriptorPool imguiDescriptorPool = nullptr;

    std::string currentModelPath;

    std::unique_ptr<BrdfLutRenderer> brdfLutRenderer;
    std::unique_ptr<EnvironmentRenderer> environmentRenderer;
    std::unique_ptr<IrradianceRenderer> irradianceRenderer;
    std::unique_ptr<PrefilterRenderer> prefilterRenderer;


    Material* getSelectedRenderableMaterial();
    int getMaterialIndex(const Material& material) const;


    bool isWireframeSupported() const;
    glm::vec3 computeSceneCenter() const;


    std::vector<std::unique_ptr<Texture2D>>     normalTextures;
    std::unique_ptr<Texture2D>                  defaultNormalTexture;

    std::vector<std::unique_ptr<Texture2D>>     metallicRoughnessTextures;
    std::unique_ptr<Texture2D>                  defaultMetallicRoughnessTexture;

    
    std::vector<std::unique_ptr<Texture2D>>     aoTextures;
    std::unique_ptr<Texture2D>                  defaultAoTexture;

    std::vector<std::unique_ptr<Texture2D>>     emissiveTextures;
    std::unique_ptr<Texture2D>                  defaultEmissiveTexture;
    
    

    

    std::unique_ptr<PostProcessRenderer> postProcessRenderer;


    std::vector<std::unique_ptr<GpuMesh>> gpuMeshes;



    std::vector<std::unique_ptr<Texture2D>> textures;
    std::vector<std::unique_ptr<Material>> materials;


    glm::vec3 getRenderableWorldPosition(const Renderable& renderable) const;

   

    
    vk::raii::DescriptorSetLayout   frameDescriptorSetLayout = nullptr;
    vk::raii::DescriptorSetLayout   materialDescriptorSetLayout = nullptr;


    
    


    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;


    void createEnvironmentCubemap(const std::array<std::string, 6>& facePaths);
    
    
    void resetEnvironmentSettings();


    vk::raii::Image environmentCubeImage{ nullptr };

    vk::raii::DeviceMemory environmentCubeMemory{ nullptr };
    vk::raii::ImageView environmentCubeView{ nullptr };
    vk::raii::Sampler environmentCubeSampler{ nullptr };

    
    


    // IBL descriptor set
    vk::raii::DescriptorSetLayout iblDescriptorSetLayout{ nullptr };
    


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

    std::unique_ptr<Texture2D> brdfLutTexture;

    vk::raii::Image irradianceCubeImage{ nullptr };
    vk::raii::DeviceMemory irradianceCubeMemory{ nullptr };
    vk::raii::ImageView irradianceCubeView{ nullptr };
    vk::raii::Sampler irradianceCubeSampler{ nullptr };

    vk::raii::Image prefilteredCubeImage{ nullptr };
    vk::raii::DeviceMemory prefilteredCubeMemory{ nullptr };
    vk::raii::ImageView prefilteredCubeView{ nullptr };
    vk::raii::Sampler prefilteredCubeSampler{ nullptr };


  

   

    bool debugSkyboxFaces = false;

    

    vk::raii::Image hdrEnvironmentImage{ nullptr };
    vk::raii::DeviceMemory hdrEnvironmentMemory{ nullptr };
    vk::raii::ImageView hdrEnvironmentView{ nullptr };
    vk::raii::Sampler hdrEnvironmentSampler{ nullptr };

    uint32_t hdrEnvironmentWidth = 0;
    uint32_t hdrEnvironmentHeight = 0;

    void createHdrEnvironmentTexture(const std::string& path);

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