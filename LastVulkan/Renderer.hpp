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
#include "RendererTypes.hpp"
#include "BufferUtils.hpp"
#include "ImageUtils.hpp"
#include "MeshData.hpp"
#include "ModelLoader.hpp"
#include "GltfLoader.hpp"
#include "Texture2D.hpp"
#include "GpuMesh.hpp"
#include "Material.hpp"
#include "Renderable.hpp"
#include "Camera.hpp"
#include "Scene.hpp"
#include "EditorUiState.hpp"
#include "EnvironmentResources.hpp"
#include "EnvironmentUtils.hpp"
#include "BrdfLutRenderer.hpp"
#include "EnvironmentRenderer.hpp"
#include "IrradianceRenderer.hpp"
#include "PrefilterRenderer.hpp"





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

    void createSwapChain();
    void createImageViews();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void createColorResources();
    void createDepthResources();

    vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates,
        vk::ImageTiling tiling,
        vk::FormatFeatureFlags features);

    vk::Format findDepthFormat();

    bool hasStencilComponent(vk::Format format);


    void drawFrame();


    void loadModel();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();

    void createMaterialDescriptorSets();
    void createBloomExtractDescriptorSet();

    void createCommandBuffers();
    void createSyncObjects();

    void initImGui();
    void shutdownImGui();
    void beginImGuiFrame();
    void buildImGui();
    void renderImGui(vk::CommandBuffer commandBuffer);
    void buildOverlay();

    void createHdrColorResources();

    void createPostProcessDescriptorSetLayout();

    void createBloomBrightResources();
    void createBloomExtractDescriptorSetLayout();
    void createBloomBlurDescriptorSetLayout();

	void drawBloomExtract(vk::raii::CommandBuffer& commandBuffer);     

	void createBloomBlurPipeline();

    void createBloomExtractPipeline();

    void createBloomBlurResources();
    void createBloomBlurDescriptorSets();

    void drawBloomBlur(
        vk::raii::CommandBuffer& commandBuffer,
        vk::ImageView outputView,
        vk::DescriptorSet inputSet,
        glm::vec2 direction);

    void focusSelectedRenderable();

    void resetDefaultSceneLayout();
    void createPostProcessSampler();
    

    void cleanupDescriptorResources();
    void createPostProcessPipeline();
    void createPostProcessDescriptorSet();

    void updateCameraControls();

    void drawPostProcessToSwapchain(
        vk::raii::CommandBuffer& commandBuffer,
        uint32_t imageIndex);

    void updateUniformBuffer(uint32_t currentFrame);
    void recordCommandBuffer(uint32_t imageIndex);

    void transitionImageLayout(
        vk::CommandBuffer cmd,
        vk::Image image,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        vk::AccessFlags srcAccessMask,
        vk::AccessFlags dstAccessMask,
        vk::PipelineStageFlags srcStage,
        vk::PipelineStageFlags dstStage,
        vk::ImageAspectFlags aspectMask);

    void transitionToColorAttachment(
        vk::CommandBuffer cmd,
        vk::Image image,
        vk::ImageLayout oldLayout);

    void transitionToPresent(
        vk::CommandBuffer cmd,
        vk::Image image);

    void transitionToDepthAttachment(
        vk::CommandBuffer cmd,
        vk::Image image,
        vk::ImageAspectFlags aspectMask);

    void transitionToShaderReadOnly(
        vk::CommandBuffer cmd,
        vk::Image image,
        vk::ImageLayout oldLayout);



    static uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities);
    static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const& availableFormats);
    static vk::PresentModeKHR chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const& availablePresentModes);
    vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR const& capabilities);

    
    

    Texture2D& getDefaultTexture();
    Material& getDefaultMaterial();








private:
    Window&             window;
    VulkanContext&      vkContext;
    BufferUtils         bufferUtils;
    ImageUtils          imageUtils;

    struct GltfTextureUploadMaps
    {
        std::vector<int> baseColor;
        std::vector<int> normal;
        std::vector<int> metallicRoughness;
        std::vector<int> occlusion;
        std::vector<int> emissive;
    };

    void clearSceneResources();
    void createDefaultMaterialTextures();
    void setupCameraDefaults();

    GltfSceneData loadCurrentGltfScene();

    GltfTextureUploadMaps uploadGltfTextures(const GltfSceneData& imported);

    void createMaterialsFromGltf(
        const GltfSceneData& imported,
        const GltfTextureUploadMaps& textureMaps);

    void createRenderablesFromGltf(const GltfSceneData& imported);


    vk::raii::Image bloomBrightImage{ nullptr };
    vk::raii::DeviceMemory bloomBrightMemory{ nullptr };
    vk::raii::ImageView bloomBrightView{ nullptr };

    vk::raii::DescriptorSetLayout bloomExtractDescriptorSetLayout{ nullptr };
    vk::raii::PipelineLayout bloomExtractPipelineLayout{ nullptr };
    vk::raii::Pipeline bloomExtractPipeline{ nullptr };

    vk::raii::DescriptorPool bloomExtractDescriptorPool{ nullptr };
    vk::raii::DescriptorSets bloomExtractDescriptorSets{ nullptr };

    vk::raii::Image bloomBlurTempImage{ nullptr };
    vk::raii::DeviceMemory bloomBlurTempMemory{ nullptr };
    vk::raii::ImageView bloomBlurTempView{ nullptr };

    vk::raii::DescriptorSetLayout bloomBlurDescriptorSetLayout{ nullptr };
    vk::raii::PipelineLayout bloomBlurPipelineLayout{ nullptr };
    vk::raii::Pipeline bloomBlurPipeline{ nullptr };

    vk::raii::DescriptorPool bloomBlurDescriptorPool{ nullptr };
    vk::raii::DescriptorSets bloomBlurFromBrightDescriptorSets{ nullptr };
    vk::raii::DescriptorSets bloomBlurFromTempDescriptorSets{ nullptr };

    bool imguiInitialized = false;
    vk::raii::DescriptorPool imguiDescriptorPool = nullptr;

    vk::raii::Pipeline solidPipeline = nullptr;
    vk::raii::Pipeline solidDoubleSidedPipeline = nullptr;
    vk::raii::Pipeline wireframePipeline = nullptr;
    vk::raii::Pipeline wireframeDoubleSidedPipeline = nullptr;

    std::string currentModelPath;
     
    
    EnvironmentResources environment;
    std::unique_ptr<BrdfLutRenderer> brdfLutRenderer;
    std::unique_ptr<EnvironmentRenderer> environmentRenderer;
    std::unique_ptr<IrradianceRenderer> irradianceRenderer;
    std::unique_ptr<PrefilterRenderer> prefilterRenderer;

    vk::raii::DescriptorSetLayout postProcessDescriptorSetLayout{ nullptr };
    vk::raii::PipelineLayout postProcessPipelineLayout{ nullptr };
    vk::raii::Pipeline postProcessPipeline{ nullptr };

    vk::raii::Sampler postProcessSampler{ nullptr };

    vk::raii::DescriptorPool postProcessDescriptorPool{ nullptr };
    vk::raii::DescriptorSets postProcessDescriptorSets{ nullptr };


    

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

    UniformBufferObject lastUbo;
    

    glm::vec3 lightColor{ 1.0f };
    float lightIntensity = 3.0f;

    glm::vec3 ambientColor{ 1.0f };
    float ambientIntensity = 0.0f;

    float frameTimeMs = 0.0f;
    float fps = 0.0f;

    

 

    Material* getSelectedRenderableMaterial();
    int getMaterialIndex(const Material& material) const;


    bool isWireframeSupported() const;
    glm::vec3 computeSceneCenter() const;

    EditorUiState uiState;

    Camera camera;

    std::vector<std::unique_ptr<Texture2D>>     normalTextures;
    std::unique_ptr<Texture2D>                  defaultNormalTexture;

    std::vector<std::unique_ptr<Texture2D>>     metallicRoughnessTextures;
    std::unique_ptr<Texture2D>                  defaultMetallicRoughnessTexture;

    
    std::vector<std::unique_ptr<Texture2D>>     aoTextures;
    std::unique_ptr<Texture2D>                  defaultAoTexture;

    std::vector<std::unique_ptr<Texture2D>>     emissiveTextures;
    std::unique_ptr<Texture2D> defaultEmissiveTexture;


    vk::raii::SwapchainKHR              swapChain = nullptr;
    std::vector<vk::Image>              swapChainImages;
    vk::SurfaceFormatKHR                swapChainSurfaceFormat;
    vk::Extent2D                        swapChainExtent;
    std::vector<vk::raii::ImageView>    swapChainImageViews;

    vk::raii::PipelineLayout            pipelineLayout = nullptr;


    vk::raii::Image             colorImage = nullptr;
    vk::raii::DeviceMemory      colorImageMemory = nullptr;
    vk::raii::ImageView         colorImageView = nullptr;

    vk::Format                  depthFormat;
    vk::ImageAspectFlags        depthAspect;

    vk::raii::Image             depthImage = nullptr;
    vk::raii::DeviceMemory      depthImageMemory = nullptr;
    vk::raii::ImageView         depthImageView = nullptr;

    vk::raii::Image hdrColorImage{ nullptr };
    vk::raii::DeviceMemory hdrColorMemory{ nullptr };
    vk::raii::ImageView hdrColorView{ nullptr };

    vk::Format hdrFormat = vk::Format::eR16G16B16A16Sfloat;

    std::vector<vk::raii::CommandBuffer> commandBuffers;

    std::vector<vk::raii::Semaphore>    presentCompleteSemaphores;
    std::vector<vk::raii::Semaphore>    renderFinishedSemaphores;
    std::vector<vk::raii::Fence>        inFlightFences;
    std::vector<vk::Fence>              imagesInFlight;

    std::vector<bool> swapChainImageInitialized;

    uint32_t frameIndex = 0;

    MeshData meshData;
    ModelLoader modelLoader;

    std::vector<std::unique_ptr<GpuMesh>> gpuMeshes;



    std::vector<std::unique_ptr<Texture2D>> textures;
    std::vector<std::unique_ptr<Material>> materials;


    glm::vec3 getRenderableWorldPosition(const Renderable& renderable) const;

    Scene scene;

    vk::raii::DescriptorPool        descriptorPool = nullptr;
    vk::raii::DescriptorSetLayout   frameDescriptorSetLayout = nullptr;
    vk::raii::DescriptorSetLayout   materialDescriptorSetLayout = nullptr;


    std::vector<vk::raii::DescriptorSet> frameDescriptorSets;
    std::vector<vk::raii::DescriptorSet> materialDescriptorSets;


    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;


    void createEnvironmentCubemap(const std::array<std::string, 6>& facePaths);
    void createSkyboxPipeline();
    void drawSkybox(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex);
    void resetEnvironmentSettings();


    vk::raii::Image environmentCubeImage{ nullptr };

    vk::raii::DeviceMemory environmentCubeMemory{ nullptr };
    vk::raii::ImageView environmentCubeView{ nullptr };
    vk::raii::Sampler environmentCubeSampler{ nullptr };

    vk::raii::PipelineLayout skyboxPipelineLayout{ nullptr };
    vk::raii::Pipeline skyboxPipeline{ nullptr };


    // IBL descriptor set
    vk::raii::DescriptorSetLayout iblDescriptorSetLayout{ nullptr };
    vk::raii::DescriptorSet iblDescriptorSet{ nullptr };

    // Fallback BRDF LUT (2D)
    std::unique_ptr<Texture2D> fallbackBrdfLut;

    // Fallback cubemap (shared for irradiance/prefiltered/environment)
    vk::raii::Image fallbackBlackCubeImage{ nullptr };
    vk::raii::DeviceMemory fallbackBlackCubeMemory{ nullptr };
    vk::raii::ImageView fallbackBlackCubeView{ nullptr };
    vk::raii::Sampler fallbackBlackCubeSampler{ nullptr };

    // --- IBL fallback setup ---
    void createFallbackIBLResources();
    void createFallbackBrdfLut();
    void createFallbackBlackCube();
    void updateIBLDescriptorSet();

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

    void createCubemapFromDDS(
        const std::string& path,
        vk::raii::Image& outImage,
        vk::raii::DeviceMemory& outMemory,
        vk::raii::ImageView& outView,
        vk::raii::Sampler& outSampler,
        bool allowMipSampling);

    void createIrradianceCubemapFromDDS(const std::string& path);
    void createPrefilteredCubemapFromDDS(const std::string& path);

    bool toneMappingEnabled = true;
    bool gammaEnabled = true;
    float postExposure = 1.0f;

    bool debugSkyboxFaces = false;

    vk::raii::Pipeline transparentPipeline = nullptr;
    vk::raii::Pipeline transparentDoubleSidedPipeline = nullptr;

    vk::raii::Image hdrEnvironmentImage{ nullptr };
    vk::raii::DeviceMemory hdrEnvironmentMemory{ nullptr };
    vk::raii::ImageView hdrEnvironmentView{ nullptr };
    vk::raii::Sampler hdrEnvironmentSampler{ nullptr };

    uint32_t hdrEnvironmentWidth = 0;
    uint32_t hdrEnvironmentHeight = 0;

    void createHdrEnvironmentTexture(const std::string& path);


    vk::DescriptorImageInfo makeImageInfo(
        vk::Sampler sampler,
        vk::ImageView view) const;

  

};