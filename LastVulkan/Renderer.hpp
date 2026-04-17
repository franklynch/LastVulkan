#pragma once



#include <memory>
#include <string>

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
#include "Texture2D.hpp"
#include "GpuMesh.hpp"
#include "Material.hpp"
#include "Renderable.hpp"
#include "Camera.hpp"
#include "Scene.hpp"
#include "EditorUiState.hpp"

class Renderer
{
public:
    Renderer(Window& window, VulkanContext& vkContext);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void drawFrame();

private:
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
       
      
    void loadModel();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();

    void createMaterialDescriptorSets();

    void createCommandBuffers();
    void createSyncObjects();

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

    

    static uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities);
    static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const& availableFormats);
    static vk::PresentModeKHR chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const& availablePresentModes);
    vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR const& capabilities);

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const;
    static std::vector<char> readFile(const std::string& filename);

    Texture2D& getDefaultTexture();
    Material& getDefaultMaterial();

    
    

private:
    Window& window;
    VulkanContext& vkContext;
    BufferUtils bufferUtils;
    ImageUtils imageUtils;

    bool imguiInitialized = false;
    vk::raii::DescriptorPool imguiDescriptorPool = nullptr;

    float rotationSpeed = 90.0f;
    

    float cameraRadius = 3.0f;
    float cameraYaw = 0.0f;
    float cameraPitch = 0.5f;
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

    float frameTimeMs = 0.0f;
    float fps = 0.0f;

    Material* getSelectedRenderableMaterial();
    int getMaterialIndex(const Material& material) const;
    bool isWireframeSupported() const;

    EditorUiState uiState;

    Camera camera;
    

    vk::raii::SwapchainKHR swapChain = nullptr;
    std::vector<vk::Image> swapChainImages;
    vk::SurfaceFormatKHR swapChainSurfaceFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;

    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;

    vk::raii::Image colorImage = nullptr;
    vk::raii::DeviceMemory colorImageMemory = nullptr;
    vk::raii::ImageView colorImageView = nullptr;

    vk::Format depthFormat;
    vk::ImageAspectFlags depthAspect;

    vk::raii::Image depthImage = nullptr;
    vk::raii::DeviceMemory depthImageMemory = nullptr;
    vk::raii::ImageView depthImageView = nullptr;

    std::vector<vk::raii::CommandBuffer> commandBuffers;

    std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;
    std::vector<vk::Fence> imagesInFlight;
    std::vector<bool> swapChainImageInitialized;

    uint32_t frameIndex = 0;

    MeshData meshData;
    ModelLoader modelLoader;

    std::vector<std::unique_ptr<GpuMesh>> gpuMeshes;

    

    std::vector<std::unique_ptr<Texture2D>> textures;
    std::vector<std::unique_ptr<Material>> materials;
   
    Scene scene;

    vk::raii::DescriptorPool descriptorPool = nullptr;
    vk::raii::DescriptorSetLayout frameDescriptorSetLayout = nullptr;
    vk::raii::DescriptorSetLayout materialDescriptorSetLayout = nullptr;

    std::vector<vk::raii::DescriptorSet> frameDescriptorSets;
    std::vector<vk::raii::DescriptorSet> materialDescriptorSets;
    
    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;
    
    
};