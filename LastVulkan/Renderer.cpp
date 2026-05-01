#include "Renderer.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cassert>

#include <cstring>
#include <iostream>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <cstdio>
#include <array>


#include <stb_image.h>

#include "EditorPanels.hpp"

#include "DdsUtils.hpp"
#include "ShaderUtils.hpp"




#include <chrono>



Renderer::Renderer(Window& window, VulkanContext& vkContext)
    : window(window)
    , vkContext(vkContext)
    , bufferUtils(vkContext)
    , imageUtils(vkContext, bufferUtils)
{
    init();
}

Renderer::~Renderer()
{
    shutdownImGui();
    vkContext.getDevice().waitIdle();

    environment.runtimeBrdfLut.pipeline = nullptr;
    environment.runtimeBrdfLut.layout = nullptr;

    environment.runtimeBrdfLut.sampler = nullptr;
    environment.runtimeBrdfLut.view = nullptr;
    environment.runtimeBrdfLut.memory = nullptr;
    environment.runtimeBrdfLut.image = nullptr;

    for (auto& view : environment.runtimeEnvironmentCubeFaces.views)
    {
        view = nullptr;
    }

    environment.runtimeEnvironmentCube.sampler = nullptr;
    environment.runtimeEnvironmentCube.view = nullptr;
    environment.runtimeEnvironmentCube.memory = nullptr;
    environment.runtimeEnvironmentCube.image = nullptr;

    for (auto& mipViews : environment.runtimePrefilteredCubeMipFaceViews)
    {
        for (auto& view : mipViews.views)
        {
            view = nullptr;
        }
    }

    environment.runtimePrefilteredCubeMipFaceViews.clear();

    environment.runtimePrefilteredCube.sampler = nullptr;
    environment.runtimePrefilteredCube.view = nullptr;
    environment.runtimePrefilteredCube.memory = nullptr;
    environment.runtimePrefilteredCube.image = nullptr;

   

    hdrEnvironmentSampler = nullptr;
    hdrEnvironmentView = nullptr;
    hdrEnvironmentMemory = nullptr;
    hdrEnvironmentImage = nullptr;

    for (auto& view : environment.runtimeIrradianceCubeFaces.views)
    {
        view = nullptr;
    }

    environment.runtimeIrradianceCube.sampler = nullptr;
    environment.runtimeIrradianceCube.view = nullptr;
    environment.runtimeIrradianceCube.memory = nullptr;
    environment.runtimeIrradianceCube.image = nullptr;

    cleanupDescriptorResources();
}

void Renderer::init()
{

    createSwapChain();
    createImageViews();
    createDescriptorSetLayout();

    depthFormat = findDepthFormat();
    depthAspect = hasStencilComponent(depthFormat)
        ? (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)
        : vk::ImageAspectFlagBits::eDepth;

    
    
    
    createPostProcessDescriptorSetLayout();
        
    createColorResources();
    createHdrColorResources();
    createDepthResources();
    
    createPostProcessSampler();

    createGraphicsPipeline();
    createSkyboxPipeline();
    createPostProcessPipeline();

    clearSceneResources();
    createDefaultMaterialTextures();
    setupCameraDefaults();

    GltfSceneData imported = loadCurrentGltfScene();

    GltfTextureUploadMaps textureMaps =
        uploadGltfTextures(imported);

    createMaterialsFromGltf(imported, textureMaps);
    createRenderablesFromGltf(imported);

    

    uiState.selectedRenderableIndex = scene.empty() ? -1 : 0;

    resetEnvironmentSettings();

    createUniformBuffers();

    brdfLutRenderer = std::make_unique<BrdfLutRenderer>(vkContext, bufferUtils);
    brdfLutRenderer->init(environment);
    createFallbackIBLResources();

    createDescriptorPool();
    createDescriptorSets();
    createMaterialDescriptorSets();
    createPostProcessDescriptorSet();

    
    

    createIrradianceCubemapFromDDS("assets/ibl/output_iem.dds");
    createPrefilteredCubemapFromDDS("assets/ibl/output_pmrem.dds");

  

    

    createEnvironmentCubemap({
    "assets/skybox/right.jpg",
    "assets/skybox/left.jpg",
    "assets/skybox/top.jpg",
    "assets/skybox/bottom.jpg",
    "assets/skybox/front.jpg",
    "assets/skybox/back.jpg" });

    // createHdrEnvironmentTexture("assets/hdr/studio.hdr");

    createHdrEnvironmentTexture("assets/hdr/citrus_orchard_road_puresky_4k.hdr");

    
    
    environmentRenderer = std::make_unique<EnvironmentRenderer>(vkContext, bufferUtils);
    environmentRenderer->init(
        environment,
        hdrEnvironmentSampler,
        hdrEnvironmentView);
    

    
    

    
    irradianceRenderer = std::make_unique<IrradianceRenderer>(vkContext, bufferUtils);
    irradianceRenderer->init(environment);
    
    prefilterRenderer = std::make_unique<PrefilterRenderer>(vkContext, bufferUtils);
    prefilterRenderer->init(environment);
       

     

    

    updateIBLDescriptorSet();

    

    createCommandBuffers();
    createSyncObjects();
    initImGui();



}

void Renderer::cleanupSwapChain()
{
    solidPipeline = nullptr;
    solidDoubleSidedPipeline = nullptr;
    wireframePipeline = nullptr;
    wireframeDoubleSidedPipeline = nullptr;
    transparentPipeline = nullptr;
    transparentDoubleSidedPipeline = nullptr;

    pipelineLayout = nullptr;

    skyboxPipeline = nullptr;
    skyboxPipelineLayout = nullptr;

    postProcessPipeline = nullptr;
    postProcessPipelineLayout = nullptr;

    postProcessDescriptorSets.clear();
    postProcessDescriptorPool = nullptr;
    postProcessSampler = nullptr;


    colorImageView = nullptr;
    colorImageMemory = nullptr;
    colorImage = nullptr;

    hdrColorView = nullptr;
    hdrColorMemory = nullptr;
    hdrColorImage = nullptr;

    depthImageView = nullptr;
    depthImageMemory = nullptr;
    depthImage = nullptr;

    swapChainImageViews.clear();
    swapChainImages.clear();
    swapChain = nullptr;
}

void Renderer::recreateSwapChain()
{

    // If at any point you add this to recreateSwapChain() :
    // createDescriptorPool()
    // createDescriptorSets()
    // createMaterialDescriptorSets()
    // cleanupDescriptorResources(); must already happen first.


    int width = 0, height = 0;
    window.getFramebufferSize(width, height);

    while (width == 0 || height == 0)
    {
        window.getFramebufferSize(width, height);
        glfwWaitEvents();
    }

    vkContext.getDevice().waitIdle();

    cleanupSwapChain();
    createSwapChain();
    createImageViews();

    depthFormat = findDepthFormat();
    depthAspect = hasStencilComponent(depthFormat)
        ? (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)
        : vk::ImageAspectFlagBits::eDepth;

    createColorResources();
    createHdrColorResources();
    createDepthResources();

    createPostProcessSampler();
    createPostProcessDescriptorSet();
    createPostProcessPipeline();

    createGraphicsPipeline();
    createSkyboxPipeline();

    // Rebuild per-image semaphores for the new swapchain.
    renderFinishedSemaphores.clear();
    renderFinishedSemaphores.reserve(swapChainImages.size());
    for (size_t i = 0; i < swapChainImages.size(); ++i)
    {
        renderFinishedSemaphores.emplace_back(vkContext.getDevice(), vk::SemaphoreCreateInfo());
    }

    imagesInFlight.assign(swapChainImages.size(), vk::Fence{});
    swapChainImageInitialized.assign(swapChainImages.size(), false);
}

void Renderer::createSwapChain()
{
    auto& physicalDevice = vkContext.getPhysicalDevice();
    auto& surface = vkContext.getSurface();
    auto& device = vkContext.getDevice();

    vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    swapChainExtent = chooseSwapExtent(surfaceCapabilities);
    uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);

    std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(*surface);
    swapChainSurfaceFormat = chooseSwapSurfaceFormat(availableFormats);

    std::vector<vk::PresentModeKHR> availablePresentModes = physicalDevice.getSurfacePresentModesKHR(*surface);
    vk::PresentModeKHR presentMode = chooseSwapPresentMode(availablePresentModes);

    vk::SwapchainCreateInfoKHR swapChainCreateInfo{};
    swapChainCreateInfo
        .setSurface(*surface)
        .setMinImageCount(minImageCount)
        .setImageFormat(swapChainSurfaceFormat.format)
        .setImageColorSpace(swapChainSurfaceFormat.colorSpace)
        .setImageExtent(swapChainExtent)
        .setImageArrayLayers(1)
        .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
        .setImageSharingMode(vk::SharingMode::eExclusive)
        .setPreTransform(surfaceCapabilities.currentTransform)
        .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
        .setPresentMode(presentMode)
        .setClipped(VK_TRUE);

    swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
    swapChainImages = swapChain.getImages();
    swapChainImageInitialized.assign(swapChainImages.size(), false);
}

void Renderer::createImageViews()
{
    assert(swapChainImageViews.empty());

    auto& device = vkContext.getDevice();

    vk::ImageSubresourceRange range{};
    range
        .setAspectMask(vk::ImageAspectFlagBits::eColor)
        .setBaseMipLevel(0)
        .setLevelCount(1)
        .setBaseArrayLayer(0)
        .setLayerCount(1);

    vk::ImageViewCreateInfo imageViewCreateInfo{};
    imageViewCreateInfo
        .setViewType(vk::ImageViewType::e2D)
        .setFormat(swapChainSurfaceFormat.format)
        .setSubresourceRange(range);

    for (const auto& image : swapChainImages)
    {
        imageViewCreateInfo.setImage(image);
        swapChainImageViews.emplace_back(device, imageViewCreateInfo);
    }
}

void Renderer::createDescriptorSetLayout()
{
    auto& device = vkContext.getDevice();

    // Set 0: per-frame UBO
    {
        vk::DescriptorSetLayoutBinding uboBinding{};
        uboBinding
            .setBinding(0)
            .setDescriptorType(vk::DescriptorType::eUniformBuffer)
            .setDescriptorCount(1)
            .setStageFlags(
                vk::ShaderStageFlagBits::eVertex |
                vk::ShaderStageFlagBits::eFragment);

        vk::DescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.setBindings(uboBinding);

        frameDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
    }

    // Set 1: per-material textures
    {
        vk::DescriptorSetLayoutBinding baseColorBinding{};
        baseColorBinding
            .setBinding(0)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment);

        vk::DescriptorSetLayoutBinding normalBinding{};
        normalBinding
            .setBinding(1)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment);

        vk::DescriptorSetLayoutBinding metallicRoughnessBinding{};
        metallicRoughnessBinding
            .setBinding(2)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment);

        vk::DescriptorSetLayoutBinding aoBinding{};
        aoBinding
            .setBinding(3)
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment);

        vk::DescriptorSetLayoutBinding emissiveBinding{};
        emissiveBinding
            .setBinding(4)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment);

        std::array<vk::DescriptorSetLayoutBinding, 5> bindings = {
            baseColorBinding,
            normalBinding,
            metallicRoughnessBinding,
            aoBinding,
            emissiveBinding
        };



        vk::DescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.setBindings(bindings);

        materialDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
    }

    // Set 2: IBL textures
    {
        vk::DescriptorSetLayoutBinding irradianceBinding{};
        irradianceBinding
            .setBinding(0)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment);

        vk::DescriptorSetLayoutBinding prefilteredBinding{};
        prefilteredBinding
            .setBinding(1)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment);

        vk::DescriptorSetLayoutBinding brdfLutBinding{};
        brdfLutBinding
            .setBinding(2)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment);

        vk::DescriptorSetLayoutBinding environmentBinding{};
        environmentBinding
            .setBinding(3)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment);

        std::array<vk::DescriptorSetLayoutBinding, 4> bindings = {
            irradianceBinding,
            prefilteredBinding,
            brdfLutBinding,
            environmentBinding
        };

        vk::DescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.setBindings(bindings);

        iblDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);


    }
}

void Renderer::createGraphicsPipeline()
{
    auto& device = vkContext.getDevice();

    vk::raii::ShaderModule vertShaderModule =
        ShaderUtils::createShaderModule(vkContext.getDevice(), "shaders/vert.spv");

    vk::raii::ShaderModule fragShaderModule =
        ShaderUtils::createShaderModule(vkContext.getDevice(), "shaders/frag.spv");

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo
        .setStage(vk::ShaderStageFlagBits::eVertex)
        .setModule(*vertShaderModule)
        .setPName("main");

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo
        .setStage(vk::ShaderStageFlagBits::eFragment)
        .setModule(*fragShaderModule)
        .setPName("main");

    std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {
        vertShaderStageInfo,
        fragShaderStageInfo
    };

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo
        .setVertexBindingDescriptions(bindingDescription)
        .setVertexAttributeDescriptions(attributeDescriptions);

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly
        .setTopology(vk::PrimitiveTopology::eTriangleList)
        .setPrimitiveRestartEnable(VK_FALSE);

    vk::PipelineViewportStateCreateInfo viewportState{};
    viewportState
        .setViewportCount(1)
        .setScissorCount(1);

    vk::PipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer
        .setDepthClampEnable(VK_FALSE)
        .setRasterizerDiscardEnable(VK_FALSE)
        .setFrontFace(vk::FrontFace::eCounterClockwise)
        .setDepthBiasEnable(VK_FALSE)
        .setLineWidth(1.0f);

    vk::PipelineMultisampleStateCreateInfo multisampling{};
    multisampling
        .setRasterizationSamples(vkContext.getMsaaSamples())
        .setSampleShadingEnable(VK_FALSE);

    vk::PipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil
        .setDepthTestEnable(VK_TRUE)
        .setDepthWriteEnable(VK_TRUE)
        .setDepthCompareOp(vk::CompareOp::eLess)
        .setDepthBoundsTestEnable(VK_FALSE)
        .setStencilTestEnable(VK_FALSE);

    vk::PipelineDepthStencilStateCreateInfo transparentDepthStencil{};
    transparentDepthStencil
        .setDepthTestEnable(VK_TRUE)
        .setDepthWriteEnable(VK_FALSE)
        .setDepthCompareOp(vk::CompareOp::eLess)
        .setDepthBoundsTestEnable(VK_FALSE)
        .setStencilTestEnable(VK_FALSE);



    vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment
        .setBlendEnable(VK_FALSE)
        .setColorWriteMask(
            vk::ColorComponentFlagBits::eR |
            vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB |
            vk::ColorComponentFlagBits::eA
        );

    vk::PipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending
        .setLogicOpEnable(VK_FALSE)
        .setLogicOp(vk::LogicOp::eCopy)
        .setAttachments(colorBlendAttachment);

    vk::PipelineColorBlendAttachmentState transparentBlendAttachment{};
    transparentBlendAttachment
        .setBlendEnable(VK_TRUE)
        .setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)
        .setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
        .setColorBlendOp(vk::BlendOp::eAdd)
        .setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
        .setDstAlphaBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
        .setAlphaBlendOp(vk::BlendOp::eAdd)
        .setColorWriteMask(
            vk::ColorComponentFlagBits::eR |
            vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB |
            vk::ColorComponentFlagBits::eA
        );

    vk::PipelineColorBlendStateCreateInfo transparentColorBlending{};
    transparentColorBlending
        .setLogicOpEnable(VK_FALSE)
        .setLogicOp(vk::LogicOp::eCopy)
        .setAttachments(transparentBlendAttachment);

    std::vector<vk::DynamicState> dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor
    };

    vk::PipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.setDynamicStates(dynamicStates);

    vk::PushConstantRange pushConstantRange{};
    pushConstantRange
        .setStageFlags(
            vk::ShaderStageFlagBits::eVertex |
            vk::ShaderStageFlagBits::eFragment)
        .setOffset(0)
        .setSize(sizeof(PushConstantData));

    // Main scene pipeline layout:
    // set 0 = frame UBO
    // set 1 = material textures
    // set 2 = IBL textures
    std::array<vk::DescriptorSetLayout, 3> setLayouts = {
        *frameDescriptorSetLayout,
        *materialDescriptorSetLayout,
        *iblDescriptorSetLayout
    };

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo
        .setSetLayouts(setLayouts)
        .setPushConstantRanges(pushConstantRange);

    pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    vk::StructureChain<
        vk::GraphicsPipelineCreateInfo,
        vk::PipelineRenderingCreateInfo
    > pipelineCreateInfoChain{};

    pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>()
        .setStages(shaderStages)
        .setPVertexInputState(&vertexInputInfo)
        .setPInputAssemblyState(&inputAssembly)
        .setPViewportState(&viewportState)
        .setPRasterizationState(&rasterizer)
        .setPMultisampleState(&multisampling)
        .setPDepthStencilState(&depthStencil)
        .setPColorBlendState(&colorBlending)
        .setPDynamicState(&dynamicState)
        .setLayout(*pipelineLayout)
        .setRenderPass(vk::RenderPass{});

    pipelineCreateInfoChain.get<vk::PipelineRenderingCreateInfo>()
        .setColorAttachmentFormats(hdrFormat)
        .setDepthAttachmentFormat(depthFormat);

    // -------------------------
    // Opaque pipelines
    // -------------------------

    // Filled, culled
    rasterizer
        .setPolygonMode(vk::PolygonMode::eFill)
        .setCullMode(vk::CullModeFlagBits::eBack);

    pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>()
        .setPDepthStencilState(&depthStencil)
        .setPColorBlendState(&colorBlending);

    solidPipeline = vk::raii::Pipeline(
        device,
        nullptr,
        pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>()
    );

    // Filled, double-sided
    rasterizer
        .setPolygonMode(vk::PolygonMode::eFill)
        .setCullMode(vk::CullModeFlagBits::eNone);

    solidDoubleSidedPipeline = vk::raii::Pipeline(
        device,
        nullptr,
        pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>()
    );

    // -------------------------
    // Transparent pipelines
    // -------------------------

    // Transparent, culled
    rasterizer
        .setPolygonMode(vk::PolygonMode::eFill)
        .setCullMode(vk::CullModeFlagBits::eBack);

    pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>()
        .setPDepthStencilState(&transparentDepthStencil)
        .setPColorBlendState(&transparentColorBlending);

    transparentPipeline = vk::raii::Pipeline(
        device,
        nullptr,
        pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>()
    );

    // Transparent, double-sided
    rasterizer
        .setPolygonMode(vk::PolygonMode::eFill)
        .setCullMode(vk::CullModeFlagBits::eNone);

    transparentDoubleSidedPipeline = vk::raii::Pipeline(
        device,
        nullptr,
        pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>()
    );

    // -------------------------
    // Wireframe opaque pipelines
    // -------------------------

    wireframePipeline = nullptr;
    wireframeDoubleSidedPipeline = nullptr;

    if (vkContext.isFillModeNonSolidEnabled())
    {
        pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>()
            .setPDepthStencilState(&depthStencil)
            .setPColorBlendState(&colorBlending);

        // Wireframe, culled
        rasterizer
            .setPolygonMode(vk::PolygonMode::eLine)
            .setCullMode(vk::CullModeFlagBits::eBack);

        wireframePipeline = vk::raii::Pipeline(
            device,
            nullptr,
            pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>()
        );

        // Wireframe, double-sided
        rasterizer
            .setPolygonMode(vk::PolygonMode::eLine)
            .setCullMode(vk::CullModeFlagBits::eNone);

        wireframeDoubleSidedPipeline = vk::raii::Pipeline(
            device,
            nullptr,
            pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>()
        );
    }
}

void Renderer::createColorResources()
{
    vk::Format colorFormat = hdrFormat;

    imageUtils.createImage(
        swapChainExtent.width,
        swapChainExtent.height,
        1,
        vkContext.getMsaaSamples(),
        colorFormat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransientAttachment |
        vk::ImageUsageFlagBits::eColorAttachment,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        colorImage,
        colorImageMemory
    );

    colorImageView = imageUtils.createImageView(colorImage, colorFormat, vk::ImageAspectFlagBits::eColor, 1);
}

void Renderer::createDepthResources()
{
    imageUtils.createImage(
        swapChainExtent.width,
        swapChainExtent.height,
        1,
        vkContext.getMsaaSamples(),
        depthFormat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        depthImage,
        depthImageMemory
    );

    depthImageView = imageUtils.createImageView(
        depthImage,
        depthFormat,
        depthAspect,
        1
    );
}

void Renderer::createDefaultMaterialTextures()
{
    const unsigned char fallbackWhitePixel[4] = { 255, 255, 255, 255 };
    const unsigned char flatNormalPixels[4] = { 128, 128, 255, 255 };
    const unsigned char defaultMRPixels[4] = { 255, 255, 255, 255 };
    const unsigned char fallbackAoPixel[4] = { 255, 255, 255, 255 };
    const unsigned char fallbackEmissivePixel[4] = { 0, 0, 0, 255 };

    textures.push_back(std::make_unique<Texture2D>(
        vkContext, bufferUtils, imageUtils,
        fallbackWhitePixel, 1, 1, 4,
        "<fallback-white-base-color>",
        vk::Format::eR8G8B8A8Srgb));

    defaultNormalTexture = std::make_unique<Texture2D>(
        vkContext, bufferUtils, imageUtils,
        flatNormalPixels, 1, 1, 4,
        "Default Flat Normal",
        vk::Format::eR8G8B8A8Unorm);

    defaultMetallicRoughnessTexture = std::make_unique<Texture2D>(
        vkContext, bufferUtils, imageUtils,
        defaultMRPixels, 1, 1, 4,
        "Default MetallicRoughness",
        vk::Format::eR8G8B8A8Unorm);

    defaultAoTexture = std::make_unique<Texture2D>(
        vkContext, bufferUtils, imageUtils,
        fallbackAoPixel, 1, 1, 4,
        "<fallback-ao>",
        vk::Format::eR8G8B8A8Unorm);

    defaultEmissiveTexture = std::make_unique<Texture2D>(
        vkContext, bufferUtils, imageUtils,
        fallbackEmissivePixel, 1, 1, 4,
        "<fallback-emissive>",
        vk::Format::eR8G8B8A8Srgb);
}

void Renderer::clearSceneResources()
{
    textures.clear();
    normalTextures.clear();
    metallicRoughnessTextures.clear();
    aoTextures.clear();
    emissiveTextures.clear();

    defaultNormalTexture.reset();
    defaultMetallicRoughnessTexture.reset();
    defaultAoTexture.reset();
    defaultEmissiveTexture.reset();

    materials.clear();
    scene.clear();
    gpuMeshes.clear();
}

void Renderer::setupCameraDefaults()
{
    camera.setTarget({ 0.0f, 0.0f, 0.0f });
    camera.setOrbit(cameraRadius, cameraYaw, cameraPitch);
    camera.setFov(cameraFov);
    camera.setNearFar(cameraNear, cameraFar);
}

GltfSceneData Renderer::loadCurrentGltfScene()
{
    currentModelPath = "models/DamagedHelmet/glTF/DamagedHelmet.gltf";

    //currentModelPath ="models/Suzanne/gLTF/Suzanne.gltf";

    std::cout << "Loading model: " << currentModelPath << std::endl;

    GltfLoader loader;
    return loader.load(currentModelPath);
}

Renderer::GltfTextureUploadMaps Renderer::uploadGltfTextures(
    const GltfSceneData& imported)
{
    std::vector<bool> imageUsedAsBaseColor(imported.images.size(), false);
    std::vector<bool> imageUsedAsNormal(imported.images.size(), false);
    std::vector<bool> imageUsedAsMR(imported.images.size(), false);
    std::vector<bool> imageUsedAsAO(imported.images.size(), false);
    std::vector<bool> imageUsedAsEmissive(imported.images.size(), false);

    GltfTextureUploadMaps maps;
    maps.baseColor.resize(imported.images.size(), -1);
    maps.normal.resize(imported.images.size(), -1);
    maps.metallicRoughness.resize(imported.images.size(), -1);
    maps.occlusion.resize(imported.images.size(), -1);
    maps.emissive.resize(imported.images.size(), -1);

    for (const auto& importedMaterial : imported.materials)
    {
        if (importedMaterial.baseColorImageIndex >= 0 &&
            importedMaterial.baseColorImageIndex < static_cast<int>(imageUsedAsBaseColor.size()))
        {
            imageUsedAsBaseColor[importedMaterial.baseColorImageIndex] = true;
        }

        if (importedMaterial.normalImageIndex >= 0 &&
            importedMaterial.normalImageIndex < static_cast<int>(imageUsedAsNormal.size()))
        {
            imageUsedAsNormal[importedMaterial.normalImageIndex] = true;
        }

        if (importedMaterial.metallicRoughnessImageIndex >= 0 &&
            importedMaterial.metallicRoughnessImageIndex < static_cast<int>(imageUsedAsMR.size()))
        {
            imageUsedAsMR[importedMaterial.metallicRoughnessImageIndex] = true;
        }

        if (importedMaterial.occlusionImageIndex >= 0 &&
            importedMaterial.occlusionImageIndex < static_cast<int>(imageUsedAsAO.size()))
        {
            imageUsedAsAO[importedMaterial.occlusionImageIndex] = true;
        }

        if (importedMaterial.emissiveImageIndex >= 0 &&
            importedMaterial.emissiveImageIndex < static_cast<int>(imageUsedAsEmissive.size()))
        {
            imageUsedAsEmissive[importedMaterial.emissiveImageIndex] = true;
        }
    }

    for (size_t i = 0; i < imported.images.size(); ++i)
    {
        if (!imageUsedAsBaseColor[i])
            continue;

        const auto& image = imported.images[i];

        if (image.pixels.empty() || image.width <= 0 || image.height <= 0)
            continue;

        if (image.channels != 3 && image.channels != 4)
        {
            std::cout << "Skipping glTF baseColor image " << i
                << " (" << image.name << ") because channels = "
                << image.channels << " (expected 3 or 4)\n";
            continue;
        }

        textures.push_back(std::make_unique<Texture2D>(
            vkContext,
            bufferUtils,
            imageUtils,
            image.pixels.data(),
            static_cast<uint32_t>(image.width),
            static_cast<uint32_t>(image.height),
            static_cast<uint32_t>(image.channels),
            image.name.empty()
            ? ("glTF baseColor image " + std::to_string(i))
            : image.name,
            vk::Format::eR8G8B8A8Srgb));

        maps.baseColor[i] = static_cast<int>(textures.size()) - 1;
    }

    for (size_t i = 0; i < imported.images.size(); ++i)
    {
        if (!imageUsedAsNormal[i])
            continue;

        const auto& image = imported.images[i];

        if (image.pixels.empty() || image.width <= 0 || image.height <= 0)
            continue;

        if (image.channels != 3 && image.channels != 4)
        {
            std::cout << "Skipping glTF normal image " << i
                << " (" << image.name << ") because channels = "
                << image.channels << " (expected 3 or 4)\n";
            continue;
        }

        normalTextures.push_back(std::make_unique<Texture2D>(
            vkContext,
            bufferUtils,
            imageUtils,
            image.pixels.data(),
            static_cast<uint32_t>(image.width),
            static_cast<uint32_t>(image.height),
            static_cast<uint32_t>(image.channels),
            image.name.empty()
            ? ("glTF normal image " + std::to_string(i))
            : image.name,
            vk::Format::eR8G8B8A8Unorm));

        maps.normal[i] = static_cast<int>(normalTextures.size()) - 1;
    }

    for (size_t i = 0; i < imported.images.size(); ++i)
    {
        if (!imageUsedAsMR[i])
            continue;

        const auto& image = imported.images[i];

        if (image.pixels.empty() || image.width <= 0 || image.height <= 0)
            continue;

        if (image.channels != 3 && image.channels != 4)
        {
            std::cout << "Skipping glTF metallicRoughness image " << i
                << " (" << image.name << ") because channels = "
                << image.channels << " (expected 3 or 4)\n";
            continue;
        }

        metallicRoughnessTextures.push_back(std::make_unique<Texture2D>(
            vkContext,
            bufferUtils,
            imageUtils,
            image.pixels.data(),
            static_cast<uint32_t>(image.width),
            static_cast<uint32_t>(image.height),
            static_cast<uint32_t>(image.channels),
            image.name.empty()
            ? ("glTF metallicRoughness image " + std::to_string(i))
            : image.name,
            vk::Format::eR8G8B8A8Unorm));

        maps.metallicRoughness[i] =
            static_cast<int>(metallicRoughnessTextures.size()) - 1;
    }

    for (size_t i = 0; i < imported.images.size(); ++i)
    {
        if (!imageUsedAsAO[i])
            continue;

        const auto& image = imported.images[i];

        if (image.pixels.empty() || image.width <= 0 || image.height <= 0)
            continue;

        if (image.channels != 3 && image.channels != 4)
        {
            std::cout << "Skipping glTF AO image " << i
                << " (" << image.name << ") because channels = "
                << image.channels << " (expected 3 or 4)\n";
            continue;
        }

        aoTextures.push_back(std::make_unique<Texture2D>(
            vkContext,
            bufferUtils,
            imageUtils,
            image.pixels.data(),
            static_cast<uint32_t>(image.width),
            static_cast<uint32_t>(image.height),
            static_cast<uint32_t>(image.channels),
            image.name.empty()
            ? ("glTF AO image " + std::to_string(i))
            : image.name,
            vk::Format::eR8G8B8A8Unorm));

        maps.occlusion[i] = static_cast<int>(aoTextures.size()) - 1;
    }

    for (size_t i = 0; i < imported.images.size(); ++i)
    {
        if (!imageUsedAsEmissive[i])
            continue;

        const auto& image = imported.images[i];

        if (image.pixels.empty() || image.width <= 0 || image.height <= 0)
            continue;

        if (image.channels != 3 && image.channels != 4)
        {
            std::cout << "Skipping glTF emissive image " << i
                << " (" << image.name << ") because channels = "
                << image.channels << " (expected 3 or 4)\n";
            continue;
        }

        emissiveTextures.push_back(std::make_unique<Texture2D>(
            vkContext,
            bufferUtils,
            imageUtils,
            image.pixels.data(),
            static_cast<uint32_t>(image.width),
            static_cast<uint32_t>(image.height),
            static_cast<uint32_t>(image.channels),
            image.name.empty()
            ? ("glTF emissive image " + std::to_string(i))
            : image.name,
            vk::Format::eR8G8B8A8Srgb));

        maps.emissive[i] = static_cast<int>(emissiveTextures.size()) - 1;
    }

    return maps;
}

void Renderer::createMaterialsFromGltf(
    const GltfSceneData& imported,
    const GltfTextureUploadMaps& textureMaps)
{
    // Slot 0 = default fallback material
    auto fallbackMaterial = std::make_unique<Material>(
        getDefaultTexture(),
        defaultNormalTexture.get(),
        defaultMetallicRoughnessTexture.get()
    );

    fallbackMaterial->setName("Default fallback material");
    fallbackMaterial->setOcclusionTexture(defaultAoTexture.get(), false);
    fallbackMaterial->setOcclusionStrength(0.0f);
    fallbackMaterial->setEmissiveTexture(defaultEmissiveTexture.get(), false);
    fallbackMaterial->setEmissiveFactor(glm::vec3(0.0f));

    materials.push_back(std::move(fallbackMaterial));

    for (const auto& importedMaterial : imported.materials)
    {
        Texture2D* assignedBaseColorTexture = &getDefaultTexture();

        if (importedMaterial.baseColorImageIndex >= 0 &&
            importedMaterial.baseColorImageIndex < static_cast<int>(textureMaps.baseColor.size()))
        {
            const int textureIndex =
                textureMaps.baseColor[importedMaterial.baseColorImageIndex];

            if (textureIndex >= 0 &&
                textureIndex < static_cast<int>(textures.size()))
            {
                assignedBaseColorTexture = textures[textureIndex].get();
            }
        }

        auto material = std::make_unique<Material>(
            *assignedBaseColorTexture,
            defaultNormalTexture.get(),
            defaultMetallicRoughnessTexture.get()
        );

        material->setBaseColorFactor(importedMaterial.baseColorFactor);
        material->setName(importedMaterial.name);
        material->setDoubleSided(importedMaterial.doubleSided);
        material->setMetallicFactor(importedMaterial.metallicFactor);
        material->setRoughnessFactor(importedMaterial.roughnessFactor);
        material->setNormalScale(importedMaterial.normalScale);
        material->setAlphaMode(importedMaterial.alphaMode);
        material->setAlphaCutoff(importedMaterial.alphaCutoff);

        Texture2D* assignedNormalTexture = defaultNormalTexture.get();
        bool hasRealNormalTexture = false;

        if (importedMaterial.normalImageIndex >= 0 &&
            importedMaterial.normalImageIndex < static_cast<int>(textureMaps.normal.size()))
        {
            const int textureIndex =
                textureMaps.normal[importedMaterial.normalImageIndex];

            if (textureIndex >= 0 &&
                textureIndex < static_cast<int>(normalTextures.size()))
            {
                assignedNormalTexture = normalTextures[textureIndex].get();
                hasRealNormalTexture = true;
            }
        }

        Texture2D* assignedMRTexture = defaultMetallicRoughnessTexture.get();
        bool hasRealMRTexture = false;

        if (importedMaterial.metallicRoughnessImageIndex >= 0 &&
            importedMaterial.metallicRoughnessImageIndex <
            static_cast<int>(textureMaps.metallicRoughness.size()))
        {
            const int textureIndex =
                textureMaps.metallicRoughness[importedMaterial.metallicRoughnessImageIndex];

            if (textureIndex >= 0 &&
                textureIndex < static_cast<int>(metallicRoughnessTextures.size()))
            {
                assignedMRTexture = metallicRoughnessTextures[textureIndex].get();
                hasRealMRTexture = true;
            }
        }

        Texture2D* assignedAOTexture = defaultAoTexture.get();
        bool hasRealAOTexture = false;

        if (importedMaterial.occlusionImageIndex >= 0 &&
            importedMaterial.occlusionImageIndex <
            static_cast<int>(textureMaps.occlusion.size()))
        {
            const int textureIndex =
                textureMaps.occlusion[importedMaterial.occlusionImageIndex];

            if (textureIndex >= 0 &&
                textureIndex < static_cast<int>(aoTextures.size()))
            {
                assignedAOTexture = aoTextures[textureIndex].get();
                hasRealAOTexture = true;
            }
        }

        Texture2D* assignedEmissiveTexture = defaultEmissiveTexture.get();
        bool hasRealEmissiveTexture = false;

        if (importedMaterial.emissiveImageIndex >= 0 &&
            importedMaterial.emissiveImageIndex <
            static_cast<int>(textureMaps.emissive.size()))
        {
            const int textureIndex =
                textureMaps.emissive[importedMaterial.emissiveImageIndex];

            if (textureIndex >= 0 &&
                textureIndex < static_cast<int>(emissiveTextures.size()))
            {
                assignedEmissiveTexture = emissiveTextures[textureIndex].get();
                hasRealEmissiveTexture = true;
            }
        }

        material->setNormalTexture(assignedNormalTexture, hasRealNormalTexture);
        material->setMetallicRoughnessTexture(assignedMRTexture, hasRealMRTexture);
        material->setOcclusionTexture(assignedAOTexture, hasRealAOTexture);
        material->setOcclusionStrength(importedMaterial.occlusionStrength);
        material->setEmissiveTexture(assignedEmissiveTexture, hasRealEmissiveTexture);
        material->setEmissiveFactor(importedMaterial.emissiveFactor);

        materials.push_back(std::move(material));
    }
}

void Renderer::createRenderablesFromGltf(const GltfSceneData& imported)
{
    if (imported.renderables.empty())
    {
        throw std::runtime_error("glTF import produced no renderables");
    }

    glm::vec3 minBounds(FLT_MAX);
    glm::vec3 maxBounds(-FLT_MAX);

    // Compute bounds
    for (const auto& importedRenderable : imported.renderables)
    {
        const glm::mat4 worldMatrix =
            importedRenderable.transform.toMatrix();

        for (const auto& vertex : importedRenderable.mesh.vertices)
        {
            glm::vec3 worldPos =
                glm::vec3(worldMatrix * glm::vec4(vertex.pos, 1.0f));

            minBounds = glm::min(minBounds, worldPos);
            maxBounds = glm::max(maxBounds, worldPos);
        }
    }

    glm::vec3 modelCenter = (minBounds + maxBounds) * 0.5f;
    glm::mat4 modelRootMatrix =
        glm::translate(glm::mat4(1.0f), -modelCenter);

    camera.frameBounds(minBounds, maxBounds);

    // Create renderables
    for (size_t i = 0; i < imported.renderables.size(); ++i)
    {
        const auto& importedRenderable = imported.renderables[i];

        gpuMeshes.push_back(std::make_unique<GpuMesh>(
            vkContext,
            bufferUtils,
            importedRenderable.mesh.vertices,
            importedRenderable.mesh.indices
        ));

        const int importedMaterialIndex =
            importedRenderable.materialIndex;

        const int rendererMaterialIndex =
            importedMaterialIndex >= 0 &&
            importedMaterialIndex + 1 < static_cast<int>(materials.size())
            ? importedMaterialIndex + 1
            : 0;

        Material& assignedMaterial =
            *materials[rendererMaterialIndex];

        Renderable& renderable = scene.addRenderable(
            *gpuMeshes.back(),
            assignedMaterial,
            "glTF " + std::to_string(i)
        );

        renderable.setMaterialIndex(
            static_cast<uint32_t>(rendererMaterialIndex));

        glm::mat4 originalMatrix =
            importedRenderable.transform.toMatrix();

        glm::mat4 finalMatrix =
            modelRootMatrix * originalMatrix;

        Transform& t = renderable.getTransform();
        t.useMatrixOverride = true;
        t.matrixOverride = finalMatrix;
    }
}


vk::Format Renderer::findSupportedFormat(const std::vector<vk::Format>& candidates,
    vk::ImageTiling tiling,
    vk::FormatFeatureFlags features)
{
    auto& physicalDevice = vkContext.getPhysicalDevice();

    auto formatIt = std::ranges::find_if(candidates, [&](auto const format) {
        vk::FormatProperties props = physicalDevice.getFormatProperties(format);
        return (((tiling == vk::ImageTiling::eLinear) && ((props.linearTilingFeatures & features) == features)) ||
            ((tiling == vk::ImageTiling::eOptimal) && ((props.optimalTilingFeatures & features) == features)));
        });

    if (formatIt == candidates.end())
    {
        throw std::runtime_error("failed to find supported format!");
    }

    return *formatIt;
}

vk::Format Renderer::findDepthFormat()
{
    return findSupportedFormat(
        { vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint },
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment
    );
}

bool Renderer::hasStencilComponent(vk::Format format)
{
    return format == vk::Format::eD32SfloatS8Uint ||
        format == vk::Format::eD24UnormS8Uint;
}


void Renderer::createCommandBuffers()
{   
    auto& device = vkContext.getDevice();

    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo
        .setCommandPool(*vkContext.getCommandPool())
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(MAX_FRAMES_IN_FLIGHT);

    commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
}

void Renderer::createHdrColorResources()
{
    auto extent = swapChainExtent;

    imageUtils.createImage(
        extent.width,
        extent.height,
        1,
        vk::SampleCountFlagBits::e1,
        hdrFormat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment |
        vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        hdrColorImage,
        hdrColorMemory);

    hdrColorView = imageUtils.createImageView(
        hdrColorImage,
        hdrFormat,
        vk::ImageAspectFlagBits::eColor,
        1);
}


void Renderer::createPostProcessDescriptorSetLayout()
{
    auto& device = vkContext.getDevice();

    vk::DescriptorSetLayoutBinding hdrBinding{};
    hdrBinding
        .setBinding(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eFragment);

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.setBindings(hdrBinding);

    postProcessDescriptorSetLayout =
        vk::raii::DescriptorSetLayout(device, layoutInfo);
}

void Renderer::createPostProcessSampler()
{
    auto& device = vkContext.getDevice();

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo
        .setMagFilter(vk::Filter::eLinear)
        .setMinFilter(vk::Filter::eLinear)
        .setAddressModeU(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeW(vk::SamplerAddressMode::eClampToEdge)
        .setAnisotropyEnable(VK_FALSE)
        .setBorderColor(vk::BorderColor::eIntOpaqueBlack)
        .setUnnormalizedCoordinates(VK_FALSE)
        .setCompareEnable(VK_FALSE)
        .setMipmapMode(vk::SamplerMipmapMode::eLinear)
        .setMinLod(0.0f)
        .setMaxLod(0.0f);

    postProcessSampler = vk::raii::Sampler(device, samplerInfo);
}

void Renderer::createPostProcessPipeline()
{
    auto& device = vkContext.getDevice();

    auto vertShaderModule =
        ShaderUtils::createShaderModule(device, "shaders/post_fullscreen.spv");

    auto fragShaderModule =
        ShaderUtils::createShaderModule(device, "shaders/post_hdr.spv");

    vk::PipelineShaderStageCreateInfo vertStage{};
    vertStage
        .setStage(vk::ShaderStageFlagBits::eVertex)
        .setModule(*vertShaderModule)
        .setPName("main");

    vk::PipelineShaderStageCreateInfo fragStage{};
    fragStage
        .setStage(vk::ShaderStageFlagBits::eFragment)
        .setModule(*fragShaderModule)
        .setPName("main");

    std::array stages = { vertStage, fragStage };

    vk::PipelineVertexInputStateCreateInfo vertexInput{};

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly
        .setTopology(vk::PrimitiveTopology::eTriangleList)
        .setPrimitiveRestartEnable(VK_FALSE);

    vk::PipelineViewportStateCreateInfo viewportState{};
    viewportState
        .setViewportCount(1)
        .setScissorCount(1);

    vk::PipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer
        .setDepthClampEnable(VK_FALSE)
        .setRasterizerDiscardEnable(VK_FALSE)
        .setPolygonMode(vk::PolygonMode::eFill)
        .setCullMode(vk::CullModeFlagBits::eNone)
        .setFrontFace(vk::FrontFace::eCounterClockwise)
        .setDepthBiasEnable(VK_FALSE)
        .setLineWidth(1.0f);

    vk::PipelineMultisampleStateCreateInfo multisampling{};
    multisampling
        .setRasterizationSamples(vk::SampleCountFlagBits::e1)
        .setSampleShadingEnable(VK_FALSE);

    vk::PipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil
        .setDepthTestEnable(VK_FALSE)
        .setDepthWriteEnable(VK_FALSE)
        .setStencilTestEnable(VK_FALSE);

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment
        .setBlendEnable(VK_FALSE)
        .setColorWriteMask(
            vk::ColorComponentFlagBits::eR |
            vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB |
            vk::ColorComponentFlagBits::eA);

    vk::PipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending
        .setLogicOpEnable(VK_FALSE)
        .setAttachments(colorBlendAttachment);

    std::vector<vk::DynamicState> dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor
    };

    vk::PipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.setDynamicStates(dynamicStates);

    vk::PushConstantRange pushRange{};
    pushRange
        .setStageFlags(vk::ShaderStageFlagBits::eFragment)
        .setOffset(0)
        .setSize(sizeof(glm::vec4));

    vk::PipelineLayoutCreateInfo layoutInfo{};
    layoutInfo
        .setSetLayouts(*postProcessDescriptorSetLayout)
        .setPushConstantRanges(pushRange);

    postProcessPipelineLayout =
        vk::raii::PipelineLayout(device, layoutInfo);

    vk::StructureChain<
        vk::GraphicsPipelineCreateInfo,
        vk::PipelineRenderingCreateInfo
    > pipelineChain{};

    pipelineChain.get<vk::GraphicsPipelineCreateInfo>()
        .setStages(stages)
        .setPVertexInputState(&vertexInput)
        .setPInputAssemblyState(&inputAssembly)
        .setPViewportState(&viewportState)
        .setPRasterizationState(&rasterizer)
        .setPMultisampleState(&multisampling)
        .setPDepthStencilState(&depthStencil)
        .setPColorBlendState(&colorBlending)
        .setPDynamicState(&dynamicState)
        .setLayout(*postProcessPipelineLayout)
        .setRenderPass(vk::RenderPass{});

    pipelineChain.get<vk::PipelineRenderingCreateInfo>()
        .setColorAttachmentFormats(swapChainSurfaceFormat.format);

    postProcessPipeline =
        vk::raii::Pipeline(
            device,
            nullptr,
            pipelineChain.get<vk::GraphicsPipelineCreateInfo>());
}

void Renderer::createPostProcessDescriptorSet()
{
    auto& device = vkContext.getDevice();

    vk::DescriptorPoolSize poolSize{};
    poolSize
        .setType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1);

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo
        .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
        .setMaxSets(1)
        .setPoolSizes(poolSize);

    postProcessDescriptorPool =
        vk::raii::DescriptorPool(device, poolInfo);

    vk::DescriptorSetLayout layout = *postProcessDescriptorSetLayout;

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo
        .setDescriptorPool(*postProcessDescriptorPool)
        .setSetLayouts(layout);

    postProcessDescriptorSets =
        vk::raii::DescriptorSets(device, allocInfo);

    vk::DescriptorImageInfo hdrInfo{};
    hdrInfo
        .setSampler(*postProcessSampler)
        .setImageView(*hdrColorView)
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

    vk::WriteDescriptorSet write{};
    write
        .setDstSet(*postProcessDescriptorSets[0])
        .setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(hdrInfo);

    device.updateDescriptorSets(write, nullptr);
}

void Renderer::loadModel()
{
    std::cout << "Loading model: " << MODEL_PATH << std::endl;

    meshData = modelLoader.load(MODEL_PATH);

    if (meshData.empty())
    {
        throw std::runtime_error("loaded mesh is empty");
    }

    std::cout << "Loaded mesh: "
        << meshData.vertices.size() << " vertices, "
        << meshData.indices.size() << " indices\n";

}

Texture2D& Renderer::getDefaultTexture()
{
    if (textures.empty() || !textures[0])
    {
        throw std::runtime_error("default texture is not available");
    }

    return *textures[0];
}

Material& Renderer::getDefaultMaterial()
{
    if (materials.empty() || !materials[0])
    {
        throw std::runtime_error("default material is not available");
    }

    return *materials[0];
}


void Renderer::createUniformBuffers()
{
    uniformBuffers.clear();
    uniformBuffersMemory.clear();
    uniformBuffersMapped.clear();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vk::DeviceSize         bufferSize = sizeof(UniformBufferObject);
        vk::raii::Buffer       buffer({});
        vk::raii::DeviceMemory bufferMem({});
        bufferUtils.createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
        uniformBuffers.emplace_back(std::move(buffer));
        uniformBuffersMemory.emplace_back(std::move(bufferMem));
        uniformBuffersMapped.emplace_back(uniformBuffersMemory[i].mapMemory(0, bufferSize));
    }
}

void Renderer::createDescriptorPool()
{
    cleanupDescriptorResources();

    uint32_t materialCount = static_cast<uint32_t>(materials.size());

    std::array poolSizes{
        vk::DescriptorPoolSize(
            vk::DescriptorType::eUniformBuffer,
            MAX_FRAMES_IN_FLIGHT),

        vk::DescriptorPoolSize(
            vk::DescriptorType::eCombinedImageSampler,
            materialCount * 5 + 4) // 5 bindings per material + 4 for IBL
    };

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo
        .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
        .setMaxSets(MAX_FRAMES_IN_FLIGHT + materialCount + 1) // +1 for IBL set
        .setPoolSizes(poolSizes);

    descriptorPool = vk::raii::DescriptorPool(vkContext.getDevice(), poolInfo);
}

void Renderer::createDescriptorSets()
{
    auto& device = vkContext.getDevice();

    // =========================
    // Set 0  Frame UBO sets
    // =========================

    std::vector<vk::DescriptorSetLayout> layouts(
        MAX_FRAMES_IN_FLIGHT,
        *frameDescriptorSetLayout
    );

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo
        .setDescriptorPool(*descriptorPool)
        .setSetLayouts(layouts);

    frameDescriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vk::DescriptorBufferInfo bufferInfo{};
        bufferInfo
            .setBuffer(*uniformBuffers[i])
            .setOffset(0)
            .setRange(sizeof(UniformBufferObject));

        vk::WriteDescriptorSet uboWrite{};
        uboWrite
            .setDstSet(*frameDescriptorSets[i])
            .setDstBinding(0)
            .setDescriptorType(vk::DescriptorType::eUniformBuffer)
            .setDescriptorCount(1)
            .setBufferInfo(bufferInfo);

        device.updateDescriptorSets(uboWrite, nullptr);
    }

    // =========================
    // Set 2  IBL descriptor set
    // =========================

    vk::DescriptorSetAllocateInfo iblAllocInfo{};
    iblAllocInfo
        .setDescriptorPool(*descriptorPool)
        .setSetLayouts(*iblDescriptorSetLayout);

    vk::raii::DescriptorSets iblSets(device, iblAllocInfo);

    iblDescriptorSet = std::move(iblSets.front());

    // NOTE:
    // We do NOT update the descriptors here yet unless
    // fallback textures already exist.
}

void Renderer::createSyncObjects()
{
    auto& device = vkContext.getDevice();

    assert(presentCompleteSemaphores.empty() &&
        renderFinishedSemaphores.empty() &&
        inFlightFences.empty());

    // Per-swapchain-image semaphores for present wait/signal lifetime.
    renderFinishedSemaphores.clear();
    renderFinishedSemaphores.reserve(swapChainImages.size());
    for (size_t i = 0; i < swapChainImages.size(); ++i)
    {
        renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
    }

    // Per-frame acquire semaphores and fences.
    presentCompleteSemaphores.clear();
    inFlightFences.clear();
    presentCompleteSemaphores.reserve(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.reserve(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());

        vk::FenceCreateInfo fenceInfo{};
        fenceInfo.setFlags(vk::FenceCreateFlagBits::eSignaled);
        inFlightFences.emplace_back(device, fenceInfo);
    }

    imagesInFlight.assign(swapChainImages.size(), vk::Fence{});
}

void Renderer::updateUniformBuffer(uint32_t currentFrame)
{
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float>(currentTime - startTime).count();

    if (animateModel)
    {
        currentAnimationAngle = time * glm::radians(rotationSpeed);
    }

    UniformBufferObject ubo{};

    const auto& extent = swapChainExtent;
    float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);

    ubo.view = camera.getViewMatrix();
    ubo.proj = camera.getProjectionMatrix(aspect);

    ubo.invView = glm::inverse(ubo.view);
    ubo.invProj = glm::inverse(ubo.proj);

    ubo.lightDirection = glm::vec4(glm::normalize(lightDirection), 0.0f);
    
    ubo.lightColor = glm::vec4(lightColor * lightIntensity, 1.0f);
    ubo.ambientColor = glm::vec4(ambientColor * ambientIntensity, 1.0f);

    ubo.cameraPosition = glm::vec4(camera.getPosition(), 1.0f);

    ubo.environmentParams0 = glm::vec4(
        skyboxExposure,
        skyboxLod,
        iblIntensity,
        showSkybox ? 1.0f : 0.0f
    );

    ubo.environmentParams1 = glm::vec4(
        diffuseIBLIntensity,
        specularIBLIntensity,
        debugReflectionOnly ? 1.0f : 0.0f,
        enableIBL ? 1.0f : 0.0f
    );

    ubo.postProcessParams = glm::vec4(
        postExposure,
        toneMappingEnabled ? 1.0f : 0.0f,
        gammaEnabled ? 1.0f : 0.0f,
        glm::radians(environmentRotationDegrees)
    );

    ubo.environmentControlParams = glm::vec4(
        rotateSkybox ? 1.0f : 0.0f,
        rotateIBLLighting ? 1.0f : 0.0f,
        debugSkyboxFaces ? 1.0f : 0.0f,
        0.0f
    );

    ubo.debugParams = glm::ivec4(uiState.debugViewMode, 0, 0, 0);

    const uint32_t mipLevels =
        prefilterRenderer
        ? prefilterRenderer->getDebugRuntimePrefilteredMipLevels()
        : 1;

    float maxPrefilterMip =
        mipLevels > 0
        ? static_cast<float>(mipLevels - 1)
        : 0.0f;

    debugSpecularMip = std::clamp(debugSpecularMip, 0.0f, maxPrefilterMip);

    ubo.specularDebugParams = glm::vec4(
        debugForceSpecularMip ? 1.0f : 0.0f,
        debugSpecularMip,
        maxPrefilterMip,
        roughnessMipScale);

    ubo.specularCurveParams = glm::vec4(
        roughnessMipBias,
        0.0f,
        0.0f,
        0.0f);

    lastUbo = ubo;


    std::memcpy(uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));





};

void Renderer::createMaterialDescriptorSets()
{
    auto& device = vkContext.getDevice();

    if (materials.empty())
    {
        materialDescriptorSets.clear();
        return;
    }

    std::vector<vk::DescriptorSetLayout> layouts(
        materials.size(),
        *materialDescriptorSetLayout
    );

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo
        .setDescriptorPool(*descriptorPool)
        .setSetLayouts(layouts);

    materialDescriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    for (size_t i = 0; i < materials.size(); ++i)
    {
        MaterialImageWrite baseColorWrite =
            materials[i]->makeImageWrite(*materialDescriptorSets[i], 0);

        MaterialImageWrite normalWrite =
            materials[i]->makeNormalImageWrite(*materialDescriptorSets[i], 1);

        MaterialImageWrite metallicRoughnessWrite =
            materials[i]->makeMetallicRoughnessImageWrite(*materialDescriptorSets[i], 2);

        MaterialImageWrite aoWrite =
            materials[i]->makeOcclusionImageWrite(*materialDescriptorSets[i], 3);

        MaterialImageWrite emissiveWrite =
            materials[i]->makeEmissiveImageWrite(*materialDescriptorSets[i], 4);

        std::array<vk::WriteDescriptorSet, 5> descriptorWrites = {
            baseColorWrite.write,
            normalWrite.write,
            metallicRoughnessWrite.write,
            aoWrite.write,
            emissiveWrite.write
        };

        device.updateDescriptorSets(descriptorWrites, nullptr);
    }
}

void Renderer::drawPostProcessToSwapchain(
    vk::raii::CommandBuffer& commandBuffer,
    uint32_t imageIndex)
{
    if (postProcessDescriptorSets.empty())
    {
        throw std::runtime_error(
            "drawPostProcessToSwapchain: postProcessDescriptorSets is empty");
    }
    
    const auto& extent = swapChainExtent;

    vk::ClearValue clearColor = vk::ClearColorValue(
        std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f });

    vk::RenderingAttachmentInfo colorAttachment{};
    colorAttachment
        .setImageView(*swapChainImageViews[imageIndex])
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(clearColor);

    vk::RenderingInfo renderingInfo{};
    renderingInfo
        .setRenderArea(vk::Rect2D{ {0, 0}, extent })
        .setLayerCount(1)
        .setColorAttachments(colorAttachment);

    commandBuffer.beginRendering(renderingInfo);

    commandBuffer.setViewport(
        0,
        vk::Viewport(
            0.0f,
            0.0f,
            static_cast<float>(extent.width),
            static_cast<float>(extent.height),
            0.0f,
            1.0f));

    commandBuffer.setScissor(
        0,
        vk::Rect2D(vk::Offset2D(0, 0), extent));

    commandBuffer.bindPipeline(
        vk::PipelineBindPoint::eGraphics,
        *postProcessPipeline);

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *postProcessPipelineLayout,
        0,
        *postProcessDescriptorSets[0],
        {});

    glm::vec4 postParams(
        postExposure,
        toneMappingEnabled ? 1.0f : 0.0f,
        gammaEnabled ? 1.0f : 0.0f,
        0.0f);

    commandBuffer.pushConstants<glm::vec4>(
        *postProcessPipelineLayout,
        vk::ShaderStageFlagBits::eFragment,
        0,
        postParams);

    commandBuffer.draw(3, 1, 0, 0);

    renderImGui(*commandBuffer);

    commandBuffer.endRendering();
}


void Renderer::drawFrame()
{
    static auto lastFrameTime = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    frameTimeMs = std::chrono::duration<float, std::milli>(now - lastFrameTime).count();
    lastFrameTime = now;
    fps = frameTimeMs > 0.0f ? 1000.0f / frameTimeMs : 0.0f;

    auto& device = vkContext.getDevice();
    auto& queue = vkContext.getQueue();

    vk::Result waitResult =
        device.waitForFences(*inFlightFences[frameIndex], VK_TRUE, UINT64_MAX);

    if (waitResult != vk::Result::eSuccess)
    {
        throw std::runtime_error("failed waiting for in-flight fence");
    }

    uint32_t imageIndex = 0;
    vk::Result result{};

    try
    {
        auto acquireResult =
            swapChain.acquireNextImage(
                UINT64_MAX,
                *presentCompleteSemaphores[frameIndex],
                nullptr);

        result = acquireResult.result;
        imageIndex = acquireResult.value;
    }
    catch (const vk::OutOfDateKHRError&)
    {
        window.resetResizedFlag();
        recreateSwapChain();
        return;
    }

    if (result == vk::Result::eErrorOutOfDateKHR)
    {
        window.resetResizedFlag();
        recreateSwapChain();
        return;
    }

    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
    {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    if (imageIndex >= swapChainImages.size())
    {
        throw std::runtime_error("imageIndex out of range for swapChainImages");
    }

    if (imageIndex >= renderFinishedSemaphores.size())
    {
        throw std::runtime_error("imageIndex out of range for renderFinishedSemaphores");
    }

    if (imageIndex >= imagesInFlight.size())
    {
        throw std::runtime_error("imageIndex out of range for imagesInFlight");
    }

    if (imagesInFlight[imageIndex])
    {
        vk::Result imageFenceWaitResult =
            device.waitForFences(imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);

        if (imageFenceWaitResult != vk::Result::eSuccess)
        {
            throw std::runtime_error("failed waiting for image fence");
        }
    }

    imagesInFlight[imageIndex] = *inFlightFences[frameIndex];

    beginImGuiFrame();
    updateCameraControls();
    buildImGui();

    updateUniformBuffer(frameIndex);

    device.resetFences(*inFlightFences[frameIndex]);

    commandBuffers[frameIndex].reset();
    recordCommandBuffer(imageIndex);

    vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;

    vk::SubmitInfo submitInfo{};
    submitInfo
        .setWaitSemaphores(*presentCompleteSemaphores[frameIndex])
        .setWaitDstStageMask(waitStage)
        .setCommandBuffers(*commandBuffers[frameIndex])
        .setSignalSemaphores(*renderFinishedSemaphores[imageIndex]);

    try
    {
        queue.submit(submitInfo, *inFlightFences[frameIndex]);
    }
    catch (const vk::SystemError& err)
    {
        std::cerr << "Queue submit failed: " << err.what() << std::endl;
        throw;
    }

    vk::PresentInfoKHR presentInfo{};
    presentInfo
        .setWaitSemaphores(*renderFinishedSemaphores[imageIndex])
        .setSwapchains(*swapChain)
        .setPImageIndices(&imageIndex);

    try
    {
        result = queue.presentKHR(presentInfo);
    }
    catch (const vk::OutOfDateKHRError&)
    {
        window.resetResizedFlag();
        recreateSwapChain();
        return;
    }

    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR ||
        window.wasResized())
    {
        window.resetResizedFlag();
        recreateSwapChain();
        return;
    }

    if (result != vk::Result::eSuccess)
    {
        throw std::runtime_error("failed to present swap chain image!");
    }

    frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
}

void Renderer::recordCommandBuffer(uint32_t imageIndex)
{
    auto& commandBuffer = commandBuffers[frameIndex];
    commandBuffer.begin(vk::CommandBufferBeginInfo{});

    vk::CommandBuffer cmd = *commandBuffer;
    const auto& extent = swapChainExtent;

    transitionToColorAttachment(
        cmd,
        *colorImage,
        vk::ImageLayout::eUndefined);

    transitionToColorAttachment(
        cmd,
        *hdrColorImage,
        vk::ImageLayout::eUndefined);

    vk::ImageLayout swapChainOldLayout =
        swapChainImageInitialized[imageIndex]
        ? vk::ImageLayout::ePresentSrcKHR
        : vk::ImageLayout::eUndefined;

    

    transitionToDepthAttachment(
        cmd,
        *depthImage,
        depthAspect);

    vk::ClearValue clearColorValue = vk::ClearColorValue(
        clearColor.r,
        clearColor.g,
        clearColor.b,
        clearColor.a);

    vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

    vk::RenderingAttachmentInfo colorAttachmentInfo{};
    colorAttachmentInfo
        .setImageView(*colorImageView)
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eDontCare)
        .setClearValue(clearColorValue)
        .setResolveMode(vk::ResolveModeFlagBits::eAverage)
        .setResolveImageView(*hdrColorView)
        .setResolveImageLayout(vk::ImageLayout::eColorAttachmentOptimal);

    vk::RenderingAttachmentInfo depthAttachmentInfo{};
    depthAttachmentInfo
        .setImageView(*depthImageView)
        .setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eDontCare)
        .setClearValue(clearDepth);

    vk::RenderingInfo renderingInfo{};
    renderingInfo
        .setRenderArea(vk::Rect2D{}.setOffset(vk::Offset2D{ 0, 0 }).setExtent(extent))
        .setLayerCount(1)
        .setColorAttachments(colorAttachmentInfo)
        .setPDepthAttachment(&depthAttachmentInfo);

    commandBuffer.beginRendering(renderingInfo);

    commandBuffer.setViewport(
        0,
        vk::Viewport(
            0.0f,
            0.0f,
            static_cast<float>(extent.width),
            static_cast<float>(extent.height),
            0.0f,
            1.0f));

    commandBuffer.setScissor(
        0,
        vk::Rect2D(vk::Offset2D(0, 0), extent));

    // Draw skybox first
    drawSkybox(commandBuffer, imageIndex);

    // -------------------------
    // Pass 1: OPAQUE + MASK
    // -------------------------
    for (auto& renderable : scene.getRenderables())
    {
        Material& renderableMaterial = renderable.getMaterial();

        const bool isBlendMaterial = (renderableMaterial.getAlphaMode() == "BLEND");
        if (isBlendMaterial)
        {
            continue;
        }

        uint32_t materialIndex = renderable.getMaterialIndex();
        assert(materialIndex < materials.size());
        
        commandBuffer.bindVertexBuffers(
            0,
            *renderable.getMesh().getVertexBuffer(),
            { 0 });

        commandBuffer.bindIndexBuffer(
            *renderable.getMesh().getIndexBuffer(),
            0,
            vk::IndexType::eUint32);

        vk::Pipeline activePipeline = *solidPipeline;

        if (uiState.wireframeRequested)
        {
            if (renderableMaterial.isDoubleSided())
            {
                if (wireframeDoubleSidedPipeline != nullptr)
                {
                    activePipeline = *wireframeDoubleSidedPipeline;
                }
                else
                {
                    activePipeline = *solidDoubleSidedPipeline;
                }
            }
            else
            {
                if (wireframePipeline != nullptr)
                {
                    activePipeline = *wireframePipeline;
                }
                else
                {
                    activePipeline = *solidPipeline;
                }
            }
        }
        else
        {
            if (renderableMaterial.isDoubleSided())
            {
                activePipeline = *solidDoubleSidedPipeline;
            }
            else
            {
                activePipeline = *solidPipeline;
            }
        }

        commandBuffer.bindPipeline(
            vk::PipelineBindPoint::eGraphics,
            activePipeline);

        std::array<vk::DescriptorSet, 3> sets = {
            *frameDescriptorSets[frameIndex],
            *materialDescriptorSets[materialIndex],
            *iblDescriptorSet
        };

        commandBuffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            *pipelineLayout,
            0,
            sets,
            {});

        PushConstantData pushData{};
        pushData.model = renderable.getTransform().toMatrix();

        if (animateModel)
        {
            pushData.model = glm::rotate(
                pushData.model,
                currentAnimationAngle,
                glm::vec3(0.0f, 0.0f, 1.0f));
        }

        glm::mat3 normalMatrix =
            glm::transpose(glm::inverse(glm::mat3(pushData.model)));

        pushData.normalMatrix = normalMatrix;
        pushData.baseColorFactor = renderableMaterial.getBaseColorFactor();

        const bool isMaskMaterial = (renderableMaterial.getAlphaMode() == "MASK");

        pushData.materialParams = glm::vec4(
            renderableMaterial.getMetallicFactor(),
            renderableMaterial.getRoughnessFactor(),
            renderableMaterial.getNormalScale(),
            renderableMaterial.getAlphaCutoff());

        pushData.alphaModeParams = glm::vec4(
            isMaskMaterial ? 1.0f : 0.0f,
            0.0f, // isBlend = false in opaque/mask pass
            renderableMaterial.getOcclusionStrength(),
            0.0f);

        
        pushData.emissiveFactor =
            glm::vec4(renderableMaterial.getEmissiveFactor(), 1.0f);

        

        cmd.pushConstants(
            *pipelineLayout,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            0,
            sizeof(PushConstantData),
            &pushData);

        commandBuffer.drawIndexed(
            renderable.getMesh().getIndexCount(),
            1,
            0,
            0,
            0);
    }

    // -------------------------
    // Pass 2: BLEND
    // -------------------------
    std::vector<Renderable*> transparentRenderables;
    transparentRenderables.reserve(scene.getRenderables().size());

    for (auto& renderable : scene.getRenderables())
    {
        if (renderable.getMaterial().getAlphaMode() == "BLEND")
        {
            transparentRenderables.push_back(&renderable);
        }
    }

    // Back-to-front sort for transparent objects
    const glm::vec3 cameraPos = camera.getPosition();

    std::sort(
        transparentRenderables.begin(),
        transparentRenderables.end(),
        [&](const Renderable* a, const Renderable* b)
        {
            glm::vec3 aPos = glm::vec3(a->getTransform().toMatrix()[3]);
            glm::vec3 bPos = glm::vec3(b->getTransform().toMatrix()[3]);

            float da = glm::dot(aPos - cameraPos, aPos - cameraPos);
            float db = glm::dot(bPos - cameraPos, bPos - cameraPos);

            return da > db; // back-to-front
        });

    for (Renderable* renderable : transparentRenderables)
    {
        Material& renderableMaterial = renderable->getMaterial();

        uint32_t materialIndex = renderable->getMaterialIndex();
        assert(materialIndex < materials.size());

        commandBuffer.bindVertexBuffers(
            0,
            *renderable->getMesh().getVertexBuffer(),
            { 0 });

        commandBuffer.bindIndexBuffer(
            *renderable->getMesh().getIndexBuffer(),
            0,
            vk::IndexType::eUint32);

        vk::Pipeline activePipeline =
            renderableMaterial.isDoubleSided()
            ? *transparentDoubleSidedPipeline
            : *transparentPipeline;

        commandBuffer.bindPipeline(
            vk::PipelineBindPoint::eGraphics,
            activePipeline);

        std::array<vk::DescriptorSet, 3> sets = {
            *frameDescriptorSets[frameIndex],
            *materialDescriptorSets[materialIndex],
            *iblDescriptorSet
        };

        commandBuffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            *pipelineLayout,
            0,
            sets,
            {});

        PushConstantData pushData{};
        pushData.model = renderable->getTransform().toMatrix();

        if (animateModel)
        {
            pushData.model = glm::rotate(
                pushData.model,
                currentAnimationAngle,
                glm::vec3(0.0f, 0.0f, 1.0f));
        }

        glm::mat3 normalMatrix =
            glm::transpose(glm::inverse(glm::mat3(pushData.model)));

        pushData.normalMatrix = normalMatrix;
        pushData.baseColorFactor = renderableMaterial.getBaseColorFactor();

        pushData.materialParams = glm::vec4(
            renderableMaterial.getMetallicFactor(),
            renderableMaterial.getRoughnessFactor(),
            renderableMaterial.getNormalScale(),
            renderableMaterial.getAlphaCutoff());

        const bool isMaskMaterial = (renderableMaterial.getAlphaMode() == "MASK");
        const bool isBlendMaterial = (renderableMaterial.getAlphaMode() == "BLEND");

        pushData.alphaModeParams = glm::vec4(
            isMaskMaterial ? 1.0f : 0.0f,
            isBlendMaterial ? 1.0f : 0.0f,
            renderableMaterial.getOcclusionStrength(),
            0.0f);

        pushData.emissiveFactor =
            glm::vec4(renderableMaterial.getEmissiveFactor(), 1.0f);

        

        cmd.pushConstants(
            *pipelineLayout,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            0,
            sizeof(PushConstantData),
            &pushData);

        commandBuffer.drawIndexed(
            renderable->getMesh().getIndexCount(),
            1,
            0,
            0,
            0);
    }

  //  renderImGui(*commandBuffer);

    commandBuffer.endRendering();

    transitionToShaderReadOnly(
        cmd,
        *hdrColorImage,
        vk::ImageLayout::eColorAttachmentOptimal);

    transitionToColorAttachment(
        cmd,
        swapChainImages[imageIndex],
        swapChainOldLayout);

    drawPostProcessToSwapchain(commandBuffer, imageIndex);

    transitionToPresent(
        cmd,
        swapChainImages[imageIndex]);

    swapChainImageInitialized[imageIndex] = true;

    commandBuffer.end();
}

void Renderer::createIrradianceCubemapFromDDS(const std::string& path)
{
    createCubemapFromDDS(
        path,
        irradianceCubeImage,
        irradianceCubeMemory,
        irradianceCubeView,
        irradianceCubeSampler,
        false
    );
}

void Renderer::createPrefilteredCubemapFromDDS(const std::string& path)
{
    createCubemapFromDDS(
        path,
        prefilteredCubeImage,
        prefilteredCubeMemory,
        prefilteredCubeView,
        prefilteredCubeSampler,
        true
    );
}

void Renderer::createCubemapFromDDS(
    const std::string& path,
    vk::raii::Image& outImage,
    vk::raii::DeviceMemory& outMemory,
    vk::raii::ImageView& outView,
    vk::raii::Sampler& outSampler,
    bool allowMipSampling)
{
    DdsCubemapData dds = DdsUtils::loadCubemapDDS(path);

    struct PackedRegion
    {
        vk::BufferImageCopy copyRegion{};
        size_t size = 0;
        const std::vector<uint8_t>* pixels = nullptr;
    };

    std::vector<PackedRegion> packedRegions;
    packedRegions.reserve(dds.subresources.size());

    vk::DeviceSize totalUploadSize = 0;

    for (const auto& sub : dds.subresources)
    {
        PackedRegion region{};
        region.size = sub.slicePitch;
        region.pixels = &sub.pixels;

        region.copyRegion
            .setBufferOffset(totalUploadSize)
            .setBufferRowLength(0)
            .setBufferImageHeight(0)
            .setImageSubresource(
                vk::ImageSubresourceLayers{}
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setMipLevel(sub.mipLevel)
                .setBaseArrayLayer(sub.arrayLayer)
                .setLayerCount(1))
            .setImageOffset(vk::Offset3D{ 0, 0, 0 })
            .setImageExtent(vk::Extent3D{ sub.width, sub.height, 1 });

        packedRegions.push_back(region);
        totalUploadSize += static_cast<vk::DeviceSize>(sub.slicePitch);
    }

    vk::raii::Buffer stagingBuffer{ nullptr };
    vk::raii::DeviceMemory stagingMemory{ nullptr };

    bufferUtils.createBuffer(
        totalUploadSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer,
        stagingMemory
    );

    {
        void* mapped = stagingMemory.mapMemory(0, totalUploadSize);
        uint8_t* dst = static_cast<uint8_t*>(mapped);

        vk::DeviceSize currentOffset = 0;
        for (const auto& region : packedRegions)
        {
            std::memcpy(dst + currentOffset, region.pixels->data(), region.size);
            currentOffset += static_cast<vk::DeviceSize>(region.size);
        }

        stagingMemory.unmapMemory();
    }

    auto& device = vkContext.getDevice();

    vk::ImageCreateInfo imageInfo{};
    imageInfo
        .setFlags(vk::ImageCreateFlagBits::eCubeCompatible)
        .setImageType(vk::ImageType::e2D)
        .setFormat(dds.format)
        .setExtent(vk::Extent3D{ dds.width, dds.height, 1 })
        .setMipLevels(dds.mipLevels)
        .setArrayLayers(6)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setTiling(vk::ImageTiling::eOptimal)
        .setUsage(
            vk::ImageUsageFlagBits::eTransferDst |
            vk::ImageUsageFlagBits::eSampled)
        .setSharingMode(vk::SharingMode::eExclusive)
        .setInitialLayout(vk::ImageLayout::eUndefined);

    outImage = vk::raii::Image(device, imageInfo);

    vk::MemoryRequirements memRequirements = outImage.getMemoryRequirements();

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo
        .setAllocationSize(memRequirements.size)
        .setMemoryTypeIndex(
            bufferUtils.findMemoryType(
                memRequirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal));

    outMemory = vk::raii::DeviceMemory(device, allocInfo);
    outImage.bindMemory(*outMemory, 0);

    auto cmd = bufferUtils.beginSingleTimeCommands();

    vk::ImageMemoryBarrier toTransfer{};
    toTransfer
        .setOldLayout(vk::ImageLayout::eUndefined)
        .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*outImage)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(dds.mipLevels)
            .setBaseArrayLayer(0)
            .setLayerCount(6))
        .setSrcAccessMask({})
        .setDstAccessMask(vk::AccessFlagBits::eTransferWrite);

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eTransfer,
        {},
        nullptr,
        nullptr,
        toTransfer
    );

    std::vector<vk::BufferImageCopy> copyRegions;
    copyRegions.reserve(packedRegions.size());

    for (const auto& region : packedRegions)
    {
        copyRegions.push_back(region.copyRegion);
    }

    cmd.copyBufferToImage(
        *stagingBuffer,
        *outImage,
        vk::ImageLayout::eTransferDstOptimal,
        copyRegions
    );

    vk::ImageMemoryBarrier toShaderRead{};
    toShaderRead
        .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
        .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*outImage)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(dds.mipLevels)
            .setBaseArrayLayer(0)
            .setLayerCount(6))
        .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader,
        {},
        nullptr,
        nullptr,
        toShaderRead
    );

    bufferUtils.endSingleTimeCommands(cmd);

    vk::ImageViewCreateInfo viewInfo{};
    viewInfo
        .setImage(*outImage)
        .setViewType(vk::ImageViewType::eCube)
        .setFormat(dds.format)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(dds.mipLevels)
            .setBaseArrayLayer(0)
            .setLayerCount(6));

    outView = vk::raii::ImageView(device, viewInfo);

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo
        .setMagFilter(vk::Filter::eLinear)
        .setMinFilter(vk::Filter::eLinear)
        .setMipmapMode(vk::SamplerMipmapMode::eLinear)
        .setAddressModeU(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeW(vk::SamplerAddressMode::eClampToEdge)
        .setMipLodBias(0.0f)
        .setAnisotropyEnable(VK_FALSE)
        .setCompareEnable(VK_FALSE)
        .setMinLod(0.0f)
        .setMaxLod(allowMipSampling ? static_cast<float>(dds.mipLevels - 1) : 0.0f)
        .setBorderColor(vk::BorderColor::eFloatOpaqueWhite)
        .setUnnormalizedCoordinates(VK_FALSE);

    outSampler = vk::raii::Sampler(device, samplerInfo);
}



void Renderer::transitionImageLayout(
    vk::CommandBuffer cmd,
    vk::Image image,
    vk::ImageLayout oldLayout,
    vk::ImageLayout newLayout,
    vk::AccessFlags srcAccessMask,
    vk::AccessFlags dstAccessMask,
    vk::PipelineStageFlags srcStage,
    vk::PipelineStageFlags dstStage,
    vk::ImageAspectFlags aspectMask)
{
    vk::ImageMemoryBarrier barrier{};
    barrier
        .setOldLayout(oldLayout)
        .setNewLayout(newLayout)
        .setSrcAccessMask(srcAccessMask)
        .setDstAccessMask(dstAccessMask)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(image);

    barrier.subresourceRange
        .setAspectMask(aspectMask)
        .setBaseMipLevel(0)
        .setLevelCount(1)
        .setBaseArrayLayer(0)
        .setLayerCount(1);

    cmd.pipelineBarrier(
        srcStage,
        dstStage,
        {},
        nullptr,
        nullptr,
        barrier);
}

void Renderer::transitionToColorAttachment(
    vk::CommandBuffer cmd,
    vk::Image image,
    vk::ImageLayout oldLayout)
{
    transitionImageLayout(
        cmd,
        image,
        oldLayout,
        vk::ImageLayout::eColorAttachmentOptimal,
        {},
        vk::AccessFlagBits::eColorAttachmentWrite,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor);
}

void Renderer::transitionToPresent(
    vk::CommandBuffer cmd,
    vk::Image image)
{
    transitionImageLayout(
        cmd,
        image,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits::eColorAttachmentWrite,
        {},
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eBottomOfPipe,
        vk::ImageAspectFlagBits::eColor);
}

void Renderer::transitionToDepthAttachment(
    vk::CommandBuffer cmd,
    vk::Image image,
    vk::ImageAspectFlags aspectMask)
{
    transitionImageLayout(
        cmd,
        image,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthAttachmentOptimal,
        {},
        vk::AccessFlagBits::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests,
        vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests,
        aspectMask);
}

void Renderer::transitionToShaderReadOnly(
    vk::CommandBuffer cmd,
    vk::Image image,
    vk::ImageLayout oldLayout)
{
    vk::ImageMemoryBarrier barrier{};
    barrier
        .setOldLayout(oldLayout)
        .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(image)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(1))
        .setSrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite)
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eFragmentShader,
        {},
        nullptr,
        nullptr,
        barrier);
}


void Renderer::resetDefaultSceneLayout()
{
    auto& renderables = scene.getRenderables();

    if (renderables.size() > 0)
    {
        renderables[0].setName("Center");
        renderables[0].getTransform().position = { 0.0f, 0.0f, 0.0f };
        renderables[0].getTransform().rotation = { 0.0f, 0.0f, 0.0f };
        renderables[0].getTransform().scale = { 1.0f, 1.0f, 1.0f };
    }

    if (renderables.size() > 1)
    {
        renderables[1].setName("Right");
        renderables[1].getTransform().position = { 1.5f, 0.0f, 0.0f };
        renderables[1].getTransform().rotation = { 0.0f, 0.0f, 0.0f };
        renderables[1].getTransform().scale = { 1.0f, 1.0f, 1.0f };
    }

    if (renderables.size() > 2)
    {
        renderables[2].setName("Left");
        renderables[2].getTransform().position = { -1.5f, 0.0f, 0.0f };
        renderables[2].getTransform().rotation = { 0.0f, 0.0f, 0.0f };
        renderables[2].getTransform().scale = { 1.0f, 1.0f, 1.0f };
    }

    for (size_t i = 3; i < renderables.size(); ++i)
    {
        renderables[i].setName("Renderable " + std::to_string(i));
        renderables[i].getTransform().position = { 0.0f, 0.0f, 0.0f };
        renderables[i].getTransform().rotation = { 0.0f, 0.0f, 0.0f };
        renderables[i].getTransform().scale = { 1.0f, 1.0f, 1.0f };
    }
}


void Renderer::createEnvironmentCubemap(const std::array<std::string, 6>& facePaths)
{
    auto& device = vkContext.getDevice();

    int texWidth = 0;
    int texHeight = 0;
    int texChannels = 0;

    std::vector<stbi_uc*> facePixels(6, nullptr);

    for (size_t i = 0; i < 6; ++i)
    {
        int w = 0, h = 0, c = 0;
        facePixels[i] = stbi_load(facePaths[i].c_str(), &w, &h, &c, STBI_rgb_alpha);
        if (!facePixels[i])
        {
            throw std::runtime_error("Failed to load cubemap face: " + facePaths[i]);
        }

        if (i == 0)
        {
            texWidth = w;
            texHeight = h;
            texChannels = 4;
        }
        else
        {
            if (w != texWidth || h != texHeight)
            {
                throw std::runtime_error("Cubemap faces must all have the same dimensions");
            }
        }
    }

    const vk::DeviceSize faceSize =
        static_cast<vk::DeviceSize>(texWidth) *
        static_cast<vk::DeviceSize>(texHeight) * 4;

    const vk::DeviceSize totalSize = faceSize * 6;

    vk::raii::Buffer stagingBuffer{ nullptr };
    vk::raii::DeviceMemory stagingMemory{ nullptr };

    bufferUtils.createBuffer(
        totalSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer,
        stagingMemory
    );

    void* mapped = stagingMemory.mapMemory(0, totalSize);
    unsigned char* dst = static_cast<unsigned char*>(mapped);

    for (size_t i = 0; i < 6; ++i)
    {
        std::memcpy(dst + i * faceSize, facePixels[i], static_cast<size_t>(faceSize));
    }

    stagingMemory.unmapMemory();

    for (auto* pixels : facePixels)
    {
        stbi_image_free(pixels);
    }

    const vk::Format format = vk::Format::eR8G8B8A8Srgb;

    vk::ImageCreateInfo imageInfo{};
    imageInfo
        .setFlags(vk::ImageCreateFlagBits::eCubeCompatible)
        .setImageType(vk::ImageType::e2D)
        .setFormat(format)
        .setExtent(vk::Extent3D{
            static_cast<uint32_t>(texWidth),
            static_cast<uint32_t>(texHeight),
            1
            })
        .setMipLevels(1)
        .setArrayLayers(6)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setTiling(vk::ImageTiling::eOptimal)
        .setUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled)
        .setSharingMode(vk::SharingMode::eExclusive)
        .setInitialLayout(vk::ImageLayout::eUndefined);

    environmentCubeImage = vk::raii::Image(device, imageInfo);

    vk::MemoryRequirements memReq = environmentCubeImage.getMemoryRequirements();

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo
        .setAllocationSize(memReq.size)
        .setMemoryTypeIndex(
            bufferUtils.findMemoryType(
                memReq.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal
            )
        );

    environmentCubeMemory = vk::raii::DeviceMemory(device, allocInfo);
    environmentCubeImage.bindMemory(*environmentCubeMemory, 0);

    auto cmd = bufferUtils.beginSingleTimeCommands();

    vk::ImageMemoryBarrier toTransfer{};
    toTransfer
        .setOldLayout(vk::ImageLayout::eUndefined)
        .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*environmentCubeImage)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(6))
        .setSrcAccessMask({})
        .setDstAccessMask(vk::AccessFlagBits::eTransferWrite);

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eTransfer,
        {},
        nullptr,
        nullptr,
        toTransfer
    );

    std::array<vk::BufferImageCopy, 6> copyRegions{};
    for (uint32_t face = 0; face < 6; ++face)
    {
        copyRegions[face]
            .setBufferOffset(face * faceSize)
            .setBufferRowLength(0)
            .setBufferImageHeight(0)
            .setImageSubresource(
                vk::ImageSubresourceLayers{}
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setMipLevel(0)
                .setBaseArrayLayer(face)
                .setLayerCount(1))
            .setImageOffset(vk::Offset3D{ 0, 0, 0 })
            .setImageExtent(vk::Extent3D{
                static_cast<uint32_t>(texWidth),
                static_cast<uint32_t>(texHeight),
                1
                });
    }

    cmd.copyBufferToImage(
        *stagingBuffer,
        *environmentCubeImage,
        vk::ImageLayout::eTransferDstOptimal,
        copyRegions
    );

    vk::ImageMemoryBarrier toShaderRead{};
    toShaderRead
        .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
        .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*environmentCubeImage)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(6))
        .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader,
        {},
        nullptr,
        nullptr,
        toShaderRead
    );

    bufferUtils.endSingleTimeCommands(cmd);

    vk::ImageViewCreateInfo viewInfo{};
    viewInfo
        .setImage(*environmentCubeImage)
        .setViewType(vk::ImageViewType::eCube)
        .setFormat(format)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(6));

    environmentCubeView = vk::raii::ImageView(device, viewInfo);

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo
        .setMagFilter(vk::Filter::eLinear)
        .setMinFilter(vk::Filter::eLinear)
        .setMipmapMode(vk::SamplerMipmapMode::eLinear)
        .setAddressModeU(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeW(vk::SamplerAddressMode::eClampToEdge)
        .setAnisotropyEnable(VK_FALSE)
        .setMaxAnisotropy(1.0f)
        .setMinLod(0.0f)
        .setMaxLod(0.0f)
        .setBorderColor(vk::BorderColor::eIntOpaqueBlack)
        .setUnnormalizedCoordinates(VK_FALSE);

    environmentCubeSampler = vk::raii::Sampler(device, samplerInfo);
}


void Renderer::cleanupDescriptorResources()
{
    iblDescriptorSet = nullptr;
    materialDescriptorSets.clear();
    frameDescriptorSets.clear();

    descriptorPool = nullptr;
}

uint32_t Renderer::chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities)
{
    auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
    if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount))
    {
        minImageCount = surfaceCapabilities.maxImageCount;
    }
    return minImageCount;
}

vk::SurfaceFormatKHR Renderer::chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const& availableFormats)
{
    assert(!availableFormats.empty());
    const auto formatIt = std::ranges::find_if(
        availableFormats,
        [](const auto& format) { return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear; });
    return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
}

vk::PresentModeKHR Renderer::chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const& availablePresentModes)
{
    assert(std::ranges::any_of(availablePresentModes, [](auto presentMode) { return presentMode == vk::PresentModeKHR::eFifo; }));
    return std::ranges::any_of(availablePresentModes,
        [](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; }) ?
        vk::PresentModeKHR::eMailbox :
        vk::PresentModeKHR::eFifo;
}

vk::Extent2D Renderer::chooseSwapExtent(vk::SurfaceCapabilitiesKHR const& capabilities)
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
    {
        return capabilities.currentExtent;
    }
    int width, height;
    window.getFramebufferSize(width, height);

    return {
        std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height) };
}

void Renderer::updateCameraControls()
{
    ImGuiIO& io = ImGui::GetIO();

    if (!io.WantCaptureMouse)
    {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Right))
        {
            cameraYaw -= io.MouseDelta.x * mouseOrbitSensitivity;
            cameraPitch -= io.MouseDelta.y * mouseOrbitSensitivity;

            cameraPitch = std::clamp(cameraPitch, -1.55f, 1.55f);
        }

        if (ImGui::IsMouseDown(ImGuiMouseButton_Middle))
        {
            glm::vec3 forward = glm::normalize(camera.getTarget() - camera.getPosition());
            glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 0.0f, 1.0f)));
            glm::vec3 up = glm::normalize(glm::cross(right, forward));

            glm::vec3 panDelta =
                (-right * io.MouseDelta.x + up * io.MouseDelta.y) *
                (mousePanSensitivity * cameraRadius);

            camera.offsetPosition(panDelta);
            camera.offsetTarget(panDelta);
        }

        if (io.MouseWheel != 0.0f)
        {
            cameraRadius -= io.MouseWheel * mouseZoomSensitivity;
            cameraRadius = std::clamp(cameraRadius, minCameraRadius, maxCameraRadius);
        }
    }

    glm::vec3 target = camera.getTarget();
    camera.setOrbit(cameraRadius, cameraYaw, cameraPitch);
    camera.setTarget(target);
    camera.setFov(cameraFov);
    camera.setNearFar(cameraNear, cameraFar);
}





Material* Renderer::getSelectedRenderableMaterial()
{
    Renderable* selected = scene.getSelectedRenderable(uiState.selectedRenderableIndex);
    if (!selected)
    {
        return nullptr;
    }

    return &selected->getMaterial();
}


glm::vec3 Renderer::getRenderableWorldPosition(const Renderable& renderable) const
{
    return glm::vec3(renderable.getTransform().toMatrix()[3]);
}

glm::vec3 Renderer::computeSceneCenter() const
{
    if (scene.empty())
    {
        return glm::vec3(0.0f);
    }

    glm::vec3 sum(0.0f);

    for (const auto& renderable : scene.getRenderables())
    {
        sum += getRenderableWorldPosition(renderable);
    }

    return sum / static_cast<float>(scene.size());
}

void Renderer::focusSelectedRenderable()
{
    Renderable* selected = scene.getSelectedRenderable(uiState.selectedRenderableIndex);

    if (!selected)
    {
        return;
    }

    camera.setTarget(getRenderableWorldPosition(*selected));
}


int Renderer::getMaterialIndex(const Material& material) const
{
    auto it = std::find_if(
        materials.begin(),
        materials.end(),
        [&](const std::unique_ptr<Material>& candidate)
        {
            return candidate.get() == &material;
        });

    if (it == materials.end())
    {
        return -1;
    }

    return static_cast<int>(std::distance(materials.begin(), it));
}

bool Renderer::isWireframeSupported() const
{
    return vkContext.isFillModeNonSolidEnabled();
}

void Renderer::createFallbackIBLResources()
{
    createFallbackBrdfLut();
    createFallbackBlackCube();
}

void Renderer::createFallbackBrdfLut()
{
    const unsigned char blackPixel[4] = { 0, 0, 0, 255 };

    Texture2D::SamplerOptions samplerOptions{};
    samplerOptions.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerOptions.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    samplerOptions.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    samplerOptions.enableAnisotropy = false;
    samplerOptions.maxLod = 0.0f;

    fallbackBrdfLut = std::make_unique<Texture2D>(
        vkContext,
        bufferUtils,
        imageUtils,
        blackPixel,
        1,
        1,
        4,
        "Fallback BRDF LUT",
        vk::Format::eR8G8B8A8Unorm,
        samplerOptions
    );
}

void Renderer::createFallbackBlackCube()
{
    auto& device = vkContext.getDevice();

    const vk::Format format = vk::Format::eR8G8B8A8Unorm;

    const std::array<unsigned char, 24> blackFaces = {
        0, 0, 0, 255,
        0, 0, 0, 255,
        0, 0, 0, 255,
        0, 0, 0, 255,
        0, 0, 0, 255,
        0, 0, 0, 255
    };

    vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(blackFaces.size());

    vk::raii::Buffer stagingBuffer{ nullptr };
    vk::raii::DeviceMemory stagingMemory{ nullptr };

    bufferUtils.createBuffer(
        imageSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer,
        stagingMemory
    );

    void* mapped = stagingMemory.mapMemory(0, imageSize);
    std::memcpy(mapped, blackFaces.data(), static_cast<size_t>(imageSize));
    stagingMemory.unmapMemory();

    vk::ImageCreateInfo imageInfo{};
    imageInfo
        .setFlags(vk::ImageCreateFlagBits::eCubeCompatible)
        .setImageType(vk::ImageType::e2D)
        .setFormat(format)
        .setExtent(vk::Extent3D{ 1, 1, 1 })
        .setMipLevels(1)
        .setArrayLayers(6)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setTiling(vk::ImageTiling::eOptimal)
        .setUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled)
        .setSharingMode(vk::SharingMode::eExclusive)
        .setInitialLayout(vk::ImageLayout::eUndefined);

    fallbackBlackCubeImage = vk::raii::Image(device, imageInfo);

    vk::MemoryRequirements memReq = fallbackBlackCubeImage.getMemoryRequirements();

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo
        .setAllocationSize(memReq.size)
        .setMemoryTypeIndex(
            bufferUtils.findMemoryType(
                memReq.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal
            )
        );

    fallbackBlackCubeMemory = vk::raii::DeviceMemory(device, allocInfo);
    fallbackBlackCubeImage.bindMemory(*fallbackBlackCubeMemory, 0);

    auto cmd = bufferUtils.beginSingleTimeCommands();

    vk::ImageMemoryBarrier toTransfer{};
    toTransfer
        .setOldLayout(vk::ImageLayout::eUndefined)
        .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*fallbackBlackCubeImage)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(6))
        .setSrcAccessMask({})
        .setDstAccessMask(vk::AccessFlagBits::eTransferWrite);

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eTransfer,
        {},
        nullptr,
        nullptr,
        toTransfer
    );

    std::array<vk::BufferImageCopy, 6> copyRegions{};
    for (uint32_t face = 0; face < 6; ++face)
    {
        copyRegions[face]
            .setBufferOffset(face * 4)
            .setBufferRowLength(0)
            .setBufferImageHeight(0)
            .setImageSubresource(
                vk::ImageSubresourceLayers{}
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setMipLevel(0)
                .setBaseArrayLayer(face)
                .setLayerCount(1))
            .setImageOffset(vk::Offset3D{ 0, 0, 0 })
            .setImageExtent(vk::Extent3D{ 1, 1, 1 });
    }

    cmd.copyBufferToImage(
        *stagingBuffer,
        *fallbackBlackCubeImage,
        vk::ImageLayout::eTransferDstOptimal,
        copyRegions
    );

    vk::ImageMemoryBarrier toShaderRead{};
    toShaderRead
        .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
        .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*fallbackBlackCubeImage)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(6))
        .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader,
        {},
        nullptr,
        nullptr,
        toShaderRead
    );

    bufferUtils.endSingleTimeCommands(cmd);

    vk::ImageViewCreateInfo viewInfo{};
    viewInfo
        .setImage(*fallbackBlackCubeImage)
        .setViewType(vk::ImageViewType::eCube)
        .setFormat(format)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(6));

    fallbackBlackCubeView = vk::raii::ImageView(device, viewInfo);

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo
        .setMagFilter(vk::Filter::eLinear)
        .setMinFilter(vk::Filter::eLinear)
        .setMipmapMode(vk::SamplerMipmapMode::eLinear)
        .setAddressModeU(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeW(vk::SamplerAddressMode::eClampToEdge)
        .setAnisotropyEnable(VK_FALSE)
        .setMaxAnisotropy(1.0f)
        .setMinLod(0.0f)
        .setMaxLod(0.0f)
        .setBorderColor(vk::BorderColor::eIntOpaqueBlack)
        .setUnnormalizedCoordinates(VK_FALSE);

    fallbackBlackCubeSampler = vk::raii::Sampler(device, samplerInfo);
}

void Renderer::updateIBLDescriptorSet()
{
    if (iblDescriptorSet == nullptr)
    {
        throw std::runtime_error("updateIBLDescriptorSet: iblDescriptorSet is null");
    }

    if (environmentCubeSampler == nullptr || environmentCubeView == nullptr)
    {
        throw std::runtime_error("updateIBLDescriptorSet: fallback environment cubemap is not initialized");
    }

    const bool hasFallbackBrdf =
        brdfLutTexture &&
        brdfLutTexture->getSampler() != nullptr &&
        brdfLutTexture->getImageView() != nullptr;

    const bool hasRuntimeBrdf =
        environment.runtimeBrdfLut.sampler != nullptr &&
        environment.runtimeBrdfLut.view != nullptr;

    if (!hasRuntimeBrdf && !hasFallbackBrdf)
    {
        throw std::runtime_error(
            "updateIBLDescriptorSet: no BRDF LUT available (runtime or fallback)");
    }

    auto& device = vkContext.getDevice();

    const bool useRuntimeIrradiance =
        environment.runtimeIrradianceCube.sampler != nullptr &&
        environment.runtimeIrradianceCube.view != nullptr;

    const bool useRuntimePrefiltered =
        environment.runtimePrefilteredCube.sampler != nullptr &&
        environment.runtimePrefilteredCube.view != nullptr;

    const bool useRuntimeBrdf =
        environment.runtimeBrdfLut.sampler != nullptr &&
        environment.runtimeBrdfLut.view != nullptr;

    const bool useRuntimeEnvironment =
        environment.runtimeEnvironmentCube.sampler != nullptr &&
        environment.runtimeEnvironmentCube.view != nullptr;

    auto irradianceInfo = useRuntimeIrradiance
        ? makeImageInfo(environment.runtimeIrradianceCube.sampler, environment.runtimeIrradianceCube.view)
        : makeImageInfo(*irradianceCubeSampler, *irradianceCubeView);

    auto prefilteredInfo = useRuntimePrefiltered
        ? makeImageInfo(environment.runtimePrefilteredCube.sampler, environment.runtimePrefilteredCube.view)
        : makeImageInfo(*prefilteredCubeSampler, *prefilteredCubeView);

    auto brdfInfo = useRuntimeBrdf
        ? makeImageInfo(environment.runtimeBrdfLut.sampler, environment.runtimeBrdfLut.view)
        : makeImageInfo(brdfLutTexture->getSampler(), brdfLutTexture->getImageView());

    auto environmentInfo = useRuntimeEnvironment
        ? makeImageInfo(environment.runtimeEnvironmentCube.sampler, environment.runtimeEnvironmentCube.view)
        : makeImageInfo(*environmentCubeSampler, *environmentCubeView);

    std::array<vk::WriteDescriptorSet, 4> writes{};

    writes[0]
        .setDstSet(*iblDescriptorSet)
        .setDstBinding(0)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(irradianceInfo);

    writes[1]
        .setDstSet(*iblDescriptorSet)
        .setDstBinding(1)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(prefilteredInfo);

    writes[2]
        .setDstSet(*iblDescriptorSet)
        .setDstBinding(2)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(brdfInfo);

    writes[3]
        .setDstSet(*iblDescriptorSet)
        .setDstBinding(3)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(environmentInfo);

    device.updateDescriptorSets(writes, {});
}

void Renderer::resetEnvironmentSettings()
{
    showSkybox = true;
    enableIBL = true;
    debugReflectionOnly = false;

    skyboxExposure = 1.0f;
    skyboxLod = 0.0f;

    iblIntensity = 1.0f;
    diffuseIBLIntensity = 1.0f;
    specularIBLIntensity = 1.0f;

    toneMappingEnabled = true;
    gammaEnabled = true;
    postExposure = 1.0f;

    environmentRotationDegrees = 0.0f;
    rotateSkybox = true;
    rotateIBLLighting = true;
}



void Renderer::createHdrEnvironmentTexture(const std::string& path)
{
    int width = 0;
    int height = 0;
    int channels = 0;

    float* pixels = stbi_loadf(path.c_str(), &width, &height, &channels, 4);

    if (!pixels)
    {
        throw std::runtime_error("Failed to load HDR environment: " + path);
    }

    hdrEnvironmentWidth = static_cast<uint32_t>(width);
    hdrEnvironmentHeight = static_cast<uint32_t>(height);

    const vk::DeviceSize imageSize =
        static_cast<vk::DeviceSize>(width) *
        static_cast<vk::DeviceSize>(height) *
        4 *
        sizeof(float);

    vk::raii::Buffer stagingBuffer{ nullptr };
    vk::raii::DeviceMemory stagingMemory{ nullptr };

    bufferUtils.createBuffer(
        imageSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer,
        stagingMemory);

    {
        void* mapped = stagingMemory.mapMemory(0, imageSize);
        std::memcpy(mapped, pixels, static_cast<size_t>(imageSize));
        stagingMemory.unmapMemory();
    }

    stbi_image_free(pixels);

    auto& device = vkContext.getDevice();

    const vk::Format hdrFormat = vk::Format::eR32G32B32A32Sfloat;

    vk::ImageCreateInfo imageInfo{};
    imageInfo
        .setImageType(vk::ImageType::e2D)
        .setFormat(hdrFormat)
        .setExtent(vk::Extent3D{
            hdrEnvironmentWidth,
            hdrEnvironmentHeight,
            1 })
            .setMipLevels(1)
        .setArrayLayers(1)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setTiling(vk::ImageTiling::eOptimal)
        .setUsage(
            vk::ImageUsageFlagBits::eTransferDst |
            vk::ImageUsageFlagBits::eSampled)
        .setSharingMode(vk::SharingMode::eExclusive)
        .setInitialLayout(vk::ImageLayout::eUndefined);

    hdrEnvironmentImage = vk::raii::Image(device, imageInfo);

    vk::MemoryRequirements memRequirements =
        hdrEnvironmentImage.getMemoryRequirements();

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo
        .setAllocationSize(memRequirements.size)
        .setMemoryTypeIndex(
            bufferUtils.findMemoryType(
                memRequirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal));

    hdrEnvironmentMemory = vk::raii::DeviceMemory(device, allocInfo);
    hdrEnvironmentImage.bindMemory(*hdrEnvironmentMemory, 0);

    auto cmd = bufferUtils.beginSingleTimeCommands();

    vk::ImageMemoryBarrier toTransfer{};
    toTransfer
        .setOldLayout(vk::ImageLayout::eUndefined)
        .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*hdrEnvironmentImage)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(1))
        .setSrcAccessMask({})
        .setDstAccessMask(vk::AccessFlagBits::eTransferWrite);

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eTransfer,
        {},
        nullptr,
        nullptr,
        toTransfer);

    vk::BufferImageCopy copyRegion{};
    copyRegion
        .setBufferOffset(0)
        .setBufferRowLength(0)
        .setBufferImageHeight(0)
        .setImageSubresource(
            vk::ImageSubresourceLayers{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setMipLevel(0)
            .setBaseArrayLayer(0)
            .setLayerCount(1))
        .setImageOffset(vk::Offset3D{ 0, 0, 0 })
        .setImageExtent(vk::Extent3D{
            hdrEnvironmentWidth,
            hdrEnvironmentHeight,
            1 });

    cmd.copyBufferToImage(
        *stagingBuffer,
        *hdrEnvironmentImage,
        vk::ImageLayout::eTransferDstOptimal,
        copyRegion);

    vk::ImageMemoryBarrier toShaderRead{};
    toShaderRead
        .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
        .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*hdrEnvironmentImage)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(1))
        .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader,
        {},
        nullptr,
        nullptr,
        toShaderRead);

    bufferUtils.endSingleTimeCommands(cmd);

    vk::ImageViewCreateInfo viewInfo{};
    viewInfo
        .setImage(*hdrEnvironmentImage)
        .setViewType(vk::ImageViewType::e2D)
        .setFormat(hdrFormat)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(1));

    hdrEnvironmentView = vk::raii::ImageView(device, viewInfo);

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo
        .setMagFilter(vk::Filter::eLinear)
        .setMinFilter(vk::Filter::eLinear)
        .setMipmapMode(vk::SamplerMipmapMode::eLinear)
        .setAddressModeU(vk::SamplerAddressMode::eRepeat)
        .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeW(vk::SamplerAddressMode::eClampToEdge)
        .setMipLodBias(0.0f)
        .setAnisotropyEnable(VK_FALSE)
        .setCompareEnable(VK_FALSE)
        .setMinLod(0.0f)
        .setMaxLod(0.0f)
        .setBorderColor(vk::BorderColor::eFloatOpaqueWhite)
        .setUnnormalizedCoordinates(VK_FALSE);

    hdrEnvironmentSampler = vk::raii::Sampler(device, samplerInfo);

    std::cout << "Loaded HDR environment: "
        << path << " "
        << hdrEnvironmentWidth << "x"
        << hdrEnvironmentHeight << "\n";
}

void Renderer::createSkyboxPipeline()
{
    auto& device = vkContext.getDevice();

    vk::raii::ShaderModule vertShaderModule =
        ShaderUtils::createShaderModule(vkContext.getDevice(), "shaders/skybox_vert.spv");

    vk::raii::ShaderModule fragShaderModule =
        ShaderUtils::createShaderModule(vkContext.getDevice(), "shaders/skybox_frag.spv");

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo
        .setStage(vk::ShaderStageFlagBits::eVertex)
        .setModule(*vertShaderModule)
        .setPName("main");

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo
        .setStage(vk::ShaderStageFlagBits::eFragment)
        .setModule(*fragShaderModule)
        .setPName("main");

    std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {
        vertShaderStageInfo,
        fragShaderStageInfo
    };

    // Fullscreen triangle: no vertex buffers, no attributes.
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly
        .setTopology(vk::PrimitiveTopology::eTriangleList)
        .setPrimitiveRestartEnable(VK_FALSE);

    vk::PipelineViewportStateCreateInfo viewportState{};
    viewportState
        .setViewportCount(1)
        .setScissorCount(1);

    vk::PipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer
        .setDepthClampEnable(VK_FALSE)
        .setRasterizerDiscardEnable(VK_FALSE)
        .setPolygonMode(vk::PolygonMode::eFill)
        .setCullMode(vk::CullModeFlagBits::eNone)
        .setFrontFace(vk::FrontFace::eCounterClockwise)
        .setDepthBiasEnable(VK_FALSE)
        .setLineWidth(1.0f);

    vk::PipelineMultisampleStateCreateInfo multisampling{};
    multisampling
        .setRasterizationSamples(vkContext.getMsaaSamples())
        .setSampleShadingEnable(VK_FALSE);

    vk::PipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil
        .setDepthTestEnable(VK_FALSE)
        .setDepthWriteEnable(VK_FALSE)

        .setDepthBoundsTestEnable(VK_FALSE)
        .setStencilTestEnable(VK_FALSE);

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment
        .setBlendEnable(VK_FALSE)
        .setColorWriteMask(
            vk::ColorComponentFlagBits::eR |
            vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB |
            vk::ColorComponentFlagBits::eA
        );

    vk::PipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending
        .setLogicOpEnable(VK_FALSE)
        .setLogicOp(vk::LogicOp::eCopy)
        .setAttachments(colorBlendAttachment);

    std::vector<vk::DynamicState> dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor
    };

    vk::PipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.setDynamicStates(dynamicStates);

    // Skybox pipeline layout:
    // set 0 = frame UBO
    // set 1 = IBL descriptor set
    std::array<vk::DescriptorSetLayout, 2> setLayouts = {
        *frameDescriptorSetLayout,
        *iblDescriptorSetLayout
    };

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setSetLayouts(setLayouts);

    skyboxPipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    vk::StructureChain<
        vk::GraphicsPipelineCreateInfo,
        vk::PipelineRenderingCreateInfo
    > pipelineCreateInfoChain{};

    pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>()
        .setStages(shaderStages)
        .setPVertexInputState(&vertexInputInfo)
        .setPInputAssemblyState(&inputAssembly)
        .setPViewportState(&viewportState)
        .setPRasterizationState(&rasterizer)
        .setPMultisampleState(&multisampling)
        .setPDepthStencilState(&depthStencil)
        .setPColorBlendState(&colorBlending)
        .setPDynamicState(&dynamicState)
        .setLayout(*skyboxPipelineLayout)
        .setRenderPass(vk::RenderPass{});

    pipelineCreateInfoChain.get<vk::PipelineRenderingCreateInfo>()
        .setColorAttachmentFormats(hdrFormat)
        .setDepthAttachmentFormat(depthFormat);

    skyboxPipeline = vk::raii::Pipeline(
        device,
        nullptr,
        pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>()
    );
}

void Renderer::drawSkybox(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
    (void)imageIndex;

    if (skyboxPipeline == nullptr || skyboxPipelineLayout == nullptr)
        return;

    if (iblDescriptorSet == nullptr)
        return;

    if (environmentCubeView == nullptr || environmentCubeSampler == nullptr)
        return;

    commandBuffer.bindPipeline(
        vk::PipelineBindPoint::eGraphics,
        *skyboxPipeline
    );

 /*   vk::Viewport viewport{};
    viewport
        .setX(0.0f)
        .setY(0.0f)
        .setWidth(static_cast<float>(swapChainExtent.width))
        .setHeight(static_cast<float>(swapChainExtent.height))
        .setMinDepth(0.0f)
        .setMaxDepth(1.0f);

    vk::Rect2D scissor{};
    scissor
        .setOffset(vk::Offset2D{ 0, 0 })
        .setExtent(swapChainExtent);

    commandBuffer.setViewport(0, viewport);
    commandBuffer.setScissor(0, scissor);

    */

    std::array<vk::DescriptorSet, 2> sets = {
        *frameDescriptorSets[frameIndex],
        *iblDescriptorSet
    };

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *skyboxPipelineLayout,
        0,
        sets,
        {}
    );

    commandBuffer.draw(3, 1, 0, 0);
}



vk::DescriptorImageInfo Renderer::makeImageInfo(
    vk::Sampler sampler,
    vk::ImageView view) const
{
    return vk::DescriptorImageInfo{}
        .setSampler(sampler)
        .setImageView(view)
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
}


void Renderer::applyIblCalibrationPreset(const IblCalibrationPreset& preset)
{
    lightIntensity = preset.lightIntensity;
    skyboxExposure = preset.skyboxExposure;
    iblIntensity = preset.iblIntensity;
    diffuseIBLIntensity = preset.diffuseIBLIntensity;
    specularIBLIntensity = preset.specularIBLIntensity;
    postExposure = preset.postExposure;
}

void Renderer::resetIblEnergyCalibration()
{
    applyIblCalibrationPreset(defaultIblCalibrationPreset);
}



void Renderer::initImGui()
{
    auto& device = vkContext.getDevice();
    auto& physicalDevice = vkContext.getPhysicalDevice();
    auto& queue = vkContext.getQueue();

    std::array<vk::DescriptorPoolSize, 11> poolSizes = {
        vk::DescriptorPoolSize(vk::DescriptorType::eSampler, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageImage, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformTexelBuffer, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageTexelBuffer, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBufferDynamic, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBufferDynamic, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eInputAttachment, 1000)
    };

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo
        .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
        .setMaxSets(1000 * static_cast<uint32_t>(poolSizes.size()))
        .setPoolSizes(poolSizes);

    imguiDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window.getHandle(), true);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_3;
    initInfo.Instance = *vkContext.getInstance();
    initInfo.PhysicalDevice = *physicalDevice;
    initInfo.Device = *device;
    initInfo.QueueFamily = vkContext.getQueueIndex();
    initInfo.Queue = *queue;
    initInfo.DescriptorPool = *imguiDescriptorPool;
    initInfo.MinImageCount = static_cast<uint32_t>(swapChainImages.size());
    initInfo.ImageCount = static_cast<uint32_t>(swapChainImages.size());
    initInfo.UseDynamicRendering = true;
    initInfo.CheckVkResultFn = nullptr;

    VkFormat colorFormat =
        static_cast<VkFormat>(swapChainSurfaceFormat.format);

    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    

    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = {};
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &colorFormat;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.depthAttachmentFormat =
        VK_FORMAT_UNDEFINED;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.stencilAttachmentFormat =
        VK_FORMAT_UNDEFINED;


    ImGui_ImplVulkan_Init(&initInfo);

    imguiInitialized = true;
}

void Renderer::shutdownImGui()
{
    if (!imguiInitialized)
        return;

    vkContext.getDevice().waitIdle();

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    imguiDescriptorPool = nullptr;
    imguiInitialized = false;
}

void Renderer::beginImGuiFrame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}


void Renderer::buildImGui()
{
    if (uiState.showDebugPanel)
    {
        ImGui::Begin("Rendering Debug", &uiState.showDebugPanel);

        uint32_t totalVertexCount = 0;
        uint32_t totalIndexCount = 0;

        for (const auto& gpuMesh : gpuMeshes)
        {
            if (!gpuMesh)
                continue;

            totalVertexCount += gpuMesh->getVertexCount();
            totalIndexCount += gpuMesh->getIndexCount();
        }



        EditorPanels::drawRendererPanel(
            vkContext,
            swapChainExtent,
            swapChainImages.size(),
            textures.empty() ? nullptr : &getDefaultTexture(),
            vkContext.getMsaaSamples(),
            gpuMeshes.size(),
            totalVertexCount,
            totalIndexCount,
            frameTimeMs,
            fps,
            scene);

        EditorPanels::drawAnimationPanel(
            animateModel,
            rotationSpeed);

        EditorPanels::drawCameraPanel(
            camera,
            cameraRadius,
            cameraYaw,
            cameraPitch,
            cameraFov,
            cameraNear,
            cameraFar,
            mouseOrbitSensitivity,
            mousePanSensitivity,
            mouseZoomSensitivity,
            minCameraRadius,
            maxCameraRadius);

        float maxPrefilterMip =
            prefilterRenderer
            ? static_cast<float>(prefilterRenderer->getDebugRuntimePrefilteredMipLevels() - 1)
            : 0.0f;

        

        EditorPanels::drawLookDevPanel(
            lightDirection,
            lightColor,
            lightIntensity,
            ambientColor,
            ambientIntensity,
            uiState.debugViewMode,

            showSkybox,
            enableIBL,
            debugReflectionOnly,
            debugSkyboxFaces,
            skyboxExposure,
            skyboxLod,
            iblIntensity,
            diffuseIBLIntensity,
            specularIBLIntensity,

            toneMappingEnabled,
            gammaEnabled,
            postExposure,

            environmentRotationDegrees,
            rotateSkybox,
            rotateIBLLighting,

            debugForceSpecularMip,
            debugSpecularMip,
            roughnessMipScale,
            roughnessMipBias,
            maxPrefilterMip,

            [this]() { resetEnvironmentSettings(); },
            [this]() { resetIblEnergyCalibration(); }
            
          
            );

        EditorPanels::drawUboInspector(lastUbo);

        Renderable* selectedRenderable = scene.getSelectedRenderable(uiState.selectedRenderableIndex);

        Material* selectedMaterial = getSelectedRenderableMaterial();

        const Texture2D* baseColorTexture =
            selectedMaterial ? &selectedMaterial->getTexture() : nullptr;

        const Texture2D* normalTexture =
            selectedMaterial ? selectedMaterial->getNormalTexture() : nullptr;

        const Texture2D* metallicRoughnessTexture =
            selectedMaterial ? selectedMaterial->getMetallicRoughnessTexture() : nullptr;


        //   Material* selectedMaterial = getSelectedRenderableMaterial();
        int selectedMaterialIndex = selectedMaterial ? getMaterialIndex(*selectedMaterial) : -1;
        const Texture2D* selectedTexture = selectedMaterial ? &selectedMaterial->getTexture() : nullptr;


        EditorPanels::drawSelectedMaterialPanel(
            selectedRenderable,
            selectedMaterial,
            selectedTexture,
            selectedMaterialIndex);


        EditorPanels::drawVerificationPanel(
            scene,
            uiState,
            currentModelPath,
            selectedRenderable,
            selectedMaterial,
            baseColorTexture,
            normalTexture,
            metallicRoughnessTexture,
            lightDirection,
            lightColor,
            ambientColor);

        EditorPanels::drawDebugPanel(
            uiState,
            isWireframeSupported());

        

        ImGui::End();
    }

    if (uiState.showDemoWindow)
    {
        ImGui::ShowDemoWindow(&uiState.showDemoWindow);
    }
}
 

    



void Renderer::renderImGui(vk::CommandBuffer commandBuffer)
{
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
}

void Renderer::buildOverlay()
{
    constexpr ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav;

    ImGui::SetNextWindowBgAlpha(0.35f);
    ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_Always);

    if (ImGui::Begin("Overlay", nullptr, flags))
    {
        ImGui::Text("FPS: %.1f", fps);
        ImGui::Text("Frame: %.2f ms", frameTimeMs);
        ImGui::Text("MSAA: %d", static_cast<int>(vkContext.getMsaaSamples()));
    }
    ImGui::End();
}