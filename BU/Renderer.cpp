#include "Renderer.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include <algorithm>
#include <cassert>

#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <cstdio>

#include <stb_image.h>

#include "EditorPanels.hpp"
#include "GltfLoader.hpp"



#include <glm/gtc/matrix_transform.hpp>
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

    createGraphicsPipeline();
    createColorResources();
    createDepthResources();

    textures.clear();
    normalTextures.clear();
    defaultNormalTexture.reset();
    materials.clear();
    scene.clear();
    gpuMeshes.clear();

    metallicRoughnessTextures.clear();
    defaultMetallicRoughnessTexture.reset();

    const unsigned char flatNormalPixels[4] = { 128, 128, 255, 255 };

    defaultNormalTexture = std::make_unique<Texture2D>(
        vkContext,
        bufferUtils,
        imageUtils,
        flatNormalPixels,
        1,
        1,
        4,
        "Default Flat Normal",
        vk::Format::eR8G8B8A8Unorm);

    const unsigned char defaultMRPixels[4] = { 255, 255, 255, 255 };

    defaultMetallicRoughnessTexture = std::make_unique<Texture2D>(
        vkContext,
        bufferUtils,
        imageUtils,
        defaultMRPixels,
        1,
        1,
        4,
        "Default MetallicRoughness",
        vk::Format::eR8G8B8A8Unorm);

    

    // Slot 0 = default fallback texture
    textures.push_back(std::make_unique<Texture2D>(
        vkContext,
        bufferUtils,
        imageUtils,
        TEXTURE_PATH
    ));

    camera.setTarget({ 0.0f, 0.0f, 0.0f });
    camera.setOrbit(cameraRadius, cameraYaw, cameraPitch);
    camera.setFov(cameraFov);
    camera.setNearFar(cameraNear, cameraFar);

    currentModelPath = "models/DamagedHelmet/glTF/DamagedHelmet.gltf";
    std::cout << "Loading model: " << currentModelPath << std::endl;

    GltfLoader loader;
    GltfSceneData imported = loader.load(currentModelPath);

       

    

    if (textures.empty())
    {
        throw std::runtime_error("default texture is not available");
    }

    std::vector<int> gltfImageToNormalTextureIndex(imported.images.size(), -1);

    std::vector<bool> imageUsedAsNormal(imported.images.size(), false);

    for (const auto& importedMaterial : imported.materials)
    {
        if (importedMaterial.normalImageIndex >= 0 &&
            importedMaterial.normalImageIndex < static_cast<int>(imageUsedAsNormal.size()))
        {
            imageUsedAsNormal[importedMaterial.normalImageIndex] = true;
        }
    }

    for (size_t i = 0; i < imported.images.size(); ++i)
    {
        if (!imageUsedAsNormal[i])
            continue;

        const auto& image = imported.images[i];

        if (image.pixels.empty() || image.width <= 0 || image.height <= 0)
            continue;

        if (image.channels != 3 && image.channels != 4)
            continue;

        normalTextures.push_back(std::make_unique<Texture2D>(
            vkContext,
            bufferUtils,
            imageUtils,
            image.pixels.data(),
            static_cast<uint32_t>(image.width),
            static_cast<uint32_t>(image.height),
            static_cast<uint32_t>(image.channels),
            image.name.empty() ? ("glTF normal image " + std::to_string(i)) : image.name,
            vk::Format::eR8G8B8A8Unorm));

        gltfImageToNormalTextureIndex[i] = static_cast<int>(normalTextures.size()) - 1;
    }

    std::vector<int> gltfImageToMRTextureIndex(imported.images.size(), -1);

    std::vector<bool> imageUsedAsMR(imported.images.size(), false);

    for (const auto& importedMaterial : imported.materials)
    {
        if (importedMaterial.metallicRoughnessImageIndex >= 0 &&
            importedMaterial.metallicRoughnessImageIndex < static_cast<int>(imageUsedAsMR.size()))
        {
            imageUsedAsMR[importedMaterial.metallicRoughnessImageIndex] = true;
        }
    }

    for (size_t i = 0; i < imported.images.size(); ++i)
    {
        if (!imageUsedAsMR[i])
            continue;

        const auto& image = imported.images[i];

        if (image.pixels.empty() || image.width <= 0 || image.height <= 0)
            continue;

        if (image.channels != 3 && image.channels != 4)
            continue;

        metallicRoughnessTextures.push_back(std::make_unique<Texture2D>(
            vkContext,
            bufferUtils,
            imageUtils,
            image.pixels.data(),
            static_cast<uint32_t>(image.width),
            static_cast<uint32_t>(image.height),
            static_cast<uint32_t>(image.channels),
            image.name.empty() ? ("glTF metallicRoughness image " + std::to_string(i)) : image.name,
            vk::Format::eR8G8B8A8Unorm));

        gltfImageToMRTextureIndex[i] = static_cast<int>(metallicRoughnessTextures.size()) - 1;
    }
    
    std::vector<int> gltfImageToTextureIndex(imported.images.size(), -1);

    for (size_t i = 0; i < imported.images.size(); ++i)
    {
        const auto& image = imported.images[i];

        if (image.pixels.empty() || image.width <= 0 || image.height <= 0)
        {
            continue;
        }

        if (image.channels != 3 && image.channels != 4)
        {
            std::cout << "Skipping glTF image " << i
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
            image.name.empty() ? ("glTF image " + std::to_string(i)) : image.name
        ));

        gltfImageToTextureIndex[i] = static_cast<int>(textures.size()) - 1;

        


    }

    // Slot 0 = default fallback material
    materials.push_back(std::make_unique<Material>(getDefaultTexture()));

    for (const auto& importedMaterial : imported.materials)
    {
        Texture2D* assignedTexture = &getDefaultTexture();

        if (importedMaterial.baseColorImageIndex >= 0 &&
            importedMaterial.baseColorImageIndex < static_cast<int>(gltfImageToTextureIndex.size()))
        {
            int textureIndex = gltfImageToTextureIndex[importedMaterial.baseColorImageIndex];
            if (textureIndex >= 0 && textureIndex < static_cast<int>(textures.size()))
            {
                assignedTexture = textures[textureIndex].get();
            }
        }

        auto material = std::make_unique<Material>(*assignedTexture);
        material->setBaseColorFactor(importedMaterial.baseColorFactor);
        material->setName(importedMaterial.name);
        material->setDoubleSided(importedMaterial.doubleSided);
        material->setMetallicFactor(importedMaterial.metallicFactor);
        material->setRoughnessFactor(importedMaterial.roughnessFactor);
        material->setNormalScale(importedMaterial.normalScale);

        material->setAlphaMode(importedMaterial.alphaMode);
        material->setAlphaCutoff(importedMaterial.alphaCutoff);

        Texture2D* assignedNormalTexture = defaultNormalTexture.get();

        if (importedMaterial.normalImageIndex >= 0 &&
            importedMaterial.normalImageIndex < static_cast<int>(gltfImageToNormalTextureIndex.size()))
        {
            int normalTextureIndex = gltfImageToNormalTextureIndex[importedMaterial.normalImageIndex];
            if (normalTextureIndex >= 0 &&
                normalTextureIndex < static_cast<int>(normalTextures.size()))
            {
                assignedNormalTexture = normalTextures[normalTextureIndex].get();
            }
        }

        Texture2D* assignedMRTexture = defaultMetallicRoughnessTexture.get();

        if (importedMaterial.metallicRoughnessImageIndex >= 0 &&
            importedMaterial.metallicRoughnessImageIndex < static_cast<int>(gltfImageToMRTextureIndex.size()))
        {
            int mrTextureIndex = gltfImageToMRTextureIndex[importedMaterial.metallicRoughnessImageIndex];
            if (mrTextureIndex >= 0 &&
                mrTextureIndex < static_cast<int>(metallicRoughnessTextures.size()))
            {
                assignedMRTexture = metallicRoughnessTextures[mrTextureIndex].get();
            }
        }

        material->setMetallicRoughnessTexture(assignedMRTexture);

        material->setNormalTexture(assignedNormalTexture);

        materials.push_back(std::move(material));

        std::cout << "Loaded textures: " << textures.size() << std::endl;


    }



    if (imported.renderables.empty())
    {
        throw std::runtime_error("glTF import produced no renderables");
    }

    std::cout << "Imported glTF renderables: "
        << imported.renderables.size()
        << std::endl;

    std::cout << "Imported glTF images: "
        << imported.images.size()
        << std::endl;

    std::cout << "Imported engine textures: "
        << textures.size()
        << std::endl;

    std::cout << "Imported engine materials: "
        << materials.size()
        << std::endl;

    for (size_t i = 0; i < imported.renderables.size(); ++i)
    {
        gpuMeshes.push_back(std::make_unique<GpuMesh>(
            vkContext,
            bufferUtils,
            imported.renderables[i].mesh.vertices,
            imported.renderables[i].mesh.indices
        ));

        Material& assignedMaterial =
            (imported.renderables[i].materialIndex >= 0 &&
                static_cast<size_t>(imported.renderables[i].materialIndex + 1) < materials.size())
            ? *materials[imported.renderables[i].materialIndex + 1]
            : getDefaultMaterial();

        Renderable& renderable = scene.addRenderable(
            *gpuMeshes.back(),
            assignedMaterial,
            "glTF " + std::to_string(i)
        );

        renderable.getTransform() = imported.renderables[i].transform;

        std::cout << "glTF primitive " << i
            << " material index: " << imported.renderables[i].materialIndex
            << std::endl;
    }

    uiState.selectedRenderableIndex = scene.empty() ? -1 : 0;

    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createMaterialDescriptorSets();

    createFallbackIBLResources();

    createEnvironmentCubemap({
    "assets/skybox/right.jpg",
    "assets/skybox/left.jpg",
    "assets/skybox/top.jpg",
    "assets/skybox/bottom.jpg",
    "assets/skybox/front.jpg",
    "assets/skybox/back.jpg"
    
    
        });


    updateIBLDescriptorSet();
       
    createSkyboxPipeline();
     
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
    pipelineLayout = nullptr;

    skyboxPipeline = nullptr;
    skyboxPipelineLayout = nullptr;

    colorImageView = nullptr;
    colorImageMemory = nullptr;
    colorImage = nullptr;

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
    createDepthResources();
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

        std::array<vk::DescriptorSetLayoutBinding, 3> bindings = {
            baseColorBinding,
            normalBinding,
            metallicRoughnessBinding
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
        createShaderModule(readFile("shaders/vert.spv"));

    vk::raii::ShaderModule fragShaderModule =
        createShaderModule(readFile("shaders/frag.spv"));

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
        .setColorAttachmentFormats(swapChainSurfaceFormat.format)
        .setDepthAttachmentFormat(depthFormat);

    // Filled, culled
    rasterizer
        .setPolygonMode(vk::PolygonMode::eFill)
        .setCullMode(vk::CullModeFlagBits::eBack);

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

    // Wireframe variants
    wireframePipeline = nullptr;
    wireframeDoubleSidedPipeline = nullptr;

    if (vkContext.isFillModeNonSolidEnabled())
    {
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
    vk::Format colorFormat = swapChainSurfaceFormat.format;

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
            materialCount * 3 + 4) // +4 for IBL
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
    // Set 0 — Frame UBO sets
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
    // Set 2 — IBL descriptor set
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
    ubo.lightColor = glm::vec4(5.0f, 5.0f, 5.0f, 1.0f);
    ubo.ambientColor = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
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

        std::array<vk::WriteDescriptorSet, 3> descriptorWrites = {
            baseColorWrite.write,
            normalWrite.write,
            metallicRoughnessWrite.write
        };

        device.updateDescriptorSets(descriptorWrites, nullptr);
    }
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

    queue.submit(submitInfo, *inFlightFences[frameIndex]);

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

    vk::ImageLayout swapChainOldLayout =
        swapChainImageInitialized[imageIndex]
        ? vk::ImageLayout::ePresentSrcKHR
        : vk::ImageLayout::eUndefined;

    transitionToColorAttachment(
        cmd,
        swapChainImages[imageIndex],
        swapChainOldLayout);

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
        .setResolveImageView(*swapChainImageViews[imageIndex])
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

    for (auto& renderable : scene.getRenderables())
    {
        commandBuffer.bindVertexBuffers(
            0,
            *renderable.getMesh().getVertexBuffer(),
            { 0 });

        commandBuffer.bindIndexBuffer(
            *renderable.getMesh().getIndexBuffer(),
            0,
            vk::IndexType::eUint32);

        Material& renderableMaterial = renderable.getMaterial();

        auto materialIt = std::find_if(
            materials.begin(),
            materials.end(),
            [&](const std::unique_ptr<Material>& candidate)
            {
                return candidate.get() == &renderableMaterial;
            });

        if (materialIt == materials.end())
        {
            throw std::runtime_error("renderable material not found in renderer materials");
        }

        size_t materialIndex = static_cast<size_t>(std::distance(materials.begin(), materialIt));

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

        // Main scene pipeline layout:
        // set 0 = frame UBO
        // set 1 = material textures
        // set 2 = IBL textures
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

        glm::mat3 normalMatrix =
            glm::transpose(glm::inverse(glm::mat3(pushData.model)));

        pushData.normalMatrix = normalMatrix;
        pushData.baseColorFactor = renderableMaterial.getBaseColorFactor();
        pushData.materialParams = glm::vec4(
            renderableMaterial.getMetallicFactor(),
            renderableMaterial.getRoughnessFactor(),
            renderableMaterial.getNormalScale(),
            0.0f);

        if (animateModel)
        {
            pushData.model = glm::rotate(
                pushData.model,
                currentAnimationAngle,
                glm::vec3(0.0f, 0.0f, 1.0f));
        }

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

    renderImGui(*commandBuffer);

    commandBuffer.endRendering();

    transitionToPresent(
        cmd,
        swapChainImages[imageIndex]);

    swapChainImageInitialized[imageIndex] = true;

    commandBuffer.end();
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


 void Renderer::    (const std::array<std::string, 6>& facePaths)
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

[[nodiscard]] vk::raii::ShaderModule Renderer::createShaderModule(const std::vector<char>& code) const
{
    vk::ShaderModuleCreateInfo createInfo{};
    createInfo
        .setCodeSize(code.size())
        .setPCode(reinterpret_cast<const uint32_t*>(code.data()));

    
    vk::raii::ShaderModule     shaderModule{ vkContext.getDevice(), createInfo };

    return shaderModule;
}

std::vector<char> Renderer::readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("failed to open file: " + filename);
    }

    std::vector<char> buffer(static_cast<size_t>(file.tellg()));
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    return buffer;
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



void Renderer::focusSelectedRenderable()
{
    Renderable* selected = scene.getSelectedRenderable(uiState.selectedRenderableIndex);
    if (!selected)
    {
        return;
    }

    camera.setTarget(selected->getTransform().position);
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


glm::vec3 Renderer::computeSceneCenter() const
{
    if (scene.empty())
    {
        return glm::vec3(0.0f);
    }

    glm::vec3 sum(0.0f);

    for (const auto& renderable : scene.getRenderables())
    {
        sum += renderable.getTransform().position;
    }

    return sum / static_cast<float>(scene.size());
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

    if (fallbackBlackCubeView == nullptr || fallbackBlackCubeSampler == nullptr)
    {
        throw std::runtime_error("updateIBLDescriptorSet: fallback black cube is not initialized");
    }

    if (!fallbackBrdfLut ||
        fallbackBrdfLut->getImageView() == nullptr ||
        fallbackBrdfLut->getSampler() == nullptr)
    {
        throw std::runtime_error("updateIBLDescriptorSet: fallback BRDF LUT is not initialized");
    }

    if (environmentCubeView == nullptr || environmentCubeSampler == nullptr)
    {
        throw std::runtime_error("updateIBLDescriptorSet: environment cubemap is not initialized");
    }

    auto& device = vkContext.getDevice();

    vk::DescriptorImageInfo irradianceInfo{};
    irradianceInfo
        .setSampler(*fallbackBlackCubeSampler)
        .setImageView(*fallbackBlackCubeView)
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

    vk::DescriptorImageInfo prefilteredInfo{};
    prefilteredInfo
        .setSampler(*fallbackBlackCubeSampler)
        .setImageView(*fallbackBlackCubeView)
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

    vk::DescriptorImageInfo brdfInfo{};
    brdfInfo
        .setSampler(*fallbackBrdfLut->getSampler())
        .setImageView(*fallbackBrdfLut->getImageView())
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

    vk::DescriptorImageInfo environmentInfo{};
    environmentInfo
        .setSampler(*environmentCubeSampler)
        .setImageView(*environmentCubeView)
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

    std::array<vk::WriteDescriptorSet, 4> writes{};

    writes[0]
        .setDstSet(*iblDescriptorSet)
        .setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(irradianceInfo);

    writes[1]
        .setDstSet(*iblDescriptorSet)
        .setDstBinding(1)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(prefilteredInfo);

    writes[2]
        .setDstSet(*iblDescriptorSet)
        .setDstBinding(2)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(brdfInfo);

    writes[3]
        .setDstSet(*iblDescriptorSet)
        .setDstBinding(3)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(environmentInfo);

    device.updateDescriptorSets(writes, {});
}





void Renderer::createSkyboxPipeline()
{
    auto& device = vkContext.getDevice();

    vk::raii::ShaderModule vertShaderModule =
        createShaderModule(readFile("shaders/skybox_vert.spv"));

    vk::raii::ShaderModule fragShaderModule =
        createShaderModule(readFile("shaders/skybox_frag.spv"));

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
        .setColorAttachmentFormats(swapChainSurfaceFormat.format)
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

    vk::Viewport viewport{};
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

    VkFormat colorFormat = static_cast<VkFormat>(swapChainSurfaceFormat.format);
   
    VkFormat imguiDepthFormat = static_cast<VkFormat>(depthFormat);

    initInfo.PipelineInfoMain.MSAASamples =
        static_cast<VkSampleCountFlagBits>(vkContext.getMsaaSamples());

    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = {};
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &colorFormat;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.depthAttachmentFormat = imguiDepthFormat;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.stencilAttachmentFormat =
        hasStencilComponent(depthFormat) ? imguiDepthFormat : VK_FORMAT_UNDEFINED;

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
        ImGui::Begin("Debug Panel", &uiState.showDebugPanel);

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

        EditorPanels::drawLightingPanel(
            lightDirection,
            lightColor,
            ambientColor);

        EditorPanels::drawEnvironmentPanel(
            showSkybox,
            enableIBL,
            debugReflectionOnly,
            skyboxExposure,
            skyboxLod,
            iblIntensity,
            diffuseIBLIntensity,
            specularIBLIntensity);

        if (!gpuMeshes.empty() && !materials.empty())
        {
            EditorPanels::drawScenePanel(
                scene,
                uiState,
                *gpuMeshes[0],
                getDefaultMaterial(),
                camera,
                [this]() { resetDefaultSceneLayout(); },
                [this]() { focusSelectedRenderable(); });
        }
        else
        {
            ImGui::TextUnformatted("Scene editing unavailable: no GPU mesh or default material loaded.");
        }


   //     const Renderable* selectedRenderable =
   //         scene.empty() ? nullptr : &scene.getRenderables()[uiState.selectedRenderableIndex];

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
     

        EditorPanels::drawAssetInspectionPanel(
            scene,
            uiState,
            gpuMeshes.size(),
            textures.size(),
            materials.size(),
            selectedMaterialIndex,
            selectedMaterial,
            selectedTexture);

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

        EditorPanels::drawSelectedMaterialPanel(
            selectedRenderable,
            selectedMaterial,
            selectedTexture,
            selectedMaterialIndex);
        
        
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