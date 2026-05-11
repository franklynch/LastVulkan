#include "Renderer.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cassert>
#include <memory>

#include <cstring>
#include <iostream>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <cstdio>
#include <array>


#include <stb_image.h>

#include "EditorPanels.hpp"


#include "ShaderUtils.hpp"
#include "TransitionUtils.hpp"
#include "PostProcessRenderer.hpp"




#include <chrono>



Renderer::Renderer(Window& window, VulkanContext& vkContext)
    : window(window)
    , vkContext(vkContext)
    , bufferUtils(vkContext)
    , imageUtils(vkContext, bufferUtils)
    , gltfSceneLoader(vkContext, bufferUtils, imageUtils)
    , swapchain(window, vkContext)
    , renderTargets(vkContext, imageUtils)
    , frameResources(vkContext)
    , scenePipelines(vkContext)
    , environmentSystem(vkContext, bufferUtils, imageUtils)
    , sceneRenderer(vkContext)
	, descriptorManager(vkContext)
    
{
    init();
}

Renderer::~Renderer()
{
    shutdownImGui();
    vkContext.getDevice().waitIdle();

    environmentSystem.resources().runtimeBrdfLut.pipeline = nullptr;
    environmentSystem.resources().runtimeBrdfLut.layout = nullptr;

    environmentSystem.resources().runtimeBrdfLut.sampler = nullptr;
    environmentSystem.resources().runtimeBrdfLut.view = nullptr;
    environmentSystem.resources().runtimeBrdfLut.memory = nullptr;
    environmentSystem.resources().runtimeBrdfLut.image = nullptr;

    for (auto& view : environmentSystem.resources().runtimeEnvironmentCubeFaces.views)
    {
        view = nullptr;
    }

    environmentSystem.resources().runtimeEnvironmentCube.sampler = nullptr;
    environmentSystem.resources().runtimeEnvironmentCube.view = nullptr;
    environmentSystem.resources().runtimeEnvironmentCube.memory = nullptr;
    environmentSystem.resources().runtimeEnvironmentCube.image = nullptr;

    for (auto& mipViews : environmentSystem.resources().runtimePrefilteredCubeMipFaceViews)
    {
        for (auto& view : mipViews.views)
        {
            view = nullptr;
        }
    }

    environmentSystem.resources().runtimePrefilteredCubeMipFaceViews.clear();

    environmentSystem.resources().runtimePrefilteredCube.sampler = nullptr;
    environmentSystem.resources().runtimePrefilteredCube.view = nullptr;
    environmentSystem.resources().runtimePrefilteredCube.memory = nullptr;
    environmentSystem.resources().runtimePrefilteredCube.image = nullptr;

   

    hdrEnvironmentSampler = nullptr;
    hdrEnvironmentView = nullptr;
    hdrEnvironmentMemory = nullptr;
    hdrEnvironmentImage = nullptr;

    for (auto& view : environmentSystem.resources().runtimeIrradianceCubeFaces.views)
    {
        view = nullptr;
    }

    environmentSystem.resources().runtimeIrradianceCube.sampler = nullptr;
    environmentSystem.resources().runtimeIrradianceCube.view = nullptr;
    environmentSystem.resources().runtimeIrradianceCube.memory = nullptr;
    environmentSystem.resources().runtimeIrradianceCube.image = nullptr;

    cleanupDescriptorResources();
}

void Renderer::init()
{

    swapchain.create();
    

    postProcessRenderer = std::make_unique<PostProcessRenderer>(
        vkContext,
        bufferUtils,
        imageUtils);

    postProcessRenderer->init(
        swapchain.extent(),
        swapchain.format());
    
    renderTargets.create(
        swapchain.extent(),
        postProcessRenderer->getHdrFormat());
    

    frameResources.createCommandBuffers(MAX_FRAMES_IN_FLIGHT);

    frameResources.createSyncObjects(
        MAX_FRAMES_IN_FLIGHT,
        swapchain.imageCount());

    createUniformBuffers();


    descriptorManager.createLayouts();

    scenePipelines.create(
        swapchain.extent(),
        postProcessRenderer->getHdrFormat(),
        renderTargets.depthFormat(),
        descriptorManager.frameLayout(),
        descriptorManager.materialLayout(),
        descriptorManager.iblLayout(),
        isWireframeSupported());

    scenePipelines.createSkybox(
        swapchain.extent(),
        postProcessRenderer->getHdrFormat(),
        renderTargets.depthFormat(),
        descriptorManager.frameLayout(),
        descriptorManager.iblLayout());

    clearSceneResources();
    createDefaultMaterialTextures();
    setupCameraDefaults();

    currentModelPath = "models/DamagedHelmet/glTF/DamagedHelmet.gltf";

    gltfSceneLoader.load(
        currentModelPath,
        scene,
        gpuMeshes,
        textures,
        normalTextures,
        metallicRoughnessTextures,
        aoTextures,
        emissiveTextures,
        materials,
        getDefaultTexture(),
        *defaultNormalTexture,
        *defaultMetallicRoughnessTexture,
        *defaultAoTexture,
        *defaultEmissiveTexture,
        camera);

    createUniformBuffers();
        
    descriptorManager.createDescriptorPool(
        MAX_FRAMES_IN_FLIGHT,
        static_cast<uint32_t>(materials.size()));

    descriptorManager.allocateFrameDescriptorSets(
        MAX_FRAMES_IN_FLIGHT);

    descriptorManager.allocateIBLDescriptorSet();

    createDescriptorSets();          // writes frame descriptors only
    createMaterialDescriptorSets();  // still Renderer-owned

    
   
    resetEnvironmentSettings();

    brdfLutRenderer = std::make_unique<BrdfLutRenderer>(vkContext, bufferUtils);

    brdfLutRenderer->init(environmentSystem.resources());

    environmentSystem.createFallbackResources();
    

    uiState.selectedRenderableIndex = scene.empty() ? -1 : 0;

   

    

    

   
    
    
    

    

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
        environmentSystem.resources(),
        hdrEnvironmentSampler,
        hdrEnvironmentView);
    

    irradianceRenderer = std::make_unique<IrradianceRenderer>(vkContext, bufferUtils);
    irradianceRenderer->init(environmentSystem.resources());
    
    prefilterRenderer = std::make_unique<PrefilterRenderer>(vkContext, bufferUtils);
    prefilterRenderer->init(environmentSystem.resources());


    environmentSystem.updateIBLDescriptorSet(
        descriptorManager.iblDescriptorSet(),
        *environmentCubeSampler,
        *environmentCubeView);

   
    initImGui();

}

void Renderer::cleanupSwapChain()
{
   scenePipelines.cleanup();
    renderTargets.cleanup();
    swapchain.cleanup();

    // frameResources.cleanupSwapchainDependent();

    
}

void Renderer::recreateSwapChain()
{
    int width = 0;
    int height = 0;
    window.getFramebufferSize(width, height);

    while (width == 0 || height == 0)
    {
        window.getFramebufferSize(width, height);
        glfwWaitEvents();
    }

    vkContext.getDevice().waitIdle();

    cleanupSwapChain();

    swapchain.create();
    
    

    postProcessRenderer->recreate(
        swapchain.extent(), 
        swapchain.format());

    renderTargets.create(
        swapchain.extent(),
        postProcessRenderer->getHdrFormat());

    
    frameResources.recreateSwapchainDependent(swapchain.imageCount());

    scenePipelines.cleanup();

    scenePipelines.create(
        swapchain.extent(),
        postProcessRenderer->getHdrFormat(),
        renderTargets.depthFormat(),
        descriptorManager.frameLayout(),
        descriptorManager.materialLayout(),
        descriptorManager.iblLayout(),
        isWireframeSupported());

    scenePipelines.createSkybox(
        swapchain.extent(),
        postProcessRenderer->getHdrFormat(),
        renderTargets.depthFormat(),
        descriptorManager.frameLayout(),
        descriptorManager.iblLayout());

    
}

void Renderer::createDescriptorSets()
{
    auto& device = vkContext.getDevice();

    auto& frameSets =
        descriptorManager.frameDescriptorSets();

    if (frameSets.size() != MAX_FRAMES_IN_FLIGHT)
    {
        throw std::runtime_error(
            "createDescriptorSets: frame descriptor sets not allocated");
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::DescriptorBufferInfo bufferInfo{};
        bufferInfo
            .setBuffer(*uniformBuffers[i])
            .setOffset(0)
            .setRange(sizeof(UniformBufferObject));

        vk::WriteDescriptorSet descriptorWrite{};
        descriptorWrite
            .setDstSet(*frameSets[i])
            .setDstBinding(0)
            .setDstArrayElement(0)
            .setDescriptorType(vk::DescriptorType::eUniformBuffer)
            .setDescriptorCount(1)
            .setBufferInfo(bufferInfo);

        device.updateDescriptorSets(descriptorWrite, nullptr);
    }
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

	const auto& extent = swapchain.extent();
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
        postProcessRenderer->postExposure,
        postProcessRenderer->toneMappingEnabled ? 1.0f : 0.0f,
        postProcessRenderer->gammaEnabled ? 1.0f : 0.0f,
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
        descriptorManager.materialLayout()
    );

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo
        .setDescriptorPool(*descriptorManager.descriptorPool())
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

void Renderer::drawFrame()
{
    static auto lastFrameTime = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    frameTimeMs = std::chrono::duration<float, std::milli>(now - lastFrameTime).count();
    lastFrameTime = now;
    fps = frameTimeMs > 0.0f ? 1000.0f / frameTimeMs : 0.0f;

    auto& device = vkContext.getDevice();
    auto& queue = vkContext.getQueue();

    uint32_t currentFrame = frameResources.currentFrameIndex();

    vk::Result waitResult =
        device.waitForFences(*frameResources.inFlightFence(currentFrame), VK_TRUE, UINT64_MAX);

    if (waitResult != vk::Result::eSuccess)
    {
        throw std::runtime_error("failed waiting for in-flight fence");
    }

    uint32_t imageIndex = 0;
    vk::Result result{};

    try
    {
        constexpr uint64_t acquireTimeoutNs = 1'000'000'000;

        auto acquireResult =
            swapchain.handle().acquireNextImage(
                acquireTimeoutNs,
                *frameResources.presentCompleteSemaphore(currentFrame),
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

    if (result == vk::Result::eTimeout)
    {
        return;
    }

    if (result == vk::Result::eErrorOutOfDateKHR)
    {
        window.resetResizedFlag();
        recreateSwapChain();
        return;
    }

    if (result != vk::Result::eSuccess &&
        result != vk::Result::eSuboptimalKHR)
    {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    if (imageIndex >= swapchain.imageCount())
    {
        throw std::runtime_error("imageIndex out of range for swapchain images");
    }

    if (frameResources.imageInFlight(imageIndex))
    {
        vk::Result imageFenceWaitResult =
            device.waitForFences(
                frameResources.imageInFlight(imageIndex),
                VK_TRUE,
                UINT64_MAX);

        if (imageFenceWaitResult != vk::Result::eSuccess)
        {
            throw std::runtime_error("failed waiting for image fence");
        }
    }

    frameResources.imageInFlight(imageIndex) = *frameResources.inFlightFence(currentFrame);

  //  beginImGuiFrame();
    updateCameraControls();
  //  buildImGui();

    updateUniformBuffer(currentFrame);

    device.resetFences(
        *frameResources.inFlightFence(currentFrame));

    auto& commandBuffer =
        frameResources.commandBuffer(currentFrame);

    commandBuffer.reset();

    recordCommandBuffer(imageIndex);

    vk::PipelineStageFlags waitStage =
        vk::PipelineStageFlagBits::eColorAttachmentOutput;

    vk::SubmitInfo submitInfo{};
    submitInfo
        .setWaitSemaphores(
            *frameResources.presentCompleteSemaphore(currentFrame))
        .setWaitDstStageMask(waitStage)
        .setCommandBuffers(*commandBuffer)
        .setSignalSemaphores(
            *frameResources.renderFinishedSemaphore(imageIndex));

    try
    {
        queue.submit(
            submitInfo,
            *frameResources.inFlightFence(currentFrame));
    }
    catch (const vk::SystemError& err)
    {
        std::cerr << "Queue submit failed: " << err.what() << std::endl;
        throw;
    }

    std::array<vk::SwapchainKHR, 1> swapchains = {
        swapchain.get()
    };

    vk::PresentInfoKHR presentInfo{};
    presentInfo
        .setWaitSemaphores(
            *frameResources.renderFinishedSemaphore(imageIndex))
        .setSwapchains(swapchains)
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



    frameResources.advanceFrame(MAX_FRAMES_IN_FLIGHT);
}

void Renderer::recordCommandBuffer(uint32_t imageIndex) 
{
    uint32_t currentFrame = frameResources.currentFrameIndex();

    auto& commandBuffer =
        frameResources.commandBuffer(currentFrame);

    commandBuffer.begin(vk::CommandBufferBeginInfo{});

    vk::CommandBuffer cmd = *commandBuffer;
	const auto extent = swapchain.extent();

    TransitionUtils::transitionToColorAttachment(
        cmd,
        *renderTargets.colorImage(),
        vk::ImageLayout::eUndefined);

    TransitionUtils::transitionToColorAttachment(
        cmd,
        postProcessRenderer->getHdrImage(),
        vk::ImageLayout::eUndefined);

    vk::ImageLayout swapChainOldLayout =
        frameResources.imageInitialized(imageIndex)
        ? vk::ImageLayout::ePresentSrcKHR
        : vk::ImageLayout::eUndefined;

    

    TransitionUtils::transitionToDepthAttachment(
        cmd,
        *renderTargets.depthImage(),
        renderTargets.depthAspect());
        

    vk::ClearValue clearColorValue = vk::ClearColorValue(
        clearColor.r,
        clearColor.g,
        clearColor.b,
        clearColor.a);

    vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

    vk::RenderingAttachmentInfo colorAttachmentInfo{};
    colorAttachmentInfo
        .setImageView(*renderTargets.colorView())
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eDontCare)
        .setClearValue(clearColorValue)
        .setResolveMode(vk::ResolveModeFlagBits::eAverage)
        .setResolveImageView(postProcessRenderer->getHdrView())
        .setResolveImageLayout(vk::ImageLayout::eColorAttachmentOptimal);

    vk::RenderingAttachmentInfo depthAttachmentInfo{};
    depthAttachmentInfo
        .setImageView(*renderTargets.depthView())
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



   

    SceneRenderer::SceneRenderContext renderContext{};

    renderContext.pipelineLayout = scenePipelines.layout();

    renderContext.solidPipeline =
        scenePipelines.solid(false);

    renderContext.solidDoubleSidedPipeline =
        scenePipelines.solid(true);

    renderContext.wireframePipeline =
        scenePipelines.wireframe(false);

    renderContext.wireframeDoubleSidedPipeline =
        scenePipelines.wireframe(true);

    renderContext.frameDescriptorSet =
        *descriptorManager.frameDescriptorSets()[frameResources.currentFrameIndex()];

    renderContext.iblDescriptorSet =
        *descriptorManager.iblDescriptorSet();

    renderContext.materialDescriptorSets =
        &materialDescriptorSets;

    renderContext.wireframeEnabled =
        uiState.wireframeRequested;

    renderContext.animateModel =
        animateModel;

    renderContext.currentAnimationAngle =
        currentAnimationAngle;

    renderContext.transparentPipeline =
        scenePipelines.transparent(false);

    renderContext.transparentDoubleSidedPipeline =
        scenePipelines.transparent(true);

    renderContext.skyboxPipeline =
        scenePipelines.skybox();

    renderContext.skyboxPipelineLayout =
        scenePipelines.skyboxLayout();

    sceneRenderer.renderSkybox(
        commandBuffer,
        renderContext);

    sceneRenderer.renderOpaque(
        commandBuffer,
        scene,
        renderContext);

    sceneRenderer.renderTransparent(
        commandBuffer,
        scene,
        camera,
        renderContext);


  //  renderImGui(*commandBuffer);

    commandBuffer.endRendering();


    postProcessRenderer->executeBloomChain(commandBuffer);

    // Final post-process pass

    postProcessRenderer->executeFinalComposite(
        commandBuffer,
        swapchain.images()[imageIndex],
        *swapchain.imageViews()[imageIndex],
        swapChainOldLayout);


    frameResources.markImageInitialized(imageIndex);

    commandBuffer.end();

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
    
    materialDescriptorSets.clear();
    

    
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

    postProcessRenderer->toneMappingEnabled = true;
    postProcessRenderer->gammaEnabled    = true;
    postProcessRenderer->postExposure = 1.0f;

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


void Renderer::applyIblCalibrationPreset(const IblCalibrationPreset& preset)
{
    lightIntensity = preset.lightIntensity;
    skyboxExposure = preset.skyboxExposure;
    iblIntensity = preset.iblIntensity;
    diffuseIBLIntensity = preset.diffuseIBLIntensity;
    specularIBLIntensity = preset.specularIBLIntensity;
    postProcessRenderer->postExposure = preset.postExposure;
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
    initInfo.MinImageCount = static_cast<uint32_t>(swapchain.images().size());
    initInfo.ImageCount = static_cast<uint32_t>(swapchain.images().size());
    initInfo.UseDynamicRendering = true;
    initInfo.CheckVkResultFn = nullptr;

    VkFormat colorFormat =
        static_cast<VkFormat>(swapchain.surfaceFormat().format);

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

            postProcessRenderer->toneMappingEnabled,
            postProcessRenderer->gammaEnabled,
            postProcessRenderer->postExposure,

            postProcessRenderer->bloomEnabled,
            postProcessRenderer->bloomThreshold,
            postProcessRenderer->bloomKnee,
            postProcessRenderer->bloomStrength,

            postProcessRenderer->bloomIntensity,
            postProcessRenderer->bloomUpsampleRadius,

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


        EditorPanels::drawRendererPanel(
            vkContext,
            swapchain.extent(),
            swapchain.images().size(),
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