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
	, materialSystem(vkContext, bufferUtils, imageUtils)
	, editorUi(window, vkContext)
    
{
    init();
}

Renderer::~Renderer()
{
    vkContext.getDevice().waitIdle();

    environmentSystem.cleanup();

   

    editorUi.shutdown();

    frameResources.cleanup();
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
        
        postProcessRenderer->getHdrFormat(),
        renderTargets.depthFormat(),
        descriptorManager.frameLayout(),
        descriptorManager.materialLayout(),
        descriptorManager.iblLayout(),
        isWireframeSupported());

    scenePipelines.createSkybox(
        
        postProcessRenderer->getHdrFormat(),
        renderTargets.depthFormat(),
        descriptorManager.frameLayout(),
        descriptorManager.iblLayout());

    clearSceneResources();
    
    materialSystem.clear();
    materialSystem.createDefaultTextures();

    setupCameraDefaults();

    currentModelPath = "models/DamagedHelmet/glTF/DamagedHelmet.gltf";

    gltfSceneLoader.load(
        currentModelPath,
        scene,
        gpuMeshes,
        materialSystem.baseColorTextures(),
        materialSystem.normalTextures(),
        materialSystem.metallicRoughnessTextures(),
        materialSystem.aoTextures(),
        materialSystem.emissiveTextures(),
        materialSystem.materials(),
        materialSystem.defaultTexture(),
        materialSystem.defaultNormalTexture(),
        materialSystem.defaultMetallicRoughnessTexture(),
        materialSystem.defaultAoTexture(),
        materialSystem.defaultEmissiveTexture(),
        camera);

    
        
    descriptorManager.createDescriptorPool(
        MAX_FRAMES_IN_FLIGHT,
        static_cast<uint32_t>(materialSystem.materials().size()));

    descriptorManager.allocateFrameDescriptorSets(
        MAX_FRAMES_IN_FLIGHT);

    descriptorManager.allocateIBLDescriptorSet();

   
    descriptorManager.updateFrameDescriptorSets(
        uniformBuffers,
        MAX_FRAMES_IN_FLIGHT);


    descriptorManager.createMaterialDescriptorSets(
        materialSystem.materials());

    
   
    resetEnvironmentSettings();

    environmentSystem.createFallbackResources();

    environmentSystem.createFallbackEnvironmentCubemap({
           "assets/skybox/right.jpg",
           "assets/skybox/left.jpg",
           "assets/skybox/top.jpg",
           "assets/skybox/bottom.jpg",
           "assets/skybox/front.jpg",
           "assets/skybox/back.jpg" });

  
    environmentSystem.loadHdrEnvironment(
        "assets/hdr/citrus_orchard_road_puresky_4k.hdr",
        descriptorManager.iblDescriptorSet());

    
    
   

    editorUi.init(
        swapchain.format(),
        swapchain.imageCount());
    

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
        
        postProcessRenderer->getHdrFormat(),
        renderTargets.depthFormat(),
        descriptorManager.frameLayout(),
        descriptorManager.materialLayout(),
        descriptorManager.iblLayout(),
        isWireframeSupported());

    scenePipelines.createSkybox(
        
        postProcessRenderer->getHdrFormat(),
        renderTargets.depthFormat(),
        descriptorManager.frameLayout(),
        descriptorManager.iblLayout());

    
}


void Renderer::clearSceneResources()
{
   
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

//    ubo.debugParams = glm::ivec4(uiState.debugViewMode, 0, 0, 0);

    const uint32_t mipLevels =
        environmentSystem.getDebugRuntimePrefilteredMipLevels();

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


void Renderer::updateFrameTiming()
{
    static auto lastFrameTime = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    frameTimeMs = std::chrono::duration<float, std::milli>(now - lastFrameTime).count();
    lastFrameTime = now;
    fps = frameTimeMs > 0.0f ? 1000.0f / frameTimeMs : 0.0f;
}

void Renderer::updateEditorUiFrame()
{
    if (editorUi.isInitialized())
    {
        editorUi.beginFrame();

        editorUi.buildMinimal(
            frameTimeMs,
            fps);

        InputState input = editorUi.captureInputState();
        updateCameraControls(input);
    }
}

Renderer::AcquiredImage Renderer::acquireSwapchainImage(
    uint32_t currentFrame)
{
    uint32_t imageIndex = 0;
    vk::Result result{};

    try
    {

        // Avoid UINT64_MAX here because validation warns when forward progress
        // cannot be guaranteed for the surface.
        constexpr uint64_t acquireTimeoutNs =
            1'000'000'000;

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
        return {};
    }

    if (result == vk::Result::eTimeout)
    {
        return {};
    }

    if (result == vk::Result::eErrorOutOfDateKHR)
    {
        window.resetResizedFlag();
        recreateSwapChain();
        return {};
    }

    if (result != vk::Result::eSuccess &&
        result != vk::Result::eSuboptimalKHR)
    {
        throw std::runtime_error(
            "failed to acquire swap chain image!");
    }

    if (imageIndex >= swapchain.imageCount())
    {
        throw std::runtime_error(
            "imageIndex out of range for swapchain images");
    }

    return {
        .imageIndex = imageIndex,
        .valid = true
    };
}

void Renderer::waitForSwapchainImageFence(
    uint32_t imageIndex,
    uint32_t currentFrame)
{
    auto& device = vkContext.getDevice();

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
}

void Renderer::submitCommandBuffer(
    uint32_t currentFrame,
    uint32_t imageIndex,
    vk::raii::CommandBuffer& commandBuffer)
{
    auto& queue = vkContext.getQueue();


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
        std::cerr << "Queue submit failed: "
            << err.what()
            << std::endl;
        throw;
    }


}

bool Renderer::presentFrame(uint32_t imageIndex)
{
    auto& queue = vkContext.getQueue();

    std::array<vk::SwapchainKHR, 1> swapchains = {
        swapchain.get()
    };

    vk::PresentInfoKHR presentInfo{};
    presentInfo
        .setWaitSemaphores(
            *frameResources.renderFinishedSemaphore(imageIndex))
        .setSwapchains(swapchains)
        .setPImageIndices(&imageIndex);

    vk::Result result{};

    try
    {
        result = queue.presentKHR(presentInfo);
    }
    catch (const vk::OutOfDateKHRError&)
    {
        window.resetResizedFlag();
        recreateSwapChain();
        return false;
    }

    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR ||
        window.wasResized())
    {
        window.resetResizedFlag();
        recreateSwapChain();
        return false;
    }

    if (result != vk::Result::eSuccess)
    {
        throw std::runtime_error(
            "failed to present swap chain image!");
    }

    return true;
}

SceneRenderer::SceneRenderContext Renderer::buildSceneRenderContext() const
{
    SceneRenderer::SceneRenderContext renderContext{};

    renderContext.pipelineLayout =
        scenePipelines.layout();

    renderContext.solidPipeline =
        scenePipelines.solid(false);

    renderContext.solidDoubleSidedPipeline =
        scenePipelines.solid(true);

    renderContext.wireframePipeline =
        scenePipelines.wireframe(false);

    renderContext.wireframeDoubleSidedPipeline =
        scenePipelines.wireframe(true);

    renderContext.transparentPipeline =
        scenePipelines.transparent(false);

    renderContext.transparentDoubleSidedPipeline =
        scenePipelines.transparent(true);

    renderContext.skyboxPipeline =
        scenePipelines.skybox();

    renderContext.skyboxPipelineLayout =
        scenePipelines.skyboxLayout();

    renderContext.frameDescriptorSet =
        *descriptorManager.frameDescriptorSets()
        [frameResources.currentFrameIndex()];

    renderContext.iblDescriptorSet =
        *descriptorManager.iblDescriptorSet();

    renderContext.materialDescriptorSets =
        &descriptorManager.materialDescriptorSets();

    renderContext.wireframeEnabled =
        editorUi.state().wireframeRequested;

    renderContext.animateModel =
        animateModel;

    renderContext.currentAnimationAngle =
        currentAnimationAngle;

    return renderContext;
}

void Renderer::recordScenePass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
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




    auto renderContext =
        buildSceneRenderContext();
    
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

    commandBuffer.endRendering();
}

void Renderer::recordBloomPass(
    vk::raii::CommandBuffer& commandBuffer)
{
    postProcessRenderer->executeBloomChain(commandBuffer);
}

void Renderer::recordFinalCompositePass(
    vk::raii::CommandBuffer& commandBuffer,
    uint32_t imageIndex)
{
    vk::CommandBuffer cmd = *commandBuffer;

    vk::ImageLayout swapchainOldLayout =
        frameResources.imageInitialized(imageIndex)
        ? vk::ImageLayout::ePresentSrcKHR
        : vk::ImageLayout::eUndefined;

    TransitionUtils::transitionToColorAttachment(
        cmd,
        swapchain.images()[imageIndex],
        swapchainOldLayout);

    postProcessRenderer->beginFinalCompositePass(
        commandBuffer,
        *swapchain.imageViews()[imageIndex]);

    postProcessRenderer->recordFinalComposite(commandBuffer);

    if (editorUi.isInitialized())
    {
        editorUi.render(*commandBuffer);
    }

    postProcessRenderer->endFinalCompositePass(commandBuffer);

    TransitionUtils::transitionToPresent(
        cmd,
        swapchain.images()[imageIndex]);
}


void Renderer::drawFrame()
{
    updateFrameTiming();

    auto& device = vkContext.getDevice();
    auto& queue = vkContext.getQueue();

    uint32_t currentFrame = frameResources.currentFrameIndex();

    vk::Result waitResult =
        device.waitForFences(*frameResources.inFlightFence(currentFrame), VK_TRUE, UINT64_MAX);

    if (waitResult != vk::Result::eSuccess)
    {
        throw std::runtime_error("failed waiting for in-flight fence");
    }
    
   
    auto acquired =
        acquireSwapchainImage(currentFrame);

    if (!acquired.valid)
    {
        return;
    }

    uint32_t imageIndex =
        acquired.imageIndex;


    

    waitForSwapchainImageFence(
        imageIndex,
        currentFrame);

    updateEditorUiFrame();

    
    updateUniformBuffer(currentFrame);

    device.resetFences(
        *frameResources.inFlightFence(currentFrame));

    auto& commandBuffer =
        frameResources.commandBuffer(currentFrame);

    commandBuffer.reset();

    recordCommandBuffer(imageIndex);

    
    submitCommandBuffer(
        currentFrame,
        imageIndex,
        commandBuffer);


    if (!presentFrame(imageIndex))
    {
        return;
    }



    frameResources.advanceFrame(MAX_FRAMES_IN_FLIGHT);
}

void Renderer::recordCommandBuffer(uint32_t imageIndex) 
{
    uint32_t currentFrame = frameResources.currentFrameIndex();

    auto& commandBuffer =
        frameResources.commandBuffer(currentFrame);

    commandBuffer.begin(vk::CommandBufferBeginInfo{});

    recordScenePass(commandBuffer, imageIndex);
    
    recordBloomPass(commandBuffer);
    
    recordFinalCompositePass(commandBuffer, imageIndex);

    frameResources.markImageInitialized(imageIndex);

    commandBuffer.end();

}

void Renderer::resetDefaultSceneLayout()
{
    auto& renderables = scene.getRenderables();

    if (renderables.size() > 0)
    {
        renderables[0].getTransform().useMatrixOverride = false;

        renderables[0].setName("Center");
        renderables[0].getTransform().position = { 0.0f, 0.0f, 0.0f };
        renderables[0].getTransform().rotation = { 0.0f, 0.0f, 0.0f };
        renderables[0].getTransform().scale = { 1.0f, 1.0f, 1.0f };
    }

    if (renderables.size() > 1)
    {
        renderables[1].getTransform().useMatrixOverride = false;
        renderables[1].setName("Right");
        renderables[1].getTransform().position = { 1.5f, 0.0f, 0.0f };
        renderables[1].getTransform().rotation = { 0.0f, 0.0f, 0.0f };
        renderables[1].getTransform().scale = { 1.0f, 1.0f, 1.0f };
    }

    if (renderables.size() > 2)
    {
        renderables[2].getTransform().useMatrixOverride = false;
        renderables[2].setName("Left");
        renderables[2].getTransform().position = { -1.5f, 0.0f, 0.0f };
        renderables[2].getTransform().rotation = { 0.0f, 0.0f, 0.0f };
        renderables[2].getTransform().scale = { 1.0f, 1.0f, 1.0f };
    }

    for (size_t i = 3; i < renderables.size(); ++i)
    {
        renderables[i].getTransform().useMatrixOverride = false;

        renderables[i].setName("Renderable " + std::to_string(i));
        renderables[i].getTransform().position = { 0.0f, 0.0f, 0.0f };
        renderables[i].getTransform().rotation = { 0.0f, 0.0f, 0.0f };
        renderables[i].getTransform().scale = { 1.0f, 1.0f, 1.0f };
    }
}






void Renderer::updateCameraControls(const InputState& input)
{
    if (!input.wantsMouseCapture)
    {
        if (input.rightMouseDown)
        {
            cameraYaw -= input.mouseDeltaX * mouseOrbitSensitivity;
            cameraPitch -= input.mouseDeltaY * mouseOrbitSensitivity;

            cameraPitch = std::clamp(cameraPitch, -1.55f, 1.55f);
        }

        if (input.middleMouseDown)
        {
            glm::vec3 forward =
                glm::normalize(camera.getTarget() - camera.getPosition());

            glm::vec3 right =
                glm::normalize(glm::cross(forward, glm::vec3(0.0f, 0.0f, 1.0f)));

            glm::vec3 up =
                glm::normalize(glm::cross(right, forward));

            glm::vec3 panDelta =
                (-right * input.mouseDeltaX + up * input.mouseDeltaY) *
                (mousePanSensitivity * cameraRadius);

            camera.offsetPosition(panDelta);
            camera.offsetTarget(panDelta);
        }

        if (input.scrollDelta != 0.0f)
        {
            cameraRadius -= input.scrollDelta * mouseZoomSensitivity;
            cameraRadius = std::clamp(
                cameraRadius,
                minCameraRadius,
                maxCameraRadius);
        }
    }

    glm::vec3 target = camera.getTarget();

    camera.setOrbit(
        cameraRadius,
        cameraYaw,
        cameraPitch);

    camera.setTarget(target);
    camera.setFov(cameraFov);
    camera.setNearFar(cameraNear, cameraFar);
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









