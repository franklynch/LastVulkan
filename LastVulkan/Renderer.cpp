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

    hdrEnvironmentSampler = nullptr;
    hdrEnvironmentView = nullptr;
    hdrEnvironmentMemory = nullptr;
    hdrEnvironmentImage = nullptr;

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

    createUniformBuffers();
        
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

    if (editorUi.isInitialized())
    {
        editorUi.beginFrame();

        editorUi.buildMinimal(
            frameTimeMs,
            fps);

        updateCameraControls();
    }

   


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
        &descriptorManager.materialDescriptorSets();

    renderContext.wireframeEnabled =
        editorUi.state().wireframeRequested;

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




    commandBuffer.endRendering();


    postProcessRenderer->executeBloomChain(commandBuffer);

    

    // Final post-process pass
    TransitionUtils::transitionToColorAttachment(
        cmd,
        swapchain.images()[imageIndex],
        swapChainOldLayout);

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
        materialSystem.materials().begin(),
        materialSystem.materials().end(),
        [&](const std::unique_ptr<Material>& candidate)
        {
            return candidate.get() == &material;
        });

    if (it == materialSystem.materials().end())
    {
        return -1;
    }

    return static_cast<int>(
        std::distance(
            materialSystem.materials().begin(),
            it));
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




void Renderer::buildImGui()
{
    
}
 




