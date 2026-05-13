#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <array>

#include "PostProcessRenderer.hpp"
#include "ShaderUtils.hpp"
#include "TransitionUtils.hpp"


PostProcessRenderer::PostProcessRenderer(
    VulkanContext& vkContext,
    BufferUtils& bufferUtils,
    ImageUtils& imageUtils)
    : vkContext(vkContext)
    , bufferUtils(bufferUtils)
    , imageUtils(imageUtils)
{
}

void PostProcessRenderer::init(
    vk::Extent2D newExtent,
    vk::Format newSwapchainFormat)
{
    extent = newExtent;
    swapchainFormat = newSwapchainFormat;

    createHdrColorResources();

    createBloomBrightResources();
    createBloomBlurResources();
    createBloomDownsampleResources();

    createPostProcessSampler();

    createBloomExtractDescriptorSetLayout();
    createBloomBlurDescriptorSetLayout();
    createPostProcessDescriptorSetLayout();

    createBloomExtractPipeline();
    createBloomBlurPipeline();
    createBloomDownsamplePipeline();
    createBloomUpsamplePipeline();
    createPostProcessPipeline();

    createBloomExtractDescriptorSet();
    createBloomBlurDescriptorSets();
    createBloomDownsampleDescriptorSets();
    createBloomUpsampleDescriptorSets();
    createPostProcessDescriptorSet();
    
}

void PostProcessRenderer::cleanup()
{
    
    postProcessPipeline = nullptr;
    postProcessPipelineLayout = nullptr;

    postProcessDescriptorSets = nullptr;
    postProcessDescriptorPool = nullptr;
    postProcessDescriptorSetLayout = nullptr;
    
    
    bloomUpsamplePipeline = nullptr;
    bloomUpsamplePipelineLayout = nullptr;

    bloomDownsamplePipeline = nullptr;
    bloomDownsamplePipelineLayout = nullptr;

    bloomUpsampleFinalDescriptorSet = nullptr;
    bloomUpsampleDescriptorSets.clear();
    bloomUpsampleDescriptorPool = nullptr;

    bloomDownsampleDescriptorSets.clear();
    bloomDownsampleDescriptorPool = nullptr;

    bloomDownsampleChain.clear();
    
    
    bloomBlurPipeline = nullptr;
    bloomBlurPipelineLayout = nullptr;

    bloomExtractPipeline = nullptr;
    bloomExtractPipelineLayout = nullptr;

    bloomBlurFromTempDescriptorSets = nullptr;
    bloomBlurFromBrightDescriptorSets = nullptr;
    bloomBlurDescriptorPool = nullptr;
    bloomBlurDescriptorSetLayout = nullptr;

    bloomExtractDescriptorSets = nullptr;
    bloomExtractDescriptorPool = nullptr;
    bloomExtractDescriptorSetLayout = nullptr;

    postProcessSampler = nullptr;

    bloomBlurTempView = nullptr;
    bloomBlurTempImage = nullptr;
    bloomBlurTempMemory = nullptr;

    bloomBrightView = nullptr;
    bloomBrightImage = nullptr;
    bloomBrightMemory = nullptr;

    hdrColorView = nullptr;
    hdrColorImage = nullptr;
    hdrColorMemory = nullptr;
}

void PostProcessRenderer::recreate(
    vk::Extent2D newExtent,
    vk::Format newSwapchainFormat)
{
    cleanup();

    extent = newExtent;
    swapchainFormat = newSwapchainFormat;

    createHdrColorResources();

    createBloomBrightResources();
    createBloomBlurResources();
    createBloomDownsampleResources();

    createPostProcessSampler();

    createBloomExtractDescriptorSetLayout();
    createBloomBlurDescriptorSetLayout();
    createPostProcessDescriptorSetLayout();

    createBloomExtractPipeline();
    createBloomBlurPipeline();
    createBloomDownsamplePipeline();
    createBloomUpsamplePipeline();
    createPostProcessPipeline();

    createBloomExtractDescriptorSet();
    createBloomBlurDescriptorSets();
    createBloomDownsampleDescriptorSets();
    createBloomUpsampleDescriptorSets();
    createPostProcessDescriptorSet();
    
    
}

void PostProcessRenderer::createHdrColorResources()
{
    

    imageUtils.createImage(
        extent.width,
        extent.height,
        1,
        vk::SampleCountFlagBits::e1,
        hdrColorFormat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment |
        vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        hdrColorImage,
        hdrColorMemory);

    hdrColorView = imageUtils.createImageView(
        hdrColorImage,
        hdrColorFormat,
        vk::ImageAspectFlagBits::eColor,
        1);
}

void PostProcessRenderer::createBloomBrightResources()
{
    imageUtils.createImage(
        extent.width,
        extent.height,
        1,
        vk::SampleCountFlagBits::e1,
        hdrColorFormat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment |
        vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        bloomBrightImage,
        bloomBrightMemory);

    bloomBrightView = imageUtils.createImageView(
        bloomBrightImage,
        hdrColorFormat,
        vk::ImageAspectFlagBits::eColor,
        1);
}

void PostProcessRenderer::createBloomBlurResources()
{
    imageUtils.createImage(
        extent.width,
        extent.height,
        1,
        vk::SampleCountFlagBits::e1,
        hdrColorFormat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment |
        vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        bloomBlurTempImage,
        bloomBlurTempMemory);

    bloomBlurTempView = imageUtils.createImageView(
        bloomBlurTempImage,
        hdrColorFormat,
        vk::ImageAspectFlagBits::eColor,
        1);
}

void PostProcessRenderer::createBloomExtractDescriptorSetLayout()
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

    bloomExtractDescriptorSetLayout =
        vk::raii::DescriptorSetLayout(device, layoutInfo);
}

void PostProcessRenderer::createBloomExtractPipeline()
{
    auto& device = vkContext.getDevice();

    vk::raii::ShaderModule vertShaderModule =
        ShaderUtils::createShaderModule(
            device,
            "shaders/post_fullscreen.spv");

    vk::raii::ShaderModule fragShaderModule =
        ShaderUtils::createShaderModule(
            device,
            "shaders/bloom_extract.spv");

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

    std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {
        vertStage,
        fragStage
    };

    vk::PipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput
        .setVertexBindingDescriptions({})
        .setVertexAttributeDescriptions({});

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
        .setDepthBoundsTestEnable(VK_FALSE)
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
        .setSetLayouts(*bloomExtractDescriptorSetLayout)
        .setPushConstantRanges(pushRange);

    bloomExtractPipelineLayout =
        vk::raii::PipelineLayout(device, layoutInfo);

    std::array<vk::Format, 1> colorFormats = {
        getHdrFormat() };

    vk::StructureChain<
        vk::GraphicsPipelineCreateInfo,
        vk::PipelineRenderingCreateInfo
    > pipelineChain{};

    pipelineChain.get<vk::GraphicsPipelineCreateInfo>()
        .setStages(shaderStages)
        .setPVertexInputState(&vertexInput)
        .setPInputAssemblyState(&inputAssembly)
        .setPViewportState(&viewportState)
        .setPRasterizationState(&rasterizer)
        .setPMultisampleState(&multisampling)
        .setPDepthStencilState(&depthStencil)
        .setPColorBlendState(&colorBlending)
        .setPDynamicState(&dynamicState)
        .setLayout(*bloomExtractPipelineLayout)
        .setRenderPass(vk::RenderPass{});

    pipelineChain.get<vk::PipelineRenderingCreateInfo>()
        .setColorAttachmentFormats(colorFormats);

    bloomExtractPipeline =
        vk::raii::Pipeline(
            device,
            nullptr,
            pipelineChain.get<vk::GraphicsPipelineCreateInfo>());
}

void PostProcessRenderer::createBloomExtractDescriptorSet()
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

    bloomExtractDescriptorPool =
        vk::raii::DescriptorPool(device, poolInfo);

    vk::DescriptorSetLayout layout = *bloomExtractDescriptorSetLayout;

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo
        .setDescriptorPool(*bloomExtractDescriptorPool)
        .setSetLayouts(layout);

    bloomExtractDescriptorSets =
        vk::raii::DescriptorSets(device, allocInfo);

    vk::DescriptorImageInfo hdrInfo{};
    hdrInfo
        .setSampler(*postProcessSampler)
        .setImageView(getHdrView())
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);






    vk::WriteDescriptorSet write{};
    write
        .setDstSet(*bloomExtractDescriptorSets[0])
        .setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(hdrInfo);

    device.updateDescriptorSets(write, nullptr);
}



void PostProcessRenderer::recordBloomExtract(
    vk::raii::CommandBuffer& commandBuffer)
{
    drawBloomExtract(commandBuffer);
}

void PostProcessRenderer::recordBloomBlurFromBright(
    vk::raii::CommandBuffer& commandBuffer,
    vk::ImageView outputView,
    glm::vec2 direction)
{
    drawBloomBlur(
        commandBuffer,
        outputView,
        *bloomBlurFromBrightDescriptorSets[0],
        direction);
}

void PostProcessRenderer::recordBloomBlurFromTemp(
    vk::raii::CommandBuffer& commandBuffer,
    vk::ImageView outputView,
    glm::vec2 direction)
{
    drawBloomBlur(
        commandBuffer,
        outputView,
        *bloomBlurFromTempDescriptorSets[0],
        direction);
}

void PostProcessRenderer::drawBloomExtract(vk::raii::CommandBuffer& commandBuffer)
{
    vk::ClearValue clearColor = vk::ClearColorValue(
        std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f });

    vk::RenderingAttachmentInfo colorAttachment{};
    colorAttachment
        .setImageView(getBloomBrightView())
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
        *bloomExtractPipeline);

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *bloomExtractPipelineLayout,
        0,
        *bloomExtractDescriptorSets[0],
        {});

    glm::vec4 params(
        ppsettings.bloomThreshold,
        ppsettings.bloomKnee,
        0.0f,
        0.0f);

    commandBuffer.pushConstants<glm::vec4>(
        *bloomExtractPipelineLayout,
        vk::ShaderStageFlagBits::eFragment,
        0,
        params);

    commandBuffer.draw(3, 1, 0, 0);

    commandBuffer.endRendering();
}

void PostProcessRenderer::createBloomBlurDescriptorSetLayout()
{
    auto& device = vkContext.getDevice();

    vk::DescriptorSetLayoutBinding inputBinding{};
    inputBinding
        .setBinding(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eFragment);

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.setBindings(inputBinding);

    bloomBlurDescriptorSetLayout =
        vk::raii::DescriptorSetLayout(device, layoutInfo);
}

void PostProcessRenderer::createBloomBlurPipeline()
{
    auto& device = vkContext.getDevice();

    vk::raii::ShaderModule vertShaderModule =
        ShaderUtils::createShaderModule(
            device,
            "shaders/post_fullscreen.spv");

    vk::raii::ShaderModule fragShaderModule =
        ShaderUtils::createShaderModule(
            device,
            "shaders/bloom_blur.spv");

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

    std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {
        vertStage,
        fragStage
    };

    vk::PipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput
        .setVertexBindingDescriptions({})
        .setVertexAttributeDescriptions({});

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
        .setDepthBoundsTestEnable(VK_FALSE)
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
        .setSetLayouts(*bloomBlurDescriptorSetLayout)
        .setPushConstantRanges(pushRange);
    ;

    bloomBlurPipelineLayout =
        vk::raii::PipelineLayout(device, layoutInfo);

    std::array<vk::Format, 1> colorFormats = {
        getHdrFormat()
    };

    vk::StructureChain<
        vk::GraphicsPipelineCreateInfo,
        vk::PipelineRenderingCreateInfo
    > pipelineChain{};

    pipelineChain.get<vk::GraphicsPipelineCreateInfo>()
        .setStages(shaderStages)
        .setPVertexInputState(&vertexInput)
        .setPInputAssemblyState(&inputAssembly)
        .setPViewportState(&viewportState)
        .setPRasterizationState(&rasterizer)
        .setPMultisampleState(&multisampling)
        .setPDepthStencilState(&depthStencil)
        .setPColorBlendState(&colorBlending)
        .setPDynamicState(&dynamicState)
        .setLayout(*bloomBlurPipelineLayout)
        .setRenderPass(vk::RenderPass{});

    pipelineChain.get<vk::PipelineRenderingCreateInfo>()
        .setColorAttachmentFormats(colorFormats);

    bloomBlurPipeline =
        vk::raii::Pipeline(
            device,
            nullptr,
            pipelineChain.get<vk::GraphicsPipelineCreateInfo>());
}



void PostProcessRenderer::createBloomBlurDescriptorSets()
{
    auto& device = vkContext.getDevice();

    vk::DescriptorPoolSize poolSize{};
    poolSize
        .setType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(2);

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo
        .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
        .setMaxSets(2)
        .setPoolSizes(poolSize);

    bloomBlurDescriptorPool =
        vk::raii::DescriptorPool(device, poolInfo);

    vk::DescriptorSetLayout layout = *bloomBlurDescriptorSetLayout;

    vk::DescriptorSetAllocateInfo allocBrightInfo{};
    allocBrightInfo
        .setDescriptorPool(*bloomBlurDescriptorPool)
        .setSetLayouts(layout);

    bloomBlurFromBrightDescriptorSets =
        vk::raii::DescriptorSets(device, allocBrightInfo);

    vk::DescriptorSetAllocateInfo allocTempInfo{};
    allocTempInfo
        .setDescriptorPool(*bloomBlurDescriptorPool)
        .setSetLayouts(layout);

    bloomBlurFromTempDescriptorSets =
        vk::raii::DescriptorSets(device, allocTempInfo);

    auto writeSet = [&](vk::DescriptorSet set, vk::ImageView view)
        {
            vk::DescriptorImageInfo info{};
            info
                .setSampler(*postProcessSampler)
                .setImageView(view)
                .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

            vk::WriteDescriptorSet write{};
            write
                .setDstSet(set)
                .setDstBinding(0)
                .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
                .setDescriptorCount(1)
                .setImageInfo(info);

            device.updateDescriptorSets(write, nullptr);
        };

    writeSet(*bloomBlurFromBrightDescriptorSets[0], getBloomBrightView());
    writeSet(*bloomBlurFromTempDescriptorSets[0], getBloomBlurTempView());
}

void PostProcessRenderer::drawBloomBlur(
    vk::raii::CommandBuffer& commandBuffer,
    vk::ImageView outputView,
    vk::DescriptorSet inputSet,
    glm::vec2 direction)
{
    vk::ClearValue clearColor = vk::ClearColorValue(
        std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f });

    vk::RenderingAttachmentInfo colorAttachment{};
    colorAttachment
        .setImageView(outputView)
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
        *bloomBlurPipeline);

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *bloomBlurPipelineLayout,
        0,
        inputSet,
        {});

    glm::vec4 params(
        direction.x,
        direction.y,
        1.0f / static_cast<float>(extent.width),
        1.0f / static_cast<float>(extent.height));

    commandBuffer.pushConstants<glm::vec4>(
        *bloomBlurPipelineLayout,
        vk::ShaderStageFlagBits::eFragment,
        0,
        params);

    commandBuffer.draw(3, 1, 0, 0);

    commandBuffer.endRendering();
}

void PostProcessRenderer::createBloomDownsampleResources()
{
    bloomDownsampleChain.clear();

    uint32_t width = extent.width / 2;
    uint32_t height = extent.height / 2;

    for (uint32_t i = 0; i < bloomDownsampleLevels; ++i)
    {
        width = std::max(1u, width);
        height = std::max(1u, height);

        BloomMipResource level{};
        level.width = width;
        level.height = height;

        imageUtils.createImage(
            width,
            height,
            1,
            vk::SampleCountFlagBits::e1,
            getHdrFormat(),
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eColorAttachment |
            vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            level.image,
            level.memory);

        level.view = imageUtils.createImageView(
            level.image,
            getHdrFormat(),
            vk::ImageAspectFlagBits::eColor,
            1);

        bloomDownsampleChain.push_back(std::move(level));

        width /= 2;
        height /= 2;
    }
}

void PostProcessRenderer::createBloomDownsamplePipeline()
{
    auto& device = vkContext.getDevice();

    vk::raii::ShaderModule vertShaderModule =
        ShaderUtils::createShaderModule(device, "shaders/post_fullscreen.spv");

    vk::raii::ShaderModule fragShaderModule =
        ShaderUtils::createShaderModule(device, "shaders/bloom_downsample.spv");

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

    std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {
        vertStage,
        fragStage
    };

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
        .setDepthBoundsTestEnable(VK_FALSE)
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

    vk::DescriptorSetLayout blurLayout =
        getBloomBlurDescriptorSetLayout();

    vk::PipelineLayoutCreateInfo layoutInfo{};
    layoutInfo
        .setSetLayouts(blurLayout)
        .setPushConstantRanges(pushRange);

    bloomDownsamplePipelineLayout =
        vk::raii::PipelineLayout(device, layoutInfo);

    std::array<vk::Format, 1> colorFormats = {
        getHdrFormat()
    };

    vk::StructureChain<
        vk::GraphicsPipelineCreateInfo,
        vk::PipelineRenderingCreateInfo
    > pipelineChain{};

    pipelineChain.get<vk::GraphicsPipelineCreateInfo>()
        .setStages(shaderStages)
        .setPVertexInputState(&vertexInput)
        .setPInputAssemblyState(&inputAssembly)
        .setPViewportState(&viewportState)
        .setPRasterizationState(&rasterizer)
        .setPMultisampleState(&multisampling)
        .setPDepthStencilState(&depthStencil)
        .setPColorBlendState(&colorBlending)
        .setPDynamicState(&dynamicState)
        .setLayout(*bloomDownsamplePipelineLayout)
        .setRenderPass(vk::RenderPass{});

    pipelineChain.get<vk::PipelineRenderingCreateInfo>()
        .setColorAttachmentFormats(colorFormats);

    bloomDownsamplePipeline =
        vk::raii::Pipeline(
            device,
            nullptr,
            pipelineChain.get<vk::GraphicsPipelineCreateInfo>());
}

void PostProcessRenderer::createBloomUpsamplePipeline()
{
    auto& device = vkContext.getDevice();

    vk::raii::ShaderModule vertShaderModule =
        ShaderUtils::createShaderModule(device, "shaders/post_fullscreen.spv");

    vk::raii::ShaderModule fragShaderModule =
        ShaderUtils::createShaderModule(device, "shaders/bloom_upsample.spv");

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

    std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {
        vertStage,
        fragStage
    };

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
        .setDepthBoundsTestEnable(VK_FALSE)
        .setStencilTestEnable(VK_FALSE);

    vk::PipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment
        .setBlendEnable(VK_TRUE)
        .setSrcColorBlendFactor(vk::BlendFactor::eOne)
        .setDstColorBlendFactor(vk::BlendFactor::eOne)
        .setColorBlendOp(vk::BlendOp::eAdd)
        .setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
        .setDstAlphaBlendFactor(vk::BlendFactor::eOne)
        .setAlphaBlendOp(vk::BlendOp::eAdd)
        .setColorWriteMask(
            vk::ColorComponentFlagBits::eR |
            vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB |
            vk::ColorComponentFlagBits::eA);

    vk::PipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending
        .setLogicOpEnable(VK_FALSE)
        .setAttachments(blendAttachment);

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

    vk::DescriptorSetLayout blurLayout =
        getBloomBlurDescriptorSetLayout();

    vk::PipelineLayoutCreateInfo layoutInfo{};
    layoutInfo
        .setSetLayouts(blurLayout)
        .setPushConstantRanges(pushRange);

    bloomUpsamplePipelineLayout =
        vk::raii::PipelineLayout(device, layoutInfo);

    std::array<vk::Format, 1> colorFormats = {
        getHdrFormat()
    };

    vk::StructureChain<
        vk::GraphicsPipelineCreateInfo,
        vk::PipelineRenderingCreateInfo
    > pipelineChain{};

    pipelineChain.get<vk::GraphicsPipelineCreateInfo>()
        .setStages(shaderStages)
        .setPVertexInputState(&vertexInput)
        .setPInputAssemblyState(&inputAssembly)
        .setPViewportState(&viewportState)
        .setPRasterizationState(&rasterizer)
        .setPMultisampleState(&multisampling)
        .setPDepthStencilState(&depthStencil)
        .setPColorBlendState(&colorBlending)
        .setPDynamicState(&dynamicState)
        .setLayout(*bloomUpsamplePipelineLayout)
        .setRenderPass(vk::RenderPass{});

    pipelineChain.get<vk::PipelineRenderingCreateInfo>()
        .setColorAttachmentFormats(colorFormats);

    bloomUpsamplePipeline =
        vk::raii::Pipeline(
            device,
            nullptr,
            pipelineChain.get<vk::GraphicsPipelineCreateInfo>());
}

void PostProcessRenderer::createBloomDownsampleDescriptorSets()
{
    bloomDownsampleDescriptorSets.clear();
      
    vk::DescriptorSetLayout blurLayout =
        getBloomBlurDescriptorSetLayout();

    const uint32_t descriptorCount =
        static_cast<uint32_t>(bloomDownsampleLevels);

    vk::DescriptorPoolSize poolSize{};
    poolSize
        .setType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(descriptorCount);

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo
        .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
        .setMaxSets(descriptorCount)
        .setPoolSizes(poolSize);

    bloomDownsampleDescriptorPool =
        vk::raii::DescriptorPool(vkContext.getDevice(), poolInfo);

    for (uint32_t i = 0; i < bloomDownsampleLevels; ++i)
    {



        vk::DescriptorSetAllocateInfo allocInfo{};
        allocInfo
            .setDescriptorPool(*bloomDownsampleDescriptorPool)
            .setSetLayouts(blurLayout);

        bloomDownsampleDescriptorSets.emplace_back(
            vkContext.getDevice(),
            allocInfo);

        vk::ImageView sourceView =
            (i == 0)
            ?  getBloomBrightView()
            : *bloomDownsampleChain[i - 1].view;

        vk::DescriptorImageInfo inputInfo{};
        inputInfo
            .setSampler(*postProcessSampler)
            .setImageView(sourceView)
            .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

        vk::WriteDescriptorSet write{};
        write
            .setDstSet(*bloomDownsampleDescriptorSets[i][0])
            .setDstBinding(0)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(1)
            .setImageInfo(inputInfo);

        vkContext.getDevice().updateDescriptorSets(write, nullptr);
    }
}

void PostProcessRenderer::createBloomUpsampleDescriptorSets()
{
    bloomUpsampleDescriptorSets.clear();
    bloomUpsampleFinalDescriptorSet.clear();

    const uint32_t chainSets =
        bloomDownsampleLevels > 1 ? bloomDownsampleLevels - 1 : 0;

    const uint32_t totalSets = chainSets + 1; // + final into bloomBright

    vk::DescriptorPoolSize poolSize{};
    poolSize
        .setType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(totalSets);

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo
        .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
        .setMaxSets(totalSets)
        .setPoolSizes(poolSize);

    bloomUpsampleDescriptorPool =
        vk::raii::DescriptorPool(vkContext.getDevice(), poolInfo);


    vk::DescriptorSetLayout layout = getBloomBlurDescriptorSetLayout();

    auto allocateAndWrite = [&](vk::ImageView sourceView) -> vk::raii::DescriptorSets
        {
            vk::DescriptorSetAllocateInfo allocInfo{};
            allocInfo
                .setDescriptorPool(*bloomUpsampleDescriptorPool)
                .setSetLayouts(layout);

            vk::raii::DescriptorSets sets(vkContext.getDevice(), allocInfo);

            vk::DescriptorImageInfo info{};
            info
                .setSampler(*postProcessSampler)
                .setImageView(sourceView)
                .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

            vk::WriteDescriptorSet write{};
            write
                .setDstSet(*sets[0])
                .setDstBinding(0)
                .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
                .setDescriptorCount(1)
                .setImageInfo(info);

            vkContext.getDevice().updateDescriptorSets(write, nullptr);

            return sets;
        };

    // For chain: src = level i, dst = level i - 1.
    // We need descriptor sets for i = 1..N-1.
    for (uint32_t i = 1; i < bloomDownsampleLevels; ++i)
    {
        bloomUpsampleDescriptorSets.push_back(
            allocateAndWrite(*bloomDownsampleChain[i].view));
    }

    // Final: src = level 0, dst = bloomBright.
    if (!bloomDownsampleChain.empty())
    {
        bloomUpsampleFinalDescriptorSet =
            allocateAndWrite(*bloomDownsampleChain[0].view);
    }
}

void PostProcessRenderer::drawBloomDownsample(
    vk::raii::CommandBuffer& commandBuffer,
    vk::ImageView outputView,
    vk::DescriptorSet inputSet,
    uint32_t outputWidth,
    uint32_t outputHeight,
    uint32_t inputWidth,
    uint32_t inputHeight)
{
    vk::ClearValue clearColor = vk::ClearColorValue(
        std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f });

    vk::RenderingAttachmentInfo colorAttachment{};
    colorAttachment
        .setImageView(outputView)
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(clearColor);

    vk::RenderingInfo renderingInfo{};
    renderingInfo
        .setRenderArea(vk::Rect2D{
            vk::Offset2D{0, 0},
            vk::Extent2D{outputWidth, outputHeight}
            })
        .setLayerCount(1)
        .setColorAttachments(colorAttachment);

    commandBuffer.beginRendering(renderingInfo);

    commandBuffer.setViewport(
        0,
        vk::Viewport(
            0.0f,
            0.0f,
            static_cast<float>(outputWidth),
            static_cast<float>(outputHeight),
            0.0f,
            1.0f));

    commandBuffer.setScissor(
        0,
        vk::Rect2D(
            vk::Offset2D{ 0, 0 },
            vk::Extent2D{ outputWidth, outputHeight }));

    commandBuffer.bindPipeline(
        vk::PipelineBindPoint::eGraphics,
        *bloomDownsamplePipeline);

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *bloomDownsamplePipelineLayout,
        0,
        inputSet,
        {});

    glm::vec4 params(
        1.0f / static_cast<float>(inputWidth),
        1.0f / static_cast<float>(inputHeight),
        0.0f,
        0.0f);

    commandBuffer.pushConstants<glm::vec4>(
        *bloomDownsamplePipelineLayout,
        vk::ShaderStageFlagBits::eFragment,
        0,
        params);

    commandBuffer.draw(3, 1, 0, 0);

    commandBuffer.endRendering();
}

void PostProcessRenderer::drawBloomUpsample(
    vk::raii::CommandBuffer& commandBuffer,
    vk::ImageView outputView,
    vk::DescriptorSet inputSet,
    uint32_t outputWidth,
    uint32_t outputHeight,
    uint32_t inputWidth,
    uint32_t inputHeight)
{
    vk::RenderingAttachmentInfo colorAttachment{};
    colorAttachment
        .setImageView(outputView)
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eLoad) // important: additive into existing image
        .setStoreOp(vk::AttachmentStoreOp::eStore);

    vk::RenderingInfo renderingInfo{};
    renderingInfo
        .setRenderArea(vk::Rect2D{
            vk::Offset2D{0, 0},
            vk::Extent2D{outputWidth, outputHeight}
            })
        .setLayerCount(1)
        .setColorAttachments(colorAttachment);

    commandBuffer.beginRendering(renderingInfo);

    commandBuffer.setViewport(
        0,
        vk::Viewport(
            0.0f,
            0.0f,
            static_cast<float>(outputWidth),
            static_cast<float>(outputHeight),
            0.0f,
            1.0f));

    commandBuffer.setScissor(
        0,
        vk::Rect2D(
            vk::Offset2D{ 0, 0 },
            vk::Extent2D{ outputWidth, outputHeight }));

    commandBuffer.bindPipeline(
        vk::PipelineBindPoint::eGraphics,
        *bloomUpsamplePipeline);

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *bloomUpsamplePipelineLayout,
        0,
        inputSet,
        {});

    /* glm::vec4 params(
        1.0f, // filterRadius / contribution scale
        0.0f,
        0.0f,
        0.0f);
        */


    glm::vec4 params(
        ppsettings.bloomIntensity,
        ppsettings.bloomUpsampleRadius,
        1.0f / static_cast<float>(inputWidth),
        1.0f / static_cast<float>(inputHeight));

    commandBuffer.pushConstants<glm::vec4>(
        *bloomUpsamplePipelineLayout,
        vk::ShaderStageFlagBits::eFragment,
        0,
        params);

    commandBuffer.draw(3, 1, 0, 0);

    commandBuffer.endRendering();
}

void PostProcessRenderer::recordBloomPyramid(
    vk::raii::CommandBuffer& commandBuffer)
{
    vk::CommandBuffer cmd = *commandBuffer;

    for (uint32_t i = 0; i < bloomDownsampleLevels; ++i)
    {
        BloomMipResource& dst = bloomDownsampleChain[i];

        TransitionUtils::transitionToColorAttachment(cmd, *dst.image, vk::ImageLayout::eUndefined);

        uint32_t inputWidth =
            (i == 0) ? extent.width : bloomDownsampleChain[i - 1].width;

        uint32_t inputHeight =
            (i == 0) ? extent.height : bloomDownsampleChain[i - 1].height;

        drawBloomDownsample(
            commandBuffer,
            *dst.view,
            *bloomDownsampleDescriptorSets[i][0],
            dst.width,
            dst.height,
            inputWidth,
            inputHeight);

        TransitionUtils::transitionToShaderReadOnly(cmd, *dst.image, vk::ImageLayout::eColorAttachmentOptimal);
    }

    if (bloomDownsampleLevels > 1)
    {
        for (int i = static_cast<int>(bloomDownsampleLevels) - 1; i > 0; --i)
        {
            BloomMipResource& src = bloomDownsampleChain[i];
            BloomMipResource& dst = bloomDownsampleChain[i - 1];

            TransitionUtils::transitionToColorAttachment(cmd, *dst.image, vk::ImageLayout::eShaderReadOnlyOptimal);

            const uint32_t descriptorIndex = static_cast<uint32_t>(i - 1);

            drawBloomUpsample(
                commandBuffer,
                *dst.view,
                *bloomUpsampleDescriptorSets[descriptorIndex][0],
                dst.width,
                dst.height,
                src.width,
                src.height);

            TransitionUtils::transitionToShaderReadOnly(cmd, *dst.image, vk::ImageLayout::eColorAttachmentOptimal);
                
        }
    }

    if (!bloomDownsampleChain.empty())
    {
        TransitionUtils::transitionToColorAttachment(cmd, *bloomBrightImage, vk::ImageLayout::eShaderReadOnlyOptimal);

        drawBloomUpsample(
            commandBuffer,
            *bloomBrightView,
            *bloomUpsampleFinalDescriptorSet[0],
            extent.width,
            extent.height,
            bloomDownsampleChain[0].width,
            bloomDownsampleChain[0].height);

        TransitionUtils::transitionToShaderReadOnly(cmd, *bloomBrightImage, vk::ImageLayout::eColorAttachmentOptimal);
          
    }
}


void PostProcessRenderer::createPostProcessPipeline()
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
        .setColorAttachmentFormats(swapchainFormat);

    postProcessPipeline =
        vk::raii::Pipeline(
            device,
            nullptr,
            pipelineChain.get<vk::GraphicsPipelineCreateInfo>());
}

void PostProcessRenderer::createPostProcessDescriptorSet()
{
    auto& device = vkContext.getDevice();

    vk::DescriptorPoolSize poolSize{};
    poolSize
        .setType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(2);

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
        .setImageView(getHdrView())
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

    vk::DescriptorImageInfo bloomInfo{};
    bloomInfo
        .setSampler(*postProcessSampler)
        .setImageView(getBloomBrightView())
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

    std::array<vk::WriteDescriptorSet, 2> writes{};

    writes[0]
        .setDstSet(*postProcessDescriptorSets[0])
        .setDstBinding(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(hdrInfo);

    writes[1]
        .setDstSet(*postProcessDescriptorSets[0])
        .setDstBinding(1)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(bloomInfo);



    device.updateDescriptorSets(writes, nullptr);
}

void PostProcessRenderer::createPostProcessDescriptorSetLayout()
{
    auto& device = vkContext.getDevice();

    // HDR scene (input)
    vk::DescriptorSetLayoutBinding hdrBinding{};
    hdrBinding
        .setBinding(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eFragment);

    // Bloom texture (input)
    vk::DescriptorSetLayoutBinding bloomBinding{};
    bloomBinding
        .setBinding(1)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eFragment);

    std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {
        hdrBinding,
        bloomBinding
    };

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.setBindings(bindings);

    postProcessDescriptorSetLayout =
        vk::raii::DescriptorSetLayout(device, layoutInfo);
}

void PostProcessRenderer::recordFinalComposite(
    vk::raii::CommandBuffer& commandBuffer)
{
    const glm::vec4 postParams = buildFinalCompositeParams();

    commandBuffer.bindPipeline(
        vk::PipelineBindPoint::eGraphics,
        *postProcessPipeline);

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *postProcessPipelineLayout,
        0,
        *postProcessDescriptorSets[0],
        {});

    commandBuffer.pushConstants<glm::vec4>(
        *postProcessPipelineLayout,
        vk::ShaderStageFlagBits::eFragment,
        0,
        postParams);

    commandBuffer.draw(3, 1, 0, 0);
}

void PostProcessRenderer::createPostProcessSampler()
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

void PostProcessRenderer::beginFinalCompositePass(
    vk::raii::CommandBuffer& commandBuffer,
    vk::ImageView swapchainImageView)
{
    vk::ClearValue clearColor = vk::ClearColorValue(
        std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f });

    vk::RenderingAttachmentInfo colorAttachment{};
    colorAttachment
        .setImageView(swapchainImageView)
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
}

void PostProcessRenderer::endFinalCompositePass(
    vk::raii::CommandBuffer& commandBuffer)
{
    commandBuffer.endRendering();
}



void PostProcessRenderer::executeBloomChain(
    vk::raii::CommandBuffer& commandBuffer)
{
    vk::CommandBuffer cmd = *commandBuffer;

    TransitionUtils::transitionToShaderReadOnly(
        cmd,
        getHdrImage(),
        vk::ImageLayout::eColorAttachmentOptimal);

    TransitionUtils::transitionToColorAttachment(
        cmd,
        getBloomBrightImage(),
        vk::ImageLayout::eUndefined);

    recordBloomExtract(commandBuffer);

    TransitionUtils::transitionToShaderReadOnly(
        cmd,
        getBloomBrightImage(),
        vk::ImageLayout::eColorAttachmentOptimal);

    TransitionUtils::transitionToColorAttachment(
        cmd,
        getBloomBlurTempImage(),
        vk::ImageLayout::eUndefined);

    recordBloomBlurFromBright(
        commandBuffer,
        getBloomBlurTempView(),
        glm::vec2(1.0f, 0.0f));

    TransitionUtils::transitionToShaderReadOnly(
        cmd,
        getBloomBlurTempImage(),
        vk::ImageLayout::eColorAttachmentOptimal);

    TransitionUtils::transitionToColorAttachment(
        cmd,
        getBloomBrightImage(),
        vk::ImageLayout::eShaderReadOnlyOptimal);

    recordBloomBlurFromTemp(
        commandBuffer,
        getBloomBrightView(),
        glm::vec2(0.0f, 1.0f));

    TransitionUtils::transitionToShaderReadOnly(
        cmd,
        getBloomBrightImage(),
        vk::ImageLayout::eColorAttachmentOptimal);

    recordBloomPyramid(commandBuffer);
}

void PostProcessRenderer::executeFinalComposite(
    vk::raii::CommandBuffer& commandBuffer,
    vk::Image swapchainImage,
    vk::ImageView swapchainImageView,
    vk::ImageLayout oldLayout)
{
    vk::CommandBuffer cmd = *commandBuffer;

    TransitionUtils::transitionToColorAttachment(
        cmd,
        swapchainImage,
        oldLayout);

    beginFinalCompositePass(
        commandBuffer,
        swapchainImageView);

    recordFinalComposite(commandBuffer);

    endFinalCompositePass(commandBuffer);

    TransitionUtils::transitionToPresent(
        cmd,
        swapchainImage);
}



glm::vec4 PostProcessRenderer::buildFinalCompositeParams() const
{
    return glm::vec4(
        ppsettings.exposure,
        ppsettings.toneMappingEnabled ? 1.0f : 0.0f,
        ppsettings.gammaEnabled ? 1.0f : 0.0f,
        ppsettings.bloomEnabled ? ppsettings.bloomStrength : 0.0f);
}













