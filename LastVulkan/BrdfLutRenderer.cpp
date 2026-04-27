#include "BrdfLutRenderer.hpp"
#include "ShaderUtils.hpp"

BrdfLutRenderer::BrdfLutRenderer(
    VulkanContext& vkContext,
    BufferUtils& bufferUtils)
    : vkContext(vkContext), bufferUtils(bufferUtils)
{
}

void BrdfLutRenderer::init(EnvironmentResources& environment)
{
    createResources(environment);
    createPipeline(environment);
    render(environment);
}

void BrdfLutRenderer::createResources(EnvironmentResources& environment)
{

    auto& device = vkContext.getDevice();

    constexpr uint32_t lutWidth = 256;
    constexpr uint32_t lutHeight = 256;
    constexpr vk::Format lutFormat = vk::Format::eR16G16Sfloat;

    vk::ImageCreateInfo imageInfo{};
    imageInfo
        .setImageType(vk::ImageType::e2D)
        .setFormat(lutFormat)
        .setExtent(vk::Extent3D{ lutWidth, lutHeight, 1 })
        .setMipLevels(1)
        .setArrayLayers(1)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setTiling(vk::ImageTiling::eOptimal)
        .setUsage(
            vk::ImageUsageFlagBits::eColorAttachment |
            vk::ImageUsageFlagBits::eSampled)
        .setSharingMode(vk::SharingMode::eExclusive)
        .setInitialLayout(vk::ImageLayout::eUndefined);

    environment.runtimeBrdfLut.image = vk::raii::Image(device, imageInfo);

    vk::MemoryRequirements memRequirements = environment.runtimeBrdfLut.image.getMemoryRequirements();

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo
        .setAllocationSize(memRequirements.size)
        .setMemoryTypeIndex(
            bufferUtils.findMemoryType(
                memRequirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal));

    environment.runtimeBrdfLut.memory = vk::raii::DeviceMemory(device, allocInfo);
    environment.runtimeBrdfLut.image.bindMemory(*environment.runtimeBrdfLut.memory, 0);

    vk::ImageViewCreateInfo viewInfo{};
    viewInfo
        .setImage(*environment.runtimeBrdfLut.image)
        .setViewType(vk::ImageViewType::e2D)
        .setFormat(lutFormat)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(1));

    environment.runtimeBrdfLut.view = vk::raii::ImageView(device, viewInfo);

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
        .setMaxLod(0.0f)
        .setBorderColor(vk::BorderColor::eFloatOpaqueWhite)
        .setUnnormalizedCoordinates(VK_FALSE);

    environment.runtimeBrdfLut.sampler = vk::raii::Sampler(device, samplerInfo);

}

void BrdfLutRenderer::createPipeline(EnvironmentResources& environment)
{

    auto& device = vkContext.getDevice();

    vk::raii::ShaderModule vertShaderModule =
        ShaderUtils::createShaderModule(vkContext.getDevice(), "shaders/brdf_lut_vert.spv");

    vk::raii::ShaderModule fragShaderModule =
        ShaderUtils::createShaderModule(vkContext.getDevice(), "shaders/brdf_lut_frag.spv");

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

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo
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
            vk::ColorComponentFlagBits::eG
        );

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

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    environment.runtimeBrdfLut.layout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

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
        .setLayout(*environment.runtimeBrdfLut.layout)
        .setRenderPass(vk::RenderPass{});

    std::array<vk::Format, 1> brdfColorFormats = {
    vk::Format::eR16G16Sfloat
    };

    pipelineCreateInfoChain.get<vk::PipelineRenderingCreateInfo>()
        .setColorAttachmentFormats(brdfColorFormats);

    environment.runtimeBrdfLut.pipeline = vk::raii::Pipeline(
        device,
        nullptr,
        pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>()
    );

}

void BrdfLutRenderer::render(EnvironmentResources& environment)
{

    if (environment.runtimeBrdfLut.image == nullptr ||
        environment.runtimeBrdfLut.view == nullptr ||
        environment.runtimeBrdfLut.pipeline == nullptr ||
        environment.runtimeBrdfLut.layout == nullptr)
    {
        throw std::runtime_error("renderBrdfLut: resources or pipeline not initialized");
    }

    constexpr uint32_t lutWidth = 256;
    constexpr uint32_t lutHeight = 256;

    auto cmd = bufferUtils.beginSingleTimeCommands();

    // Undefined -> Color Attachment
    vk::ImageMemoryBarrier toColorAttachment{};
    toColorAttachment
        .setOldLayout(vk::ImageLayout::eUndefined)
        .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*environment.runtimeBrdfLut.image)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(1))
        .setSrcAccessMask({})
        .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        {},
        nullptr,
        nullptr,
        toColorAttachment);

    vk::ClearValue clearValue{};
    clearValue.setColor(vk::ClearColorValue(std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f }));

    vk::RenderingAttachmentInfo colorAttachment{};
    colorAttachment
        .setImageView(*environment.runtimeBrdfLut.view)
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(clearValue);

    vk::RenderingInfo renderingInfo{};
    renderingInfo
        .setRenderArea(
            vk::Rect2D{}
            .setOffset(vk::Offset2D{ 0, 0 })
            .setExtent(vk::Extent2D{ lutWidth, lutHeight }))
        .setLayerCount(1)
        .setColorAttachments(colorAttachment);

    cmd.beginRendering(renderingInfo);

    cmd.setViewport(
        0,
        vk::Viewport(
            0.0f,
            0.0f,
            static_cast<float>(lutWidth),
            static_cast<float>(lutHeight),
            0.0f,
            1.0f));

    cmd.setScissor(
        0,
        vk::Rect2D(
            vk::Offset2D{ 0, 0 },
            vk::Extent2D{ lutWidth, lutHeight }));

    cmd.bindPipeline(
        vk::PipelineBindPoint::eGraphics,
        *environment.runtimeBrdfLut.pipeline);

    cmd.draw(3, 1, 0, 0);

    cmd.endRendering();

    // Color Attachment -> Shader Read
    vk::ImageMemoryBarrier toShaderRead{};
    toShaderRead
        .setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*environment.runtimeBrdfLut.image)
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
        toShaderRead);

    bufferUtils.endSingleTimeCommands(cmd);


}