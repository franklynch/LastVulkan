#include "IrradianceRenderer.hpp"
#include "EnvironmentUtils.hpp"
#include "ShaderUtils.hpp"


#include <glm/gtc/matrix_transform.hpp>


#include <iostream>
#include <stdexcept>




struct EquirectToCubePushConstants
{
    glm::mat4 viewProj{ 1.0f };
};

IrradianceRenderer::IrradianceRenderer(
    VulkanContext& vkContext,
    BufferUtils& bufferUtils)
    : vkContext(vkContext)
    , bufferUtils(bufferUtils)
{
}

void IrradianceRenderer::init(EnvironmentResources& environment)
{
    createResources(environment);
    createFaceViews(environment);
    createDescriptorResources();
    updateDescriptorSet(environment);
    createPipeline();
    render(environment);
}

void IrradianceRenderer::createResources(EnvironmentResources& environment)
{
    createCubemapResource(
        vkContext,
        bufferUtils,
        environment.runtimeIrradianceCube,
        runtimeIrradianceCubeSize,
        1,
        vk::Format::eR16G16B16A16Sfloat,
        vk::ImageUsageFlagBits::eColorAttachment |
        vk::ImageUsageFlagBits::eSampled);

    std::cout << "Prefilter source = runtime environment cube\n";

    std::cout << "Created runtime irradiance cubemap: "
        << runtimeIrradianceCubeSize << "x"
        << runtimeIrradianceCubeSize << "\n";
}

void IrradianceRenderer::createFaceViews(EnvironmentResources& environment)
{
    if (environment.runtimeIrradianceCube.image == nullptr)
    {
        throw std::runtime_error(
            "createRuntimeIrradianceCubemapFaceViews: runtimeIrradianceCubeImage is null");
    }

    auto& device = vkContext.getDevice();

    const vk::Format cubeFormat = vk::Format::eR16G16B16A16Sfloat;

    for (uint32_t face = 0; face < 6; ++face)
    {
        vk::ImageViewCreateInfo viewInfo{};
        viewInfo
            .setImage(*environment.runtimeIrradianceCube.image)
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(cubeFormat)
            .setSubresourceRange(
                vk::ImageSubresourceRange{}
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setBaseMipLevel(0)
                .setLevelCount(1)
                .setBaseArrayLayer(face)
                .setLayerCount(1));

        environment.runtimeIrradianceCubeFaces.views[face] =
            vk::raii::ImageView(device, viewInfo);
    }

    std::cout << "Created runtime irradiance cubemap face views\n";
}

void IrradianceRenderer::createDescriptorResources()
{
    auto& device = vkContext.getDevice();

    vk::DescriptorSetLayoutBinding envBinding{};
    envBinding
        .setBinding(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eFragment);

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.setBindings(envBinding);

    irradianceDescriptorSetLayout =
        vk::raii::DescriptorSetLayout(device, layoutInfo);

    vk::DescriptorPoolSize poolSize{};
    poolSize
        .setType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1);

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo
        .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
        .setMaxSets(1)
        .setPoolSizes(poolSize);

    irradianceDescriptorPool =
        vk::raii::DescriptorPool(device, poolInfo);

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo
        .setDescriptorPool(*irradianceDescriptorPool)
        .setSetLayouts(*irradianceDescriptorSetLayout);

    auto sets = vk::raii::DescriptorSets(device, allocInfo);
    irradianceDescriptorSet = std::move(sets.front());
}

void IrradianceRenderer::updateDescriptorSet(EnvironmentResources& environment)
{
    if (environment.runtimeEnvironmentCube.sampler == nullptr ||
        environment.runtimeEnvironmentCube.view == nullptr)
    {
        throw std::runtime_error(
            "updateIrradianceDescriptorSet: runtime environment cubemap is not initialized");
    }

    vk::DescriptorImageInfo envInfo{};
    envInfo
        .setSampler(*environment.runtimeEnvironmentCube.sampler)
        .setImageView(*environment.runtimeEnvironmentCube.view)
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

    vk::WriteDescriptorSet write{};
    write
        .setDstSet(*irradianceDescriptorSet)
        .setDstBinding(0)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(envInfo);

    vkContext.getDevice().updateDescriptorSets(write, {});
}

void IrradianceRenderer::createPipeline()
{
    auto& device = vkContext.getDevice();

    vk::raii::ShaderModule vertShaderModule =
        ShaderUtils::createShaderModule(vkContext.getDevice(), "shaders/irradiance_vert.spv");

    vk::raii::ShaderModule fragShaderModule =
        ShaderUtils::createShaderModule(vkContext.getDevice(), "shaders/irradiance_frag.spv");

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

    std::array<vk::PipelineShaderStageCreateInfo, 2> stages = {
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

    vk::PipelineColorBlendAttachmentState colorAttachment{};
    colorAttachment
        .setBlendEnable(VK_FALSE)
        .setColorWriteMask(
            vk::ColorComponentFlagBits::eR |
            vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB |
            vk::ColorComponentFlagBits::eA);

    vk::PipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending
        .setLogicOpEnable(VK_FALSE)
        .setAttachments(colorAttachment);

    std::vector<vk::DynamicState> dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor
    };

    vk::PipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.setDynamicStates(dynamicStates);

    vk::PushConstantRange pushRange{};
    pushRange
        .setStageFlags(vk::ShaderStageFlagBits::eVertex)
        .setOffset(0)
        .setSize(sizeof(EquirectToCubePushConstants));

    std::array<vk::DescriptorSetLayout, 1> setLayouts = {
        *irradianceDescriptorSetLayout
    };

    vk::PipelineLayoutCreateInfo layoutInfo{};
    layoutInfo
        .setSetLayouts(setLayouts)
        .setPushConstantRanges(pushRange);

    irradiancePipelineLayout =
        vk::raii::PipelineLayout(device, layoutInfo);

    std::array<vk::Format, 1> colorFormats = {
        vk::Format::eR16G16B16A16Sfloat
    };

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
        .setLayout(*irradiancePipelineLayout)
        .setRenderPass(vk::RenderPass{});

    pipelineChain.get<vk::PipelineRenderingCreateInfo>()
        .setColorAttachmentFormats(colorFormats);

    irradiancePipeline =
        vk::raii::Pipeline(
            device,
            nullptr,
            pipelineChain.get<vk::GraphicsPipelineCreateInfo>());

    std::cout << "Created irradiance pipeline\n";
    std::cout << "irradiancePipelineLayout: " << (irradiancePipelineLayout != nullptr) << "\n";
    std::cout << "irradiancePipeline: " << (irradiancePipeline != nullptr) << "\n";
}

void IrradianceRenderer::render(EnvironmentResources& environment)
{
    if (environment.runtimeIrradianceCube.image == nullptr ||
        irradiancePipeline == nullptr ||
        irradiancePipelineLayout == nullptr)
    {
        throw std::runtime_error("renderRuntimeIrradianceCubemap: resources not initialized");
    }

    std::cout << "runtimeIrradianceCubeImage: " << (environment.runtimeIrradianceCube.image != nullptr) << "\n";
    std::cout << "irradiancePipelineLayout: " << (irradiancePipelineLayout != nullptr) << "\n";
    std::cout << "irradiancePipeline: " << (irradiancePipeline != nullptr) << "\n";

    const uint32_t cubeSize = runtimeIrradianceCubeSize;

    auto captureProjection = getCubemapCaptureProjection();
    auto captureViews = getCubemapCaptureViews();

    auto cmd = bufferUtils.beginSingleTimeCommands();

    vk::ImageMemoryBarrier toColorAttachment{};
    toColorAttachment
        .setOldLayout(vk::ImageLayout::eUndefined)
        .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*environment.runtimeIrradianceCube.image)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(6))
        .setSrcAccessMask({})
        .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        {},
        nullptr,
        nullptr,
        toColorAttachment);

    for (uint32_t face = 0; face < 6; ++face)
    {
        vk::ClearValue clearValue{};
        clearValue.setColor(
            vk::ClearColorValue(std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f }));

        vk::RenderingAttachmentInfo colorAttachment{};
        colorAttachment
            .setImageView(*environment.runtimeIrradianceCubeFaces.views[face])
            .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setClearValue(clearValue);

        vk::RenderingInfo renderingInfo{};
        renderingInfo
            .setRenderArea(
                vk::Rect2D{}
                .setOffset(vk::Offset2D{ 0, 0 })
                .setExtent(vk::Extent2D{ cubeSize, cubeSize }))
            .setLayerCount(1)
            .setColorAttachments(colorAttachment);

        cmd.beginRendering(renderingInfo);

        cmd.setViewport(
            0,
            vk::Viewport(
                0.0f,
                0.0f,
                static_cast<float>(cubeSize),
                static_cast<float>(cubeSize),
                0.0f,
                1.0f));

        cmd.setScissor(
            0,
            vk::Rect2D(
                vk::Offset2D{ 0, 0 },
                vk::Extent2D{ cubeSize, cubeSize }));

        cmd.bindPipeline(
            vk::PipelineBindPoint::eGraphics,
            *irradiancePipeline);

        cmd.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            *irradiancePipelineLayout,
            0,
            *irradianceDescriptorSet,
            {});

        EquirectToCubePushConstants push{};
        push.viewProj = captureProjection * captureViews[face];

        cmd.pushConstants<EquirectToCubePushConstants>(
            *irradiancePipelineLayout,
            vk::ShaderStageFlagBits::eVertex,
            0,
            push);

        cmd.draw(36, 1, 0, 0);

        cmd.endRendering();
    }

    vk::ImageMemoryBarrier toShaderRead{};
    toShaderRead
        .setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*environment.runtimeIrradianceCube.image)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(6))
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

    std::cout << "Rendered runtime irradiance cubemap\n";
}

