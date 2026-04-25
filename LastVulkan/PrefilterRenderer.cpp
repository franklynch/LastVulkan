#include "PrefilterRenderer.hpp"
#include "EnvironmentUtils.hpp"
#include "ShaderUtils.hpp"



#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <stdexcept>

struct PrefilterPushConstants
{
    glm::mat4 viewProj;
    glm::vec4 params;
};

PrefilterRenderer::PrefilterRenderer(
    VulkanContext& vkContext,
    BufferUtils& bufferUtils)
    : vkContext(vkContext)
    , bufferUtils(bufferUtils)
{
}

void PrefilterRenderer::init(EnvironmentResources& environment)
{
    createResources(environment);
    createFaceViews(environment);

    createDescriptorResources();
    updateDescriptorSet(environment);
    createPipeline();
    render(environment);
}

void PrefilterRenderer::createResources(EnvironmentResources& environment)
{
    createCubemapResource(
        vkContext,
        bufferUtils,
        environment.runtimePrefilteredCube,
        runtimePrefilteredCubeSize,
        runtimePrefilteredMipLevels,
        vk::Format::eR16G16B16A16Sfloat,
        vk::ImageUsageFlagBits::eColorAttachment |
        vk::ImageUsageFlagBits::eSampled);

    std::cout << "Created runtime prefiltered cubemap: "
        << runtimePrefilteredCubeSize << "x"
        << runtimePrefilteredCubeSize
        << " mips=" << runtimePrefilteredMipLevels << "\n";
}

void PrefilterRenderer::createFaceViews(EnvironmentResources& environment)
{
    if (environment.runtimePrefilteredCube.image == nullptr)
    {
        throw std::runtime_error(
            "createRuntimePrefilteredCubemapFaceViews: runtimePrefilteredCubeImage is null");
    }

    auto& device = vkContext.getDevice();

    const vk::Format cubeFormat = vk::Format::eR16G16B16A16Sfloat;

    environment.runtimePrefilteredCubeMipFaceViews.clear();
    environment.runtimePrefilteredCubeMipFaceViews.resize(runtimePrefilteredMipLevels);

    for (uint32_t mip = 0; mip < runtimePrefilteredMipLevels; ++mip)
    {
        for (uint32_t face = 0; face < 6; ++face)
        {
            vk::ImageViewCreateInfo viewInfo{};
            viewInfo
                .setImage(*environment.runtimePrefilteredCube.image)
                .setViewType(vk::ImageViewType::e2D)
                .setFormat(cubeFormat)
                .setSubresourceRange(
                    vk::ImageSubresourceRange{}
                    .setAspectMask(vk::ImageAspectFlagBits::eColor)
                    .setBaseMipLevel(mip)
                    .setLevelCount(1)
                    .setBaseArrayLayer(face)
                    .setLayerCount(1));

            environment.runtimePrefilteredCubeMipFaceViews[mip].views[face] =
                vk::raii::ImageView(device, viewInfo);
        }
    }

    std::cout << "Created runtime prefiltered cubemap mip/face views\n";
}

void PrefilterRenderer::createDescriptorResources()
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

    prefilterDescriptorSetLayout =
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

    prefilterDescriptorPool =
        vk::raii::DescriptorPool(device, poolInfo);

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo
        .setDescriptorPool(*prefilterDescriptorPool)
        .setSetLayouts(*prefilterDescriptorSetLayout);

    auto sets = vk::raii::DescriptorSets(device, allocInfo);
    prefilterDescriptorSet = std::move(sets.front());
}

void PrefilterRenderer::updateDescriptorSet(EnvironmentResources& environment)
{
    if (environment.runtimeEnvironmentCube.sampler == nullptr ||
        environment.runtimeEnvironmentCube.view == nullptr)
    {
        throw std::runtime_error(
            "updatePrefilterDescriptorSet: runtime environment cubemap is not initialized");
    }

    vk::DescriptorImageInfo envInfo{};
    envInfo
        .setSampler(*environment.runtimeEnvironmentCube.sampler)
        .setImageView(*environment.runtimeEnvironmentCube.view)
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

    vk::WriteDescriptorSet write{};
    write
        .setDstSet(*prefilterDescriptorSet)
        .setDstBinding(0)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(envInfo);

    vkContext.getDevice().updateDescriptorSets(write, {});
}

void PrefilterRenderer::createPipeline()
{
    auto& device = vkContext.getDevice();

    vk::raii::ShaderModule vertShaderModule =
        createShaderModule(vkContext.getDevice(), readFile("shaders/prefilter_vert.spv"));

    vk::raii::ShaderModule fragShaderModule =
        createShaderModule(vkContext.getDevice(), readFile("shaders/prefilter_frag.spv"));

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
        .setStageFlags(
            vk::ShaderStageFlagBits::eVertex |
            vk::ShaderStageFlagBits::eFragment)
        .setOffset(0)
        .setSize(sizeof(PrefilterPushConstants));

    std::array<vk::DescriptorSetLayout, 1> setLayouts = {
        *prefilterDescriptorSetLayout
    };

    vk::PipelineLayoutCreateInfo layoutInfo{};
    layoutInfo
        .setSetLayouts(setLayouts)
        .setPushConstantRanges(pushRange);

    prefilterPipelineLayout =
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
        .setLayout(*prefilterPipelineLayout)
        .setRenderPass(vk::RenderPass{});

    pipelineChain.get<vk::PipelineRenderingCreateInfo>()
        .setColorAttachmentFormats(colorFormats);

    prefilterPipeline =
        vk::raii::Pipeline(
            device,
            nullptr,
            pipelineChain.get<vk::GraphicsPipelineCreateInfo>());

}

void PrefilterRenderer::render(EnvironmentResources& environment)
{



    if (environment.runtimePrefilteredCube.image == nullptr ||
        prefilterPipeline == nullptr ||
        prefilterPipelineLayout == nullptr)
    {
        throw std::runtime_error("renderRuntimePrefilteredCubemap: resources not initialized");
    }

    if (environment.runtimePrefilteredCubeMipFaceViews.size() < runtimePrefilteredMipLevels)
    {
        throw std::runtime_error("renderRuntimePrefilteredCubemap: mip face views not initialized");
    }

    const uint32_t baseSize = runtimePrefilteredCubeSize;
    const uint32_t mipLevels = runtimePrefilteredMipLevels;

    auto captureProjection = getCubemapCaptureProjection();
    auto captureViews = getCubemapCaptureViews();

    auto cmd = bufferUtils.beginSingleTimeCommands();

    vk::ImageMemoryBarrier toColorAttachment{};
    toColorAttachment
        .setOldLayout(vk::ImageLayout::eUndefined)
        .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*environment.runtimePrefilteredCube.image)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(mipLevels)
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

    cmd.bindPipeline(
        vk::PipelineBindPoint::eGraphics,
        *prefilterPipeline);

    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *prefilterPipelineLayout,
        0,
        *prefilterDescriptorSet,
        {});

    for (uint32_t mip = 0; mip < mipLevels; ++mip)
    {
        const uint32_t mipSize = std::max(1u, baseSize >> mip);

        const float roughness =
            static_cast<float>(mip) /
            static_cast<float>(mipLevels - 1);

        for (uint32_t face = 0; face < 6; ++face)
        {
            vk::ClearValue clearValue{};
            clearValue.setColor(
                vk::ClearColorValue(std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f }));

            vk::RenderingAttachmentInfo colorAttachment{};
            colorAttachment
                .setImageView(*environment.runtimePrefilteredCubeMipFaceViews[mip].views[face])
                .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
                .setLoadOp(vk::AttachmentLoadOp::eClear)
                .setStoreOp(vk::AttachmentStoreOp::eStore)
                .setClearValue(clearValue);

            vk::RenderingInfo renderingInfo{};
            renderingInfo
                .setRenderArea(
                    vk::Rect2D{}
                    .setOffset(vk::Offset2D{ 0, 0 })
                    .setExtent(vk::Extent2D{ mipSize, mipSize }))
                .setLayerCount(1)
                .setColorAttachments(colorAttachment);

            cmd.beginRendering(renderingInfo);

            cmd.setViewport(
                0,
                vk::Viewport(
                    0.0f,
                    0.0f,
                    static_cast<float>(mipSize),
                    static_cast<float>(mipSize),
                    0.0f,
                    1.0f));

            cmd.setScissor(
                0,
                vk::Rect2D(
                    vk::Offset2D{ 0, 0 },
                    vk::Extent2D{ mipSize, mipSize }));

            PrefilterPushConstants push{};
            push.viewProj = captureProjection * captureViews[face];
            push.params = glm::vec4(roughness, 0.0f, 0.0f, 0.0f);

            cmd.pushConstants<PrefilterPushConstants>(
                *prefilterPipelineLayout,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                0,
                push);

            cmd.draw(36, 1, 0, 0);

            cmd.endRendering();
        }
    }

    vk::ImageMemoryBarrier toShaderRead{};
    toShaderRead
        .setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*environment.runtimePrefilteredCube.image)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(mipLevels)
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

    std::cout << "Rendered runtime prefiltered cubemap / PMREM\n";
}


