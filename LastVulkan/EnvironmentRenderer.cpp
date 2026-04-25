#include "EnvironmentRenderer.hpp"
#include "EnvironmentUtils.hpp"
#include "ShaderUtils.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <array>
#include <iostream>
#include <stdexcept>

struct EquirectToCubePushConstants
{
    glm::mat4 viewProj;
};

EnvironmentRenderer::EnvironmentRenderer(
    VulkanContext& vkContext,
    BufferUtils& bufferUtils)
    : vkContext(vkContext), bufferUtils(bufferUtils)
{
}

void EnvironmentRenderer::init(
    EnvironmentResources& environment,
    const vk::raii::Sampler& hdrSampler,
    const vk::raii::ImageView& hdrView)
{
    createResources(environment);
    createFaceViews(environment);
    createDescriptorResources();
    updateDescriptorSet(environment, hdrSampler, hdrView);
    createPipeline();
    render(environment);
}

void EnvironmentRenderer::createResources(EnvironmentResources& environment)
{

    createCubemapResource(
        vkContext,
        bufferUtils,
        environment.runtimeEnvironmentCube,
        runtimeEnvironmentCubeSize,
        1,
        vk::Format::eR16G16B16A16Sfloat,
        vk::ImageUsageFlagBits::eColorAttachment |
        vk::ImageUsageFlagBits::eSampled);

    std::cout << "Created runtime environment cubemap: "
        << runtimeEnvironmentCubeSize << "x"
        << runtimeEnvironmentCubeSize << "\n";

}

void EnvironmentRenderer::createFaceViews(EnvironmentResources& environment)
{
    if (environment.runtimeEnvironmentCube.image == nullptr)
    {
        throw std::runtime_error(
            "createRuntimeEnvironmentCubemapFaceViews: environment.runtimeEnvironmentCube.image is null");
    }

    auto& device = vkContext.getDevice();

    const vk::Format cubeFormat = vk::Format::eR16G16B16A16Sfloat;

    for (uint32_t face = 0; face < 6; ++face)
    {
        vk::ImageViewCreateInfo viewInfo{};
        viewInfo
            .setImage(*environment.runtimeEnvironmentCube.image)
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(cubeFormat)
            .setSubresourceRange(
                vk::ImageSubresourceRange{}
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setBaseMipLevel(0)
                .setLevelCount(1)
                .setBaseArrayLayer(face)
                .setLayerCount(1));

        environment.runtimeEnvironmentCubeFaces.views[face] =
            vk::raii::ImageView(device, viewInfo);
    }

    std::cout << "Created runtime environment cubemap face views\n";

}

void EnvironmentRenderer::createDescriptorResources()
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

    equirectToCubeDescriptorSetLayout =
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

    equirectToCubeDescriptorPool =
        vk::raii::DescriptorPool(device, poolInfo);

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo
        .setDescriptorPool(*equirectToCubeDescriptorPool)
        .setSetLayouts(*equirectToCubeDescriptorSetLayout);

    auto sets = vk::raii::DescriptorSets(device, allocInfo);
    equirectToCubeDescriptorSet = std::move(sets.front());
}

void EnvironmentRenderer::updateDescriptorSet(
    EnvironmentResources& environment,
    const vk::raii::Sampler& hdrSampler,
    const vk::raii::ImageView& hdrView)
{
    if (hdrSampler == nullptr || hdrView == nullptr)
    {
        throw std::runtime_error(
            "updateEquirectToCubeDescriptorSet: HDR environment texture is not initialized");
    }

    vk::DescriptorImageInfo hdrInfo{};
    hdrInfo
        .setSampler(*hdrSampler)
        .setImageView(*hdrView)
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

    vk::WriteDescriptorSet write{};
    write
        .setDstSet(*equirectToCubeDescriptorSet)
        .setDstBinding(0)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(hdrInfo);

    vkContext.getDevice().updateDescriptorSets(write, {});
}

void EnvironmentRenderer::createPipeline()
{

    auto& device = vkContext.getDevice();

    vk::raii::ShaderModule vertShaderModule =
        createShaderModule(vkContext.getDevice(), readFile("shaders/equirect_to_cube_vert.spv"));

    vk::raii::ShaderModule fragShaderModule =
        createShaderModule(vkContext.getDevice(), readFile("shaders/equirect_to_cube_frag.spv"));

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
        *equirectToCubeDescriptorSetLayout
    };

    vk::PipelineLayoutCreateInfo layoutInfo{};
    layoutInfo
        .setSetLayouts(setLayouts)
        .setPushConstantRanges(pushRange);

    equirectToCubePipelineLayout =
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
        .setLayout(*equirectToCubePipelineLayout)
        .setRenderPass(vk::RenderPass{});

    pipelineChain.get<vk::PipelineRenderingCreateInfo>()
        .setColorAttachmentFormats(colorFormats);

    equirectToCubePipeline =
        vk::raii::Pipeline(
            device,
            nullptr,
            pipelineChain.get<vk::GraphicsPipelineCreateInfo>());

}

void EnvironmentRenderer::render(EnvironmentResources& environment)
{

    if (environment.runtimeEnvironmentCube.image == nullptr ||
        equirectToCubePipeline == nullptr ||
        equirectToCubePipelineLayout == nullptr)
    {
        throw std::runtime_error("renderEquirectToCubemap: resources not initialized");
    }

    constexpr vk::Format cubeFormat = vk::Format::eR16G16B16A16Sfloat;
    const uint32_t cubeSize = runtimeEnvironmentCubeSize;

    auto captureProjection = getCubemapCaptureProjection();
    auto captureViews = getCubemapCaptureViews();

    auto cmd = bufferUtils.beginSingleTimeCommands();

    vk::ImageMemoryBarrier toColorAttachment{};
    toColorAttachment
        .setOldLayout(vk::ImageLayout::eUndefined)
        .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*environment.runtimeEnvironmentCube.image)
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
        clearValue.setColor(vk::ClearColorValue(std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f }));

        vk::RenderingAttachmentInfo colorAttachment{};
        colorAttachment
            .setImageView(*environment.runtimeEnvironmentCubeFaces.views[face])
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
            *equirectToCubePipeline);

        cmd.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            *equirectToCubePipelineLayout,
            0,
            *equirectToCubeDescriptorSet,
            {});

        EquirectToCubePushConstants push{};
        push.viewProj = captureProjection * captureViews[face];

        cmd.pushConstants<EquirectToCubePushConstants>(
            *equirectToCubePipelineLayout,
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
        .setImage(*environment.runtimeEnvironmentCube.image)
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

    std::cout << "Rendered HDR equirectangular map to runtime cubemap\n";

}

std::array<glm::mat4, 6> EnvironmentRenderer::getCubemapCaptureViews() const
{
    return {
        glm::lookAt(glm::vec3(0.0f), glm::vec3(1,0,0),  glm::vec3(0,-1,0)),
        glm::lookAt(glm::vec3(0.0f), glm::vec3(-1,0,0), glm::vec3(0,-1,0)),
        glm::lookAt(glm::vec3(0.0f), glm::vec3(0,1,0),  glm::vec3(0,0,1)),
        glm::lookAt(glm::vec3(0.0f), glm::vec3(0,-1,0), glm::vec3(0,0,-1)),
        glm::lookAt(glm::vec3(0.0f), glm::vec3(0,0,1),  glm::vec3(0,-1,0)),
        glm::lookAt(glm::vec3(0.0f), glm::vec3(0,0,-1), glm::vec3(0,-1,0))
    };
}

glm::mat4 EnvironmentRenderer::getCubemapCaptureProjection() const
{
    glm::mat4 proj = glm::perspective(
        glm::radians(90.0f),
        1.0f,
        0.1f,
        10.0f);

    proj[1][1] *= -1.0f;
    return proj;
}