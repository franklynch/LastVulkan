#include "ScenePipelines.hpp"
#include "ShaderUtils.hpp"
#include "RendererTypes.hpp"

#include <fstream>
#include <stdexcept>
#include <vector>




ScenePipelines::ScenePipelines(VulkanContext& vkContext)
    : vkContext(vkContext)
{

}

void ScenePipelines::cleanup()
{
    m_skyboxPipeline = nullptr;
    m_skyboxPipelineLayout = nullptr;

    m_transparentDoubleSidedPipeline = nullptr;
    m_transparentPipeline = nullptr;

    m_wireframeDoubleSidedPipeline = nullptr;
    m_wireframePipeline = nullptr;

    m_solidDoubleSidedPipeline = nullptr;
    m_solidPipeline = nullptr;

    m_pipelineLayout = nullptr;
}

void ScenePipelines::create(
    
    vk::Format hdrColorFormat,
    vk::Format depthFormat,
    vk::DescriptorSetLayout frameLayout,
    vk::DescriptorSetLayout materialLayout,
    vk::DescriptorSetLayout iblLayout,
    bool wireframeSupported)
{
    
    auto& device = vkContext.getDevice();
    
    vk::raii::ShaderModule vertShaderModule =
        ShaderUtils::createShaderModule(
            vkContext.getDevice(),
            "shaders/vert.spv");

    vk::raii::ShaderModule fragShaderModule =
        ShaderUtils::createShaderModule(
            vkContext.getDevice(),
            "shaders/frag.spv");
        

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

    // Main scene pipeline layout: set 0 = frame UBO  set 1 = material textures  set 2 = IBL textures
    std::array<vk::DescriptorSetLayout, 3> setLayouts = {
        frameLayout,
        materialLayout,
        iblLayout
    };

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo
        .setSetLayouts(setLayouts)
        .setPushConstantRanges(pushConstantRange);

    m_pipelineLayout =
        vk::raii::PipelineLayout(device, pipelineLayoutInfo);

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
        .setLayout(*m_pipelineLayout)
        .setRenderPass(vk::RenderPass{});

    std::array<vk::Format, 1> colorAttachmentFormats = {
            hdrColorFormat
            };

    pipelineCreateInfoChain.get<vk::PipelineRenderingCreateInfo>()
        .setColorAttachmentFormats(colorAttachmentFormats)
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

    m_solidPipeline = vk::raii::Pipeline(
        device,
        nullptr,
        pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>()
    );

    // Filled, double-sided
    rasterizer
        .setPolygonMode(vk::PolygonMode::eFill)
        .setCullMode(vk::CullModeFlagBits::eNone);

    m_solidDoubleSidedPipeline = vk::raii::Pipeline(
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

    m_transparentPipeline = vk::raii::Pipeline(
        device,
        nullptr,
        pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>()
    );

    // Transparent, double-sided
    rasterizer
        .setPolygonMode(vk::PolygonMode::eFill)
        .setCullMode(vk::CullModeFlagBits::eNone);

    m_transparentDoubleSidedPipeline = vk::raii::Pipeline(
        device,
        nullptr,
        pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>()
    );

    // -------------------------
    // Wireframe opaque pipelines
    // -------------------------

    m_wireframePipeline = nullptr;
    m_wireframeDoubleSidedPipeline = nullptr;

    if (vkContext.isFillModeNonSolidEnabled())
    {
        pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>()
            .setPDepthStencilState(&depthStencil)
            .setPColorBlendState(&colorBlending);

        // Wireframe, culled
        rasterizer
            .setPolygonMode(vk::PolygonMode::eLine)
            .setCullMode(vk::CullModeFlagBits::eBack);

        m_wireframePipeline = vk::raii::Pipeline(
            device,
            nullptr,
            pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>()
        );

        // Wireframe, double-sided
        rasterizer
            .setPolygonMode(vk::PolygonMode::eLine)
            .setCullMode(vk::CullModeFlagBits::eNone);

        m_wireframeDoubleSidedPipeline = vk::raii::Pipeline(
            device,
            nullptr,
            pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>()
        );
    }

    
}

void ScenePipelines::createSkybox(
    
    vk::Format hdrColorFormat,
    vk::Format depthFormat,
    vk::DescriptorSetLayout frameLayout,
    vk::DescriptorSetLayout iblLayout)
{
    auto& device = vkContext.getDevice();
    
    vk::raii::ShaderModule vertShaderModule =
        ShaderUtils::createShaderModule(
            vkContext.getDevice(),
            "shaders/skybox_vert.spv");

    vk::raii::ShaderModule fragShaderModule =
        ShaderUtils::createShaderModule(
            vkContext.getDevice(),
            "shaders/skybox_frag.spv");

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

    // Skybox pipeline layout: // set 0 = frame UBO // set 1 = IBL descriptor set
    std::array<vk::DescriptorSetLayout, 2> setLayouts = {
        frameLayout,
        iblLayout
    };

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setSetLayouts(setLayouts);

    m_skyboxPipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

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
        .setLayout(*m_skyboxPipelineLayout)
        .setRenderPass(vk::RenderPass{});

    std::array<vk::Format, 1> colorAttachmentFormats = {
        hdrColorFormat };

    pipelineCreateInfoChain.get<vk::PipelineRenderingCreateInfo>()
        .setColorAttachmentFormats(colorAttachmentFormats)
        .setDepthAttachmentFormat(depthFormat);

    m_skyboxPipeline = vk::raii::Pipeline(
        device,
        nullptr,
        pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>()
    );
}