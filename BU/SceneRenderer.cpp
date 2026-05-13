#include "SceneRenderer.hpp"

#include <array>
#include <cassert>

#include <algorithm>
#include <vector>

#include "RendererTypes.hpp"

SceneRenderer::SceneRenderer(VulkanContext& vkContext)
    : vkContext(vkContext)
{
}

void SceneRenderer::renderOpaque(
    vk::raii::CommandBuffer& commandBuffer,
    const Scene& scene,
    const SceneRenderContext& context)
{
    assert(context.materialDescriptorSets != nullptr);

    vk::CommandBuffer cmd = *commandBuffer;

    for (auto& renderable : scene.getRenderables())
    {
        Material& renderableMaterial = renderable.getMaterial();

        const bool isBlendMaterial =
            renderableMaterial.getAlphaMode() == "BLEND";

        if (isBlendMaterial)
        {
            continue;
        }

        uint32_t materialIndex = renderable.getMaterialIndex();

        assert(materialIndex < context.materialDescriptorSets->size());

        commandBuffer.bindVertexBuffers(
            0,
            *renderable.getMesh().getVertexBuffer(),
            { 0 });

        commandBuffer.bindIndexBuffer(
            *renderable.getMesh().getIndexBuffer(),
            0,
            vk::IndexType::eUint32);

        vk::Pipeline activePipeline{};

        if (context.wireframeEnabled)
        {
            activePipeline = renderableMaterial.isDoubleSided()
                ? context.wireframeDoubleSidedPipeline
                : context.wireframePipeline;
        }
        else
        {
            activePipeline = renderableMaterial.isDoubleSided()
                ? context.solidDoubleSidedPipeline
                : context.solidPipeline;
        }

        if (activePipeline == nullptr)
        {
            activePipeline = renderableMaterial.isDoubleSided()
                ? context.solidDoubleSidedPipeline
                : context.solidPipeline;
        }

        commandBuffer.bindPipeline(
            vk::PipelineBindPoint::eGraphics,
            activePipeline);

        std::array<vk::DescriptorSet, 3> sets = {
            context.frameDescriptorSet,
            *(*context.materialDescriptorSets)[materialIndex],
            context.iblDescriptorSet
        };

        commandBuffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            context.pipelineLayout,
            0,
            sets,
            {});

        PushConstantData pushData{};
        pushData.model = renderable.getTransform().toMatrix();

        if (context.animateModel)
        {
            pushData.model = glm::rotate(
                pushData.model,
                context.currentAnimationAngle,
                glm::vec3(0.0f, 0.0f, 1.0f));
        }

        glm::mat3 normalMatrix =
            glm::transpose(glm::inverse(glm::mat3(pushData.model)));

        pushData.normalMatrix = normalMatrix;
        pushData.baseColorFactor =
            renderableMaterial.getBaseColorFactor();

        const bool isMaskMaterial =
            renderableMaterial.getAlphaMode() == "MASK";

        pushData.materialParams = glm::vec4(
            renderableMaterial.getMetallicFactor(),
            renderableMaterial.getRoughnessFactor(),
            renderableMaterial.getNormalScale(),
            renderableMaterial.getAlphaCutoff());

        pushData.alphaModeParams = glm::vec4(
            isMaskMaterial ? 1.0f : 0.0f,
            0.0f,
            renderableMaterial.getOcclusionStrength(),
            0.0f);

        pushData.emissiveFactor =
            glm::vec4(renderableMaterial.getEmissiveFactor(), 1.0f);

        cmd.pushConstants(
            context.pipelineLayout,
            vk::ShaderStageFlagBits::eVertex |
            vk::ShaderStageFlagBits::eFragment,
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
}

void SceneRenderer::renderTransparent(
    vk::raii::CommandBuffer& commandBuffer,
    const Scene& scene,
    const Camera& camera,
    const SceneRenderContext& context)
{
    assert(context.materialDescriptorSets != nullptr);

    vk::CommandBuffer cmd = *commandBuffer;

    std::vector<const Renderable*> transparentRenderables;

    transparentRenderables.reserve(scene.getRenderables().size());

    for (const auto& renderable : scene.getRenderables())
    {
        if (renderable.getMaterial().getAlphaMode() == "BLEND")
        {
            transparentRenderables.push_back(&renderable);
        }
    }

    const glm::vec3 cameraPos = camera.getPosition();

    std::sort(
        transparentRenderables.begin(),
        transparentRenderables.end(),
        [&](const Renderable* a, const Renderable* b)
        {
            glm::vec3 aPos = glm::vec3(a->getTransform().toMatrix()[3]);
            glm::vec3 bPos = glm::vec3(b->getTransform().toMatrix()[3]);

            float da = glm::dot(aPos - cameraPos, aPos - cameraPos);
            float db = glm::dot(bPos - cameraPos, bPos - cameraPos);

            return da > db;
        });

    for (const Renderable* renderable : transparentRenderables)
    {
        const Material& renderableMaterial = renderable->getMaterial();

        uint32_t materialIndex = renderable->getMaterialIndex();
        assert(materialIndex < context.materialDescriptorSets->size());

        commandBuffer.bindVertexBuffers(
            0,
            *renderable->getMesh().getVertexBuffer(),
            { 0 });

        commandBuffer.bindIndexBuffer(
            *renderable->getMesh().getIndexBuffer(),
            0,
            vk::IndexType::eUint32);

        vk::Pipeline activePipeline =
            renderableMaterial.isDoubleSided()
            ? context.transparentDoubleSidedPipeline
            : context.transparentPipeline;

        commandBuffer.bindPipeline(
            vk::PipelineBindPoint::eGraphics,
            activePipeline);

        std::array<vk::DescriptorSet, 3> sets = {
            context.frameDescriptorSet,
            *(*context.materialDescriptorSets)[materialIndex],
            context.iblDescriptorSet
        };

        commandBuffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            context.pipelineLayout,
            0,
            sets,
            {});

        PushConstantData pushData{};
        pushData.model = renderable->getTransform().toMatrix();

        if (context.animateModel)
        {
            pushData.model = glm::rotate(
                pushData.model,
                context.currentAnimationAngle,
                glm::vec3(0.0f, 0.0f, 1.0f));
        }

        glm::mat3 normalMatrix =
            glm::transpose(glm::inverse(glm::mat3(pushData.model)));

        pushData.normalMatrix = normalMatrix;
        pushData.baseColorFactor =
            renderableMaterial.getBaseColorFactor();

        pushData.materialParams = glm::vec4(
            renderableMaterial.getMetallicFactor(),
            renderableMaterial.getRoughnessFactor(),
            renderableMaterial.getNormalScale(),
            renderableMaterial.getAlphaCutoff());

        const bool isMaskMaterial =
            renderableMaterial.getAlphaMode() == "MASK";

        const bool isBlendMaterial =
            renderableMaterial.getAlphaMode() == "BLEND";

        pushData.alphaModeParams = glm::vec4(
            isMaskMaterial ? 1.0f : 0.0f,
            isBlendMaterial ? 1.0f : 0.0f,
            renderableMaterial.getOcclusionStrength(),
            0.0f);

        pushData.emissiveFactor =
            glm::vec4(renderableMaterial.getEmissiveFactor(), 1.0f);

        cmd.pushConstants(
            context.pipelineLayout,
            vk::ShaderStageFlagBits::eVertex |
            vk::ShaderStageFlagBits::eFragment,
            0,
            sizeof(PushConstantData),
            &pushData);

        commandBuffer.drawIndexed(
            renderable->getMesh().getIndexCount(),
            1,
            0,
            0,
            0);
    }
}

void SceneRenderer::renderSkybox(
    vk::raii::CommandBuffer& commandBuffer,
    const SceneRenderContext& context)
{
    if (context.skyboxPipeline == nullptr ||
        context.skyboxPipelineLayout == nullptr ||
        context.frameDescriptorSet == nullptr ||
        context.iblDescriptorSet == nullptr)
    {
        return;
    }

    commandBuffer.bindPipeline(
        vk::PipelineBindPoint::eGraphics,
        context.skyboxPipeline);

    std::array<vk::DescriptorSet, 2> sets = {
        context.frameDescriptorSet,
        context.iblDescriptorSet
    };

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        context.skyboxPipelineLayout,
        0,
        sets,
        {});

    commandBuffer.draw(3, 1, 0, 0);
}

