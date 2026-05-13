#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#   include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include <vector>
#include <glm/glm.hpp>

#include "VulkanContext.hpp"
#include "BufferUtils.hpp"
#include "ImageUtils.hpp"

class PostProcessRenderer
{
public:
    PostProcessRenderer(
        VulkanContext& vkContext,
        BufferUtils& bufferUtils,
        ImageUtils& imageUtils);

    ~PostProcessRenderer() = default;

    PostProcessRenderer(const PostProcessRenderer&) = delete;
    PostProcessRenderer& operator=(const PostProcessRenderer&) = delete;

    void init(vk::Extent2D extent, vk::Format swapchainFormat);
    void cleanup();
    void recreate(vk::Extent2D extent, vk::Format swapchainFormat);

    vk::ImageView getHdrView() const { return *hdrColorView; }
    vk::Image getHdrImage() const { return *hdrColorImage; }
    vk::Format getHdrFormat() const { return hdrColorFormat; }

    vk::ImageView getBloomBrightView() const { return *bloomBrightView; }
    vk::Image getBloomBrightImage() const { return *bloomBrightImage; }

    vk::ImageView getBloomBlurTempView() const { return *bloomBlurTempView; }
    vk::Image getBloomBlurTempImage() const { return *bloomBlurTempImage; }

    vk::DescriptorSetLayout getBloomBlurDescriptorSetLayout() const
    {
        return *bloomBlurDescriptorSetLayout;
    }

    vk::DescriptorSet getBloomBlurFromBrightDescriptorSet()
    {
        return *bloomBlurFromBrightDescriptorSets[0];
    }

    vk::DescriptorSet getBloomBlurFromTempDescriptorSet()
    {
        return *bloomBlurFromTempDescriptorSets[0];
    }

        
    void recordBloomExtract(
        vk::raii::CommandBuffer& commandBuffer);

    void recordBloomBlurFromBright(
        vk::raii::CommandBuffer& commandBuffer,
        vk::ImageView outputView,
        glm::vec2 direction);

    void recordBloomBlurFromTemp(
        vk::raii::CommandBuffer& commandBuffer,
        vk::ImageView outputView,
        glm::vec2 direction);

    void recordBloomPyramid(vk::raii::CommandBuffer& commandBuffer);

 

    float bloomThreshold = 1.0f;
    float bloomKnee = 0.5f;

    float bloomIntensity = 1.0f;
    float bloomUpsampleRadius = 0.005f;

    float bloomStrength = 0.15f;
    bool  bloomEnabled = true;

    bool    toneMappingEnabled = true;
    bool    gammaEnabled = true;
    float   postExposure = 1.0f;

    glm::vec4 buildFinalCompositeParams() const;

    void beginFinalCompositePass(
        vk::raii::CommandBuffer& commandBuffer,
        vk::ImageView swapchainImageView);

    void recordFinalComposite(
        vk::raii::CommandBuffer& commandBuffer);

    void endFinalCompositePass(
        vk::raii::CommandBuffer& commandBuffer);



    void executeBloomChain(
        vk::raii::CommandBuffer& commandBuffer);

    void executeFinalComposite(
        vk::raii::CommandBuffer& commandBuffer,
        vk::Image swapchainImage,
        vk::ImageView swapchainImageView,
        vk::ImageLayout oldLayout);
;

    

private:
    VulkanContext& vkContext;
    BufferUtils& bufferUtils;
    ImageUtils& imageUtils;

    vk::Extent2D extent{};
    vk::Format swapchainFormat{};

    vk::raii::Image hdrColorImage{ nullptr };
    vk::raii::DeviceMemory hdrColorMemory{ nullptr };
    vk::raii::ImageView hdrColorView{ nullptr };

    vk::Format hdrColorFormat = vk::Format::eR16G16B16A16Sfloat;

    void createHdrColorResources();

    vk::raii::Image bloomBrightImage{ nullptr };
    vk::raii::DeviceMemory bloomBrightMemory{ nullptr };
    vk::raii::ImageView bloomBrightView{ nullptr };

    vk::raii::Image bloomBlurTempImage{ nullptr };
    vk::raii::DeviceMemory bloomBlurTempMemory{ nullptr };
    vk::raii::ImageView bloomBlurTempView{ nullptr };

   

    void createBloomBrightResources();
    void createBloomBlurResources();

    vk::raii::DescriptorSetLayout bloomExtractDescriptorSetLayout{ nullptr };
    vk::raii::DescriptorPool bloomExtractDescriptorPool{ nullptr };
    vk::raii::DescriptorSets bloomExtractDescriptorSets{ nullptr };
   
    vk::raii::PipelineLayout bloomExtractPipelineLayout{ nullptr };
    vk::raii::Pipeline bloomExtractPipeline{ nullptr };

    vk::raii::DescriptorSetLayout bloomBlurDescriptorSetLayout{ nullptr };
    vk::raii::DescriptorPool bloomBlurDescriptorPool{ nullptr };

    vk::raii::DescriptorSets bloomBlurFromBrightDescriptorSets{ nullptr };
    vk::raii::DescriptorSets bloomBlurFromTempDescriptorSets{ nullptr };

    vk::raii::PipelineLayout bloomBlurPipelineLayout{ nullptr };
    vk::raii::Pipeline bloomBlurPipeline{ nullptr };

    
    void createBloomExtractDescriptorSetLayout();
    void createBloomExtractPipeline();
    void createBloomExtractDescriptorSet();
    void drawBloomExtract(vk::raii::CommandBuffer& commandBuffer);

    void createBloomBlurDescriptorSetLayout();
    void createBloomBlurPipeline();
    void createBloomBlurDescriptorSets();

    void drawBloomBlur(
        vk::raii::CommandBuffer& commandBuffer,
        vk::ImageView outputView,
        vk::DescriptorSet inputSet,
        glm::vec2 direction);

   

    struct BloomMipResource
    {
        vk::raii::Image image{ nullptr };
        vk::raii::DeviceMemory memory{ nullptr };
        vk::raii::ImageView view{ nullptr };
        uint32_t width = 0;
        uint32_t height = 0;
    };

    std::vector<BloomMipResource> bloomDownsampleChain;
    uint32_t bloomDownsampleLevels = 3;

    vk::raii::DescriptorPool bloomDownsampleDescriptorPool{ nullptr };
    std::vector<vk::raii::DescriptorSets> bloomDownsampleDescriptorSets;

    vk::raii::DescriptorPool bloomUpsampleDescriptorPool{ nullptr };
    std::vector<vk::raii::DescriptorSets> bloomUpsampleDescriptorSets;
    vk::raii::DescriptorSets bloomUpsampleFinalDescriptorSet{ nullptr };

    vk::raii::PipelineLayout bloomDownsamplePipelineLayout{ nullptr };
    vk::raii::Pipeline bloomDownsamplePipeline{ nullptr };

    vk::raii::PipelineLayout bloomUpsamplePipelineLayout{ nullptr };
    vk::raii::Pipeline bloomUpsamplePipeline{ nullptr };

    void createBloomDownsampleResources();

    void createBloomDownsamplePipeline();
    void createBloomUpsamplePipeline();

    void createBloomDownsampleDescriptorSets();
    void createBloomUpsampleDescriptorSets();

    void drawBloomDownsample(
        vk::raii::CommandBuffer& commandBuffer,
        vk::ImageView outputView,
        vk::DescriptorSet inputSet,
        uint32_t outputWidth,
        uint32_t outputHeight,
        uint32_t inputWidth,
        uint32_t inputHeight);

    void drawBloomUpsample(
        vk::raii::CommandBuffer& commandBuffer,
        vk::ImageView outputView,
        vk::DescriptorSet inputSet,
        uint32_t outputWidth,
        uint32_t outputHeight,
        uint32_t inputWidth,
        uint32_t inputHeight);

    

   

    vk::raii::DescriptorSetLayout postProcessDescriptorSetLayout{ nullptr };
    vk::raii::DescriptorPool postProcessDescriptorPool{ nullptr };
    vk::raii::DescriptorSets postProcessDescriptorSets{ nullptr };

    vk::raii::PipelineLayout postProcessPipelineLayout{ nullptr };
    vk::raii::Pipeline postProcessPipeline{ nullptr };


    void createPostProcessDescriptorSetLayout();
    void createPostProcessPipeline();
    void createPostProcessDescriptorSet();

    vk::raii::Sampler postProcessSampler{ nullptr };

    void createPostProcessSampler();

};