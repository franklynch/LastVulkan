#pragma once

#include <memory>

#include "EnvironmentResources.hpp"
#include "VulkanContext.hpp"
#include "BufferUtils.hpp"
#include "ImageUtils.hpp"
#include "Texture2D.hpp"

#include "BrdfLutRenderer.hpp"
#include "EnvironmentRenderer.hpp"
#include "IrradianceRenderer.hpp"
#include "PrefilterRenderer.hpp"

class EnvironmentSystem
{
public:
    EnvironmentSystem(VulkanContext& vkContext, BufferUtils& bufferUtils, ImageUtils& imageUtils);
    
    

    void createFallbackResources();
    void createFallbackBrdfLut();
    void createFallbackBlackCube();

    void cleanup();

    EnvironmentResources& resources() { return environment; }
    const EnvironmentResources& resources() const { return environment; }

    Texture2D* fallbackBrdfLutTexture() const
    {
        return fallbackBrdfLut.get();
    }
    
    vk::Sampler getFallbackBlackCubeSampler() const
    {
        return *fallbackBlackCubeSampler;
    }

    vk::ImageView getFallbackBlackCubeView() const
    {
        return *fallbackBlackCubeView;
    }

    void createHdrEnvironmentTexture(const std::string& path);

    vk::raii::Sampler& getHdrEnvironmentSampler()
    {
        return hdrEnvironmentSampler;
    }

    vk::raii::ImageView& getHdrEnvironmentView()
    {
        return hdrEnvironmentView;
    }

    const vk::raii::Sampler& getHdrEnvironmentSampler() const
    {
        return hdrEnvironmentSampler;
    }

    const vk::raii::ImageView& getHdrEnvironmentView() const
    {
        return hdrEnvironmentView;
    }

    void initRenderers();

    void generateRuntimeEnvironmentCubemap();
    void generateRuntimeIrradianceCubemap();
    void generateRuntimePrefilteredCubemap();

    void loadHdrEnvironment(const std::string& path,
                            vk::raii::DescriptorSet& iblDescriptorSet,
                            vk::Sampler fallbackEnvironmentSampler,
                            vk::ImageView fallbackEnvironmentView);


    void updateIBLDescriptorSet(
        vk::raii::DescriptorSet& iblDescriptorSet,
        vk::Sampler fallbackEnvironmentSampler,
        vk::ImageView fallbackEnvironmentView);

    uint32_t getDebugRuntimePrefilteredMipLevels() const;

private:
    VulkanContext& vkContext;
    BufferUtils& bufferUtils;
    ImageUtils& imageUtils;

    EnvironmentResources environment;


    // Fallback BRDF LUT (2D)
    std::unique_ptr<Texture2D> fallbackBrdfLut;

    // Fallback cubemap (shared for irradiance/prefiltered/environment)
    vk::raii::Image fallbackBlackCubeImage{ nullptr };
    vk::raii::DeviceMemory fallbackBlackCubeMemory{ nullptr };
    vk::raii::ImageView fallbackBlackCubeView{ nullptr };
    vk::raii::Sampler fallbackBlackCubeSampler{ nullptr };

    vk::DescriptorImageInfo makeImageInfo(
        vk::Sampler sampler,
        vk::ImageView view) const;


    vk::raii::Image hdrEnvironmentImage{ nullptr };
    vk::raii::DeviceMemory hdrEnvironmentMemory{ nullptr };
    vk::raii::ImageView hdrEnvironmentView{ nullptr };
    vk::raii::Sampler hdrEnvironmentSampler{ nullptr };

    uint32_t hdrEnvironmentWidth = 0;
    uint32_t hdrEnvironmentHeight = 0;

    std::unique_ptr<BrdfLutRenderer>        brdfLutRenderer;
    std::unique_ptr<EnvironmentRenderer>    environmentRenderer;
    std::unique_ptr<IrradianceRenderer>     irradianceRenderer;
    std::unique_ptr<PrefilterRenderer>      prefilterRenderer;

    

    


};