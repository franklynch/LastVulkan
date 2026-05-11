#pragma once

#include <memory>

#include "EnvironmentResources.hpp"
#include "VulkanContext.hpp"
#include "BufferUtils.hpp"
#include "ImageUtils.hpp"
#include "Texture2D.hpp"

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

 


    void updateIBLDescriptorSet(
        vk::raii::DescriptorSet& iblDescriptorSet,
        vk::Sampler fallbackEnvironmentSampler,
        vk::ImageView fallbackEnvironmentView);

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


};