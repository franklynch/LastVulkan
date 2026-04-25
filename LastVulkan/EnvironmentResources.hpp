#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#   include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include <array>
#include <vector>

struct Cubemap
{
    vk::raii::Image image{ nullptr };
    vk::raii::DeviceMemory memory{ nullptr };
    vk::raii::ImageView view{ nullptr };
    vk::raii::Sampler sampler{ nullptr };
};

struct CubemapFaceViews
{
    std::array<vk::raii::ImageView, 6> views = {
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr
    };
};

struct BrdfLut
{
    vk::raii::Image image{ nullptr };
    vk::raii::DeviceMemory memory{ nullptr };
    vk::raii::ImageView view{ nullptr };
    vk::raii::Sampler sampler{ nullptr };

    vk::raii::Pipeline pipeline{ nullptr };
    vk::raii::PipelineLayout layout{ nullptr };
};

struct EnvironmentResources
{
    Cubemap runtimeEnvironmentCube;
    CubemapFaceViews runtimeEnvironmentCubeFaces;

    Cubemap runtimeIrradianceCube;
    CubemapFaceViews runtimeIrradianceCubeFaces;

    Cubemap runtimePrefilteredCube;
    std::vector<CubemapFaceViews> runtimePrefilteredCubeMipFaceViews;

    BrdfLut runtimeBrdfLut;
};