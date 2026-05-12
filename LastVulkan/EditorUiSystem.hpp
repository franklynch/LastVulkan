#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#   include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include "Window.hpp"
#include "VulkanContext.hpp"




#include "EditorUiState.hpp"
#include "MaterialSystem.hpp"
#include "Scene.hpp"



class EditorUiSystem
{
public:
    EditorUiSystem(
        Window& window,
        VulkanContext& vkContext);

    void init(
        vk::Format swapchainFormat,
        uint32_t imageCount);

    void shutdown();

    void beginFrame();
    void render(vk::CommandBuffer commandBuffer);

    void buildMinimal(
        float frameTimeMs,
        float fps);

    bool isInitialized() const
    {
        return imguiInitialized;
    }

    EditorUiState& state()
    {
        return uiState;
    }

    const EditorUiState& state() const
    {
        return uiState;
    }

   


private:
    Window& window;
    VulkanContext& vkContext;

    EditorUiState uiState;

    bool imguiInitialized = false;
    vk::raii::DescriptorPool imguiDescriptorPool = nullptr;
};