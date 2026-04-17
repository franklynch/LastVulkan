#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <ranges>
#include <iostream>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include "Window.hpp"

class VulkanContext
{
public:
    VulkanContext(
        Window& window,
        const std::vector<const char*>& validationLayers,
        const std::vector<const char*>& requiredDeviceExtensions,
        bool enableValidationLayers);

    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    vk::raii::Instance& getInstance() { return instance; }
    const vk::raii::Instance& getInstance() const { return instance; }

    vk::raii::SurfaceKHR& getSurface() { return surface; }
    const vk::raii::SurfaceKHR& getSurface() const { return surface; }

    vk::raii::PhysicalDevice& getPhysicalDevice() { return physicalDevice; }
    const vk::raii::PhysicalDevice& getPhysicalDevice() const { return physicalDevice; }

    vk::raii::Device& getDevice() { return device; }
    const vk::raii::Device& getDevice() const { return device; }

    vk::raii::Queue& getQueue() { return queue; }
    const vk::raii::Queue& getQueue() const { return queue; }

    vk::raii::CommandPool& getCommandPool() { return commandPool; }
    const vk::raii::CommandPool& getCommandPool() const { return commandPool; }

    uint32_t getQueueIndex() const { return queueIndex; }
    vk::SampleCountFlagBits getMsaaSamples() const { return msaaSamples; }

    [[nodiscard]] bool isFillModeNonSolidEnabled() const { return fillModeNonSolidEnabled; }

private:
    void createInstance();
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createCommandPool();

    bool isDeviceSuitable(vk::raii::PhysicalDevice const& physicalDevice);
    std::vector<const char*> getRequiredInstanceExtensions() const;
    void printLogicalDeviceInfo(uint32_t graphicsIndex) const;
    vk::SampleCountFlagBits getMaxUsableSampleCount() const;

    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
        vk::DebugUtilsMessageTypeFlagsEXT type,
        const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void*);

private:
    Window& window;
    const std::vector<const char*>& validationLayers;
    const std::vector<const char*>& requiredDeviceExtensions;
    bool enableValidationLayers = false;

    vk::raii::Context context;
    vk::raii::Instance instance = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;

    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;
    vk::raii::Queue queue = nullptr;
    vk::raii::CommandPool commandPool = nullptr;

    bool fillModeNonSolidEnabled = false;

    uint32_t queueIndex = ~0u;
    vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;
};