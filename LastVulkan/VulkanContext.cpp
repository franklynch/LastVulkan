#include "VulkanContext.hpp"

VulkanContext::VulkanContext(
    Window& window,
    const std::vector<const char*>& validationLayers,
    const std::vector<const char*>& requiredDeviceExtensions,
    bool enableValidationLayers)
    : window(window)
    , validationLayers(validationLayers)
    , requiredDeviceExtensions(requiredDeviceExtensions)
    , enableValidationLayers(enableValidationLayers)
{
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createCommandPool();
}

VulkanContext::~VulkanContext()
{
    device.waitIdle();
}

void VulkanContext::createInstance()
{
    vk::ApplicationInfo appInfo{};
    appInfo
        .setPApplicationName("Hello Triangle")
        .setApplicationVersion(vk::makeApiVersion(0, 1, 0, 0))
        .setPEngineName("No Engine")
        .setEngineVersion(vk::makeApiVersion(0, 1, 0, 0))
        .setApiVersion(VK_API_VERSION_1_3);

    std::vector<const char*> requiredLayers;
    if (enableValidationLayers)
    {
        requiredLayers.assign(validationLayers.begin(), validationLayers.end());
    }

    auto layerProperties = context.enumerateInstanceLayerProperties();
    auto unsupportedLayerIt = std::ranges::find_if(
        requiredLayers,
        [&layerProperties](auto const& requiredLayer)
        {
            return std::ranges::none_of(
                layerProperties,
                [requiredLayer](auto const& layerProperty)
                {
                    return std::strcmp(layerProperty.layerName, requiredLayer) == 0;
                });
        });

    if (unsupportedLayerIt != requiredLayers.end())
    {
        throw std::runtime_error("Required layer not supported: " + std::string(*unsupportedLayerIt));
    }

    auto requiredExtensions = getRequiredInstanceExtensions();

    auto extensionProperties = context.enumerateInstanceExtensionProperties();
    auto unsupportedPropertyIt = std::ranges::find_if(
        requiredExtensions,
        [&extensionProperties](auto const& requiredExtension)
        {
            return std::ranges::none_of(
                extensionProperties,
                [requiredExtension](auto const& extensionProperty)
                {
                    return std::strcmp(extensionProperty.extensionName, requiredExtension) == 0;
                });
        });

    if (unsupportedPropertyIt != requiredExtensions.end())
    {
        throw std::runtime_error("Required extension not supported: " + std::string(*unsupportedPropertyIt));
    }

    vk::InstanceCreateInfo createInfo{};
    createInfo
        .setPApplicationInfo(&appInfo)
        .setEnabledLayerCount(static_cast<uint32_t>(requiredLayers.size()))
        .setPpEnabledLayerNames(requiredLayers.data())
        .setEnabledExtensionCount(static_cast<uint32_t>(requiredExtensions.size()))
        .setPpEnabledExtensionNames(requiredExtensions.data());

    instance = vk::raii::Instance(context, createInfo);
}

void VulkanContext::setupDebugMessenger()
{
    if (!enableValidationLayers)
        return;

    vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);

    vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);

    vk::DebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo
        .setMessageSeverity(severityFlags)
        .setMessageType(messageTypeFlags)
        .setPfnUserCallback(&debugCallback);

    debugMessenger = instance.createDebugUtilsMessengerEXT(createInfo);
}

void VulkanContext::createSurface()
{
    VkSurfaceKHR rawSurface;
    if (glfwCreateWindowSurface(*instance, window.getHandle(), nullptr, &rawSurface) != 0)
    {
        throw std::runtime_error("failed to create window surface!");
    }

    surface = vk::raii::SurfaceKHR(instance, rawSurface);
}

bool VulkanContext::isDeviceSuitable(vk::raii::PhysicalDevice const& physicalDevice)
{
    bool supportsVulkan1_3 =
        physicalDevice.getProperties().apiVersion >= VK_API_VERSION_1_3;

    auto queueFamilies = physicalDevice.getQueueFamilyProperties();
    bool supportsGraphicsAndPresent = false;
    for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilies.size()); ++i)
    {
        if ((queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
            physicalDevice.getSurfaceSupportKHR(i, *surface))
        {
            supportsGraphicsAndPresent = true;
            break;
        }
    }

    auto availableDeviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();
    bool supportsAllRequiredExtensions =
        std::ranges::all_of(
            requiredDeviceExtensions,
            [&availableDeviceExtensions](auto const& requiredExt)
            {
                return std::ranges::any_of(
                    availableDeviceExtensions,
                    [requiredExt](auto const& availableExt)
                    {
                        return std::strcmp(availableExt.extensionName, requiredExt) == 0;
                    });
            });

    auto features =
        physicalDevice.template getFeatures2<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan11Features,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();

    bool supportsRequiredFeatures =
        features.get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
        features.get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
        features.get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
        features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
        features.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

    return supportsVulkan1_3 &&
        supportsGraphicsAndPresent &&
        supportsAllRequiredExtensions &&
        supportsRequiredFeatures;
}

void VulkanContext::pickPhysicalDevice()
{
    auto physicalDevices = instance.enumeratePhysicalDevices();
    auto devIter = std::ranges::find_if(
        physicalDevices,
        [&](auto const& physicalDevice)
        {
            return isDeviceSuitable(physicalDevice);
        });

    if (devIter == physicalDevices.end())
    {
        throw std::runtime_error("failed to find a suitable GPU!");
    }

    physicalDevice = *devIter;
}

vk::SampleCountFlagBits VulkanContext::getMaxUsableSampleCount() const
{
    vk::PhysicalDeviceProperties props = physicalDevice.getProperties();

    vk::SampleCountFlags counts =
        props.limits.framebufferColorSampleCounts &
        props.limits.framebufferDepthSampleCounts;

    if (counts & vk::SampleCountFlagBits::e64) return vk::SampleCountFlagBits::e64;
    if (counts & vk::SampleCountFlagBits::e32) return vk::SampleCountFlagBits::e32;
    if (counts & vk::SampleCountFlagBits::e16) return vk::SampleCountFlagBits::e16;
    if (counts & vk::SampleCountFlagBits::e8)  return vk::SampleCountFlagBits::e8;
    if (counts & vk::SampleCountFlagBits::e4)  return vk::SampleCountFlagBits::e4;
    if (counts & vk::SampleCountFlagBits::e2)  return vk::SampleCountFlagBits::e2;

    return vk::SampleCountFlagBits::e1;
}

void VulkanContext::createLogicalDevice()
{
    auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

    msaaSamples = getMaxUsableSampleCount();

    for (uint32_t qfpIndex = 0; qfpIndex < static_cast<uint32_t>(queueFamilyProperties.size()); ++qfpIndex)
    {
        if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
            physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface))
        {
            queueIndex = qfpIndex;
            break;
        }
    }

    if (queueIndex == ~0u)
    {
        throw std::runtime_error("Could not find a queue for graphics and present");
    }

    vk::StructureChain<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan11Features,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
    > featureChain{};

    featureChain.get<vk::PhysicalDeviceVulkan11Features>()
        .setShaderDrawParameters(VK_TRUE);

    featureChain.get<vk::PhysicalDeviceFeatures2>()
        .features.samplerAnisotropy = VK_TRUE;

    featureChain.get<vk::PhysicalDeviceVulkan13Features>()
        .setSynchronization2(VK_TRUE)
        .setDynamicRendering(VK_TRUE);

    featureChain.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>()
        .setExtendedDynamicState(VK_TRUE);

    float queuePriority = 1.0f;

    vk::DeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo
        .setQueueFamilyIndex(queueIndex)
        .setQueueCount(1)
        .setPQueuePriorities(&queuePriority);

    vk::DeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo
        .setPNext(&featureChain.get<vk::PhysicalDeviceFeatures2>())
        .setQueueCreateInfos(queueCreateInfo)
        .setPEnabledExtensionNames(requiredDeviceExtensions);

    device = vk::raii::Device(physicalDevice, deviceCreateInfo);
    queue = vk::raii::Queue(device, queueIndex, 0);

    printLogicalDeviceInfo(queueIndex);
}

void VulkanContext::createCommandPool()
{
    vk::CommandPoolCreateInfo poolInfo{};
    poolInfo
        .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
        .setQueueFamilyIndex(queueIndex);

    commandPool = vk::raii::CommandPool(device, poolInfo);
}

std::vector<const char*> VulkanContext::getRequiredInstanceExtensions() const
{
    uint32_t glfwExtensionCount = 0;
    auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    if (enableValidationLayers)
    {
        extensions.push_back(vk::EXTDebugUtilsExtensionName);
    }

    return extensions;
}

void VulkanContext::printLogicalDeviceInfo(uint32_t graphicsIndex) const
{
    vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();

    std::cout << "Logical device created successfully\n";
    std::cout << "Selected GPU: " << properties.deviceName << '\n';
    std::cout << "Vendor ID: " << properties.vendorID << '\n';
    std::cout << "Device ID: " << properties.deviceID << '\n';
    std::cout << "API Version: "
        << VK_API_VERSION_MAJOR(properties.apiVersion) << "."
        << VK_API_VERSION_MINOR(properties.apiVersion) << "."
        << VK_API_VERSION_PATCH(properties.apiVersion) << '\n';
    std::cout << "Graphics Queue Family Index: " << graphicsIndex << '\n';
}

VKAPI_ATTR vk::Bool32 VKAPI_CALL VulkanContext::debugCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT type,
    const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void*)
{
    if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError ||
        severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
    {
        std::cerr << "validation layer: type " << to_string(type)
            << " msg: " << pCallbackData->pMessage << std::endl;
    }

    return vk::False;
}