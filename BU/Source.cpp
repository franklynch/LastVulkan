#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <vector>

#include "Window.hpp"
#include "VulkanContext.hpp"
#include "Renderer.hpp"
#include "RendererTypes.hpp"

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

const std::vector<char const*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

const std::vector<const char*> requiredDeviceExtension = {
    vk::KHRSwapchainExtensionName
};

class Application
{
public:
    void run();

private:
    void init();
    void mainLoop();

private:
    std::unique_ptr<Window> window;
    std::unique_ptr<VulkanContext> vkContext;
    std::unique_ptr<Renderer> renderer;
};

void Application::run()
{
    init();
    mainLoop();
}

void Application::init()
{
    window = std::make_unique<Window>(WIDTH, HEIGHT, "Vulkan");

    vkContext = std::make_unique<VulkanContext>(
        *window,
        validationLayers,
        requiredDeviceExtension,
        enableValidationLayers
    );

    renderer = std::make_unique<Renderer>(*window, *vkContext);
}

void Application::mainLoop()
{
    while (!window->shouldClose())
    {
        window->pollEvents();
        renderer->drawFrame();
    }

    vkContext->getDevice().waitIdle();
}

int main()
{
    try
    {
        Application app;
        app.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}