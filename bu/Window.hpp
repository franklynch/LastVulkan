#pragma once

#include <cstdint>
#include <stdexcept>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

class Window
{
public:
    Window(uint32_t width, uint32_t height, const char* title);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    GLFWwindow* getHandle() const { return window; }

    bool shouldClose() const;
    void pollEvents() const;

    bool wasResized() const { return framebufferResized; }
    void resetResizedFlag() { framebufferResized = false; }

    void getFramebufferSize(int& width, int& height) const;

private:
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

private:
    GLFWwindow* window = nullptr;
    bool framebufferResized = false;
};