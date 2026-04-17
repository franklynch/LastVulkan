#include "Window.hpp"

Window::Window(uint32_t width, uint32_t height, const char* title)
{
    if (!glfwInit())
    {
        throw std::runtime_error("failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(
        static_cast<int>(width),
        static_cast<int>(height),
        title,
        nullptr,
        nullptr);

    if (!window)
    {
        glfwTerminate();
        throw std::runtime_error("failed to create GLFW window");
    }

    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, Window::framebufferResizeCallback);
}

Window::~Window()
{
    if (window)
    {
        glfwDestroyWindow(window);
        window = nullptr;
    }

    glfwTerminate();
}

bool Window::shouldClose() const
{
    return glfwWindowShouldClose(window);
}

void Window::pollEvents() const
{
    glfwPollEvents();
}

void Window::getFramebufferSize(int& width, int& height) const
{
    glfwGetFramebufferSize(window, &width, &height);
}

void Window::framebufferResizeCallback(GLFWwindow* glfwWindow, int, int)
{
    auto* window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(glfwWindow));
    if (window)
    {
        window->framebufferResized = true;
    }
}