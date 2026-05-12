#include "EditorUiSystem.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

EditorUiSystem::EditorUiSystem(
    Window& window,
    VulkanContext& vkContext)
    : window(window),
    vkContext(vkContext)
{
}

void EditorUiSystem::init(vk::Format swapchainFormat, uint32_t imageCount)
{
    auto& device = vkContext.getDevice();
    auto& physicalDevice = vkContext.getPhysicalDevice();
    auto& queue = vkContext.getQueue();

    std::array<vk::DescriptorPoolSize, 11> poolSizes = {
        vk::DescriptorPoolSize(vk::DescriptorType::eSampler, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageImage, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformTexelBuffer, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageTexelBuffer, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBufferDynamic, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBufferDynamic, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eInputAttachment, 1000)
    };

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo
        .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
        .setMaxSets(1000 * static_cast<uint32_t>(poolSizes.size()))
        .setPoolSizes(poolSizes);

    imguiDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window.getHandle(), true);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_3;
    initInfo.Instance = *vkContext.getInstance();
    initInfo.PhysicalDevice = *physicalDevice;
    initInfo.Device = *device;
    initInfo.QueueFamily = vkContext.getQueueIndex();
    initInfo.Queue = *queue;
    initInfo.DescriptorPool = *imguiDescriptorPool;
    initInfo.MinImageCount = imageCount;
    initInfo.ImageCount = imageCount;
    initInfo.UseDynamicRendering = true;
    initInfo.CheckVkResultFn = nullptr;

    VkFormat colorFormat =
        static_cast<VkFormat>(swapchainFormat);

    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;



    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = {};
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &colorFormat;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.depthAttachmentFormat =
        VK_FORMAT_UNDEFINED;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.stencilAttachmentFormat =
        VK_FORMAT_UNDEFINED;


    ImGui_ImplVulkan_Init(&initInfo);

    imguiInitialized = true;
}

void EditorUiSystem::shutdown()
{
    if (!imguiInitialized)
        return;

    vkContext.getDevice().waitIdle();

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    imguiDescriptorPool = nullptr;
    imguiInitialized = false;
}

void EditorUiSystem::beginFrame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void EditorUiSystem::render(vk::CommandBuffer commandBuffer)
{

    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
}

void EditorUiSystem::buildMinimal(
    float frameTimeMs,
    float fps)
{
    ImGui::Begin("Renderer");

    ImGui::Text("Frame time: %.3f ms", frameTimeMs);
    ImGui::Text("FPS: %.1f", fps);

    ImGui::Separator();

    ImGui::Checkbox("Wireframe", &uiState.wireframeRequested);
    

    ImGui::End();

    if (uiState.showDemoWindow)
    {
        ImGui::ShowDemoWindow(&uiState.showDemoWindow);
    }
}



