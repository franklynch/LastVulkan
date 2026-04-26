#include "EnvironmentUtils.hpp"

#include <glm/gtc/matrix_transform.hpp>

void createCubemapResource(
    VulkanContext& vkContext,
    BufferUtils& bufferUtils,
    Cubemap& cubemap,
    uint32_t size,
    uint32_t mipLevels,
    vk::Format format,
    vk::ImageUsageFlags usage)
{
    auto& device = vkContext.getDevice();

    vk::ImageCreateInfo imageInfo{};
    imageInfo
        .setFlags(vk::ImageCreateFlagBits::eCubeCompatible)
        .setImageType(vk::ImageType::e2D)
        .setFormat(format)
        .setExtent(vk::Extent3D{ size, size, 1 })
        .setMipLevels(mipLevels)
        .setArrayLayers(6)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setTiling(vk::ImageTiling::eOptimal)
        .setUsage(usage)
        .setSharingMode(vk::SharingMode::eExclusive)
        .setInitialLayout(vk::ImageLayout::eUndefined);

    cubemap.image = vk::raii::Image(device, imageInfo);

    vk::MemoryRequirements memRequirements =
        cubemap.image.getMemoryRequirements();

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo
        .setAllocationSize(memRequirements.size)
        .setMemoryTypeIndex(
            bufferUtils.findMemoryType(
                memRequirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal));

    cubemap.memory = vk::raii::DeviceMemory(device, allocInfo);
    cubemap.image.bindMemory(*cubemap.memory, 0);

    vk::ImageViewCreateInfo viewInfo{};
    viewInfo
        .setImage(*cubemap.image)
        .setViewType(vk::ImageViewType::eCube)
        .setFormat(format)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(mipLevels)
            .setBaseArrayLayer(0)
            .setLayerCount(6));

    cubemap.view = vk::raii::ImageView(device, viewInfo);

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo
        .setMagFilter(vk::Filter::eLinear)
        .setMinFilter(vk::Filter::eLinear)
        .setMipmapMode(vk::SamplerMipmapMode::eLinear)
        .setAddressModeU(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeW(vk::SamplerAddressMode::eClampToEdge)
        .setMinLod(0.0f)
        .setMaxLod(static_cast<float>(mipLevels - 1));

    cubemap.sampler = vk::raii::Sampler(device, samplerInfo);

    std::cout << "createCubemapResource size=" << size
        << " mips=" << mipLevels << "\n";
}


std::array<glm::mat4, 6> getCubemapCaptureViews()
{
    return {
        glm::lookAt(glm::vec3(0.0f), glm::vec3(1,  0,  0), glm::vec3(0, -1,  0)),
        glm::lookAt(glm::vec3(0.0f), glm::vec3(-1,  0,  0), glm::vec3(0, -1,  0)),
        glm::lookAt(glm::vec3(0.0f), glm::vec3(0,  1,  0), glm::vec3(0,  0,  1)),
        glm::lookAt(glm::vec3(0.0f), glm::vec3(0, -1,  0), glm::vec3(0,  0, -1)),
        glm::lookAt(glm::vec3(0.0f), glm::vec3(0,  0,  1), glm::vec3(0, -1,  0)),
        glm::lookAt(glm::vec3(0.0f), glm::vec3(0,  0, -1), glm::vec3(0, -1,  0))
    };
}

glm::mat4 getCubemapCaptureProjection()
{
    glm::mat4 proj = glm::perspective(
        glm::radians(90.0f),
        1.0f,
        0.1f,
        10.0f);

    proj[1][1] *= -1.0f;
    return proj;
}
