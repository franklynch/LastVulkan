#include  "EnvironmentSystem.hpp"  

#include <stb_image.h>

EnvironmentSystem::EnvironmentSystem(VulkanContext& vkContext, BufferUtils& bufferUtils, ImageUtils& imageUtils)
    : vkContext(vkContext)
    , bufferUtils(bufferUtils)
    , imageUtils(imageUtils)
{

}

void EnvironmentSystem::createFallbackResources()

{   
    createFallbackBrdfLut();
    createFallbackBlackCube();
}

void EnvironmentSystem::createFallbackBrdfLut()
{
    const unsigned char blackPixel[4] = { 0, 0, 0, 255 };

    Texture2D::SamplerOptions samplerOptions{};
    samplerOptions.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerOptions.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    samplerOptions.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    samplerOptions.enableAnisotropy = false;
    samplerOptions.maxLod = 0.0f;

    fallbackBrdfLut = std::make_unique<Texture2D>(
        vkContext,
        bufferUtils,
        imageUtils,
        blackPixel,
        1,
        1,
        4,
        "Fallback BRDF LUT",
        vk::Format::eR8G8B8A8Unorm,
        samplerOptions
    );
}

void EnvironmentSystem::createFallbackBlackCube()
{
    auto& device = vkContext.getDevice();

    const vk::Format format = vk::Format::eR8G8B8A8Unorm;

    const std::array<unsigned char, 24> blackFaces = {
        0, 0, 0, 255,
        0, 0, 0, 255,
        0, 0, 0, 255,
        0, 0, 0, 255,
        0, 0, 0, 255,
        0, 0, 0, 255
    };

    vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(blackFaces.size());

    vk::raii::Buffer stagingBuffer{ nullptr };
    vk::raii::DeviceMemory stagingMemory{ nullptr };

    bufferUtils.createBuffer(
        imageSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer,
        stagingMemory
    );

    void* mapped = stagingMemory.mapMemory(0, imageSize);
    std::memcpy(mapped, blackFaces.data(), static_cast<size_t>(imageSize));
    stagingMemory.unmapMemory();

    vk::ImageCreateInfo imageInfo{};
    imageInfo
        .setFlags(vk::ImageCreateFlagBits::eCubeCompatible)
        .setImageType(vk::ImageType::e2D)
        .setFormat(format)
        .setExtent(vk::Extent3D{ 1, 1, 1 })
        .setMipLevels(1)
        .setArrayLayers(6)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setTiling(vk::ImageTiling::eOptimal)
        .setUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled)
        .setSharingMode(vk::SharingMode::eExclusive)
        .setInitialLayout(vk::ImageLayout::eUndefined);

    fallbackBlackCubeImage = vk::raii::Image(device, imageInfo);

    vk::MemoryRequirements memReq = fallbackBlackCubeImage.getMemoryRequirements();

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo
        .setAllocationSize(memReq.size)
        .setMemoryTypeIndex(
            bufferUtils.findMemoryType(
                memReq.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal
            )
        );

    fallbackBlackCubeMemory = vk::raii::DeviceMemory(device, allocInfo);
    fallbackBlackCubeImage.bindMemory(*fallbackBlackCubeMemory, 0);

    auto cmd = bufferUtils.beginSingleTimeCommands();

    vk::ImageMemoryBarrier toTransfer{};
    toTransfer
        .setOldLayout(vk::ImageLayout::eUndefined)
        .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*fallbackBlackCubeImage)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(6))
        .setSrcAccessMask({})
        .setDstAccessMask(vk::AccessFlagBits::eTransferWrite);

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eTransfer,
        {},
        nullptr,
        nullptr,
        toTransfer
    );

    std::array<vk::BufferImageCopy, 6> copyRegions{};
    for (uint32_t face = 0; face < 6; ++face)
    {
        copyRegions[face]
            .setBufferOffset(face * 4)
            .setBufferRowLength(0)
            .setBufferImageHeight(0)
            .setImageSubresource(
                vk::ImageSubresourceLayers{}
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setMipLevel(0)
                .setBaseArrayLayer(face)
                .setLayerCount(1))
            .setImageOffset(vk::Offset3D{ 0, 0, 0 })
            .setImageExtent(vk::Extent3D{ 1, 1, 1 });
    }

    cmd.copyBufferToImage(
        *stagingBuffer,
        *fallbackBlackCubeImage,
        vk::ImageLayout::eTransferDstOptimal,
        copyRegions
    );

    vk::ImageMemoryBarrier toShaderRead{};
    toShaderRead
        .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
        .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*fallbackBlackCubeImage)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(6))
        .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader,
        {},
        nullptr,
        nullptr,
        toShaderRead
    );

    bufferUtils.endSingleTimeCommands(cmd);

    vk::ImageViewCreateInfo viewInfo{};
    viewInfo
        .setImage(*fallbackBlackCubeImage)
        .setViewType(vk::ImageViewType::eCube)
        .setFormat(format)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(6));

    fallbackBlackCubeView = vk::raii::ImageView(device, viewInfo);

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo
        .setMagFilter(vk::Filter::eLinear)
        .setMinFilter(vk::Filter::eLinear)
        .setMipmapMode(vk::SamplerMipmapMode::eLinear)
        .setAddressModeU(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeW(vk::SamplerAddressMode::eClampToEdge)
        .setAnisotropyEnable(VK_FALSE)
        .setMaxAnisotropy(1.0f)
        .setMinLod(0.0f)
        .setMaxLod(0.0f)
        .setBorderColor(vk::BorderColor::eIntOpaqueBlack)
        .setUnnormalizedCoordinates(VK_FALSE);

    fallbackBlackCubeSampler = vk::raii::Sampler(device, samplerInfo);
}

vk::DescriptorImageInfo EnvironmentSystem::makeImageInfo(
    vk::Sampler sampler,
    vk::ImageView view) const
{
    return vk::DescriptorImageInfo{}
        .setSampler(sampler)
        .setImageView(view)
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
}

void EnvironmentSystem::updateIBLDescriptorSet(
    vk::raii::DescriptorSet& iblDescriptorSet,
    vk::Sampler fallbackEnvironmentSampler,
    vk::ImageView fallbackEnvironmentView)
{
    if (iblDescriptorSet == nullptr)
    {
        throw std::runtime_error("updateIBLDescriptorSet: iblDescriptorSet is null");
    }

    auto* fallbackBrdf = fallbackBrdfLut.get();

    const bool hasFallbackBrdf =
        fallbackBrdf &&
        fallbackBrdf->getSampler() != nullptr &&
        fallbackBrdf->getImageView() != nullptr;

    const bool hasRuntimeBrdf =
        environment.runtimeBrdfLut.sampler != nullptr &&
        environment.runtimeBrdfLut.view != nullptr;

    if (!hasRuntimeBrdf && !hasFallbackBrdf)
    {
        throw std::runtime_error(
            "updateIBLDescriptorSet: no BRDF LUT available");
    }

    auto& device = vkContext.getDevice();

    const bool useRuntimeIrradiance =
        environment.runtimeIrradianceCube.sampler != nullptr &&
        environment.runtimeIrradianceCube.view != nullptr;

    const bool useRuntimePrefiltered =
        environment.runtimePrefilteredCube.sampler != nullptr &&
        environment.runtimePrefilteredCube.view != nullptr;

    const bool useRuntimeBrdf =
        environment.runtimeBrdfLut.sampler != nullptr &&
        environment.runtimeBrdfLut.view != nullptr;

    const bool useRuntimeEnvironment =
        environment.runtimeEnvironmentCube.sampler != nullptr &&
        environment.runtimeEnvironmentCube.view != nullptr;

    auto irradianceInfo = useRuntimeIrradiance
        ? makeImageInfo(
            *environment.runtimeIrradianceCube.sampler,
            *environment.runtimeIrradianceCube.view)
        : makeImageInfo(
            getFallbackBlackCubeSampler(),
            getFallbackBlackCubeView());

    auto prefilteredInfo = useRuntimePrefiltered
        ? makeImageInfo(
            *environment.runtimePrefilteredCube.sampler,
            *environment.runtimePrefilteredCube.view)
        : makeImageInfo(
            getFallbackBlackCubeSampler(),
            getFallbackBlackCubeView());

    auto brdfInfo = useRuntimeBrdf
        ? makeImageInfo(
            *environment.runtimeBrdfLut.sampler,
            *environment.runtimeBrdfLut.view)
        : makeImageInfo(
            fallbackBrdf->getSampler(),
            fallbackBrdf->getImageView());

    auto environmentInfo = useRuntimeEnvironment
        ? makeImageInfo(
            *environment.runtimeEnvironmentCube.sampler,
            *environment.runtimeEnvironmentCube.view)
        : makeImageInfo(
            fallbackEnvironmentSampler,
            fallbackEnvironmentView);

    std::array<vk::WriteDescriptorSet, 4> writes{};

    writes[0]
        .setDstSet(*iblDescriptorSet)
        .setDstBinding(0)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(irradianceInfo);

    writes[1]
        .setDstSet(*iblDescriptorSet)
        .setDstBinding(1)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(prefilteredInfo);

    writes[2]
        .setDstSet(*iblDescriptorSet)
        .setDstBinding(2)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(brdfInfo);

    writes[3]
        .setDstSet(*iblDescriptorSet)
        .setDstBinding(3)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(1)
        .setImageInfo(environmentInfo);

    device.updateDescriptorSets(writes, {});
}

uint32_t EnvironmentSystem::getDebugRuntimePrefilteredMipLevels() const
{
    if (!prefilterRenderer)
    {
        return 1;
    }

    return prefilterRenderer->getDebugRuntimePrefilteredMipLevels();
}

void EnvironmentSystem::createFallbackEnvironmentCubemap(const std::array<std::string, 6>& facePaths)
{
    auto& device = vkContext.getDevice();

    int texWidth = 0;
    int texHeight = 0;
    int texChannels = 0;

    std::vector<stbi_uc*> facePixels(6, nullptr);

    for (size_t i = 0; i < 6; ++i)
    {
        int w = 0, h = 0, c = 0;
        facePixels[i] = stbi_load(facePaths[i].c_str(), &w, &h, &c, STBI_rgb_alpha);
        if (!facePixels[i])
        {
            throw std::runtime_error("Failed to load cubemap face: " + facePaths[i]);
        }

        if (i == 0)
        {
            texWidth = w;
            texHeight = h;
            texChannels = 4;
        }
        else
        {
            if (w != texWidth || h != texHeight)
            {
                throw std::runtime_error("Cubemap faces must all have the same dimensions");
            }
        }
    }

    const vk::DeviceSize faceSize =
        static_cast<vk::DeviceSize>(texWidth) *
        static_cast<vk::DeviceSize>(texHeight) * 4;

    const vk::DeviceSize totalSize = faceSize * 6;

    vk::raii::Buffer stagingBuffer{ nullptr };
    vk::raii::DeviceMemory stagingMemory{ nullptr };

    bufferUtils.createBuffer(
        totalSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer,
        stagingMemory
    );

    void* mapped = stagingMemory.mapMemory(0, totalSize);
    unsigned char* dst = static_cast<unsigned char*>(mapped);

    for (size_t i = 0; i < 6; ++i)
    {
        std::memcpy(dst + i * faceSize, facePixels[i], static_cast<size_t>(faceSize));
    }

    stagingMemory.unmapMemory();

    for (auto* pixels : facePixels)
    {
        stbi_image_free(pixels);
    }

    const vk::Format format = vk::Format::eR8G8B8A8Srgb;

    vk::ImageCreateInfo imageInfo{};
    imageInfo
        .setFlags(vk::ImageCreateFlagBits::eCubeCompatible)
        .setImageType(vk::ImageType::e2D)
        .setFormat(format)
        .setExtent(vk::Extent3D{
            static_cast<uint32_t>(texWidth),
            static_cast<uint32_t>(texHeight),
            1
            })
        .setMipLevels(1)
        .setArrayLayers(6)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setTiling(vk::ImageTiling::eOptimal)
        .setUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled)
        .setSharingMode(vk::SharingMode::eExclusive)
        .setInitialLayout(vk::ImageLayout::eUndefined);

    fallbackEnvironmentCubeImage = vk::raii::Image(device, imageInfo);

    vk::MemoryRequirements memReq = fallbackEnvironmentCubeImage.getMemoryRequirements();

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo
        .setAllocationSize(memReq.size)
        .setMemoryTypeIndex(
            bufferUtils.findMemoryType(
                memReq.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal
            )
        );

    fallbackEnvironmentCubeMemory = vk::raii::DeviceMemory(device, allocInfo);
    fallbackEnvironmentCubeImage.bindMemory(*fallbackEnvironmentCubeMemory, 0);

    auto cmd = bufferUtils.beginSingleTimeCommands();

    vk::ImageMemoryBarrier toTransfer{};
    toTransfer
        .setOldLayout(vk::ImageLayout::eUndefined)
        .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*fallbackEnvironmentCubeImage)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(6))
        .setSrcAccessMask({})
        .setDstAccessMask(vk::AccessFlagBits::eTransferWrite);

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eTransfer,
        {},
        nullptr,
        nullptr,
        toTransfer
    );

    std::array<vk::BufferImageCopy, 6> copyRegions{};
    for (uint32_t face = 0; face < 6; ++face)
    {
        copyRegions[face]
            .setBufferOffset(face * faceSize)
            .setBufferRowLength(0)
            .setBufferImageHeight(0)
            .setImageSubresource(
                vk::ImageSubresourceLayers{}
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setMipLevel(0)
                .setBaseArrayLayer(face)
                .setLayerCount(1))
            .setImageOffset(vk::Offset3D{ 0, 0, 0 })
            .setImageExtent(vk::Extent3D{
                static_cast<uint32_t>(texWidth),
                static_cast<uint32_t>(texHeight),
                1
                });
    }

    cmd.copyBufferToImage(
        *stagingBuffer,
        *fallbackEnvironmentCubeImage,
        vk::ImageLayout::eTransferDstOptimal,
        copyRegions
    );

    vk::ImageMemoryBarrier toShaderRead{};
    toShaderRead
        .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
        .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*fallbackEnvironmentCubeImage)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(6))
        .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader,
        {},
        nullptr,
        nullptr,
        toShaderRead
    );

    bufferUtils.endSingleTimeCommands(cmd);

    vk::ImageViewCreateInfo viewInfo{};
    viewInfo
        .setImage(*fallbackEnvironmentCubeImage)
        .setViewType(vk::ImageViewType::eCube)
        .setFormat(format)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(6));

    fallbackEnvironmentCubeView = vk::raii::ImageView(device, viewInfo);

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo
        .setMagFilter(vk::Filter::eLinear)
        .setMinFilter(vk::Filter::eLinear)
        .setMipmapMode(vk::SamplerMipmapMode::eLinear)
        .setAddressModeU(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeW(vk::SamplerAddressMode::eClampToEdge)
        .setAnisotropyEnable(VK_FALSE)
        .setMaxAnisotropy(1.0f)
        .setMinLod(0.0f)
        .setMaxLod(0.0f)
        .setBorderColor(vk::BorderColor::eIntOpaqueBlack)
        .setUnnormalizedCoordinates(VK_FALSE);

    fallbackEnvironmentCubeSampler = vk::raii::Sampler(device, samplerInfo);
}

void EnvironmentSystem::cleanup()
{
    environment.runtimeBrdfLut.pipeline = nullptr;
    environment.runtimeBrdfLut.layout = nullptr;
    environment.runtimeBrdfLut.sampler = nullptr;
    environment.runtimeBrdfLut.view = nullptr;
    environment.runtimeBrdfLut.memory = nullptr;
    environment.runtimeBrdfLut.image = nullptr;

    for (auto& view : environment.runtimeEnvironmentCubeFaces.views)
    {
        view = nullptr;
    }

    environment.runtimeEnvironmentCube.sampler = nullptr;
    environment.runtimeEnvironmentCube.view = nullptr;
    environment.runtimeEnvironmentCube.memory = nullptr;
    environment.runtimeEnvironmentCube.image = nullptr;

    for (auto& mipViews : environment.runtimePrefilteredCubeMipFaceViews)
    {
        for (auto& view : mipViews.views)
        {
            view = nullptr;
        }
    }

    environment.runtimePrefilteredCubeMipFaceViews.clear();

    environment.runtimePrefilteredCube.sampler = nullptr;
    environment.runtimePrefilteredCube.view = nullptr;
    environment.runtimePrefilteredCube.memory = nullptr;
    environment.runtimePrefilteredCube.image = nullptr;

    for (auto& view : environment.runtimeIrradianceCubeFaces.views)
    {
        view = nullptr;
    }

    environment.runtimeIrradianceCube.sampler = nullptr;
    environment.runtimeIrradianceCube.view = nullptr;
    environment.runtimeIrradianceCube.memory = nullptr;
    environment.runtimeIrradianceCube.image = nullptr;

    fallbackBrdfLut.reset();

    fallbackBlackCubeSampler = nullptr;
    fallbackBlackCubeView = nullptr;
    fallbackBlackCubeMemory = nullptr;
    fallbackBlackCubeImage = nullptr;

    hdrEnvironmentSampler = nullptr;
    hdrEnvironmentView = nullptr;
    hdrEnvironmentMemory = nullptr;
    hdrEnvironmentImage = nullptr;

    fallbackEnvironmentCubeSampler = nullptr;
    fallbackEnvironmentCubeView = nullptr;
    fallbackEnvironmentCubeMemory = nullptr;
    fallbackEnvironmentCubeImage = nullptr;

}

void EnvironmentSystem::createHdrEnvironmentTexture(const std::string& path)
{
    int width = 0;
    int height = 0;
    int channels = 0;

    float* pixels = stbi_loadf(path.c_str(), &width, &height, &channels, 4);

    if (!pixels)
    {
        throw std::runtime_error("Failed to load HDR environment: " + path);
    }

    hdrEnvironmentWidth = static_cast<uint32_t>(width);
    hdrEnvironmentHeight = static_cast<uint32_t>(height);

    const vk::DeviceSize imageSize =
        static_cast<vk::DeviceSize>(width) *
        static_cast<vk::DeviceSize>(height) *
        4 *
        sizeof(float);

    vk::raii::Buffer stagingBuffer{ nullptr };
    vk::raii::DeviceMemory stagingMemory{ nullptr };

    bufferUtils.createBuffer(
        imageSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer,
        stagingMemory);

    {
        void* mapped = stagingMemory.mapMemory(0, imageSize);
        std::memcpy(mapped, pixels, static_cast<size_t>(imageSize));
        stagingMemory.unmapMemory();
    }

    stbi_image_free(pixels);

    auto& device = vkContext.getDevice();

    const vk::Format hdrFormat = vk::Format::eR32G32B32A32Sfloat;

    vk::ImageCreateInfo imageInfo{};
    imageInfo
        .setImageType(vk::ImageType::e2D)
        .setFormat(hdrFormat)
        .setExtent(vk::Extent3D{
            hdrEnvironmentWidth,
            hdrEnvironmentHeight,
            1 })
            .setMipLevels(1)
        .setArrayLayers(1)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setTiling(vk::ImageTiling::eOptimal)
        .setUsage(
            vk::ImageUsageFlagBits::eTransferDst |
            vk::ImageUsageFlagBits::eSampled)
        .setSharingMode(vk::SharingMode::eExclusive)
        .setInitialLayout(vk::ImageLayout::eUndefined);

    hdrEnvironmentImage = vk::raii::Image(device, imageInfo);

    vk::MemoryRequirements memRequirements =
        hdrEnvironmentImage.getMemoryRequirements();

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo
        .setAllocationSize(memRequirements.size)
        .setMemoryTypeIndex(
            bufferUtils.findMemoryType(
                memRequirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal));

    hdrEnvironmentMemory = vk::raii::DeviceMemory(device, allocInfo);
    hdrEnvironmentImage.bindMemory(*hdrEnvironmentMemory, 0);

    auto cmd = bufferUtils.beginSingleTimeCommands();

    vk::ImageMemoryBarrier toTransfer{};
    toTransfer
        .setOldLayout(vk::ImageLayout::eUndefined)
        .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*hdrEnvironmentImage)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(1))
        .setSrcAccessMask({})
        .setDstAccessMask(vk::AccessFlagBits::eTransferWrite);

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eTransfer,
        {},
        nullptr,
        nullptr,
        toTransfer);

    vk::BufferImageCopy copyRegion{};
    copyRegion
        .setBufferOffset(0)
        .setBufferRowLength(0)
        .setBufferImageHeight(0)
        .setImageSubresource(
            vk::ImageSubresourceLayers{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setMipLevel(0)
            .setBaseArrayLayer(0)
            .setLayerCount(1))
        .setImageOffset(vk::Offset3D{ 0, 0, 0 })
        .setImageExtent(vk::Extent3D{
            hdrEnvironmentWidth,
            hdrEnvironmentHeight,
            1 });

    cmd.copyBufferToImage(
        *stagingBuffer,
        *hdrEnvironmentImage,
        vk::ImageLayout::eTransferDstOptimal,
        copyRegion);

    vk::ImageMemoryBarrier toShaderRead{};
    toShaderRead
        .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
        .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(*hdrEnvironmentImage)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(1))
        .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader,
        {},
        nullptr,
        nullptr,
        toShaderRead);

    bufferUtils.endSingleTimeCommands(cmd);

    vk::ImageViewCreateInfo viewInfo{};
    viewInfo
        .setImage(*hdrEnvironmentImage)
        .setViewType(vk::ImageViewType::e2D)
        .setFormat(hdrFormat)
        .setSubresourceRange(
            vk::ImageSubresourceRange{}
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(1));

    hdrEnvironmentView = vk::raii::ImageView(device, viewInfo);

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo
        .setMagFilter(vk::Filter::eLinear)
        .setMinFilter(vk::Filter::eLinear)
        .setMipmapMode(vk::SamplerMipmapMode::eLinear)
        .setAddressModeU(vk::SamplerAddressMode::eRepeat)
        .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeW(vk::SamplerAddressMode::eClampToEdge)
        .setMipLodBias(0.0f)
        .setAnisotropyEnable(VK_FALSE)
        .setCompareEnable(VK_FALSE)
        .setMinLod(0.0f)
        .setMaxLod(0.0f)
        .setBorderColor(vk::BorderColor::eFloatOpaqueWhite)
        .setUnnormalizedCoordinates(VK_FALSE);

    hdrEnvironmentSampler = vk::raii::Sampler(device, samplerInfo);

    std::cout << "Loaded HDR environment: "
        << path << " "
        << hdrEnvironmentWidth << "x"
        << hdrEnvironmentHeight << "\n";
}

void EnvironmentSystem::initRenderers()
{
    brdfLutRenderer =
        std::make_unique<BrdfLutRenderer>(vkContext, bufferUtils);

    brdfLutRenderer->init(environment);

    environmentRenderer =
        std::make_unique<EnvironmentRenderer>(vkContext, bufferUtils);

    environmentRenderer->init(
        environment,
        hdrEnvironmentSampler,
        hdrEnvironmentView);

    irradianceRenderer =
        std::make_unique<IrradianceRenderer>(vkContext, bufferUtils);

    irradianceRenderer->init(environment);

    prefilterRenderer =
        std::make_unique<PrefilterRenderer>(vkContext, bufferUtils);

    prefilterRenderer->init(environment);
}

void EnvironmentSystem::generateRuntimeEnvironmentCubemap()
{
    if (!environmentRenderer)
    {
        throw std::runtime_error("EnvironmentRenderer not initialized");
    }

    environmentRenderer->render(environment);
}

void EnvironmentSystem::generateRuntimeIrradianceCubemap()
{

    if (!irradianceRenderer)
    {
        throw std::runtime_error("IrradianceRenderer not initialized");
    }
	irradianceRenderer->render(environment);
}

void EnvironmentSystem::generateRuntimePrefilteredCubemap()
{
    if (!prefilterRenderer)
    {
        throw std::runtime_error("PrefilterRenderer not initialized");
    }
	prefilterRenderer->render(environment);
}

void EnvironmentSystem::loadHdrEnvironment(
    const std::string& path,
    vk::raii::DescriptorSet& iblDescriptorSet)
{
    createHdrEnvironmentTexture(path);

    initRenderers();

    generateRuntimeEnvironmentCubemap();
    generateRuntimeIrradianceCubemap();
    generateRuntimePrefilteredCubemap();

    updateIBLDescriptorSet(
        iblDescriptorSet,
        fallbackEnvironmentSampler(),
        fallbackEnvironmentView());
}

