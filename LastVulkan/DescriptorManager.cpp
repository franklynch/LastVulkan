#include "DescriptorManager.hpp"

DescriptorManager::DescriptorManager(VulkanContext& vkContext)
    : vkContext(vkContext)
{
}

void DescriptorManager::cleanupLayouts()
{
    m_iblDescriptorSetLayout = nullptr;
    m_materialDescriptorSetLayout = nullptr;
    m_frameDescriptorSetLayout = nullptr;
}



void DescriptorManager::createLayouts()
{
    auto& device = vkContext.getDevice();

    // Set 0: per-frame UBO
    {
        vk::DescriptorSetLayoutBinding uboBinding{};
        uboBinding
            .setBinding(0)
            .setDescriptorType(vk::DescriptorType::eUniformBuffer)
            .setDescriptorCount(1)
            .setStageFlags(
                vk::ShaderStageFlagBits::eVertex |
                vk::ShaderStageFlagBits::eFragment);

        vk::DescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.setBindings(uboBinding);

        m_frameDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
    }

    // Set 1: per-material textures
    {
        vk::DescriptorSetLayoutBinding baseColorBinding{};
        baseColorBinding
            .setBinding(0)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment);

        vk::DescriptorSetLayoutBinding normalBinding{};
        normalBinding
            .setBinding(1)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment);

        vk::DescriptorSetLayoutBinding metallicRoughnessBinding{};
        metallicRoughnessBinding
            .setBinding(2)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment);

        vk::DescriptorSetLayoutBinding aoBinding{};
        aoBinding
            .setBinding(3)
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment);

        vk::DescriptorSetLayoutBinding emissiveBinding{};
        emissiveBinding
            .setBinding(4)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment);

        std::array<vk::DescriptorSetLayoutBinding, 5> bindings = {
            baseColorBinding,
            normalBinding,
            metallicRoughnessBinding,
            aoBinding,
            emissiveBinding
        };



        vk::DescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.setBindings(bindings);

        m_materialDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
    }

    // Set 2: IBL textures
    {
        vk::DescriptorSetLayoutBinding irradianceBinding{};
        irradianceBinding
            .setBinding(0)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment);

        vk::DescriptorSetLayoutBinding prefilteredBinding{};
        prefilteredBinding
            .setBinding(1)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment);

        vk::DescriptorSetLayoutBinding brdfLutBinding{};
        brdfLutBinding
            .setBinding(2)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment);

        vk::DescriptorSetLayoutBinding environmentBinding{};
        environmentBinding
            .setBinding(3)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment);

        std::array<vk::DescriptorSetLayoutBinding, 4> bindings = {
            irradianceBinding,
            prefilteredBinding,
            brdfLutBinding,
            environmentBinding
        };

        vk::DescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.setBindings(bindings);

        m_iblDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);



    }
}


void DescriptorManager::createDescriptorPool(
    uint32_t maxFramesInFlight,
    uint32_t materialCount)
{
    
    if (maxFramesInFlight == 0)
    {
        throw std::runtime_error("createDescriptorPool: maxFramesInFlight is zero");
    }
    
    auto& device = vkContext.getDevice();
    
    const uint32_t materialSamplerCount =
        materialCount > 0 ? materialCount * 5 : 1;

    std::array<vk::DescriptorPoolSize, 4> poolSizes{};

    poolSizes[0]
        .setType(vk::DescriptorType::eUniformBuffer)
        .setDescriptorCount(maxFramesInFlight);

    poolSizes[1]
        .setType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(materialSamplerCount);

    poolSizes[2]
        .setType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(4);

    poolSizes[3]
        .setType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(16);

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo
        .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
        .setPoolSizes(poolSizes)
        .setMaxSets(maxFramesInFlight + std::max<uint32_t>(1, materialCount) + 1 + 8);

    m_descriptorPool =
        vk::raii::DescriptorPool(device, poolInfo);
    
}

void DescriptorManager::allocateFrameDescriptorSets(
    uint32_t maxFramesInFlight)
{
    
    if (m_descriptorPool == nullptr)
    {
        throw std::runtime_error("allocateFrameDescriptorSets: descriptor pool is null");
    }

    if (maxFramesInFlight == 0)
    {
        throw std::runtime_error("allocateFrameDescriptorSets: maxFramesInFlight is zero");
    }
    
    auto& device = vkContext.getDevice();

    std::vector<vk::DescriptorSetLayout> layouts(
        maxFramesInFlight,
        frameLayout());

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo
        .setDescriptorPool(*m_descriptorPool)
        .setSetLayouts(layouts);

    m_frameDescriptorSets =
        device.allocateDescriptorSets(allocInfo);
}

void DescriptorManager::allocateIBLDescriptorSet()
{
    auto& device = vkContext.getDevice();

    std::array<vk::DescriptorSetLayout, 1> layouts = {
        iblLayout()
    };

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo
        .setDescriptorPool(*m_descriptorPool)
        .setSetLayouts(layouts);

    auto sets = device.allocateDescriptorSets(allocInfo);

    m_iblDescriptorSet = std::move(sets[0]);
}

void DescriptorManager::createFrameDescriptorSets(
    uint32_t maxFramesInFlight)
{
    std::vector<vk::DescriptorSetLayout> layouts(
        maxFramesInFlight,
        frameLayout());

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo
        .setDescriptorPool(*m_descriptorPool)
        .setSetLayouts(layouts);

    m_frameDescriptorSets =
        vkContext.getDevice().allocateDescriptorSets(allocInfo);
}

void DescriptorManager::createIBLDescriptorSet()
{
    std::array<vk::DescriptorSetLayout, 1> layouts = {
        iblLayout()
    };

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo
        .setDescriptorPool(*m_descriptorPool)
        .setSetLayouts(layouts);

    auto sets =
        vkContext.getDevice().allocateDescriptorSets(allocInfo);

    m_iblDescriptorSet = std::move(sets.front());
}


