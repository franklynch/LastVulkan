#pragma once

#include <vector>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#   include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include "VulkanContext.hpp"
#include "Material.hpp"

class DescriptorManager
{
public:
    explicit DescriptorManager(VulkanContext& vkContext);

    ~DescriptorManager();

    

    void createLayouts();
    void cleanup();

    void createDescriptorPool(
        uint32_t maxFramesInFlight,
        uint32_t materialCount);

    

    void allocateFrameDescriptorSets(uint32_t maxFramesInFlight);
    void allocateIBLDescriptorSet();

    void updateFrameDescriptorSets(
        const std::vector<vk::raii::Buffer>& uniformBuffers,
        uint32_t maxFramesInFlight);

    void createMaterialDescriptorSets(
        const std::vector<std::unique_ptr<Material>>& materials);

    vk::DescriptorSetLayout frameLayout() const { return *m_frameDescriptorSetLayout; }
    vk::DescriptorSetLayout materialLayout() const { return *m_materialDescriptorSetLayout; }
    vk::DescriptorSetLayout iblLayout() const { return *m_iblDescriptorSetLayout; }

    const vk::raii::DescriptorSetLayout& frameLayoutRaii() const { return m_frameDescriptorSetLayout; }
    const vk::raii::DescriptorSetLayout& materialLayoutRaii() const { return m_materialDescriptorSetLayout; }
    const vk::raii::DescriptorSetLayout& iblLayoutRaii() const { return m_iblDescriptorSetLayout; }
    const std::vector<vk::raii::DescriptorSet>& materialDescriptorSets() const { return m_materialDescriptorSets; }

    std::vector<vk::raii::DescriptorSet>& materialDescriptorSets()
    {
        return m_materialDescriptorSets;
    }

    const vk::raii::DescriptorPool& descriptorPool() const
    {
        return m_descriptorPool;
    }
    const std::vector<vk::raii::DescriptorSet>& frameDescriptorSets() const
    {
        return m_frameDescriptorSets;
    }
    
    const vk::raii::DescriptorSet& iblDescriptorSet() const
    {
        return m_iblDescriptorSet;
    }

    vk::raii::DescriptorSet& iblDescriptorSet()
    {
        return m_iblDescriptorSet;
    }

    

private:
    VulkanContext& vkContext;

    vk::raii::DescriptorSetLayout m_frameDescriptorSetLayout = nullptr;
    vk::raii::DescriptorSetLayout m_materialDescriptorSetLayout = nullptr;
    vk::raii::DescriptorSetLayout m_iblDescriptorSetLayout = nullptr;

    vk::raii::DescriptorPool m_descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> m_frameDescriptorSets;
    vk::raii::DescriptorSet m_iblDescriptorSet = nullptr;

    std::vector<vk::raii::DescriptorSet> m_materialDescriptorSets;

    static constexpr uint32_t extraDescriptorSetHeadroom = 8;

    
};