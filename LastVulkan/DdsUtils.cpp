#include "DdsUtils.hpp"

#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>

#include <DirectXTex.h>

namespace
{
    vk::Format dxgiToVkFormat(DXGI_FORMAT format)
    {
        switch (format)
        {
        case DXGI_FORMAT_R8G8B8A8_UNORM:
            return vk::Format::eR8G8B8A8Unorm;

        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
            return vk::Format::eR8G8B8A8Srgb;

        case DXGI_FORMAT_B8G8R8A8_UNORM:
            return vk::Format::eB8G8R8A8Unorm;

        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
            return vk::Format::eB8G8R8A8Srgb;

        case DXGI_FORMAT_R16G16B16A16_FLOAT:
            return vk::Format::eR16G16B16A16Sfloat;

        case DXGI_FORMAT_R32G32B32A32_FLOAT:
            return vk::Format::eR32G32B32A32Sfloat;

        case DXGI_FORMAT_BC1_UNORM:
            return vk::Format::eBc1RgbaUnormBlock;

        case DXGI_FORMAT_BC1_UNORM_SRGB:
            return vk::Format::eBc1RgbaSrgbBlock;

        case DXGI_FORMAT_BC3_UNORM:
            return vk::Format::eBc3UnormBlock;

        case DXGI_FORMAT_BC3_UNORM_SRGB:
            return vk::Format::eBc3SrgbBlock;

        case DXGI_FORMAT_BC6H_UF16:
            return vk::Format::eBc6HUfloatBlock;

        case DXGI_FORMAT_BC6H_SF16:
            return vk::Format::eBc6HSfloatBlock;

        case DXGI_FORMAT_BC7_UNORM:
            return vk::Format::eBc7UnormBlock;

        case DXGI_FORMAT_BC7_UNORM_SRGB:
            return vk::Format::eBc7SrgbBlock;

        default:
            return vk::Format::eUndefined;
        }
    }

    bool isCompressedDxgiFormat(DXGI_FORMAT format)
    {
        switch (format)
        {
        case DXGI_FORMAT_BC1_UNORM:
        case DXGI_FORMAT_BC1_UNORM_SRGB:
        case DXGI_FORMAT_BC3_UNORM:
        case DXGI_FORMAT_BC3_UNORM_SRGB:
        case DXGI_FORMAT_BC6H_UF16:
        case DXGI_FORMAT_BC6H_SF16:
        case DXGI_FORMAT_BC7_UNORM:
        case DXGI_FORMAT_BC7_UNORM_SRGB:
            return true;

        default:
            return false;
        }
    }
}

DdsCubemapData DdsUtils::loadCubemapDDS(const std::string& path)
{
    DirectX::ScratchImage scratchImage;
    DirectX::TexMetadata metadata{};

    std::wstring widePath(path.begin(), path.end());

    HRESULT hr = DirectX::LoadFromDDSFile(
        widePath.c_str(),
        DirectX::DDS_FLAGS_NONE,
        &metadata,
        scratchImage
    );

    if (FAILED(hr))
    {
        throw std::runtime_error("DdsUtils::loadCubemapDDS failed to load DDS file: " + path);
    }

    if (!metadata.IsCubemap())
    {
        throw std::runtime_error("DdsUtils::loadCubemapDDS expected a cubemap DDS: " + path);
    }

    if (metadata.arraySize != 6)
    {
        throw std::runtime_error("DdsUtils::loadCubemapDDS expected 6 cubemap faces: " + path);
    }

    vk::Format vkFormat = dxgiToVkFormat(metadata.format);
    if (vkFormat == vk::Format::eUndefined)
    {
        throw std::runtime_error("DdsUtils::loadCubemapDDS unsupported DDS format: " + path);
    }

    DdsCubemapData result{};
    result.format = vkFormat;
    result.width = static_cast<uint32_t>(metadata.width);
    result.height = static_cast<uint32_t>(metadata.height);
    result.mipLevels = static_cast<uint32_t>(metadata.mipLevels);
    result.arrayLayers = static_cast<uint32_t>(metadata.arraySize);
    result.isCompressed = isCompressedDxgiFormat(metadata.format);

    result.subresources.reserve(result.mipLevels * result.arrayLayers);

    for (size_t mip = 0; mip < metadata.mipLevels; ++mip)
    {
        for (size_t face = 0; face < metadata.arraySize; ++face)
        {
            const DirectX::Image* image = scratchImage.GetImage(mip, face, 0);

            if (!image || !image->pixels)
            {
                throw std::runtime_error("DdsUtils::loadCubemapDDS missing subresource pixels: " + path);
            }

            DdsSubresource subresource{};
            subresource.mipLevel = static_cast<uint32_t>(mip);
            subresource.arrayLayer = static_cast<uint32_t>(face);
            subresource.width = static_cast<uint32_t>(image->width);
            subresource.height = static_cast<uint32_t>(image->height);
            subresource.rowPitch = image->rowPitch;
            subresource.slicePitch = image->slicePitch;
            subresource.pixels.resize(image->slicePitch);
            std::memcpy(subresource.pixels.data(), image->pixels, image->slicePitch);

            result.subresources.push_back(subresource);
        }
    }

    return result;
}