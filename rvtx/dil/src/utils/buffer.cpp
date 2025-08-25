// rvtx/dil/utils/buffer.cpp

#include "rvtx/dil/utils/buffer.hpp"

#include <algorithm>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <memory>

// Diligent
#include "Common/interface/RefCntAutoPtr.hpp"
#include "Primitives/interface/BasicTypes.h"
#include "Graphics/GraphicsEngine/interface/GraphicsTypes.h"
#include "Graphics/GraphicsEngine/interface/Buffer.h"
#include "Graphics/GraphicsEngine/interface/BufferView.h"
#include "Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "Graphics/GraphicsEngine/interface/ShaderResourceBinding.h"
#include "MapHelper.hpp"

using namespace Diligent;

namespace rvtx::dil
{
    // ---------- Helpers internes ----------
    using MapByteHelper = MapHelper<Uint8>;
    static std::unordered_map<IBuffer*, std::unique_ptr<MapByteHelper>> g_MapHelpers;

    static BUFFER_MODE modeFrom(bool structured)
    {
        return structured ? BUFFER_MODE_STRUCTURED : BUFFER_MODE_UNDEFINED;
    }

    // ---------- Conversion de nos enums vers Diligent ----------
    BIND_FLAGS Buffer::toDgBindFlags(BufferBind b)
    {
        BIND_FLAGS f = BIND_NONE;
        if ((b & BufferBind::Uniform) != BufferBind::None) f |= BIND_UNIFORM_BUFFER;
        if ((b & BufferBind::ShaderResource) != BufferBind::None) f |= BIND_SHADER_RESOURCE;
        if ((b & BufferBind::Unordered) != BufferBind::None) f |= BIND_UNORDERED_ACCESS;
        if ((b & BufferBind::Vertex) != BufferBind::None) f |= BIND_VERTEX_BUFFER;
        if ((b & BufferBind::Index) != BufferBind::None) f |= BIND_INDEX_BUFFER;
        return f;
    }

    USAGE Buffer::toDgUsage(BufferUsage u)
    {
        switch (u)
        {
        case BufferUsage::Immutable: return USAGE_IMMUTABLE;
        case BufferUsage::Default:   return USAGE_DEFAULT;
        case BufferUsage::Dynamic:   return USAGE_DYNAMIC;
        default:                     return USAGE_DEFAULT;
        }
    }

    // ---------- Création interne ----------
    void Buffer::create_(IRenderDevice* device,
        const void* initData,
        std::size_t    byteSize,
        BufferBind     binds,
        BufferUsage    usage,
        Uint32         elementStride,
        bool           structured)
    {
        mSize = byteSize;
        mBindFlags = toDgBindFlags(binds);
        mUsage = toDgUsage(usage);
        mStride = elementStride;
        mStructured = structured;

        BufferDesc desc;
        desc.Name = "rvtx::dil::Buffer";
        desc.Size = static_cast<Uint64>(byteSize);
        desc.Usage = mUsage;
        desc.BindFlags = mBindFlags;
        desc.CPUAccessFlags = (mUsage == USAGE_DYNAMIC) ? CPU_ACCESS_WRITE : CPU_ACCESS_NONE;
        desc.Mode = modeFrom(structured);
        desc.ElementByteStride = structured ? elementStride : 0;

        BufferData  bd{};
        BufferData* pInit = nullptr;
        if (initData != nullptr)
        {
            bd.pData = initData;
            bd.DataSize = static_cast<Uint64>(byteSize);
            pInit = &bd;
        }

        device->CreateBuffer(desc, pInit, &mBuf);

        // Vues auto SRV/UAV selon les flags
        if ((mBindFlags & BIND_SHADER_RESOURCE) != 0)
        {
            BufferViewDesc srv{};
            srv.ViewType = BUFFER_VIEW_SHADER_RESOURCE;
            srv.ByteOffset = 0;
            srv.ByteWidth = 0; // 0 => toute la taille
            mBuf->CreateView(srv, &mSRV);
        }

        if ((mBindFlags & BIND_UNORDERED_ACCESS) != 0)
        {
            BufferViewDesc uav{};
            uav.ViewType = BUFFER_VIEW_UNORDERED_ACCESS;
            uav.ByteOffset = 0;
            uav.ByteWidth = 0;
            mBuf->CreateView(uav, &mUAV);
        }
    }

    // ---------- Ctors ----------
    Buffer::Buffer(IRenderDevice* device,
        rvtx::ConstSpan<uint8_t> data,
        BufferBind binds,
        BufferUsage usage,
        Uint32 elementStride,
        bool structured)
    {
        create_(device, data.ptr, data.size, binds, usage, elementStride, structured);
    }

    Buffer::Buffer(IRenderDevice* device,
        std::size_t byteSize,
        BufferBind binds,
        BufferUsage usage,
        Uint32 elementStride,
        bool structured,
        bool zeroInit)
    {
        if (zeroInit)
        {
            std::vector<uint8_t> zeros(byteSize, 0u);
            create_(device, zeros.data(), byteSize, binds, usage, elementStride, structured);
        }
        else
        {
            create_(device, nullptr, byteSize, binds, usage, elementStride, structured);
        }
    }

    Buffer::Buffer(Buffer&& other) noexcept { *this = std::move(other); }

    Buffer& Buffer::operator=(Buffer&& other) noexcept
    {
        mBuf = std::move(other.mBuf);
        mSRV = std::move(other.mSRV);
        mUAV = std::move(other.mUAV);
        mSize = other.mSize;
        mBindFlags = other.mBindFlags;
        mUsage = other.mUsage;
        mStride = other.mStride;
        mStructured = other.mStructured;
        other.mSize = 0;
        return *this;
    }

    // ---------- Accesseurs ----------
    IBufferView* Buffer::srv() const { return mSRV; }
    IBufferView* Buffer::uav() const { return mUAV; }
    BufferDesc   Buffer::desc() const { return mBuf ? mBuf->GetDesc() : BufferDesc{}; }

    // ---------- Map / Unmap ----------
    uint8_t* Buffer::map(IDeviceContext* ctx,
        MapAccess acc, MapHint hint,
        std::size_t offset, std::size_t /*length*/)
    {
        if (!mBuf) return nullptr;

        MAP_TYPE  mapType = MAP_WRITE;
        MAP_FLAGS mapFlags = MAP_FLAG_NONE;

        switch (acc)
        {
        case MapAccess::Read:      mapType = MAP_READ;       break;
        case MapAccess::Write:     mapType = MAP_WRITE;      break;
        case MapAccess::ReadWrite: mapType = MAP_READ_WRITE; break;
        }
        if (hint == MapHint::Discard)       mapFlags = MAP_FLAG_DISCARD;
        else if (hint == MapHint::NoOverwrite) mapFlags = MAP_FLAG_NO_OVERWRITE;

        // Map RAII grâce à MapHelper; conservation dans une table pour unmap ultérieur.
        std::unique_ptr<MapByteHelper> helper(new MapByteHelper(ctx, mBuf, mapType, mapFlags));

        Uint8* base = *helper;
        g_MapHelpers[mBuf.RawPtr()] = std::move(helper);

        return base ? base + offset : nullptr;
    }

    void Buffer::unmap(IDeviceContext* /*ctx*/)
    {
        if (!mBuf) return;
        auto it = g_MapHelpers.find(mBuf.RawPtr());
        if (it != g_MapHelpers.end())
        {
            // Destruction du MapHelper => Unmap automatique
            g_MapHelpers.erase(it);
        }
    }

    // ---------- Resize ----------
    void Buffer::resize(IRenderDevice* device,
        IDeviceContext* ctx,
        std::size_t newSize,
        bool preserveData,
        bool zeroPad)
    {
        if (newSize == mSize) return;

        RefCntAutoPtr<IBuffer>     newBuf;
        RefCntAutoPtr<IBufferView> newSRV, newUAV;

        BufferDesc d = desc();
        d.Size = static_cast<Uint64>(newSize);

        device->CreateBuffer(d, nullptr, &newBuf);

        if ((mBindFlags & BIND_SHADER_RESOURCE) != 0)
        {
            BufferViewDesc srv{};
            srv.ViewType = BUFFER_VIEW_SHADER_RESOURCE;
            newBuf->CreateView(srv, &newSRV);
        }
        if ((mBindFlags & BIND_UNORDERED_ACCESS) != 0)
        {
            BufferViewDesc uav{};
            uav.ViewType = BUFFER_VIEW_UNORDERED_ACCESS;
            newBuf->CreateView(uav, &newUAV);
        }

        if (preserveData && mBuf)
        {
            const Uint64 copySz = static_cast<Uint64>(std::min(mSize, newSize));
            if (copySz > 0)
            {
                ctx->CopyBuffer(mBuf, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                    newBuf, 0, copySz,
                    RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
            }
            if (zeroPad && newSize > mSize && mUsage != USAGE_IMMUTABLE)
            {
                std::unique_ptr<MapByteHelper> helper(new MapByteHelper(ctx, newBuf, MAP_WRITE, MAP_FLAG_DISCARD));
                Uint8* p = *helper;
                if (p) std::memset(p + mSize, 0, newSize - mSize);
            }
        }

        mBuf = std::move(newBuf);
        mSRV = std::move(newSRV);
        mUAV = std::move(newUAV);
        mSize = newSize;
    }

    // ---------- Binding ----------
    void Buffer::bindToSRB(IShaderResourceBinding* srb,
        const char* varName,
        SHADER_TYPE stage,
        bool asUav) const
    {
        if (!srb || !mBuf) return;

        // Cas UBO
        if ((mBindFlags & BIND_UNIFORM_BUFFER) != 0)
        {
            if (auto* var = srb->GetVariableByName(stage, varName))
                var->Set(mBuf);
            return;
        }

        if (asUav)
        {
            if (mUAV)
                if (auto* var = srb->GetVariableByName(stage, varName))
                    var->Set(mUAV);
        }
        else
        {
            if (mSRV)
                if (auto* var = srb->GetVariableByName(stage, varName))
                    var->Set(mSRV);
        }
    }

    void Buffer::setAsVertexBuffer(IDeviceContext* ctx, Uint32 slot, Uint64 offset) const
    {
        if (!mBuf) return;
        IBuffer* vbs[] = { mBuf };
        Uint64   offs[] = { offset };
        ctx->SetVertexBuffers(slot, 1, vbs, offs,
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
            SET_VERTEX_BUFFERS_FLAG_RESET);
    }

    void Buffer::setAsIndexBuffer(IDeviceContext* ctx, Uint64 offset, VALUE_TYPE /*indexType*/) const
    {
        if (!mBuf) return;
        ctx->SetIndexBuffer(mBuf, offset, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

} // namespace rvtx::dil
