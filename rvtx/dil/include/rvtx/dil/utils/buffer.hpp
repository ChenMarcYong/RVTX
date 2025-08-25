#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <functional>

#include "rvtx/core/meta.hpp"
#include "rvtx/core/type.hpp"

// Diligent headers
#include "BasicTypes.h"
#include "Buffer.h"
#include "DeviceContext.h"
#include "RenderDevice.h"
#include "ShaderResourceBinding.h"
#include "RefCntAutoPtr.hpp"

namespace rvtx::dil {

    // Flags abstraits côté RVTX (convertis en Diligent dans toDgBindFlags)
    enum class BufferBind : uint32_t
    {
        None = 0,
        Uniform = 1u << 0,  // Diligent::BIND_UNIFORM_BUFFER
        ShaderResource = 1u << 1,  // Diligent::BIND_SHADER_RESOURCE
        Unordered = 1u << 2,  // Diligent::BIND_UNORDERED_ACCESS
        Vertex = 1u << 3,  // Diligent::BIND_VERTEX_BUFFER
        Index = 1u << 4,  // Diligent::BIND_INDEX_BUFFER
    };
    RVTX_DEFINE_ENUM_BITWISE_FUNCTION(BufferBind)

        enum class BufferUsage : uint8_t
    {
        Immutable,   // données fixes (init data)
        Default,     // maj via Update/Copy
        Dynamic      // map(write/discard)
    };

    enum class MapAccess : uint8_t { Read, Write, ReadWrite };
    enum class MapHint : uint8_t { None, Discard, NoOverwrite };

    class Buffer
    {
    public:
        Buffer() = default;

        // Création depuis un bloc mémoire
        Buffer(Diligent::IRenderDevice* device,
            rvtx::ConstSpan<uint8_t> data,
            BufferBind               binds,
            BufferUsage              usage = BufferUsage::Immutable,
            Diligent::Uint32         elementStride = 0,
            bool                     structured = false);

        // Allocation “vide”
        Buffer(Diligent::IRenderDevice* device,
            std::size_t              byteSize,
            BufferBind               binds,
            BufferUsage              usage = BufferUsage::Default,
            Diligent::Uint32         elementStride = 0,
            bool                     structured = false,
            bool                     zeroInit = true);

        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
        Buffer(Buffer&&) noexcept;
        Buffer& operator=(Buffer&&) noexcept;
        ~Buffer() = default;

        // ------ Helpers Typed<T> (implémenté dans buffer.inl) ------
        template<class T>
        static Buffer Typed(Diligent::IRenderDevice* device,
            rvtx::ConstSpan<T>       data,
            BufferBind               binds,
            BufferUsage              usage = BufferUsage::Immutable,
            bool                     structured = false);

        // Accès bruts
        Diligent::IBuffer* get()  const { return mBuf; }
        Diligent::IBufferView* srv()  const;
        Diligent::IBufferView* uav()  const;
        std::size_t            size() const { return mSize; }
        Diligent::BufferDesc   desc() const;

        // Map / Unmap (non-template -> implémenté dans .cpp)
        uint8_t* map(Diligent::IDeviceContext* ctx,
            MapAccess acc,
            MapHint hint = MapHint::None,
            std::size_t offset = 0,
            std::size_t length = 0);
        void     unmap(Diligent::IDeviceContext* ctx);

        // RAII mapping
        template<class T>
        using ScopedMapping = std::unique_ptr<T, std::function<void(T*)>>;

        template<class T>
        ScopedMapping<T> scopedMap(Diligent::IDeviceContext* ctx,
            MapAccess acc,
            MapHint hint = MapHint::None,
            std::size_t offset = 0,
            std::size_t length = 0)
        {
            T* ptr = reinterpret_cast<T*>(map(ctx, acc, hint, offset, length));
            auto deleter = [this, ctx](T*) { this->unmap(ctx); };
            return ScopedMapping<T>(ptr, deleter);
        }

        // Resize (recrée et copie optionnellement l’ancien contenu)
        void resize(Diligent::IRenderDevice* device,
            Diligent::IDeviceContext* ctx,
            std::size_t newSize,
            bool preserveData = true,
            bool zeroPad = false);

        // Binding
        void bindToSRB(Diligent::IShaderResourceBinding* srb,
            const char* varName,
            Diligent::SHADER_TYPE stage,
            bool asUav = false) const;

        void setAsVertexBuffer(Diligent::IDeviceContext* ctx,
            Diligent::Uint32 slot,
            Diligent::Uint64 offset = 0) const;

        void setAsIndexBuffer(Diligent::IDeviceContext* ctx,
            Diligent::Uint64 offset = 0,
            Diligent::VALUE_TYPE indexType = Diligent::VT_UINT32) const;

    private:
        void create_(Diligent::IRenderDevice* device,
            const void* initData,
            std::size_t              byteSize,
            BufferBind               binds,
            BufferUsage              usage,
            Diligent::Uint32         elementStride,
            bool                     structured);

        static Diligent::BIND_FLAGS toDgBindFlags(BufferBind);
        static Diligent::USAGE      toDgUsage(BufferUsage);

    private:
        Diligent::RefCntAutoPtr<Diligent::IBuffer>     mBuf;
        Diligent::RefCntAutoPtr<Diligent::IBufferView> mSRV;
        Diligent::RefCntAutoPtr<Diligent::IBufferView> mUAV;
        std::size_t                                    mSize = 0;
        Diligent::BIND_FLAGS                           mBindFlags = Diligent::BIND_NONE;
        Diligent::USAGE                                mUsage = Diligent::USAGE_DEFAULT;
        Diligent::Uint32                               mStride = 0;   // pour StructuredBuffer
        bool                                           mStructured = false;
    };

} // namespace rvtx::dil

#include "rvtx/dil/utils/buffer.inl"
