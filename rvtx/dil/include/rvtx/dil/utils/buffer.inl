// rvtx/dil/utils/buffer.inl
#pragma once

namespace rvtx::dil
{
    template<class T>
    Buffer Buffer::Typed(Diligent::IRenderDevice* device,
        rvtx::ConstSpan<T>       data,
        BufferBind               binds,
        BufferUsage              usage,
        bool                     structured)
    {
        Buffer out;

        // Renseigne l'état interne (mêmes noms que dans le .hpp)
        out.mSize = data.size * sizeof(T);
        out.mBindFlags = toDgBindFlags(binds);
        out.mUsage = toDgUsage(usage);
        out.mStride = structured ? static_cast<Diligent::Uint32>(sizeof(T)) : 0;
        out.mStructured = structured;

        // Desc Diligent
        Diligent::BufferDesc desc{};
        desc.Name = "rvtx::dil::Buffer::Typed";
        desc.Size = static_cast<Diligent::Uint64>(out.mSize);
        desc.BindFlags = out.mBindFlags;
        desc.Usage = out.mUsage;
        desc.CPUAccessFlags = (out.mUsage == Diligent::USAGE_DYNAMIC)
            ? Diligent::CPU_ACCESS_WRITE
            : Diligent::CPU_ACCESS_NONE;

        if (structured)
        {
            desc.Mode = Diligent::BUFFER_MODE_STRUCTURED;
            desc.ElementByteStride = static_cast<Diligent::Uint32>(sizeof(T));
        }

        // Données initiales
        Diligent::BufferData init{};
        init.pData = static_cast<const void*>(data.ptr);
        init.DataSize = static_cast<Diligent::Uint64>(out.mSize);

        // Création
        device->CreateBuffer(desc, &init, &out.mBuf);

        // Vues auto si besoin (SRV/UAV)
        if ((out.mBindFlags & Diligent::BIND_SHADER_RESOURCE) != 0)
        {
            Diligent::BufferViewDesc v{};
            v.ViewType = Diligent::BUFFER_VIEW_SHADER_RESOURCE;
            out.mBuf->CreateView(v, &out.mSRV);
        }
        if ((out.mBindFlags & Diligent::BIND_UNORDERED_ACCESS) != 0)
        {
            Diligent::BufferViewDesc v{};
            v.ViewType = Diligent::BUFFER_VIEW_UNORDERED_ACCESS;
            out.mBuf->CreateView(v, &out.mUAV);
        }

        return out;
    }
} // namespace rvtx::dil
