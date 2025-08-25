#ifndef RVTX_DIL_PASS_PASS_HPP
#define RVTX_DIL_PASS_PASS_HPP

#include <cstdint>

namespace Diligent { class IRenderDevice; }

namespace rvtx::dil
{
    class Pass
    {
    public:
        Pass() = default;
        Pass(uint32_t width, uint32_t height);
        virtual ~Pass() = default;

        virtual void resize(Diligent::IRenderDevice* pDevice, uint32_t width, uint32_t height) = 0;


        uint32_t width()  const noexcept { return m_width; }
        uint32_t height() const noexcept { return m_height; }


    protected:
        uint32_t m_width;
        uint32_t m_height;
    };
} // namespace rvtx::dil

#endif // RVTX_DIL_PASS_PASS_HPP
