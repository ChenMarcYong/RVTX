#ifndef RVTX_SYSTEM_VISIBILITY_HPP
#define RVTX_SYSTEM_VISIBILITY_HPP

namespace rvtx
{
    struct Visibility
    {
        static constexpr bool in_place_delete = true;

        bool visible { true };
    };
} // namespace rvtx

#endif // RVTX_SYSTEM_VISIBILITY_HPP
