#ifndef RVTX_CORE_IMAGE_HPP
#define RVTX_CORE_IMAGE_HPP

#include <filesystem>

#include "rvtx/core/type.hpp"

namespace rvtx
{
    std::vector<float> loadExr( std::filesystem::path path, uint32_t & width, uint32_t & height, uint32_t & channels );

    void save( std::filesystem::path path,
               ConstSpan<uint32_t>   data,
               uint32_t              width,
               uint32_t              height,
               std::string_view      text = "",
               bool                  flip = true );
} // namespace rvtx

#endif // RVTX_CORE_IMAGE_HPP
