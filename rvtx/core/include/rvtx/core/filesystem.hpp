#ifndef RVTX_CORE_FILESYSTEM_HPP
#define RVTX_CORE_FILESYSTEM_HPP

#include <filesystem>

#include "rvtx/core/type.hpp"

namespace rvtx
{
    std::string read( const std::filesystem::path & path );
    void        save( const std::filesystem::path & path, std::string_view data );
    void        save( const std::filesystem::path & path, ConstSpan<char> data );
} // namespace rvtx

#endif // RVTX_CORE_FILESYSTEM_HPP
