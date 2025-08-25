#include "rvtx/core/filesystem.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>

namespace rvtx
{
    std::string read( const std::filesystem::path & path )
    {
        std::ifstream file;
        file.open( path, std::ios::in );

        if ( !file.is_open() )
            throw std::runtime_error( "Cannot open file: " + path.string() );

        const uintmax_t size = std::filesystem::file_size( path );
        std::string     result {};
        result.resize( size, '\0' );

        file.read( result.data(), static_cast<std::streamsize>( size ) );
        file.close();

        return result;
    }

    void save( const std::filesystem::path & path, std::string_view data )
    {
        save( path, ConstSpan<char> { data.data(), data.size() } );
    }

    void save( const std::filesystem::path & path, ConstSpan<char> data )
    {
        std::ofstream file { path, std::ios::out | std::ios::binary };
        std::copy_n( data.ptr, data.size, std::ostream_iterator<char>( file ) );
    }
} // namespace rvtx
