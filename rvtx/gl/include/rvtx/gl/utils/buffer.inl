#include "rvtx/gl/utils/buffer.hpp"

namespace rvtx::gl
{
    template<class Type>
    Buffer Buffer::Typed( rvtx::ConstSpan<Type> data, BufferAuthorization authorization )
    {
        return { rvtx::ConstSpan<uint8_t>( reinterpret_cast<const uint8_t *>( data.ptr ), data.size * sizeof( Type ) ),
                 authorization };
    }

    template<class MappingType>
    MappingType * Buffer::map( std::size_t startingPoint, std::size_t mappingSize, BufferAuthorization mappingType )
    {
        return reinterpret_cast<MappingType *>( map( startingPoint, mappingSize, mappingType ) );
    }

    template<class Type>
    Buffer::ScopedMapping<Type> Buffer::scopedMap( std::size_t         startingPoint,
                                                   std::size_t         mappingSize,
                                                   BufferAuthorization mappingType )
    {
        Type * ptr = map<Type>( startingPoint, mappingSize, mappingType );
        return ScopedMapping<Type>( ptr, [ this ]( Type * ) { unmap(); } );
    }
} // namespace rvtx::gl
