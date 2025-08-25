#include "rvtx/core/type.hpp"

namespace rvtx
{
    template<class Type>
    Span<Type>::Span( Type * pptr, std::size_t psize ) : ptr( pptr ), size( psize )
    {
    }
    template<class Type>
    Span<Type>::Span( std::vector<Type> & data ) : ptr( data.data() ), size( data.size() )
    {
    }

    template<class Type>
    const Type & Span<Type>::operator[]( std::size_t index ) const
    {
        return ptr[ index ];
    }

    template<class Type>
    ConstSpan<Type>::ConstSpan( const Type * pptr, std::size_t psize ) : ptr( pptr ), size( psize )
    {
    }
    template<class Type>
    ConstSpan<Type>::ConstSpan( const std::vector<Type> & data ) : ptr( data.data() ), size( data.size() )
    {
    }
    template<class Type>
    ConstSpan<Type>::ConstSpan( const Span<Type> data ) : ptr( data.ptr ), size( data.size )
    {
    }

    template<class Type>
    const Type & ConstSpan<Type>::operator[]( std::size_t index ) const
    {
        return ptr[ index ];
    }

    template<class HandleType>
    HandleSpan<HandleType>::HandleSpan( HandleType phandle, std::size_t poffset, std::size_t psize ) :
        handle( phandle ), offset( poffset ), size( psize )
    {
    }

    template<class HandleType>
    HandleSpan<HandleType>::HandleSpan( HandleType phandle, std::size_t psize ) :
        handle( phandle ), offset( 0 ), size( psize )
    {
    }

    template<class Type>
    inline RangeBase<Type>::RangeBase( Type start, Type end ) : start( start ), end( end )
    {
    }

    template<class Type>
    Type RangeBase<Type>::size() const
    {
        return end - start;
    }

    template<class Type>
    Reference<Type>::Reference( Type & data ) : m_data( &data )
    {
    }

    template<class Type>
    Type * Reference<Type>::operator->()
    {
        return m_data;
    }

    template<class Type>
    Type & Reference<Type>::operator*()
    {
        return *m_data;
    }

    template<class Type>
    const Type * ConstReference<Type>::operator->() const
    {
        return m_data;
    }

    template<class Type>
    const Type & ConstReference<Type>::operator*() const
    {
        return *m_data;
    }

    template<class Type>
    ConstReference<Type>::operator Reference<Type>() const
    {
        return *m_data;
    }
} // namespace rvtx
