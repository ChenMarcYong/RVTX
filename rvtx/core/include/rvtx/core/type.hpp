#ifndef RVTX_CORE_TYPES_HPP
#define RVTX_CORE_TYPES_HPP

#include <cstddef>
#include <vector>

namespace rvtx
{
    template<class Type>
    struct Span
    {
        Type *      ptr  = nullptr;
        std::size_t size = 0;

        Span() = default;
        Span( Type * ptr, std::size_t size );
        Span( std::vector<Type> & data );

        const Type & operator[]( std::size_t index ) const;
    };

    template<class Type>
    struct ConstSpan
    {
        const Type * ptr  = nullptr;
        std::size_t  size = 0;

        ConstSpan() = default;
        ConstSpan( const Type * ptr, std::size_t size );
        ConstSpan( const std::vector<Type> & data );
        ConstSpan( const Span<Type> data );

        const Type & operator[]( std::size_t index ) const;
    };

    template<class HandleType>
    struct HandleSpan
    {
        HandleType  handle;
        std::size_t offset = 0;
        std::size_t size   = 0;

        HandleSpan() = default;
        HandleSpan( HandleType handle, std::size_t offset, std::size_t size );
        HandleSpan( HandleType handle, std::size_t size );
    };

    template<class Type>
    struct RangeBase
    {
        RangeBase() = default;
        RangeBase( Type start, Type end );

        Type start;
        Type end;

        Type size() const;
    };

    using Range = RangeBase<std::size_t>;

    template<class Type>
    class Reference
    {
      public:
        Reference() = default;
        Reference( Type & data );

        Reference( const Reference & )             = default;
        Reference & operator=( const Reference & ) = default;

        Reference( Reference && )             = default;
        Reference & operator=( Reference && ) = default;

        ~Reference() = default;

        Type * operator->();
        Type & operator*();

      private:
        Type * m_data = nullptr;
    };

    template<class Type>
    class ConstReference
    {
      public:
        ConstReference() = default;
        ConstReference( const Type & data );
        ConstReference( Reference<Type> data );

        ConstReference( const ConstReference & )             = default;
        ConstReference & operator=( const ConstReference & ) = default;

        ConstReference( ConstReference && )             = default;
        ConstReference & operator=( ConstReference && ) = default;

        ~ConstReference() = default;

        const Type * operator->() const;
        const Type & operator*() const;
        operator Reference<Type>() const;

      private:
        Type * m_data = nullptr;
    };
} // namespace rvtx

#include "rvtx/core/type.inl"

#endif // RVTX_CORE_TYPES_HPP
