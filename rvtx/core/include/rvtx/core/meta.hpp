#ifndef RVTX_CORE_META_HPP
#define RVTX_CORE_META_HPP

#define RVTX_DEFINE_ENUM_BITWISE_FUNCTION( Type )                                      \
    inline constexpr Type operator&( const Type l, const Type r )                      \
    {                                                                                  \
        return static_cast<Type>( rvtx::toUnderlying( l ) & rvtx::toUnderlying( r ) ); \
    }                                                                                  \
                                                                                       \
    inline constexpr Type operator|( const Type l, const Type r )                      \
    {                                                                                  \
        return static_cast<Type>( rvtx::toUnderlying( l ) | rvtx::toUnderlying( r ) ); \
    }                                                                                  \
                                                                                       \
    inline constexpr Type operator~( const Type m )                                    \
    {                                                                                  \
        return static_cast<Type>( ~rvtx::toUnderlying( m ) );                          \
    }

namespace rvtx
{
    template<class Enum>
    constexpr typename std::underlying_type<Enum>::type toUnderlying( const Enum e ) noexcept
    {
        return static_cast<typename std::underlying_type<Enum>::type>( e );
    }
} // namespace rvtx

#endif // RVTX_CORE_META_HPP
