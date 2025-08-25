#ifndef RVTX_CORE_LOGGER_HPP
#define RVTX_CORE_LOGGER_HPP

#include <fmt/format.h>

#include "logger_formatters.hpp"

namespace rvtx::logger
{
    enum class Level
    {
        Info,
        Debug,
        Warning,
        Error
    };

    void log( const Level level, std::string_view str );

    template<typename... T>
    void log( const Level level, std::string_view str, T &&... args );

    template<typename... T>
    void debug( std::string_view str, T &&... args );

    template<typename... T>
    void info( std::string_view str, T &&... args );

    template<typename... T>
    void warning( std::string_view str, T &&... args );

    template<typename... T>
    void error( std::string_view str, T &&... args );
} // namespace rvtx::logger

#define LOG( ... )             \
    fmt::print( __VA_ARGS__ ); \
    fmt::print( "\n" )
#define LOG_INFO( ... ) rvtx::logger::info( __VA_ARGS__ )
#define LOG_WARNING( ... ) rvtx::logger::warning( __VA_ARGS__ )
#define LOG_ERROR( ... ) rvtx::logger::error( __VA_ARGS__ )

#define __rvtx_LOG_CRITICAL_ERROR( H, C, ... )                                                                   \
    LOG_ERROR( fmt::format( "{}\n -> File : '{}', line {}\n -> Condition : ({}\n -> Message :\n{}", H, __FILE__, \
                            __LINE__, C, fmt::format( __VA_ARGS__ ) ) );

#define RVTX_ERROR( C, ... )                                                            \
    do {                                                                               \
        if ( C )                                                                       \
        {                                                                              \
            __rvtx_LOG_CRITICAL_ERROR( "CRITICAL ERROR", #C##") is true", __VA_ARGS__ ); \
            std::terminate();                                                          \
        }                                                                              \
    } while ( false )

#ifndef RVTX_DIST
#define RVTX_ASSERT( C, ... )                                                   \
        do {                                                                       \
            if ( !( C ) )                                                          \
            {                                                                      \
                __rvtx_LOG_CRITICAL_ERROR( "ASSERT", #C##") is not true", __VA_ARGS__ ); \
                std::terminate();                                                  \
            }                                                                      \
        } while ( false )
#else
#define RVTX_ASSERT( C, ... ) \
        do {                 \
        } while ( false )
#endif

#include "rvtx/core/logger.inl"

#endif // RVTX_CORE_LOGGER_HPP
