#include "rvtx/core/logger.hpp"

#include <fmt/chrono.h>
#include <fmt/color.h>

#include "rvtx/core/time.hpp"

namespace rvtx::logger
{
    static fmt::text_style toColor( Level level )
    {
        switch ( level )
        {
        case Level::Info: return fg( fmt::color::green );
        case Level::Debug: return fg( fmt::color::light_gray );
        case Level::Warning: return fg( fmt::color::orange );
        case Level::Error: return fg( fmt::color::red );
        default: return fg( fmt::color::white );
        }
    }

    static std::string_view toString( Level level )
    {
        switch ( level )
        {
        case Level::Info: return "Info";
        case Level::Debug: return "Debug";
        case Level::Warning: return "Warning";
        case Level::Error: return "Error";
        default: return "";
        }
    }

    void log( const Level level, std::string_view str )
    {
        fmt::print( toColor( level ), "[{:%Y-%m-%d %H:%M:%S}] [{}] {}\n", time::now(), toString( level ), str );
    }

} // namespace rvtx::logger
