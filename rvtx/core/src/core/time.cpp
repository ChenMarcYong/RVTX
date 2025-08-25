#include "rvtx/core/time.hpp"

#include <chrono>

namespace rvtx
{
    std::tm time::now()
    {
        const std::time_t currentTime = std::chrono::system_clock::to_time_t( std::chrono::system_clock::now() );

        // From https://github.com/gabime/spdlog/blob/master/include/spdlog/details/os.h#L73
        std::tm tm;
#ifdef _WIN32
        // https://en.cppreference.com/w/c/chrono/localtime
        // "The implementation of localtime_s in Microsoft CRT is incompatible with the C standard since it has reversed
        // parameter order and returns errno_t."
        ::localtime_s( &tm, &currentTime );
#else
        ::localtime_r( &currentTime, &tm );
#endif

        return tm;
    }
} // namespace rvtx
