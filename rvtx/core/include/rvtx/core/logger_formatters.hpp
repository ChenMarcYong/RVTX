#ifndef RVTX_CORE_LOGGER_FORMATTERS_HPP
#define RVTX_CORE_LOGGER_FORMATTERS_HPP

#include <fmt/format.h>
#include "glm/fwd.hpp"

namespace fmt
{
    template<glm::length_t L, typename T, glm::qualifier Q>
    struct formatter<glm::vec<L, T, Q>> : formatter<T>
    {
        formatter<T> type_formatter;

      public:
        template<typename ParseContext>
        constexpr auto parse( ParseContext & ctx )
        {
            return type_formatter.parse( ctx );
        }

        template<typename FmtContext>
        auto format( glm::vec<L, T, Q> const & vec, FmtContext & ctx )
        {
            auto out = std::copy_n( "(", 1, ctx.out() );
            for ( glm::length_t l = 0; l < L - 1; ++l )
            {
                out = type_formatter.format( vec[ l ], ctx );
                out = std::copy_n( ", ", 2, out );
                ctx.advance_to( out );
            }
            out = type_formatter.format( vec[ L - 1 ], ctx );
            return std::copy_n( ")", 1, out );
        }
    };
    
    template<typename T, glm::qualifier Q>
    struct formatter<glm::qua<T, Q>> : formatter<T>
    {
        formatter<T> type_formatter;

      public:
        template<typename ParseContext>
        constexpr auto parse( ParseContext & ctx )
        {
            return type_formatter.parse( ctx );
        }

        template<typename FmtContext>
        auto format( glm::qua<T, Q> const & vec, FmtContext & ctx )
        {
            auto out = std::copy_n( "(", 1, ctx.out() );
            for ( glm::length_t l = 0; l < 3; ++l )
            {
                out = type_formatter.format( vec[ l ], ctx );
                out = std::copy_n( ", ", 2, out );
                ctx.advance_to( out );
            }
            out = type_formatter.format( vec[ 3 ], ctx );
            return std::copy_n( ")", 1, out );
        }
    };
} // namespace fmt

#endif // RVTX_CORE_LOGGER_FORMATTERS_HPP
