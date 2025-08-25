#include "rvtx/core/benchmark.hpp"

#include <glm/ext/quaternion_common.hpp>
#include <glm/vec3.hpp>

#include "fmt/color.h"
#include "fmt/printf.h"
#include "plf_nanotimer.h"
#include "rvtx/core/time.hpp"

namespace rvtx
{
    Benchmark::Benchmark( std::string benchmarkName ) : m_name( std::move( benchmarkName ) )
    {
#ifdef WIN32
        HANDLE hOut   = GetStdHandle( STD_OUTPUT_HANDLE );
        DWORD  dwMode = 0;
        GetConsoleMode( hOut, &dwMode );
        dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        SetConsoleMode( hOut, dwMode );
#endif // WIN32
    }

#if defined( _MSC_VER )
#pragma optimize( "", off )
    void Benchmark::doNotOptimizeSink( void const * ) {}
#pragma optimize( "", on )
#endif

    double Benchmark::timer_ns( const Task & task )
    {
        plf::nanotimer timer;
        timer.start();

        task();

        return timer.get_elapsed_ns();
    }

    double Benchmark::timer_us( const Task & task )
    {
        plf::nanotimer timer;
        timer.start();

        task();

        return timer.get_elapsed_us();
    }

    double Benchmark::timer_ms( const Task & task )
    {
        plf::nanotimer timer;
        timer.start();

        task();

        return timer.get_elapsed_ms();
    }

    std::vector<double> Benchmark::run( const Task & task ) const
    {
        if ( m_printProgress )
            return runPrintInternal( task );
        return runInternal( task );
    }

    std::vector<double> Benchmark::runInternal( const Task & task ) const
    {
        if ( m_warmups > 0 )
        {
            for ( std::size_t i = 0; i < m_warmups; i++ )
            {
                doNotOptimize( m_timerFunction( task ) );
            }
        }

        std::vector<double> results {};
        results.resize( m_iterations );
        for ( std::size_t i = 0; i < m_iterations; i++ )
        {
            const double currentTime = m_timerFunction( task );
            results[ i ]             = currentTime;
        }

        return results;
    }

    fmt::color colorFromProgress( const std::size_t index, const std::size_t count )
    {
        const glm::vec3 red    = glm::vec3( 1.f, 0.f, 0.f );
        const glm::vec3 orange = glm::vec3( 1.f, 0.5f, 0.f );
        const glm::vec3 yellow = glm::vec3( 1.f, 1.f, 0.f );

        float progress = index / static_cast<float>( count );

        glm::vec3 color;
        if ( progress < 0.5f )
            color = glm::mix( red, orange, progress / 0.5f );
        else
            color = glm::mix( orange, yellow,
                              ( progress - 0.5f ) / 0.5f ); // ORANGE to YELLOW

        const uint8_t r = static_cast<uint8_t>( color.r * 255 );
        const uint8_t g = static_cast<uint8_t>( color.g * 255 );
        const uint8_t b = static_cast<uint8_t>( color.b * 255 );

        return static_cast<fmt::color>( r << 16 | g << 8 | b );
    }

    inline double computeMean( std::vector<double> a )
    {
        double total = 0.f;
        for ( const double & i : a )
            total += i;
        return total / a.size();
    }

    std::vector<double> Benchmark::runPrintInternal( const Task & task ) const
    {
        // This function certainly needs refactoring...

        plf::nanotimer timer;
        std::string    progressBar;
        float          progressStep = 30.f / m_warmups;
        float          progress     = 0.f;

        fmt::print( fg( fmt::color::white ), "> " );
        fmt::print( fg( fmt::color::white ) | fmt::emphasis::underline | fmt::emphasis::bold, "{}", m_name );
        fmt::print( fg( fmt::color::white ), " ({} warmups, {} iterations)\n", m_warmups, m_iterations );

        std::vector<double> itTimes {};
        itTimes.reserve( m_warmups + m_iterations );

        std::size_t remainingIterations = m_warmups + m_iterations;
        if ( m_warmups > 0 )
        {
            timer.start();
            plf::nanotimer itTimer;
            for ( std::size_t i = 0; i < m_warmups; i++ )
            {
                itTimer.start();
                doNotOptimize( m_timerFunction( task ) );

                fmt::print( "\33[2K\rWarmup {}/{} [", i + 1, m_warmups );
                progress += progressStep;
                if ( progress >= 1.f - 0.0001f )
                {
                    for ( ; progress > 1.f; progress -= 1.f )
                        progressBar += static_cast<char>( 219 );
                }

                fmt::print( fg( colorFromProgress( i, m_warmups ) ), "{:30}", progressBar );
                fmt::print( "] {:.3} %", ( i + 1 ) * 100.f / m_warmups );

                itTimes.emplace_back( itTimer.get_elapsed_ms() );
                const double meanItTime = computeMean( itTimes ) / 1000.0;
                fmt::print( fg( fmt::color::coral ),
                            "\n\33[2K ~ Estimated remaining time : {:.3}s {}\033[F",
                            meanItTime * --remainingIterations,
                            m_printIterationStats
                                ? fmt::format( "[{:.3}s/it | {:.3}it/s]", meanItTime, 1.f / meanItTime )
                                : "" );
            }
            progress += progressStep;
            if ( progress >= 1.f - 0.0001f )
            {
                for ( ; progress > 1.f; progress -= 1.f )
                    progressBar += static_cast<char>( 219 );
            }
            fmt::print( fmt::emphasis::italic, "\33[2K\rWarmup {}/{} ", m_warmups, m_warmups );
            fmt::print( "[" );
            fmt::print( fg( fmt::color::green ), "{:30}", progressBar );
            fmt::print( "]" );
            fmt::print( fmt::emphasis::italic, " 100 %" );
            fmt::print( " {} Done ({:.3}s).\n", static_cast<char>( 175 ), timer.get_elapsed_ms() / 1000 );
        }

        std::vector<double> results {};
        results.resize( m_iterations );

        progressBar.clear();
        progressStep = 30.f / m_iterations;
        progress     = 0.f;
        timer.start();
        double         totalTime = 0.;
        plf::nanotimer itTimer;
        for ( std::size_t i = 0; i < m_iterations; i++ )
        {
            itTimer.start();
            const double currentTime = m_timerFunction( task );
            results[ i ]             = currentTime;
            totalTime += currentTime;

            fmt::print( "\33[2K\rBenchmark {}/{} [", i + 1, m_iterations );
            progress += progressStep;
            if ( progress >= 1.f - 0.0001f )
            {
                for ( ; progress > 1.f; progress -= 1.f )
                    progressBar += static_cast<char>( 219 );
            }
            fmt::print( fg( colorFromProgress( i, m_iterations ) ), "{:30}", progressBar );
            fmt::print( "] {:.3} %", ( i + 1 ) * 100.f / m_iterations );

            itTimes.emplace_back( itTimer.get_elapsed_ms() );
            const double meanItTime = computeMean( itTimes ) / 1000.0;
            fmt::print(
                fg( fmt::color::coral ),
                "\n\33[2K ~ Estimated remaining time : {:.3}s {}\033[F",
                meanItTime * --remainingIterations,
                m_printIterationStats ? fmt::format( "[{:.3}s/it | {:.3}it/s]", meanItTime, 1.f / meanItTime ) : "" );
        }
        progress += progressStep;
        if ( progress >= 1.f - 0.0001f )
        {
            for ( ; progress > 1.f; progress -= 1.f )
                progressBar += static_cast<char>( 219 );
        }
        fmt::print( fmt::emphasis::italic, "\33[2K\rBenchmark {}/{} ", m_iterations, m_iterations );
        fmt::print( "[" );
        fmt::print( fg( fmt::color::green ), "{:30}", progressBar );
        fmt::print( "]" );
        fmt::print( fmt::emphasis::italic, " 100 %" );
        fmt::print( " {} Done ({:.3}s).\n", static_cast<char>( 175 ), timer.get_elapsed_ms() / 1000 );
        fmt::print( "\33[2K\r" );

        if ( m_printStats )
        {
            float  mean     = totalTime / m_iterations;
            double variance = 0.;
            double fastest  = std::numeric_limits<double>::max();
            double slowest  = std::numeric_limits<double>::lowest();
            for ( const double time : results )
            {
                variance += ( time - mean ) * ( time - mean );
                slowest = std::max( slowest, time );
                fastest = std::min( fastest, time );
            }
            variance /= m_iterations - 1;

            double standardDeviation = std::sqrt( variance );

            fmt::print( fg( fmt::color::white ) | fmt::emphasis::underline | fmt::emphasis::underline,
                        "\nExecution time" );
            fmt::print( "\n  Total time          {:.2f} {}", totalTime, m_timerUnit );
            fmt::print( fg( fmt::color::light_yellow ), "\n  Mean time           {:.2f} {}", mean, m_timerUnit );
            fmt::print( fg( fmt::color::light_blue ), "\n  Slowest             {:.2f} {}", slowest, m_timerUnit );
            fmt::print( " ({:+.2} {} ({:+.1f} %))", slowest - mean, m_timerUnit, ( slowest - mean ) * 100. / mean );
            fmt::print( fg( fmt::color::light_coral ), "\n  Fastest             {:.2f} {}", fastest, m_timerUnit );
            fmt::print( " ({:+.2} {} ({:+.1f} %))", fastest - mean, m_timerUnit, ( fastest - mean ) * 100. / mean );
            fmt::print( fg( fmt::color::white ) | fmt::emphasis::underline | fmt::emphasis::underline, "\nStats" );
            fmt::print( "\n  Variance            {:.5}", variance );
            fmt::print( "\n  Standard deviation  {:.5}", standardDeviation );

            fmt::print( "\n\n" );
        }

        return results;
    }
} // namespace rvtx
