#include <stdexcept>

#include <glm/glm.hpp>

#include "rvtx/core/path.hpp"

namespace rvtx
{
    template<typename T>
    Path<T>::Path( const std::vector<T> & values, const float duration, const SplineType splineType ) :
        m_values( values ), m_duration( duration ), m_segmentDuration( m_duration / ( m_values.size() - 1 ) ),
        splineType( splineType )
    {
        if ( m_values.size() < 2 )
        {
            throw std::runtime_error( "Path must have at least 2 values." );
        }
    }

    template<typename T>
    T Path<T>::at( const float time ) const
    {
        const std::size_t index = time / m_segmentDuration;
        const float       t     = ( time - ( index * m_segmentDuration ) ) / m_segmentDuration;

        switch ( splineType )
        {
        case SplineType::Linear: return atLinear( index, t );
        case SplineType::CatmullRom: return atCatmullRom( index, t );
        default: assert( false && "Input spline type does not exist." ); return T(); // Should not happen
        }
    }

    template<typename T>
    float Path<T>::getDuration() const
    {
        return m_duration;
    }

    template<typename T>
    void Path<T>::setDuration( const float duration )
    {
        m_duration        = duration;
        m_segmentDuration = m_duration / m_values.size();
    }

    template<typename T>
    const std::vector<T> & Path<T>::getValues() const
    {
        return m_values;
    }

    template<typename T>
    inline void Path<T>::setValues( const std::vector<T> & values )
    {
        m_values          = values;
        m_segmentDuration = m_duration / ( m_values.size() - 1 );
    }

    template<typename T>
    const T & Path<T>::getValueAt( const std::size_t index ) const
    {
        return m_values[ index ];
    }

    template<typename T>
    std::vector<T> Path<T>::sample( const std::size_t numSamples ) const
    {
        std::vector<T> samples;
        samples.reserve( numSamples );

        for ( std::size_t i = 0; i < numSamples; ++i )
            samples.emplace_back( at( i * m_duration / ( numSamples - 1 ) ) );

        return samples;
    }

    template<typename T>
    T Path<T>::atCatmullRom( const std::size_t index, const float t ) const
    {
        const size_t i1 = std::min<size_t>( index, m_values.size() - 1 );
        const size_t i0 = glm::max<long long>( static_cast<long long>( i1 ) - 1, 0 );
        const size_t i2 = glm::min<size_t>( i1 + 1, m_values.size() - 1 );
        const size_t i3 = glm::min<size_t>( i1 + 2, m_values.size() - 1 );

        return catmullRom( m_values[ i0 ], m_values[ i1 ], m_values[ i2 ], m_values[ i3 ], t );
    }

    template<typename T>
    T Path<T>::atLinear( const std::size_t index, const float t ) const
    {
        const size_t i1 = std::min<size_t>( index, m_values.size() - 1 );
        const size_t i2 = glm::min<size_t>( i1 + 1, m_values.size() - 1 );

        return linear( m_values[ i1 ], m_values[ i2 ], t );
    }
} // namespace rvtx
