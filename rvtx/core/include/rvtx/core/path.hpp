#ifndef RVTX_CORE_PATH_HPP
#define RVTX_CORE_PATH_HPP

#include "rvtx/core/spline.hpp"
#include "rvtx/core/type.hpp"

namespace rvtx
{
    template<typename T>
    class Path
    {
      public:
        Path() = default;
        Path( const std::vector<T> & values,
              const float            duration   = 1.f,
              const SplineType       splineType = SplineType::CatmullRom );

        T at( const float time ) const;

        float                  getDuration() const;
        void                   setDuration( const float duration );
        const std::vector<T> & getValues() const;
        void                   setValues( const std::vector<T> & values );
        const T &              getValueAt( const std::size_t index ) const;

        std::vector<T> sample( const std::size_t numSamples ) const;

        SplineType splineType;

      private:
        std::vector<T> m_values;
        float          m_duration;
        float          m_segmentDuration;

        T atCatmullRom( const std::size_t index, const float t ) const;
        T atLinear( const std::size_t index, const float t ) const;
    };
} // namespace rvtx

#include "rvtx/core/path.inl"

#endif // RVTX_CORE_PATH_HPP
