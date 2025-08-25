#include "rvtx/system/path_interpolator.hpp"

namespace rvtx
{
    template<typename T>
    PathInterpolator<T>::PathInterpolator( Path<T> * path ) : m_path( path )
    {
    }

    template<typename T>
    PathTimeInterpolator<T>::PathTimeInterpolator( Path<T> * path ) :
        PathInterpolator<T>( path ), m_currentTime( 0.f ), m_nextTime( 0.f )
    {
    }

    template<typename T>
    T PathTimeInterpolator<T>::current()
    {
        return PathInterpolator<T>::m_path->at( m_nextTime );
    }

    template<typename T>
    float PathTimeInterpolator<T>::currentTime() const
    {
        return m_nextTime;
    }

    template<typename T>
    inline void PathTimeInterpolator<T>::setCurrentTime( const float currentTime )
    {
        m_currentTime = currentTime;
        m_nextTime    = m_currentTime;
    }

    template<typename T>
    void PathTimeInterpolator<T>::reset()
    {
        m_nextTime    = 0;
        m_currentTime = 0;
    }

    template<typename T>
    bool PathTimeInterpolator<T>::ended() const
    {
        return m_nextTime - 1.f >= -1e-6;
    }

    template<typename T>
    void PathTimeInterpolator<T>::step( const float timeStep )
    {
        m_currentTime = m_nextTime;
        m_nextTime    = glm::clamp( m_currentTime + timeStep, 0.f, 1.f );
    }

    template<typename T>
    PathKeyframeInterpolator<T>::PathKeyframeInterpolator( Path<T> * path, const float frameRate ) :
        PathInterpolator<T>( path ), m_frameRate( frameRate ),
        m_frameCount( m_frameRate * PathInterpolator<T>::m_path->getDuration() ),
        m_frameTimeStep( PathInterpolator<T>::m_path->getDuration() / ( m_frameCount - 1 ) ), m_currentFrame( 0 ),
        m_nextFrame( 0 )
    {
    }

    template<typename T>
    T PathKeyframeInterpolator<T>::current()
    {
        return PathInterpolator<T>::m_path->at( m_nextFrame * m_frameTimeStep );
    }

    template<typename T>
    inline uint32_t PathKeyframeInterpolator<T>::currentFrame() const
    {
        return m_nextFrame;
    }

    template<typename T>
    inline void PathKeyframeInterpolator<T>::setCurrentFrame( const uint32_t frame )
    {
        m_currentFrame = frame;
        m_nextFrame    = m_currentFrame;
    }

    template<typename T>
    void PathKeyframeInterpolator<T>::reset()
    {
        m_nextFrame    = 0;
        m_currentFrame = 0;
    }

    template<typename T>
    bool PathKeyframeInterpolator<T>::ended() const
    {
        return m_currentFrame >= m_frameCount - 1;
    }

    template<typename T>
    void PathKeyframeInterpolator<T>::step()
    {
        m_currentFrame = m_nextFrame;
        m_nextFrame    = glm::min<uint32_t>( m_currentFrame + 1, m_frameCount );
    }

    template<typename T>
    uint32_t PathKeyframeInterpolator<T>::getFrameCount() const
    {
        return m_frameCount;
    }

    template<typename T>
    T PathKeyframeInterpolator<T>::valueAt( const uint32_t frameIndex )
    {
        assert( frameIndex <= m_frameCount );
        return PathInterpolator<T>::m_path->at( frameIndex * m_frameTimeStep );
    }
} // namespace rvtx
