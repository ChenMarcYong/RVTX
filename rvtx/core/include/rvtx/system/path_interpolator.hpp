#ifndef RVTX_CORE_PATH_INTERPOLATOR_HPP
#define RVTX_CORE_PATH_INTERPOLATOR_HPP

#include "rvtx/core/path.hpp"

namespace rvtx
{
    template<typename T>
    class PathInterpolator
    {
      public:
        PathInterpolator() = default;
        PathInterpolator( Path<T> * path );
        virtual ~PathInterpolator() = default;

        virtual T    current()     = 0;
        virtual void reset()       = 0;
        virtual bool ended() const = 0;

      protected:
        Path<T> * m_path;
    };

    /**
     * \brief A path interpolator based on time.
     *
     * You can use the interpolator like so :
     * Path<glm::vec3> path {...};
     * PathTimeInterpolator pathInterpolator{ path };
     * do {
     *    glm::vec3 value = path.current();
     *
     *    // Do things with value
     *
     *    path.step( timeStep ); // --> ! Do not forget to step or it will lead to infinite loop !
     * } while ( !pathInterpolator.ended() );
     *
     * Note : current gives the value of the time with the last 'step' added. This is done so
     * it is guaranteed that the past goes from exactly from the first point to the last point.
     */
    template<typename T>
    class PathTimeInterpolator : public PathInterpolator<T>
    {
      public:
        PathTimeInterpolator() = default;
        PathTimeInterpolator( Path<T> * path );

        T     current() override;                        // Gets current value
        float currentTime() const;                       // Gets current time
        void  setCurrentTime( const float currentTime ); // Gets current time
        void  reset() override;                          // Resets interpolator time
        bool  ended() const override;                    // Checks if the path has ended
        void  step( const float timeStep );              // Queue 'timeStep' step for next 'current()' call

      protected:
        float m_currentTime;
        float m_nextTime;
    };

    /**
     * \brief A path interpolator based on a predefined number of frames.
     *
     * You can use the interpolator like so :
     * Path<glm::vec3> path {...};
     * PathKeyframeInterpolator pathInterpolator{ path };
     * do {
     *    glm::vec3 value = path.current();
     *
     *    // Do things with value
     *
     *    path.step(); // --> ! Do not forget to step or it will lead to infinite loop !
     * } while ( !pathInterpolator.ended() );
     *
     * Note : current gives the value of the frame with the last 'step' frame added. This is done so
     * it is guaranteed that the past goes from exactly from the first point to the last point. It is
     * also guaranteed that the path will exactly be 'getFrameCount()' frames from start to finish.
     */
    template<typename T>
    class PathKeyframeInterpolator : public PathInterpolator<T>
    {
      public:
        PathKeyframeInterpolator( Path<T> * path, const float frameRate );

        T        current() override;                      // Gets current value
        uint32_t currentFrame() const;                    // Gets current frame
        void     setCurrentFrame( const uint32_t frame ); // Sets current frame
        void     reset() override;                        // Resets interpolator time
        bool     ended() const override;                  // Checks if the path has ended
        void     step();                                  // Queue frame step for next 'current()' call

        uint32_t getFrameCount() const;                // Returns the frame count of the interpolator
        T        valueAt( const uint32_t frameIndex ); // Returns the value at 'frameIndex'

      protected:
        float    m_frameRate;
        uint32_t m_frameCount;
        float    m_frameTimeStep;
        uint32_t m_currentFrame;
        uint32_t m_nextFrame;
    };
} // namespace rvtx

#include "rvtx/system/path_interpolator.inl"

#endif // RVTX_CORE_PATH_INTERPOLATOR_HPP
