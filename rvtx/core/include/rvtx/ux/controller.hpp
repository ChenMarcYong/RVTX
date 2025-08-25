#ifndef RVTX_UX_CONTROLLER_HPP
#define RVTX_UX_CONTROLLER_HPP

#include <memory>
#include <vector>

namespace rvtx
{
    class Input;

    class Controller
    {
      public:
        virtual bool update( const Input & ) { return false; }

        virtual ~Controller() = default;
    };

    class ControllerForwarder : public Controller
    {
      public:
        ControllerForwarder()           = default;
        ~ControllerForwarder() override = default;

        template<class DerivedController, class... Args>
        DerivedController & add( Args &&... args )
        {
            static_assert( std::is_base_of<Controller, DerivedController>::value,
                           "DerivedController must be based on Controller." );

            auto derivedController  = std::make_unique<DerivedController>( std::forward<Args>( args )... );
            DerivedController * ptr = derivedController.get();
            controllers.emplace_back( std::move( derivedController ) );
            return *ptr;
        }

        bool update( const Input & input ) override
        {
            bool result = false;
            for ( auto & controller : controllers )
                result |= controller->update( input );
            return result;
        }

      private:
        std::vector<std::unique_ptr<Controller>> controllers;
    };
} // namespace rvtx

#endif // RVTX_UX_CONTROLLER_HPP
