#ifndef PYRVTX_PY_GLM_HPP
#define PYRVTX_PY_GLM_HPP

#include <glm/glm.hpp>
#include <nanobind/ndarray.h>

NAMESPACE_BEGIN( NB_NAMESPACE )
NAMESPACE_BEGIN( detail )

template<glm::length_t L, typename T, glm::qualifier Q>
struct type_caster<glm::vec<L, T, Q>>
{
    using VecType       = glm::vec<L, T, Q>;
    using NDArray       = ndarray<T, numpy, shape<L>>;
    using NDArrayCaster = make_caster<NDArray>;

    NB_TYPE_CASTER( VecType, const_name( "Vec" ) + const_name<L>() + const_name<T>() )

    bool from_python( handle src, uint8_t flags, cleanup_list * cleanup ) noexcept
    {
        PyObject *  temp;
        PyObject ** o = seq_get_with_size( src.ptr(), L, &temp );

        if ( o == nullptr )
        {
            Py_XDECREF( temp );
            return false;
        }

        make_caster<T> caster;

        bool success = true;
        for ( size_t i = 0; i < L; i++ )
        {
            if ( !caster.from_python( o[ i ], flags, cleanup ) )
            {
                success = false;
                break;
            }
            value[ i ] = caster.value;
        }

        Py_XDECREF( temp );

        return success;
    }

    static handle from_cpp( VecType && v, rv_policy policy, cleanup_list * cleanup ) noexcept
    {
        if ( policy == rv_policy::automatic || policy == rv_policy::automatic_reference )
            policy = rv_policy::move;

        return from_cpp( v, policy, cleanup );
    }

    static handle from_cpp( const VecType & v, rv_policy policy, cleanup_list * cleanup ) noexcept
    {
        switch ( policy )
        {
        case rv_policy::automatic: policy = rv_policy::copy; break;
        case rv_policy::automatic_reference: policy = rv_policy::reference; break;
        case rv_policy::move: policy = rv_policy::copy; break; // Trivially copyable
        default: break;
        }

        object owner;
        if ( policy == rv_policy::reference_internal )
        {
            owner  = borrow( cleanup->self() );
            policy = rv_policy::reference;
        }

        size_t pshape[] = { L };
        object o        = steal( NDArrayCaster::from_cpp( NDArray( (void *)&v, 1, pshape, owner ), policy, cleanup ) );

        return o.release();
    }
};

template<typename T, glm::qualifier Q>
struct type_caster<glm::qua<T, Q>>
{
    using QuatType      = glm::qua<T, Q>;
    using NDArray       = ndarray<T, numpy, shape<4>>;
    using NDArrayCaster = make_caster<NDArray>;

    NB_TYPE_CASTER( QuatType, const_name( "Quat" ) + const_name<T>() )

    bool from_python( handle src, uint8_t flags, cleanup_list * cleanup ) noexcept
    {
        PyObject *  temp;
        PyObject ** o = seq_get_with_size( src.ptr(), 4, &temp );

        if ( o == nullptr )
        {
            Py_XDECREF( temp );
            return false;
        }

        make_caster<T> caster;

        bool success = true;
        for ( size_t i = 0; i < 4; i++ )
        {
            if ( !caster.from_python( o[ i ], flags, cleanup ) )
            {
                success = false;
                break;
            }

            value[ i ] = caster.value;
        }

        Py_XDECREF( temp );

        return success;
    }

    static handle from_cpp( QuatType && v, rv_policy policy, cleanup_list * cleanup ) noexcept
    {
        if ( policy == rv_policy::automatic || policy == rv_policy::automatic_reference )
            policy = rv_policy::move;

        return from_cpp( v, policy, cleanup );
    }

    static handle from_cpp( const QuatType & v, rv_policy policy, cleanup_list * cleanup ) noexcept
    {
        switch ( policy )
        {
        case rv_policy::automatic: policy = rv_policy::copy; break;
        case rv_policy::automatic_reference: policy = rv_policy::reference; break;
        case rv_policy::move: policy = rv_policy::copy; break;
        default: break;
        }

        object owner;
        if ( policy == rv_policy::reference_internal )
        {
            owner  = borrow( cleanup->self() );
            policy = rv_policy::reference;
        }

        size_t pshape[] = { 4 };
        object o        = steal( NDArrayCaster::from_cpp( NDArray( (void *)&v, 1, pshape, owner ), policy, cleanup ) );

        return o.release();
    }
};

template<glm::length_t C, glm::length_t R, typename T, glm::qualifier Q>
struct type_caster<glm::mat<C, R, T, Q>>
{
    using MatType       = glm::mat<C, R, T, Q>;
    using NDArray       = ndarray<T, numpy, shape<C, R>, f_contig>; // f_contig = column major
    using NDArrayCaster = make_caster<NDArray>;

    NB_TYPE_CASTER( MatType,
                    const_name( "Mat" ) + const_name<C>() + const_name( "x" ) + const_name<R>() + const_name<T>() )

    bool from_python( handle src, uint8_t flags, cleanup_list * cleanup ) noexcept
    {
        PyObject *  temp;
        PyObject ** o = seq_get_with_size( src.ptr(), C, &temp );

        if ( o == nullptr )
        {
            Py_XDECREF( temp );
            return false;
        }

        make_caster<glm::vec<R, T, Q>> caster;

        bool success = true;
        for ( size_t i = 0; i < C; i++ )
        {
            if ( !caster.from_python( o[ i ], flags, cleanup ) )
            {
                success = false;
                break;
            }

            memcpy( (void *)( &value[ i ] ), (void *)&caster.value, R * sizeof( T ) );
        }

        Py_XDECREF( temp );

        return success;
    }

    static handle from_cpp( MatType && v, rv_policy policy, cleanup_list * cleanup ) noexcept
    {
        if ( policy == rv_policy::automatic || policy == rv_policy::automatic_reference )
            policy = rv_policy::move;

        return from_cpp( v, policy, cleanup );
    }

    static handle from_cpp( const MatType & v, rv_policy policy, cleanup_list * cleanup ) noexcept
    {
        switch ( policy )
        {
        case rv_policy::automatic: policy = rv_policy::copy; break;
        case rv_policy::automatic_reference: policy = rv_policy::reference; break;
        case rv_policy::move: policy = rv_policy::copy; break;
        default: break;
        }

        object owner;
        if ( policy == rv_policy::reference_internal )
        {
            owner  = borrow( cleanup->self() );
            policy = rv_policy::reference;
        }

        size_t pshape[] = { C, R };
        object o        = steal( NDArrayCaster::from_cpp( NDArray( (void *)&v, 2, pshape, owner ), policy, cleanup ) );

        return o.release();
    }
};

NAMESPACE_END( detail )

NAMESPACE_END( NB_NAMESPACE )

#endif