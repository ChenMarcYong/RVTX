#ifndef RVTX_MOLECULE_LOADER_HPP
#define RVTX_MOLECULE_LOADER_HPP

#include "rvtx/core/filesystem.hpp"
#include "rvtx/molecule/molecule.hpp"

namespace rvtx
{
    Molecule load( const std::filesystem::path & path );
    Molecule load( const std::string_view buffer, const std::string & extension );
} // namespace rvtx

#endif // RVTX_MOLECULE_LOADER_HPP
