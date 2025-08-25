#include "rvtx/molecule/molecule.hpp"

#include <array>
#include <unordered_set>
#include <utility>

namespace rvtx
{
    Molecule::Molecule( std::string            id,
                        std::string            name,
                        std::vector<glm::vec4> data,
                        std::vector<Atom>      atoms,
                        std::vector<Bond>      bonds,
                        std::vector<Residue>   residues,
                        std::vector<Chain>     chains,
                        std::vector<float>     charge,
                        Range                  peptideBonds,
                        Range                  residentAtoms,
                        Aabb                   aabb ) :
        id { id },
        name { name }, data { std::move( data ) }, atoms { std::move( atoms ) }, bonds { std::move( bonds ) },
        residues { std::move( residues ) }, chains { std::move( chains ) }, charge { std::move( charge ) },
        peptideBonds { peptideBonds }, residentAtoms { residentAtoms }, aabb { aabb }, as {}
    {
        if ( aabb.isInvalid() )
            computeAabb();
    }

    Molecule::Molecule( Molecule && other ) noexcept
    {
        id            = std::exchange( other.id, "" );
        name          = std::exchange( other.name, "" );
        data          = std::exchange( other.data, {} );
        atoms         = std::exchange( other.atoms, {} );
        bonds         = std::exchange( other.bonds, {} );
        residues      = std::exchange( other.residues, {} );
        chains        = std::exchange( other.chains, {} );
        charge        = std::exchange( other.charge, {} );
        peptideBonds  = std::exchange( other.peptideBonds, {} );
        residentAtoms = std::exchange( other.residentAtoms, {} );
        aabb          = std::exchange( other.aabb, {} );
        as            = std::exchange( other.as, {} );
    }

    Molecule & Molecule::operator=( Molecule && other ) noexcept
    {
        std::swap( id, other.id );
        std::swap( name, other.name );
        std::swap( data, other.data );
        std::swap( atoms, other.atoms );
        std::swap( bonds, other.bonds );
        std::swap( residues, other.residues );
        std::swap( chains, other.chains );
        std::swap( charge, other.charge );
        std::swap( peptideBonds, other.peptideBonds );
        std::swap( residentAtoms, other.residentAtoms );
        std::swap( aabb, other.aabb );
        std::swap( as, other.as );

        return *this;
    }

    Aabb Molecule::getAabb() const { return aabb; }

    Aabb Molecule::getAABB() const { return getAabb(); }

    void Molecule::computeAabb()
    {
        for ( std::size_t i = residentAtoms.start; i < residentAtoms.end; i++ )
            aabb.update( data[ i ] );
    }

    std::string_view Atom::getName() const { return rvtx::getName( symbol ); }
    float            Atom::getRadius() const { return rvtx::getRadius( symbol ); }

    static constexpr std::array<std::string_view, 119> SymbolName {
        "Unknown",       // UNKNOWN = 0,
        "Hydrogen",      // H		= 1,
        "Helium",        // HE		= 2,
        "Lithium",       // LI		= 3,
        "Beryllium",     // BE		= 4,
        "Boron",         // B		= 5,
        "Carbon",        // C		= 6,
        "Nitrogen",      // N		= 7,
        "Oxygen",        // O		= 8,
        "Fluorine",      // F		= 9,
        "Neon",          // NE		= 10,
        "Sodium",        // NA		= 11,
        "Magnesium",     // MG		= 12,
        "Aluminum",      // AL		= 13,
        "Silicon",       // SI		= 14,
        "Phosphorus",    // P		= 15,
        "Sulfur",        // S		= 16,
        "Chlorine",      // CL		= 17,
        "Argon",         // AR		= 18,
        "Potassium",     // K		= 19,
        "Calcium",       // CA		= 20,
        "Scandium",      // SC		= 21,
        "Titanium",      // TI		= 22,
        "Vanadium",      // V		= 23,
        "Chromium",      // CR		= 24,
        "Manganese",     // MN		= 25,
        "Iron",          // FE		= 26,
        "Cobalt",        // CO		= 27,
        "Nickel",        // NI		= 28,
        "Copper",        // CU		= 29,
        "Zinc",          // ZN		= 30,
        "Gallium",       // GA		= 31,
        "Germanium",     // GE		= 32,
        "Arsenic",       // AS		= 33,
        "Selenium",      // SE		= 34,
        "Bromine",       // BR		= 35,
        "Krypton",       // KR		= 36,
        "Rubidium",      // RB		= 37,
        "Strontium",     // SR		= 38,
        "Yttrium",       // Y		= 39,
        "Zirconium",     // ZR		= 40,
        "Niobium",       // NB		= 41,
        "Molybdenum",    // MO		= 42,
        "Technetium",    // TC		= 43,
        "Ruthenium",     // RU		= 44,
        "Rhodium",       // RH		= 45,
        "Palladium",     // PD		= 46,
        "Silver",        // AG		= 47,
        "Cadmium",       // CD		= 48,
        "Indium",        // IN		= 49,
        "Tin",           // SN		= 50,
        "Antimony",      // SB		= 51,
        "Tellurium",     // TE		= 52,
        "Iodine",        // I		= 53,
        "Xenon",         // XE		= 54,
        "Cesium",        // CS		= 55,
        "Barium",        // BA		= 56,
        "Lanthanum",     // LA		= 57,
        "Cerium",        // CE		= 58,
        "Praseodymium",  // PR		= 59,
        "Neodymium",     // ND		= 60,
        "Promethium",    // PM		= 61,
        "Samarium",      // SM		= 62,
        "Europium",      // EU		= 63,
        "Gadolinium",    // GD		= 64,
        "Terbium",       // TB		= 65,
        "Dysprosium",    // DY		= 66,
        "Holmium",       // HO		= 67,
        "Erbium",        // ER		= 68,
        "Thulium",       // TM		= 69,
        "Ytterbium",     // YB		= 70,
        "Lutetium",      // LU		= 71,
        "Hafnium",       // HF		= 72,
        "Tantalum",      // TA		= 73,
        "Tungsten",      // W		= 74,
        "Rhenium",       // RE		= 75,
        "Osmium",        // OS		= 76,
        "Iridium",       // IR		= 77,
        "Platinum",      // PT		= 78,
        "Gold",          // AU		= 79,
        "Mercury",       // HG		= 80,
        "Thallium",      // TL		= 81,
        "Lead",          // PB		= 82,
        "Bismuth",       // BI		= 83,
        "Polonium",      // PO		= 84,
        "Astatine",      // AT		= 85,
        "Radon",         // RN		= 86,
        "Francium",      // FR		= 87,
        "Radium",        // RA		= 88,
        "Actinium",      // AC		= 89,
        "Thorium",       // TH		= 90,
        "Protactinium",  // PA		= 91,
        "Uranium",       // U		= 92,
        "Neptunium",     // NP		= 93,
        "Plutonium",     // PU		= 94,
        "Americium",     // AM		= 95,
        "Curium",        // CM		= 96,
        "Berkelium",     // BK		= 97,
        "Californium",   // CF		= 98,
        "Einsteinium",   // ES		= 99,
        "Fermium",       // FM		= 100,
        "Mendelevium",   // MD		= 101,
        "Nobelium",      // NO		= 102,
        "Lawrencium",    // LR		= 103,
        "Rutherfordium", // RF		= 104,
        "Dubnium",       // DD		= 105,
        "Seaborgium",    // SG		= 106,
        "Bohrium",       // BHJ		= 107,
        "Hassium",       // HS		= 108,
        "Meitnerium",    // MT		= 109,
        "Darmstadtium",  // DS		= 110,
        "Roentgenium",   // RG		= 111,
        "Ununbium",      // UUB		= 112,
        "Ununtrium",     // UUT		= 113,
        "Ununquadium",   // UUQ		= 114,
        "Ununpentium",   // UUP		= 115,
        "Ununhexium",    // UUH		= 116,
        "Ununseptium",   // UUS		= 117,
        "Ununoctium"     // UUO		= 118,
    };

    static constexpr std::array<float, 119> SymbolRadius {
        1.20f, // UNKNOWN	= 0,
        1.20f, // H			= 1,
        1.43f, // HE		= 2,
        2.12f, // LI		= 3,
        1.98f, // BE		= 4,
        1.91f, // B			= 5,
        1.77f, // C			= 6,
        1.66f, // N			= 7,
        1.50f, // O			= 8,
        1.46f, // F			= 9,
        1.58f, // NE		= 10,
        2.50f, // NA		= 11,
        2.51f, // MG		= 12,
        2.25f, // AL		= 13,
        2.19f, // SI		= 14,
        1.90f, // P			= 15,
        1.89f, // S			= 16,
        1.82f, // CL		= 17,
        1.83f, // AR		= 18,
        2.73f, // K			= 19,
        2.62f, // CA		= 20,
        2.58f, // SC		= 21,
        2.46f, // TI		= 22,
        2.42f, // V			= 23,
        2.45f, // CR		= 24,
        2.45f, // MN		= 25,
        2.44f, // FE		= 26,
        2.40f, // CO		= 27,
        2.40f, // NI		= 28,
        2.38f, // CU		= 29,
        2.39f, // ZN		= 30,
        2.32f, // GA		= 31,
        2.29f, // GE		= 32,
        1.88f, // AS		= 33,
        1.82f, // SE		= 34,
        1.86f, // BR		= 35,
        2.25f, // KR		= 36,
        3.21f, // RB		= 37,
        2.84f, // SR		= 38,
        2.75f, // Y			= 39,
        2.52f, // ZR		= 40,
        2.56f, // NB		= 41,
        2.45f, // MO		= 42,
        2.44f, // TC		= 43,
        2.46f, // RU		= 44,
        2.44f, // RH		= 45,
        2.15f, // PD		= 46,
        2.53f, // AG		= 47,
        2.49f, // CD		= 48,
        2.43f, // IN		= 49,
        2.42f, // SN		= 50,
        2.47f, // SB		= 51,
        1.99f, // TE		= 52,
        2.04f, // I			= 53,
        2.06f, // XE		= 54,
        3.48f, // CS		= 55,
        3.03f, // BA		= 56,
        2.98f, // LA		= 57,
        2.88f, // CE		= 58,
        2.92f, // PR		= 59,
        2.95f, // ND		= 60,
        0.00f, // PM		= 61,
        2.90f, // SM		= 62,
        2.87f, // EU		= 63,
        2.83f, // GD		= 64,
        2.79f, // TB		= 65,
        2.87f, // DY		= 66,
        2.81f, // HO		= 67,
        2.83f, // ER		= 68,
        2.79f, // TM		= 69,
        2.80f, // YB		= 70,
        2.74f, // LU		= 71,
        2.63f, // HF		= 72,
        2.53f, // TA		= 73,
        2.57f, // W			= 74,
        2.49f, // RE		= 75,
        2.48f, // OS		= 76,
        2.41f, // IR		= 77,
        2.29f, // PT		= 78,
        2.32f, // AU		= 79,
        2.45f, // HG		= 80,
        2.47f, // TL		= 81,
        2.60f, // PB		= 82,
        2.54f, // BI		= 83,
        0.00f, // PO		= 84,
        0.00f, // AT		= 85,
        0.00f, // RN		= 86,
        0.00f, // FR		= 87,
        0.00f, // RA		= 88,
        2.80f, // AC		= 89,
        2.93f, // TH		= 90,
        2.88f, // PA		= 91,
        2.71f, // U			= 92,
        2.82f, // NP		= 93,
        2.81f, // PU		= 94,
        2.83f, // AM		= 95,
        3.05f, // CM		= 96,
        3.40f, // BK		= 97,
        3.05f, // CF		= 98,
        2.70f, // ES		= 99,
        0.00f, // FM		= 100,
        0.00f, // MD		= 101,
        0.00f, // NO		= 102,
        0.00f, // LR		= 103,
        0.00f, // RF		= 104,
        0.00f, // DD		= 105,
        0.00f, // SG		= 106,
        0.00f, // BHJ		= 107,
        0.00f, // HS		= 108,
        0.00f, // MT		= 109,
        0.00f, // DS		= 110,
        0.00f, // RG		= 111,
        0.00f, // UUB		= 112,
        0.00f, // UUT		= 113,
        0.00f, // UUQ		= 114,
        0.00f, // UUP		= 115,
        0.00f, // UUH		= 116,
        0.00f, // UUS		= 117,
        0.00f  // UUO		= 118,
    };

    // Reference: https://github.com/molstar/molstar/blob/master/src/mol-model/structure/model/types/ions.ts
    const std::unordered_set<std::string_view> Ions
        = { "118", "119", "543", "1AL", "1CU", "2FK", "2HP", "2OF", "3CO", "3MT", "3NI", "3OF", "3P8", "4MO", "4PU",
            "4TI", "6MO", "ACT", "AG",  "AL",  "ALF", "AM",  "ATH", "AU",  "AU3", "AUC", "AZI", "BA",  "BCT", "BEF",
            "BF4", "BO4", "BR",  "BS3", "BSY", "CA",  "CAC", "CD",  "CD1", "CD3", "CD5", "CE",  "CF",  "CHT", "CL",
            "CO",  "CO3", "CO5", "CON", "CR",  "CS",  "CSB", "CU",  "CU1", "CU3", "CUA", "CUZ", "CYN", "DME", "DMI",
            "DSC", "DTI", "DY",  "E4N", "EDR", "EMC", "ER3", "EU",  "EU3", "F",   "FE",  "FE2", "FPO", "GA",  "GD3",
            "GEP", "HAI", "HG",  "HGC", "IN",  "IOD", "IR",  "IR3", "IRI", "IUM", "K",   "KO4", "LA",  "LCO", "LCP",
            "LI",  "LU",  "MAC", "MG",  "MH2", "MH3", "MLI", "MMC", "MN",  "MN3", "MN5", "MN6", "MO1", "MO2", "MO3",
            "MO4", "MO5", "MO6", "MOO", "MOS", "MOW", "MW1", "MW2", "MW3", "NA",  "NA2", "NA5", "NA6", "NAO", "NAW",
            "ND",  "NET", "NH4", "NI",  "NI1", "NI2", "NI3", "NO2", "NO3", "NRU", "O4M", "OAA", "OC1", "OC2", "OC3",
            "OC4", "OC5", "OC6", "OC7", "OC8", "OCL", "OCM", "OCN", "OCO", "OF1", "OF2", "OF3", "OH",  "OS",  "OS4",
            "OXL", "PB",  "PBM", "PD",  "PDV", "PER", "PI",  "PO3", "PO4", "PR",  "PT",  "PT4", "PTN", "RB",  "RH3",
            "RHD", "RU",  "SB",  "SCN", "SE4", "SEK", "SM",  "SMO", "SO3", "SO4", "SR",  "T1A", "TB",  "TBA", "TCN",
            "TEA", "TH",  "THE", "TL",  "TMA", "TRA", "UNX", "V",   "VN3", "VO4", "W",   "WO5", "Y1",  "YB",  "YB2",
            "YH",  "YT3", "ZCM", "ZN",  "ZN2", "ZN3", "ZNO", "ZO3", "ZR",  "NCO", "OHX" };

    std::string_view getName( Symbol symbol ) { return SymbolName[ static_cast<uint8_t>( symbol ) ]; }
    float            getRadius( Symbol symbol ) { return SymbolRadius[ static_cast<uint8_t>( symbol ) ]; }
    bool             isIon( std::string_view name ) { return Ions.find( name ) != Ions.end(); }
    bool             isH2O( std::string_view name ) { return name == "HOH"; }

    void Molecule::buildAccelerationStructure( const std::size_t gridSize ) { as.build( *this, gridSize ); }

    float Molecule::computeCharge( const glm::vec3 & position, const float minDist ) const
    {
        float averageCharge = 0;

        if ( as.isBuilt() )
        {
            const std::vector<std::size_t> neighborsIds = as.getNear( position, minDist );

            for ( const std::size_t i : neighborsIds )
            {
                const glm::vec3 currentAtomPosition = { data[ i ].x, data[ i ].y, data[ i ].z };
                const float     currentAtomCharge   = charge[ i ];

                const float distance = glm::distance( position, currentAtomPosition );

                averageCharge += currentAtomCharge / ( distance * distance );
            }
        }
        else
        {
            for ( std::size_t i = 0; i < atoms.size(); i++ )
            {
                const glm::vec3 currentAtomPosition = { data[ i ].x, data[ i ].y, data[ i ].z };
                const float     currentAtomCharge   = charge[ i ];

                const float distance = glm::distance( position, currentAtomPosition );

                if ( distance <= minDist )
                    averageCharge += currentAtomCharge / ( distance * distance );
            }
        }

        return averageCharge;
    }
} // namespace rvtx
