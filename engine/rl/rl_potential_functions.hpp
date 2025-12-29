// ==========================================================================
// SimulationCraft RL Potential Functions for Reward Shaping
// ==========================================================================

#pragma once

#include "dbc/specialization.hpp"
#include "rl_interface.hpp"

namespace rl
{

// ============================================================================
// Potential functions for reward shaping
// ============================================================================
// Potential-based reward shaping uses: r' = r + γ*Φ(s') - Φ(s)
// This is provably policy-invariant (optimal policy unchanged).
// These placeholder functions return 0 (no shaping effect) until implemented.

/// Placeholder potential function for Feral Druid.
/// Returns 0 (no shaping). Future: return positive values for good states
/// (e.g., buffs active, dots ticking, high combo points).
double feral_potential( const observation_t& obs, const player_t* player );

/// Get the default potential function for a specialization.
/// Returns nullptr if no shaping is defined for that spec.
potential_fn_t get_default_potential_fn( specialization_e spec );

}  // namespace rl
