// ==========================================================================
// SimulationCraft RL Potential Functions for Reward Shaping
// ==========================================================================

#include "rl_potential_functions.hpp"

namespace rl
{

double feral_potential( const observation_t& obs, const player_t* /*player*/ )
{
  // Placeholder: no shaping yet (returns 0)
  //
  // Future implementation ideas for Feral Druid:
  // - Return positive value when Tiger's Fury is active (encourage using it)
  // - Return positive value for each dot ticking on target (Rip, Rake)
  // - Return bonus for high combo points when ready to use finisher
  // - Return negative value for letting Rip/Rake fall off
  //
  // Example (not yet active):
  // double potential = 0.0;
  // if (obs.buff_remains_norm.size() > 0 && obs.buff_remains_norm[0] > 0)  // Tiger's Fury
  //   potential += 1000.0;
  // for (double dot_remain : obs.dot_remains_norm)
  //   if (dot_remain > 0) potential += 500.0;
  // return potential;

  (void)obs;  // Suppress unused parameter warning
  return 0.0;
}

potential_fn_t get_default_potential_fn( specialization_e spec )
{
  switch ( spec )
  {
    case DRUID_FERAL:
      return &feral_potential;
    default:
      return nullptr;
  }
}

}  // namespace rl
