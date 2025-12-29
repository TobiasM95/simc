// ==========================================================================
// SimulationCraft RL Spec-Specific Observation Configuration
// ==========================================================================

#pragma once

#include "config.hpp"

#include "dbc/specialization.hpp"
#include "sc_enums.hpp"

#include <string>
#include <vector>

struct buff_t;
struct dot_t;
struct player_t;

namespace rl
{

// ============================================================================
// Spec-specific observation configuration
// ============================================================================

/// Configuration defining which buffs, dots, and resources to observe for a spec.
/// Each spec has a static list of buff/dot names; pointers are cached on first use.
struct spec_obs_config_t
{
  // Names of buffs/dots to track (defines observation vector order)
  std::vector<std::string> buff_names;  // Player buffs to track
  std::vector<std::string> dot_names;   // Dots on target to track (by dot name string)
  resource_e special_resource;          // Primary special resource (e.g., RESOURCE_COMBO_POINT)

  // Cached pointers (populated on first use, per-player)
  // Mutable because caching happens during const observation building
  mutable std::vector<buff_t*> cached_buffs;
  mutable std::vector<dot_t*> cached_dots;
  mutable bool buffs_cached                 = false;
  mutable bool dots_cached                  = false;
  mutable const player_t* cached_for_player = nullptr;
  mutable const player_t* cached_for_target = nullptr;

  // Clear cached pointers (call when player/target changes)
  void clear_cache() const
  {
    cached_buffs.clear();
    cached_dots.clear();
    buffs_cached      = false;
    dots_cached       = false;
    cached_for_player = nullptr;
    cached_for_target = nullptr;
  }
};

/// Get the observation config for a specialization (read-only).
/// Returns a reference to a static config; empty config for unknown specs.
const spec_obs_config_t& get_spec_obs_config( specialization_e spec );

/// Get a mutable reference to allow caching (called during observation building).
spec_obs_config_t& get_mutable_spec_obs_config( specialization_e spec );

/// Cache buff pointers for a player (call once at first RL decision).
/// Searches player->buff_list for each buff name and caches the pointers.
void cache_buff_pointers( spec_obs_config_t& config, const player_t& player );

/// Cache dot pointers for a player's target (call once per target).
/// Searches target->dot_list for each dot name and caches the pointers.
void cache_dot_pointers( spec_obs_config_t& config, const player_t& player, const player_t* target );

}  // namespace rl
