// ==========================================================================
// SimulationCraft RL Spec-Specific Observation Configuration
// ==========================================================================

#include "rl_spec_config.hpp"

#include "action/dot.hpp"
#include "buff/buff.hpp"
#include "player/player.hpp"

#include <unordered_map>

namespace rl
{

namespace
{

// Default empty config for unknown specs
spec_obs_config_t g_default_config = {
    {},            // No buffs
    {},            // No dots
    RESOURCE_NONE  // No special resource
};

// Static configs per spec (populated on first access)
std::unordered_map<specialization_e, spec_obs_config_t> g_spec_configs;

void init_spec_configs()
{
  static bool initialized = false;
  if ( initialized )
    return;
  initialized = true;

  // ============================================================================
  // DRUID_FERAL
  // ============================================================================
  g_spec_configs[ DRUID_FERAL ] = { // Player buffs to track (order matters - defines observation indices)
                                    {
                                        "tigers_fury",
                                        "bloodtalons",
                                        "clearcasting_cat",
                                        "berserk_cat",
                                        "incarnation_cat",
                                        "predatory_swiftness",
                                        "sudden_ambush",
                                        "apex_predators_craving",
                                        "savage_fury",
                                        "prowl",  // Stealth state
                                    },
                                    // Dots on target to track
                                    {
                                        "rip",
                                        "rake",
                                        "thrash_cat",
                                        "lunar_inspiration",  // Moonfire in cat form
                                        "feral_frenzy_tick",  // The actual dot name per sc_druid.cpp
                                    },
                                    RESOURCE_COMBO_POINT };

  // ============================================================================
  // TODO: Add other specs as needed
  // ============================================================================
  // Example template for future specs:
  //
  // g_spec_configs[ ROGUE_ASSASSINATION ] = {
  //     { "slice_and_dice", "envenom", ... },
  //     { "rupture", "garrote", "deadly_poison", ... },
  //     RESOURCE_COMBO_POINT
  // };
}

}  // anonymous namespace

const spec_obs_config_t& get_spec_obs_config( specialization_e spec )
{
  init_spec_configs();
  auto it = g_spec_configs.find( spec );
  if ( it != g_spec_configs.end() )
    return it->second;
  return g_default_config;
}

spec_obs_config_t& get_mutable_spec_obs_config( specialization_e spec )
{
  init_spec_configs();
  auto it = g_spec_configs.find( spec );
  if ( it != g_spec_configs.end() )
    return it->second;
  return g_default_config;
}

void cache_buff_pointers( spec_obs_config_t& config, const player_t& player )
{
  if ( config.buffs_cached && config.cached_for_player == &player )
    return;

  config.cached_buffs.clear();
  config.cached_buffs.reserve( config.buff_names.size() );

  for ( const auto& name : config.buff_names )
  {
    // buff_t::find searches player->buff_list by name
    // source=nullptr means match any source
    buff_t* b = buff_t::find( const_cast<player_t*>( &player ), name, nullptr );
    config.cached_buffs.push_back( b );  // nullptr if not found (not talented, etc.)
  }

  config.buffs_cached      = true;
  config.cached_for_player = &player;
}

void cache_dot_pointers( spec_obs_config_t& config, const player_t& player, const player_t* target )
{
  if ( !target )
    return;
  if ( config.dots_cached && config.cached_for_target == target && config.cached_for_player == &player )
    return;

  config.cached_dots.clear();
  config.cached_dots.reserve( config.dot_names.size() );

  for ( const auto& name : config.dot_names )
  {
    // target->find_dot(name, source) searches target's dot_list for dots from this source
    dot_t* d = target->find_dot( name, const_cast<player_t*>( &player ) );
    config.cached_dots.push_back( d );  // nullptr if not found (dot never applied)
  }

  config.dots_cached       = true;
  config.cached_for_target = target;
}

}  // namespace rl
