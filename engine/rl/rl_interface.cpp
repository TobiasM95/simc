// ==========================================================================
// SimulationCraft RL interface (experimental)
// ==========================================================================

#include "rl_interface.hpp"

#include "action/action.hpp"
#include "player/player.hpp"
#include "sim/cooldown.hpp"
#include "sim/sim.hpp"
#include "util/io.hpp"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <set>

namespace
{
// ============================================================================
// Wait pseudo-actions
// ============================================================================
// We expose wait actions with fixed durations as pseudo-actions.
// These allow the RL agent to intentionally wait/pool for specific times.
constexpr double WAIT_DURATIONS[]             = { 0.1, 0.2, 0.5, 1.0, 1.5 };
constexpr std::size_t NUM_WAIT_PSEUDO_ACTIONS = sizeof( WAIT_DURATIONS ) / sizeof( WAIT_DURATIONS[ 0 ] );

const char* WAIT_LABELS[] = { "wait_0.1", "wait_0.2", "wait_0.5", "wait_1.0", "wait_1.5" };
// ============================================================================
// Tracing helpers
// ============================================================================
thread_local std::unique_ptr<io::ofstream> tl_trace_stream;
thread_local std::string tl_trace_path;

std::string json_escape( const std::string& s )
{
  std::string out;
  out.reserve( s.size() + 8 );
  for ( unsigned char c : s )
  {
    switch ( c )
    {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if ( c < 0x20 )
        {
          char buf[ 7 ];
          std::snprintf( buf, sizeof( buf ), "\\u%04x", static_cast<unsigned>( c ) );
          out += buf;
        }
        else
        {
          out.push_back( static_cast<char>( c ) );
        }
        break;
    }
  }
  return out;
}

std::string trace_path_for_sim( const sim_t& sim )
{
  if ( sim.threads > 0 )
    return fmt::format( "{}.{}.jsonl", sim.rl_trace_file, sim.thread_index );
  return sim.rl_trace_file;
}

io::ofstream* get_trace_stream( const sim_t& sim )
{
  if ( !sim.rl_trace )
    return nullptr;

  const std::string desired_path = trace_path_for_sim( sim );
  if ( desired_path.empty() )
    return nullptr;

  if ( !tl_trace_stream || tl_trace_path != desired_path )
  {
    tl_trace_stream = std::make_unique<io::ofstream>();
    tl_trace_stream->open( desired_path, std::ios_base::out | std::ios_base::app );
    tl_trace_path = desired_path;
  }

  if ( !tl_trace_stream || !tl_trace_stream->is_open() )
    return nullptr;

  return tl_trace_stream.get();
}

// ============================================================================
// Utility
// ============================================================================
rl::policy_fn_t g_policy = &rl::dummy_policy;
void* g_policy_user_data = nullptr;

// Mutex for stdio operations (stdin/stdout) to ensure thread-safety
std::mutex g_stdio_mutex;

double clamp01( double x )
{
  return std::max( 0.0, std::min( 1.0, x ) );
}

double norm01( double value, double denom )
{
  if ( denom <= 0.0 )
    return 0.0;
  return clamp01( value / denom );
}

}  // anonymous namespace

namespace rl
{

// ============================================================================
// Policy management
// ============================================================================
policy_fn_t get_policy()
{
  return g_policy;
}

void* get_policy_user_data()
{
  return g_policy_user_data;
}

void set_policy( policy_fn_t fn, void* user_data )
{
  g_policy           = fn ? fn : &dummy_policy;
  g_policy_user_data = user_data;
}

std::size_t dummy_policy( const step_input_t& input, void* /*user_data*/ )
{
  // Default: pick a pseudo-random legal action using a thread-local round-robin counter
  std::vector<std::size_t> legal;
  for ( std::size_t i = 0; i < input.action_space.action_mask.size(); ++i )
  {
    if ( input.action_space.action_mask[ i ] )
      legal.push_back( i );
  }

  if ( legal.empty() )
    return input.action_space.actions.size();  // None legal

  static thread_local uint64_t tl_counter = 0;
  return legal[ ( tl_counter++ ) % legal.size() ];
}

// ============================================================================
// Stdio bridge policy (for external Python RL agents)
// ============================================================================
namespace
{
void write_step_json_to_stream( std::ostream& out, const step_input_t& input )
{
  out << "{";
  out << "\"type\":\"step\",";
  out << "\"t\":" << input.observation.time_s << ",";
  out << "\"fight_len\":" << input.observation.fight_length_s << ",";
  out << "\"time_rem\":" << input.observation.time_remaining_s << ",";
  out << "\"time_rem_n\":" << input.observation.time_remaining_norm << ",";
  out << "\"ttd\":" << input.observation.target_ttd_s << ",";
  out << "\"ttd_n\":" << input.observation.target_ttd_norm << ",";
  out << "\"gcd_rem\":" << input.observation.gcd_remaining_s << ",";
  out << "\"gcd_rem_n\":" << input.observation.gcd_remaining_norm << ",";
  out << "\"reward\":" << input.reward << ",";

  // Resources
  out << "\"resource_pct\":[";
  for ( std::size_t i = 0; i < input.observation.resource_pct.size(); ++i )
  {
    if ( i )
      out << ",";
    out << input.observation.resource_pct[ i ];
  }
  out << "],";

  // Action space
  out << "\"n\":" << input.action_space.actions.size() << ",";

  out << "\"mask\":[";
  for ( std::size_t i = 0; i < input.action_space.action_mask.size(); ++i )
  {
    if ( i )
      out << ",";
    out << static_cast<int>( input.action_space.action_mask[ i ] );
  }
  out << "],";

  out << "\"labels\":[";
  for ( std::size_t i = 0; i < input.action_space.action_labels.size(); ++i )
  {
    if ( i )
      out << ",";
    out << "\"" << json_escape( input.action_space.action_labels[ i ] ) << "\"";
  }
  out << "],";

  out << "\"cd_rem_s\":[";
  for ( std::size_t i = 0; i < input.action_space.cooldown_remains_s.size(); ++i )
  {
    if ( i )
      out << ",";
    out << input.action_space.cooldown_remains_s[ i ];
  }
  out << "],";

  out << "\"cd_rem_n\":[";
  for ( std::size_t i = 0; i < input.action_space.cooldown_remains_norm.size(); ++i )
  {
    if ( i )
      out << ",";
    out << input.action_space.cooldown_remains_norm[ i ];
  }
  out << "],";

  out << "\"cd_charges_f\":[";
  for ( std::size_t i = 0; i < input.action_space.cooldown_charges_frac.size(); ++i )
  {
    if ( i )
      out << ",";
    out << input.action_space.cooldown_charges_frac[ i ];
  }
  out << "]";

  out << "}\n";
  out.flush();
}
}  // anonymous namespace

std::size_t stdio_policy( const step_input_t& input, void* /*user_data*/ )
{
  std::lock_guard<std::mutex> lock( g_stdio_mutex );

  // Write state as JSON line to stdout
  write_step_json_to_stream( std::cout, input );

  // Read action index from stdin
  std::size_t action_index = input.action_space.actions.size();  // Default to invalid (no action)
  if ( std::cin >> action_index )
  {
    // Validate action is legal
    if ( action_index < input.action_space.action_mask.size() && input.action_space.action_mask[ action_index ] )
    {
      return action_index;
    }
  }

  // Fallback: return first legal action
  for ( std::size_t i = 0; i < input.action_space.action_mask.size(); ++i )
  {
    if ( input.action_space.action_mask[ i ] )
      return i;
  }

  return input.action_space.actions.size();
}

void write_episode_end( double total_damage, double fight_length_s )
{
  std::lock_guard<std::mutex> lock( g_stdio_mutex );
  std::cout << "{\"type\":\"done\",\"total_damage\":" << total_damage << ",\"fight_length\":" << fight_length_s
            << "}\n";
  std::cout.flush();
}

// ============================================================================
// Action filtering
// ============================================================================
bool is_exposed_action( const action_t& a )
{
  // Exclude background/internal actions
  if ( a.background )
    return false;

  // Exclude proc-triggered actions (e.g., convoke sub-spells, trinket procs)
  // These are not directly castable by the player but triggered by other effects.
  if ( a.proc )
    return false;

  // Exclude dual/child actions (e.g., tick actions, impact actions)
  // These are auxiliary actions that are part of another action's execution.
  if ( a.dual )
    return false;

  // Exclude actions not from APL parsing (secondary/internal actions).
  // Actions created programmatically (get_secondary_action, etc.) have action_list == nullptr.
  // Only actions parsed from APL strings have action_list set.
  if ( a.action_list == nullptr )
    return false;

  // Exclude APL control-flow actions
  if ( a.type == ACTION_CALL || a.type == ACTION_VARIABLE || a.type == ACTION_SEQUENCE )
    return false;

  // Exclude wait barriers
  if ( a.option.wait_on_ready == 1 )
    return false;

  // Exclude wait and pool_resource actions - we expose these as pseudo-actions with
  // fixed durations instead (wait_0.1, wait_0.2, etc.) to give the RL agent control.
  if ( a.name_str == "wait" || a.name_str == "pool_resource" )
    return false;

  // Exclude actions that require a resource this player doesn't use.
  // This prevents issues with e.g., Balance druid spells on a Feral character.
  resource_e res = a.current_resource();
  if ( res != RESOURCE_NONE && a.base_cost() > 0 )
  {
    if ( !a.player->resources.is_active( res ) )
      return false;
  }

  // Keep combat actions: spells, attacks, heals, absorbs, on-use items
  if ( a.type == ACTION_SPELL || a.type == ACTION_ATTACK || a.type == ACTION_HEAL || a.type == ACTION_ABSORB ||
       a.type == ACTION_USE || a.type == ACTION_OTHER )
    return true;

  return false;
}

// ============================================================================
// Observation building
// ============================================================================
observation_t build_observation( const player_t& player )
{
  observation_t obs;

  const sim_t* sim = player.sim;
  if ( !sim )
    return obs;

  obs.time_s         = sim->current_time().total_seconds();
  obs.fight_length_s = sim->expected_max_time();

  // GCD remaining
  obs.gcd_remaining_s    = std::max( 0.0, ( player.gcd_ready - sim->current_time() ).total_seconds() );
  const double base_gcd  = std::max( 0.0, player.base_gcd.total_seconds() );
  obs.gcd_remaining_norm = norm01( obs.gcd_remaining_s, base_gcd > 0.0 ? base_gcd : 1.5 );

  // Time remaining
  obs.time_remaining_s    = std::max( 0.0, obs.fight_length_s - obs.time_s );
  obs.time_remaining_norm = norm01( obs.time_remaining_s, obs.fight_length_s );

  // Target TTD
  if ( sim->target )
    obs.target_ttd_s = sim->target->time_to_percent( 0.0 ).total_seconds();
  else
    obs.target_ttd_s = obs.time_remaining_s;
  obs.target_ttd_norm = norm01( obs.target_ttd_s, obs.fight_length_s );

  // Resources
  for ( resource_e r = RESOURCE_NONE; r < RESOURCE_MAX; ++r )
    obs.resource_pct[ r ] = player.resources.pct( r );

  return obs;
}

// ============================================================================
// Action list building (one-time, deduplicated by name)
// ============================================================================
void build_action_list( const player_t& player, std::vector<action_t*>& out_actions,
                        std::vector<std::string>& out_labels )
{
  out_actions.clear();
  out_labels.clear();

  std::set<std::string> seen_names;

  for ( action_t* a : player.action_list )
  {
    if ( !a )
      continue;
    if ( !is_exposed_action( *a ) )
      continue;

    // Deduplicate by name
    if ( seen_names.count( a->name_str ) )
      continue;
    seen_names.insert( a->name_str );

    out_actions.push_back( a );
    out_labels.push_back( a->name_str );
  }

  // Append wait pseudo-actions (nullptr action pointer, special label)
  // These allow the RL agent to intentionally wait for specific durations.
  for ( std::size_t i = 0; i < NUM_WAIT_PSEUDO_ACTIONS; ++i )
  {
    out_actions.push_back( nullptr );  // No actual action_t for pseudo-actions
    out_labels.push_back( WAIT_LABELS[ i ] );
  }
}

std::size_t get_num_wait_pseudo_actions()
{
  return NUM_WAIT_PSEUDO_ACTIONS;
}

double get_wait_pseudo_action_duration( std::size_t pseudo_index )
{
  if ( pseudo_index < NUM_WAIT_PSEUDO_ACTIONS )
    return WAIT_DURATIONS[ pseudo_index ];
  return 0.0;
}

// ============================================================================
// Feature update (per-step)
// ============================================================================
void update_action_features( const player_t& player, execute_type context, util::span<action_t* const> actions,
                             std::vector<uint8_t>& out_mask, std::vector<double>& out_cd_remains_s,
                             std::vector<double>& out_cd_remains_norm, std::vector<double>& out_cd_charges_frac )
{
  const std::size_t n = actions.size();
  out_mask.resize( n );
  out_cd_remains_s.resize( n );
  out_cd_remains_norm.resize( n );
  out_cd_charges_frac.resize( n );

  const double fight_len = player.sim ? player.sim->expected_max_time() : 60.0;
  const double cd_norm   = fight_len > 0.0 ? fight_len : 60.0;

  for ( std::size_t i = 0; i < n; ++i )
  {
    action_t* a = actions[ i ];

    // Handle wait pseudo-actions (nullptr action pointers)
    if ( a == nullptr )
    {
      // Wait pseudo-actions are legal in FOREGROUND context, never in OFF_GCD or CWC
      bool legal               = ( context == execute_type::FOREGROUND );
      out_mask[ i ]            = static_cast<uint8_t>( legal ? 1 : 0 );
      out_cd_remains_s[ i ]    = 0.0;
      out_cd_remains_norm[ i ] = 0.0;
      out_cd_charges_frac[ i ] = 1.0;
      continue;
    }

    // First check if this action is valid for the current execution context.
    // This is critical: off_gcd context can only use off_gcd actions, etc.
    bool context_valid = false;
    switch ( context )
    {
      case execute_type::OFF_GCD:
        // Off-GCD actions must have use_off_gcd=true and trigger_gcd=0
        context_valid = a->use_off_gcd && a->trigger_gcd == timespan_t::zero();
        break;
      case execute_type::CAST_WHILE_CASTING:
        // Cast-while-casting actions must have usable_while_casting=true
        context_valid = a->usable_while_casting && a->use_while_casting;
        break;
      case execute_type::FOREGROUND:
      default:
        // Foreground: action must trigger a GCD (or be instant but not off-gcd only)
        // Simplification: allow actions that aren't restricted to off_gcd/cwc contexts
        context_valid = true;
        break;
    }

    bool legal = false;
    if ( context_valid )
    {
      // Check that the action's resource is usable by this player.
      // If an action costs a resource this player spec doesn't use (e.g., Astral Power on a Feral Druid),
      // calling ready() would trigger a debug assertion in resource_available().
      resource_e res = a->current_resource();
      if ( res != RESOURCE_NONE && a->base_cost() > 0 )
      {
        // If the player doesn't have this resource active, skip this action
        if ( !a->player->resources.is_active( res ) )
        {
          out_mask[ i ]            = 0;
          out_cd_remains_s[ i ]    = 0.0;
          out_cd_remains_norm[ i ] = 0.0;
          out_cd_charges_frac[ i ] = 1.0;
          continue;
        }
      }

      // Check game legality: cooldown/resources/target
      player_t* t = a->target ? a->target : a->player->target;
      legal       = a->ready() && t && a->target_ready( t );
    }

    out_mask[ i ] = static_cast<uint8_t>( legal ? 1 : 0 );

    // Cooldown features
    const double cd_s        = a->cooldown ? a->cooldown->remains().total_seconds() : 0.0;
    out_cd_remains_s[ i ]    = std::max( 0.0, cd_s );
    out_cd_remains_norm[ i ] = norm01( std::max( 0.0, cd_s ), cd_norm );
    out_cd_charges_frac[ i ] = a->cooldown ? a->cooldown->charges_fractional() : 1.0;
  }
}

// ============================================================================
// Tracing
// ============================================================================
void trace_decision( const step_input_t& input, std::size_t chosen_index )
{
  if ( !input.player || !input.player->sim )
    return;

  const sim_t& sim  = *input.player->sim;
  io::ofstream* out = get_trace_stream( sim );
  if ( !out )
    return;

  const std::string chosen_label = ( chosen_index < input.action_space.action_labels.size() )
                                       ? input.action_space.action_labels[ chosen_index ]
                                       : std::string();

  ( *out ) << "{";
  ( *out ) << "\"t\":" << input.observation.time_s << ",";
  ( *out ) << "\"fight_len\":" << input.observation.fight_length_s << ",";
  ( *out ) << "\"time_rem\":" << input.observation.time_remaining_s << ",";
  ( *out ) << "\"time_rem_n\":" << input.observation.time_remaining_norm << ",";
  ( *out ) << "\"ttd\":" << input.observation.target_ttd_s << ",";
  ( *out ) << "\"ttd_n\":" << input.observation.target_ttd_norm << ",";
  ( *out ) << "\"gcd_rem\":" << input.observation.gcd_remaining_s << ",";
  ( *out ) << "\"gcd_rem_n\":" << input.observation.gcd_remaining_norm << ",";
  ( *out ) << "\"reward\":" << input.reward << ",";

  // Resources
  ( *out ) << "\"resource_pct\":[";
  for ( std::size_t i = 0; i < input.observation.resource_pct.size(); ++i )
  {
    if ( i )
      ( *out ) << ",";
    ( *out ) << input.observation.resource_pct[ i ];
  }
  ( *out ) << "],";

  // Action space
  ( *out ) << "\"n\":" << input.action_space.actions.size() << ",";
  ( *out ) << "\"chosen\":" << chosen_index << ",";
  ( *out ) << "\"chosen_label\":\"" << json_escape( chosen_label ) << "\",";

  ( *out ) << "\"mask\":[";
  for ( std::size_t i = 0; i < input.action_space.action_mask.size(); ++i )
  {
    if ( i )
      ( *out ) << ",";
    ( *out ) << static_cast<int>( input.action_space.action_mask[ i ] );
  }
  ( *out ) << "],";

  ( *out ) << "\"labels\":[";
  for ( std::size_t i = 0; i < input.action_space.action_labels.size(); ++i )
  {
    if ( i )
      ( *out ) << ",";
    ( *out ) << "\"" << json_escape( input.action_space.action_labels[ i ] ) << "\"";
  }
  ( *out ) << "],";

  ( *out ) << "\"cd_rem_s\":[";
  for ( std::size_t i = 0; i < input.action_space.cooldown_remains_s.size(); ++i )
  {
    if ( i )
      ( *out ) << ",";
    ( *out ) << input.action_space.cooldown_remains_s[ i ];
  }
  ( *out ) << "],";

  ( *out ) << "\"cd_rem_n\":[";
  for ( std::size_t i = 0; i < input.action_space.cooldown_remains_norm.size(); ++i )
  {
    if ( i )
      ( *out ) << ",";
    ( *out ) << input.action_space.cooldown_remains_norm[ i ];
  }
  ( *out ) << "],";

  ( *out ) << "\"cd_charges_f\":[";
  for ( std::size_t i = 0; i < input.action_space.cooldown_charges_frac.size(); ++i )
  {
    if ( i )
      ( *out ) << ",";
    ( *out ) << input.action_space.cooldown_charges_frac[ i ];
  }
  ( *out ) << "]";

  ( *out ) << "}\n";
  out->flush();
}

}  // namespace rl
