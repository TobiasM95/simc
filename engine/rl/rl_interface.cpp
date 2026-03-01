// ==========================================================================
// SimulationCraft RL interface (experimental)
// ==========================================================================

#include "rl_interface.hpp"

#include "action/action.hpp"
#include "action/dot.hpp"
#include "buff/buff.hpp"
#include "dbc/class_spells.hpp"
#include "dbc/trait_data.hpp"
#include "player/player.hpp"
#include "sim/cooldown.hpp"
#include "sim/sim.hpp"
#include "util/io.hpp"
#include "util/util.hpp"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <set>
#include <unordered_set>

namespace
{
constexpr double WAIT_DURATIONS[]             = { 0.1, 0.2, 0.5, 1.0, 1.5 };
constexpr std::size_t NUM_WAIT_PSEUDO_ACTIONS = sizeof( WAIT_DURATIONS ) / sizeof( WAIT_DURATIONS[ 0 ] );
constexpr std::size_t NUM_PASS_PSEUDO_ACTIONS = 1;
constexpr const char* PASS_LABEL              = "pass";
constexpr const char* RL_BASELINE_APL_NAME    = "_rl_baseline";

const char* WAIT_LABELS[] = { "wait_0.1", "wait_0.2", "wait_0.5", "wait_1.0", "wait_1.5" };

rl::policy_fn_t g_policy = &rl::dummy_policy;
void* g_policy_user_data = nullptr;
std::mutex g_stdio_mutex;

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

std::vector<std::string> expanded_labels( const std::vector<std::string>& raw )
{
  std::vector<std::string> out;
  std::unordered_set<std::string> seen;
  for ( const auto& entry : raw )
  {
    for ( auto part : util::string_split<util::string_view>( entry, "," ) )
    {
      const std::string token = util::tokenize_fn( part );
      if ( token.empty() || seen.count( token ) )
        continue;
      seen.insert( token );
      out.push_back( token );
    }
  }
  return out;
}

const buff_t* find_buff_by_token( const player_t& player, const std::string& token )
{
  for ( const auto* buff : player.buff_list )
  {
    if ( buff && util::tokenize_fn( buff->name_str ) == token )
      return buff;
  }
  return nullptr;
}

const dot_t* find_dot_by_token( const player_t& target, const std::string& token )
{
  for ( const auto* dot : target.dot_list )
  {
    if ( dot && util::tokenize_fn( dot->name_str ) == token )
      return dot;
  }
  return nullptr;
}

std::set<std::string> discover_class_actions( const player_t& player )
{
  std::set<std::string> action_names;

  const bool ptr          = player.is_ptr();
  const unsigned class_id = static_cast<unsigned>( util::class_id( player.type ) );
  const unsigned spec_id  = static_cast<unsigned>( player.specialization() );

  for ( const auto& spell : active_class_spell_t::data( ptr ) )
  {
    if ( spell.class_id != class_id )
      continue;
    if ( spell.spec_id != 0 && spell.spec_id != spec_id )
      continue;
    if ( !spell.name || spell.name[ 0 ] == '\0' )
      continue;
    const std::string tokenized = util::tokenize_fn( spell.name );
    if ( !tokenized.empty() )
      action_names.insert( tokenized );
  }

  for ( auto tree : { talent_tree::CLASS, talent_tree::SPECIALIZATION, talent_tree::HERO } )
  {
    for ( const auto& trait : trait_data_t::data( class_id, tree, ptr ) )
    {
      if ( trait.id_spell == 0 || !trait.name || trait.name[ 0 ] == '\0' )
        continue;

      bool spec_matches = ( trait.id_spec[ 0 ] == 0 );
      if ( !spec_matches )
      {
        for ( unsigned s : trait.id_spec )
        {
          if ( s == spec_id )
          {
            spec_matches = true;
            break;
          }
        }
      }
      if ( !spec_matches )
        continue;

      const std::string tokenized = util::tokenize_fn( trait.name );
      if ( !tokenized.empty() )
        action_names.insert( tokenized );
    }
  }

  for ( const auto& [ tree, trait_node_entry_id, rank ] : player.player_traits )
  {
    if ( rank == 0 )
      continue;
    const trait_data_t* trait = trait_data_t::find( trait_node_entry_id, ptr );
    if ( !trait || trait->id_spell == 0 || !trait->name || trait->name[ 0 ] == '\0' )
      continue;
    const std::string tokenized = util::tokenize_fn( trait->name );
    if ( !tokenized.empty() )
      action_names.insert( tokenized );
  }

  action_names.insert( "auto_attack" );
  action_names.insert( "cancelform" );
  action_names.insert( "cancel_buff" );

  return action_names;
}

void create_baseline_actions( player_t& player )
{
  if ( player.is_enemy() || player.is_pet() )
    return;

  action_priority_list_t* rl_apl = player.get_action_priority_list( RL_BASELINE_APL_NAME );
  std::set<std::string> discovered_actions = discover_class_actions( player );

  std::set<std::string> existing_action_names;
  for ( action_t* a : player.action_list )
  {
    if ( a )
      existing_action_names.insert( a->name_str );
  }

  const std::size_t initial_action_count = player.action_list.size();

  for ( const auto& action_name : discovered_actions )
  {
    if ( existing_action_names.count( action_name ) )
      continue;

    action_t* a = player.create_action( action_name, "" );
    if ( !a )
      continue;

    if ( a->background || a->proc || a->dual )
      continue;

    a->action_list = rl_apl;
  }

  for ( std::size_t i = initial_action_count; i < player.action_list.size(); ++i )
  {
    action_t* a = player.action_list[ i ];
    if ( !a || a->initialized )
      continue;

    try
    {
      a->init();
      a->init_finished();
    }
    catch ( const std::exception& )
    {
      a->background = true;
    }
  }
}

void write_step_json_to_stream( std::ostream& out, const rl::step_input_t& input )
{
  out << "{";
  out << "\"type\":\"step\",";
  out << "\"t\":" << input.observation.time_s << ",";
  out << "\"fight_len\":" << input.observation.fight_length_s << ",";
  out << "\"ttd\":" << input.observation.target_ttd_s << ",";
  out << "\"gcd_rem\":" << input.observation.gcd_remaining_s << ",";
  out << "\"reward\":" << input.reward << ",";
  out << "\"spec_id\":" << input.observation.spec_id << ",";

  out << "\"resource_pct\":[";
  for ( std::size_t i = 0; i < input.observation.resource_pct.size(); ++i )
  {
    if ( i )
      out << ",";
    out << input.observation.resource_pct[ i ];
  }
  out << "],";

  out << "\"buff_labels\":[";
  for ( std::size_t i = 0; i < input.observation.buff_labels.size(); ++i )
  {
    if ( i )
      out << ",";
    out << "\"" << json_escape( input.observation.buff_labels[ i ] ) << "\"";
  }
  out << "],";

  out << "\"buff_remains\":[";
  for ( std::size_t i = 0; i < input.observation.buff_remains.size(); ++i )
  {
    if ( i )
      out << ",";
    out << input.observation.buff_remains[ i ];
  }
  out << "],";

  out << "\"buff_stacks\":[";
  for ( std::size_t i = 0; i < input.observation.buff_stacks.size(); ++i )
  {
    if ( i )
      out << ",";
    out << input.observation.buff_stacks[ i ];
  }
  out << "],";

  out << "\"dot_labels\":[";
  for ( std::size_t i = 0; i < input.observation.dot_labels.size(); ++i )
  {
    if ( i )
      out << ",";
    out << "\"" << json_escape( input.observation.dot_labels[ i ] ) << "\"";
  }
  out << "],";

  out << "\"dot_remains\":[";
  for ( std::size_t i = 0; i < input.observation.dot_remains.size(); ++i )
  {
    if ( i )
      out << ",";
    out << input.observation.dot_remains[ i ];
  }
  out << "],";

  out << "\"dot_stacks\":[";
  for ( std::size_t i = 0; i < input.observation.dot_stacks.size(); ++i )
  {
    if ( i )
      out << ",";
    out << input.observation.dot_stacks[ i ];
  }
  out << "],";

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

}  // namespace

namespace rl
{

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

std::size_t dummy_policy( const step_input_t& input, void* )
{
  static thread_local std::size_t rr = 0;
  for ( std::size_t attempt = 0; attempt < input.action_space.action_mask.size(); ++attempt )
  {
    const std::size_t idx = ( rr + attempt ) % input.action_space.action_mask.size();
    if ( input.action_space.action_mask[ idx ] )
    {
      rr = idx + 1;
      return idx;
    }
  }
  return input.action_space.actions.size();
}

std::size_t stdio_policy( const step_input_t& input, void* )
{
  std::lock_guard<std::mutex> lock( g_stdio_mutex );
  write_step_json_to_stream( std::cout, input );

  std::size_t action_index = input.action_space.actions.size();
  if ( std::cin >> action_index )
  {
    if ( action_index < input.action_space.action_mask.size() && input.action_space.action_mask[ action_index ] )
      return action_index;
  }

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

bool is_exposed_action( const action_t& a )
{
  if ( a.background || a.proc || a.dual )
    return false;
  if ( a.action_list == nullptr )
    return false;
  if ( a.option.wait_on_ready == 1 )
    return false;
  if ( a.type == ACTION_CALL || a.type == ACTION_VARIABLE || a.type == ACTION_SEQUENCE )
    return false;
  if ( a.name_str == "wait" || a.name_str == "pool_resource" )
    return false;
  if ( a.use_off_gcd || a.use_while_casting )
    return false;

  resource_e res = a.current_resource();
  if ( res != RESOURCE_NONE && a.base_cost() > 0 && !a.player->resources.is_active( res ) )
    return false;

  return a.type == ACTION_SPELL || a.type == ACTION_ATTACK || a.type == ACTION_HEAL || a.type == ACTION_ABSORB ||
         a.type == ACTION_USE || a.type == ACTION_OTHER;
}

observation_t build_observation( const player_t& player )
{
  observation_t obs;

  const sim_t* sim = player.sim;
  if ( !sim )
    return obs;

  obs.time_s         = sim->current_time().total_seconds();
  obs.fight_length_s = sim->expected_max_time();
  obs.gcd_remaining_s = std::max( 0.0, ( player.gcd_ready - sim->current_time() ).total_seconds() );

  if ( sim->target )
    obs.target_ttd_s = sim->target->time_to_percent( 0.0 ).total_seconds();
  else
    obs.target_ttd_s = std::max( 0.0, obs.fight_length_s - obs.time_s );

  for ( resource_e r = RESOURCE_NONE; r < RESOURCE_MAX; ++r )
    obs.resource_pct[ r ] = player.resources.pct( r );

  obs.spec_id = static_cast<int>( player.specialization() );

  const auto buff_labels = expanded_labels( sim->rl_observe_buffs );
  const auto dot_labels  = expanded_labels( sim->rl_observe_dots );

  obs.buff_labels = buff_labels;
  obs.dot_labels  = dot_labels;

  constexpr double NORM_WINDOW_S = 30.0;

  for ( const auto& name : buff_labels )
  {
    const buff_t* b = find_buff_by_token( player, name );
    if ( b && b->check() )
    {
      obs.buff_remains.push_back( norm01( b->remains().total_seconds(), NORM_WINDOW_S ) );
      obs.buff_stacks.push_back( b->check() );
    }
    else
    {
      obs.buff_remains.push_back( 0.0 );
      obs.buff_stacks.push_back( 0 );
    }
  }

  const player_t* target = player.target;
  for ( const auto& name : dot_labels )
  {
    const dot_t* d = target ? find_dot_by_token( *target, name ) : nullptr;
    if ( d && d->is_ticking() )
    {
      obs.dot_remains.push_back( norm01( d->remains().total_seconds(), NORM_WINDOW_S ) );
      obs.dot_stacks.push_back( d->current_stack() );
    }
    else
    {
      obs.dot_remains.push_back( 0.0 );
      obs.dot_stacks.push_back( 0 );
    }
  }

  return obs;
}

void build_action_list( const player_t& player, std::vector<action_t*>& out_actions, std::vector<std::string>& out_labels )
{
  out_actions.clear();
  out_labels.clear();

  create_baseline_actions( const_cast<player_t&>( player ) );

  std::set<std::string> seen_names;
  for ( action_t* a : player.action_list )
  {
    if ( !a || !is_exposed_action( *a ) )
      continue;
    if ( seen_names.count( a->name_str ) )
      continue;
    seen_names.insert( a->name_str );
    out_actions.push_back( a );
    out_labels.push_back( a->name_str );
  }

  for ( std::size_t i = 0; i < NUM_WAIT_PSEUDO_ACTIONS; ++i )
  {
    out_actions.push_back( nullptr );
    out_labels.push_back( WAIT_LABELS[ i ] );
  }

  out_actions.push_back( nullptr );
  out_labels.push_back( PASS_LABEL );
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

std::size_t get_num_pass_pseudo_actions()
{
  return NUM_PASS_PSEUDO_ACTIONS;
}

std::size_t get_total_pseudo_actions()
{
  return NUM_WAIT_PSEUDO_ACTIONS + NUM_PASS_PSEUDO_ACTIONS;
}

bool is_pass_pseudo_action( std::size_t action_index, std::size_t total_actions )
{
  const std::size_t num_pseudo   = get_total_pseudo_actions();
  const std::size_t real_actions = total_actions - num_pseudo;
  const std::size_t pass_index   = real_actions + NUM_WAIT_PSEUDO_ACTIONS;
  return action_index == pass_index;
}

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

    if ( a == nullptr )
    {
      out_mask[ i ]            = static_cast<uint8_t>( ( context == execute_type::FOREGROUND ) ? 1 : 0 );
      out_cd_remains_s[ i ]    = 0.0;
      out_cd_remains_norm[ i ] = 0.0;
      out_cd_charges_frac[ i ] = 1.0;
      continue;
    }

    bool context_valid = false;
    switch ( context )
    {
      case execute_type::OFF_GCD:
        context_valid = a->use_off_gcd && a->trigger_gcd == timespan_t::zero();
        break;
      case execute_type::CAST_WHILE_CASTING:
        context_valid = a->usable_while_casting && a->use_while_casting;
        break;
      case execute_type::FOREGROUND:
      default:
        context_valid = !( a->use_off_gcd || a->use_while_casting );
        break;
    }

    bool legal = false;
    if ( context_valid )
    {
      resource_e res = a->current_resource();
      if ( res == RESOURCE_NONE || a->base_cost() <= 0 || a->player->resources.is_active( res ) )
      {
        player_t* t = a->target ? a->target : a->player->target;
        legal       = a->ready() && t && a->target_ready( t );
      }
    }

    out_mask[ i ] = static_cast<uint8_t>( legal ? 1 : 0 );

    const double cd_s        = a->cooldown ? a->cooldown->remains().total_seconds() : 0.0;
    out_cd_remains_s[ i ]    = std::max( 0.0, cd_s );
    out_cd_remains_norm[ i ] = norm01( std::max( 0.0, cd_s ), cd_norm );
    out_cd_charges_frac[ i ] = a->cooldown ? a->cooldown->charges_fractional() : 1.0;
  }
}

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
  ( *out ) << "\"ttd\":" << input.observation.target_ttd_s << ",";
  ( *out ) << "\"gcd_rem\":" << input.observation.gcd_remaining_s << ",";
  ( *out ) << "\"reward\":" << input.reward << ",";
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
  ( *out ) << "]";
  ( *out ) << "}\n";
  out->flush();
}

}  // namespace rl
