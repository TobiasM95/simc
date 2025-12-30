// ==========================================================================
// SimulationCraft RL interface (experimental)
// ==========================================================================

#pragma once

#include "config.hpp"

#include "sc_enums.hpp"
#include "util/generic.hpp"
#include "util/span.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct action_t;
struct player_t;

namespace rl
{

// ============================================================================
// Observation: current game state visible to the RL agent
// ============================================================================
struct observation_t
{
  double time_s         = 0.0;
  double fight_length_s = 0.0;

  double gcd_remaining_s    = 0.0;
  double gcd_remaining_norm = 0.0;

  double time_remaining_s    = 0.0;
  double time_remaining_norm = 0.0;

  double target_ttd_s    = 0.0;
  double target_ttd_norm = 0.0;

  std::array<double, RESOURCE_MAX> resource_pct{};

  // Spec-specific observations (populated based on specialization)
  std::vector<double> buff_remains_norm;  // Normalized buff durations [0,1] (by 30s)
  std::vector<int> buff_stacks;           // Raw buff stack counts
  std::vector<double> dot_remains_norm;   // Normalized dot durations on target [0,1]
  std::vector<int> dot_stacks;            // Raw dot stack counts

  // Metadata for Python to interpret the vectors
  int spec_id = 0;                       // specialization_e cast to int
  std::vector<std::string> buff_labels;  // Names matching buff_remains_norm indices
  std::vector<std::string> dot_labels;   // Names matching dot_remains_norm indices
};

// ============================================================================
// Action space: all available abilities (static per player, built once)
// ============================================================================
struct action_space_view_t
{
  // Pointers to the canonical action for each slot (stable for the fight)
  util::span<action_t* const> actions;

  // Labels (ability names, aligned with actions)
  util::span<const std::string> action_labels;

  // Legality mask: 1 = can use right now, 0 = cannot (updated each step)
  util::span<const uint8_t> action_mask;

  // Per-action features (aligned with actions, updated each step)
  util::span<const double> cooldown_remains_s;
  util::span<const double> cooldown_remains_norm;
  util::span<const double> cooldown_charges_frac;
};

// ============================================================================
// Step input: everything the policy needs to make a decision
// ============================================================================
struct step_input_t
{
  const player_t* player = nullptr;

  observation_t observation;
  action_space_view_t action_space;

  // Reward attributed to the *previous* decision step (damage delta since last decision).
  double reward = 0.0;
};

// ============================================================================
// Policy callback: given state, return action index
// ============================================================================
using policy_fn_t = std::size_t ( * )( const step_input_t&, void* user_data );

policy_fn_t get_policy();
void* get_policy_user_data();
void set_policy( policy_fn_t fn, void* user_data = nullptr );

std::size_t dummy_policy( const step_input_t& input, void* user_data );

/// Stdio bridge policy: writes state JSON to stdout, reads action index from stdin.
/// Use with rl_stdio=1 option. Requires single-threaded execution (threads=1).
std::size_t stdio_policy( const step_input_t& input, void* user_data );

/// Write episode termination signal to stdout (for stdio bridge mode).
/// Called at the end of each iteration when rl_stdio is enabled.
void write_episode_end( double total_damage, double fight_length_s );

// ============================================================================
// Action space building (APL-independent, uses player_t::action_list)
// ============================================================================

/// Returns true if the action should be exposed to the RL agent.
bool is_exposed_action( const action_t& a );

/// Build the observation from current player/sim state.
observation_t build_observation( const player_t& player );

/// Build the static action list (call once per player, at first RL decision).
/// Deduplicates by ability name; populates actions and labels.
/// Appends wait pseudo-actions and pass pseudo-action (with nullptr action pointers) at the end.
void build_action_list( const player_t& player, std::vector<action_t*>& out_actions,
                        std::vector<std::string>& out_labels );

/// Get the number of wait pseudo-actions appended to the action list.
std::size_t get_num_wait_pseudo_actions();

/// Get the wait duration for a pseudo-action (0-indexed from the start of pseudo-actions).
/// Returns 0.0 if the index is out of range.
double get_wait_pseudo_action_duration( std::size_t pseudo_index );

/// Get the number of pass pseudo-actions (currently always 1).
std::size_t get_num_pass_pseudo_actions();

/// Get the total number of pseudo-actions (wait + pass).
std::size_t get_total_pseudo_actions();

/// Check if a given action index is the pass pseudo-action.
/// @param action_index The index of the action in the action list.
/// @param total_actions The total number of actions including all pseudo-actions.
bool is_pass_pseudo_action( std::size_t action_index, std::size_t total_actions );

/// Update the mask and per-action features for the current game state.
/// The execute_type determines which actions are valid in the current context.
void update_action_features( const player_t& player, execute_type context, util::span<action_t* const> actions,
                             std::vector<uint8_t>& out_mask, std::vector<double>& out_cd_remains_s,
                             std::vector<double>& out_cd_remains_norm, std::vector<double>& out_cd_charges_frac );

// ============================================================================
// Tracing (JSONL output for validation)
// ============================================================================
void trace_decision( const step_input_t& input, std::size_t chosen_index );

// ============================================================================
// Reward shaping (potential-based)
// ============================================================================

/// Potential function type: maps observation → scalar potential value
/// For reward shaping: r' = r + γ*Φ(s') - Φ(s)
using potential_fn_t = double ( * )( const observation_t& obs, const player_t* player );

/// Get the current potential function (nullptr = no shaping, default)
potential_fn_t get_potential_fn();

/// Set a custom potential function for reward shaping
void set_potential_fn( potential_fn_t fn );

/// Compute shaped reward given raw reward and observations.
/// If no potential function is set, returns raw_reward unchanged.
/// Formula: r' = raw_reward + gamma * Φ(curr_obs) - Φ(prev_obs)
double compute_shaped_reward( double raw_reward, const observation_t& prev_obs, const observation_t& curr_obs,
                              double gamma, const player_t* player );

}  // namespace rl
