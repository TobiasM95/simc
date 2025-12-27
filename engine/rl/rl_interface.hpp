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

// ============================================================================
// Action space building (APL-independent, uses player_t::action_list)
// ============================================================================

/// Returns true if the action should be exposed to the RL agent.
bool is_exposed_action( const action_t& a );

/// Build the observation from current player/sim state.
observation_t build_observation( const player_t& player );

/// Build the static action list (call once per player, at first RL decision).
/// Deduplicates by ability name; populates actions and labels.
/// Appends wait pseudo-actions (with nullptr action pointers) at the end.
void build_action_list( const player_t& player, std::vector<action_t*>& out_actions,
                        std::vector<std::string>& out_labels );

/// Get the number of wait pseudo-actions appended to the action list.
std::size_t get_num_wait_pseudo_actions();

/// Get the wait duration for a pseudo-action (0-indexed from the start of pseudo-actions).
/// Returns 0.0 if the index is out of range.
double get_wait_pseudo_action_duration( std::size_t pseudo_index );

/// Update the mask and per-action features for the current game state.
/// The execute_type determines which actions are valid in the current context.
void update_action_features( const player_t& player, execute_type context, util::span<action_t* const> actions,
                             std::vector<uint8_t>& out_mask, std::vector<double>& out_cd_remains_s,
                             std::vector<double>& out_cd_remains_norm, std::vector<double>& out_cd_charges_frac );

// ============================================================================
// Tracing (JSONL output for validation)
// ============================================================================
void trace_decision( const step_input_t& input, std::size_t chosen_index );

}  // namespace rl
