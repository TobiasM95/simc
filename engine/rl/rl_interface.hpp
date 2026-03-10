// ==========================================================================
// SimulationCraft RL interface (experimental)
// ==========================================================================

#pragma once

#include "config.hpp"

#include "sc_enums.hpp"
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

struct observation_t
{
  double time_s         = 0.0;
  double fight_length_s = 0.0;

  double gcd_remaining_s = 0.0;
  double target_ttd_s    = 0.0;

  std::array<double, RESOURCE_MAX> resource_pct{};

  int spec_id = 0;

  std::vector<std::string> buff_labels;
  std::vector<std::string> dot_labels;
  std::vector<double> buff_remains;
  std::vector<int> buff_stacks;
  std::vector<double> dot_remains;
  std::vector<int> dot_stacks;
};

struct action_space_view_t
{
  util::span<action_t* const> actions;
  util::span<const std::string> action_labels;
  util::span<const uint8_t> action_mask;
  util::span<const double> cooldown_remains_s;
  util::span<const double> cooldown_remains_norm;
  util::span<const double> cooldown_charges_frac;
};

struct step_input_t
{
  const player_t* player = nullptr;
  observation_t observation;
  action_space_view_t action_space;
  int teacher_idx = -1;
  std::string teacher_label;
  double reward = 0.0;
};

using policy_fn_t = std::size_t ( * )( const step_input_t&, void* user_data );

policy_fn_t get_policy();
void* get_policy_user_data();
void set_policy( policy_fn_t fn, void* user_data = nullptr );

std::size_t dummy_policy( const step_input_t& input, void* user_data );
std::size_t stdio_policy( const step_input_t& input, void* user_data );
void write_episode_end( double total_damage, double fight_length_s );

bool is_exposed_action( const action_t& a );
observation_t build_observation( const player_t& player );
void build_action_list( const player_t& player, std::vector<action_t*>& out_actions,
                        std::vector<std::string>& out_labels );

std::size_t get_num_wait_pseudo_actions();
double get_wait_pseudo_action_duration( std::size_t pseudo_index );
std::size_t get_num_pass_pseudo_actions();
std::size_t get_total_pseudo_actions();
bool is_pass_pseudo_action( std::size_t action_index, std::size_t total_actions );

void update_action_features( const player_t& player, execute_type context, util::span<action_t* const> actions,
                             std::vector<uint8_t>& out_mask, std::vector<double>& out_cd_remains_s,
                             std::vector<double>& out_cd_remains_norm, std::vector<double>& out_cd_charges_frac );

void trace_decision( const step_input_t& input, std::size_t chosen_index );

}  // namespace rl

