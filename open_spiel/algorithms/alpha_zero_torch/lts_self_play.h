// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_LTS_SELF_PLAY_H_
#define OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_LTS_SELF_PLAY_H_

#include <iostream>
#include <string>
#include <vector>

#include "open_spiel/utils/file.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/utils/thread.h"

namespace open_spiel {
    namespace algorithms {
        namespace torch_az {

            struct LTSConfig {
                std::string game;
                std::string path;
                std::string graph_def;
                std::string nn_model;
                int nn_width;
                int nn_depth;
                std::string devices;

                bool explicit_learning;
                double learning_rate;
                double weight_decay;
                int train_batch_size;
                int inference_batch_size;
                int inference_threads;
                int inference_cache;
                int replay_buffer_size;
                int replay_buffer_reuse;
                int checkpoint_freq;
                int evaluation_window;
                int max_simulations;
                int actors;
                int evaluators;
                int max_steps;

                json::Object ToJson() const {
                    return json::Object({
                                                {"game", game},
                                                {"path", path},
                                                {"graph_def", graph_def},
                                                {"nn_model", nn_model},
                                                {"nn_width", nn_width},
                                                {"nn_depth", nn_depth},
                                                {"devices", devices},
                                                {"explicit_learning", explicit_learning},
                                                {"learning_rate", learning_rate},
                                                {"weight_decay", weight_decay},
                                                {"train_batch_size", train_batch_size},
                                                {"inference_batch_size", inference_batch_size},
                                                {"inference_threads", inference_threads},
                                                {"inference_cache", inference_cache},
                                                {"replay_buffer_size", replay_buffer_size},
                                                {"replay_buffer_reuse", replay_buffer_reuse},
                                                {"checkpoint_freq", checkpoint_freq},
                                                {"evaluation_window", evaluation_window},
                                                {"max_simulations", max_simulations},
                                                {"actors", actors},
                                                {"evaluators", evaluators},
                                                {"max_steps", max_steps},
                                        });
                }

                void FromJson(const json::Object& config_json) {
                    game = config_json.at("game").GetString();
                    path = config_json.at("path").GetString();
                    graph_def = config_json.at("graph_def").GetString();
                    nn_model = config_json.at("nn_model").GetString();
                    nn_width = config_json.at("nn_width").GetInt();
                    nn_depth = config_json.at("nn_depth").GetInt();
                    devices = config_json.at("devices").GetString();
                    explicit_learning = config_json.at("explicit_learning").GetBool();
                    learning_rate = config_json.at("learning_rate").GetDouble();
                    weight_decay = config_json.at("weight_decay").GetDouble();
                    train_batch_size = config_json.at("train_batch_size").GetInt();
                    inference_batch_size = config_json.at("inference_batch_size").GetInt();
                    inference_threads = config_json.at("inference_threads").GetInt();
                    inference_cache = config_json.at("inference_cache").GetInt();
                    replay_buffer_size = config_json.at("replay_buffer_size").GetInt();
                    replay_buffer_reuse = config_json.at("replay_buffer_reuse").GetInt();
                    checkpoint_freq = config_json.at("checkpoint_freq").GetInt();
                    evaluation_window = config_json.at("evaluation_window").GetInt();
                    max_simulations = config_json.at("max_simulations").GetInt();
                    actors = config_json.at("actors").GetInt();
                    evaluators = config_json.at("evaluators").GetInt();
                    max_steps = config_json.at("max_steps").GetInt();
                }
            };

            bool LTSSelfPlay(LTSConfig config, StopToken* stop, bool resuming);

        }  // namespace torch_az
    }  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_LTS_SELF_PLAY_H_
