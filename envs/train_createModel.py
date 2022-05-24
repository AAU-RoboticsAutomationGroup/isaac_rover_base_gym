# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import isaacgym

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from utils.reformat import omegaconf_to_dict, print_dict
from utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator

from utils.utils import set_np_formatting, set_seed
import onnx
import onnxruntime as ort
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
import rl_games.torch_runner 
from utils import flatten
import torch
import numpy
import yaml



## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
# num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    # `create_rlgpu_env` is environment construction function which is passed to RL Games and called internally.
    # We use the helper function here to specify the environment config.
    create_rlgpu_env = get_rlgames_env_creator(
        omegaconf_to_dict(cfg.task),
        cfg.task_name,
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        multi_gpu=cfg.multi_gpu,
    )

    # register the rl-games adapter to use inside the runner
    vecenv.register('RLGPU',
                    lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    })

    rlg_config_dict = omegaconf_to_dict(cfg.train)

    # convert CLI arguments into dictionory
    # create runner and set the settings
    runner = Runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict
    experiment_dir = os.path.join('runs', cfg.train.params.config.name)
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    agent = runner.create_player()
    agent.restore('runs/testModel.pth')

    inputs = {
        'obs' : torch.zeros((1,) + agent.obs_shape).to(agent.device),
        'rnn_states' : agent.states
    }
    
    with torch.no_grad():
        adapter = flatten.TracingAdapter(agent.model.a2c_network, inputs,allow_non_tensor=True)
        traced = torch.jit.trace(adapter, adapter.flattened_inputs,check_trace=False)
        flattened_outputs = traced(*adapter.flattened_inputs)
        print(flattened_outputs)
    torch.onnx.export(traced, *adapter.flattened_inputs, "exomy.onnx", verbose=True, input_names=['obs'], output_names=['mu', 'sigma', 'value'], example_outputs=flattened_outputs)
    onnx_model = onnx.load("exomy.onnx")
    onnx.checker.check_model(onnx_model)
    
    observation = torch.tensor([ 0.0200, -0.5400, -0.3805,  0.4010,  0.0090, -1.0108, -1.0094, -1.0085,
        -1.0064, -1.0043, -1.0028, -1.0013, -0.9998, -0.9986, -0.9972, -0.6981,
        -0.6973, -0.6967, -0.6961, -0.6954, -0.6946, -0.6938, -0.6930, -0.6921,
        -0.6916, -0.5334, -0.5330, -0.5325, -0.5320, -0.5315, -0.5311, -0.5306,
        -0.5301, -0.5296, -0.5292, -0.4315, -0.4312, -0.4309, -0.4305, -0.4302,
        -0.4300, -0.4296, -0.4293, -0.4290, -0.4287, -0.3622, -0.3620, -0.3618,
        -0.3616, -0.3614, -0.3612, -0.3610, -0.3608, -0.3606, -0.3603]).unsqueeze(0).numpy()
    
    
    
    ort_model = ort.InferenceSession("exomy.onnx")

    outputs = ort_model.run(
        None,
        {"obs": observation},
    )
    print(outputs)

if __name__ == "__main__":
    launch_rlg_hydra()
