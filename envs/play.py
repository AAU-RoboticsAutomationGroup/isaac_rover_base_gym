import isaacgym
from rl_games.torch_runner import Runner
import ray
import gym
import yaml
import torch
import matplotlib.pyplot as plt

# from IPython import display
import numpy as np
import onnx
import onnxruntime as ort

from utils.utils import set_np_formatting, set_seed

from utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator
from utils.reformat import omegaconf_to_dict, print_dict

from rl_games.common import env_configurations, vecenv


ray.init(object_store_memory=1024*1024*1000)


cfg = './cfg/config.yaml'

# cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

create_rlgpu_env = get_rlgames_env_creator(omegaconf_to_dict(cfg.task))

vecenv.register('RLGPU',
                lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'vecenv_type': 'RLGPU',
    'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
})

rlg_config_dict = omegaconf_to_dict(cfg.train)

# runner = Runner(RLGPUAlgoObserver())
# runner.load(rlg_config_dict)
# runner.reset()


with open(rlg_config_dict, 'r') as stream:
    config = yaml.safe_load(stream)
    config['params']['config']['full_experiment_name'] = 'exomy_onnx'
runner = Runner(RLGPUAlgoObserver())
runner.load(config)
runner.run({
    'train': False,
    'play': True
})

agent = runner.create_player()
agent.restore('runs/exomy_onnx/nn/ExomyV7SorensPC.pth')

import rl_games.algos_torch.flatten as flatten
inputs = {
    'obs' : torch.zeros((1,) + agent.obs_shape).to(agent.device),
    'rnn_states' : agent.states
}
with torch.no_grad():
    adapter = flatten.TracingAdapter(agent.model.a2c_network, inputs,allow_non_tensor=True)
    traced = torch.jit.trace(adapter, adapter.flattened_inputs,check_trace=False)
    flattened_outputs = traced(*adapter.flattened_inputs)
    print(flattened_outputs)
    
torch.onnx.export(traced, *adapter.flattened_inputs, "exomy.onnx", verbose=True, input_names=['obs'], output_names=['logits', 'value'])

onnx_model = onnx.load("exomy.onnx")

# Check that the model is well formed
onnx.checker.check_model(onnx_model)


ort_model = ort.InferenceSession("exomy.onnx")

outputs = ort_model.run(
    None,
    {"obs": np.zeros((1, 4)).astype(np.float32)},
)
print(outputs)


is_done = False
env = agent.env
obs = env.reset()
#prev_screen = env.render(mode='rgb_array')
#plt.imshow(prev_screen)
total_reward = 0
num_steps = 0
while not is_done:
    outputs = ort_model.run(None, {"obs": np.expand_dims(obs, axis=0).astype(np.float32)},)
    action = np.argmax(outputs[0])
    obs, reward, done, info = env.step(action)
    total_reward += reward
    num_steps += 1
    is_done = done
    screen = env.render(mode='rgb_array')
    #plt.imshow(screen)
    #display.display(plt.gcf())    
    #display.clear_output(wait=True)
print(total_reward, num_steps)
#ipythondisplay.clear_output(wait=True)