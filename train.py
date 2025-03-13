import os
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = '/usr/lib/x86_64-linux-gnu/mesa/libEGL.so.1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'mesa'
import gymnasium
import argparse
import numpy as np
from einops import rearrange
import torch
from collections import deque
from tqdm import tqdm
import colorama

import pandas as pd

from utils import seed_np_torch, WandbLogger
from replay_buffer import ReplayBuffer
import agents
from sub_models.world_models import WorldModel
from mamba_ssm import InferenceParams
from line_profiler import profile
import yaml
from envs.my_memory_maze import MemoryMaze
from envs.my_atari import Atari
from envs.dmc_wrapper import DMControl
from eval import eval_episodes
import warnings
import ast


@profile
def train_world_model_step(replay_buffer: ReplayBuffer, world_model: WorldModel, batch_size, batch_length, logger, epoch, global_step):
    epoch_reconstruction_loss_list = []
    epoch_reward_loss_list = []
    epoch_termination_loss_list = []
    epoch_dynamics_loss_list = []
    epoch_dynamics_real_kl_div_list = []
    epoch_representation_loss_list = []
    epoch_representation_real_kl_div_list = []
    epoch_total_loss_list = []
    for e in range(epoch):
        obs, action, reward, termination = replay_buffer.sample(batch_size, batch_length, imagine=False)
        reconstruction_loss, reward_loss, termination_loss, \
        dynamics_loss, dynamics_real_kl_div, representation_loss, \
        representation_real_kl_div, total_loss = world_model.update(obs, action, reward, termination, global_step=global_step, epoch_step=e, logger=logger)

        epoch_reconstruction_loss_list.append(reconstruction_loss)
        epoch_reward_loss_list.append(reward_loss)
        epoch_termination_loss_list.append(termination_loss)
        epoch_dynamics_loss_list.append(dynamics_loss)
        epoch_dynamics_real_kl_div_list.append(dynamics_real_kl_div)
        epoch_representation_loss_list.append(representation_loss)
        epoch_representation_real_kl_div_list.append(representation_real_kl_div)
        epoch_total_loss_list.append(total_loss)
    if logger is not None:
        logger.log("WorldModel/reconstruction_loss", np.mean(epoch_reconstruction_loss_list), global_step=global_step)
        # logger.log("WorldModel/augmented_reconstruction_loss", augmented_reconstruction_loss.item(), global_step=global_step)
        logger.log("WorldModel/reward_loss",np.mean(epoch_reward_loss_list), global_step=global_step)
        logger.log("WorldModel/termination_loss", np.mean(epoch_termination_loss_list), global_step=global_step)
        logger.log("WorldModel/dynamics_loss", np.mean(epoch_dynamics_loss_list), global_step=global_step)
        logger.log("WorldModel/dynamics_real_kl_div", np.mean(epoch_dynamics_real_kl_div_list), global_step=global_step)
        logger.log("WorldModel/representation_loss", np.mean(epoch_representation_loss_list), global_step=global_step)
        logger.log("WorldModel/representation_real_kl_div", np.mean(epoch_representation_real_kl_div_list), global_step=global_step)
        logger.log("WorldModel/total_loss", np.mean(epoch_total_loss_list), global_step=global_step)    

@profile
@torch.no_grad()
def world_model_imagine_data(replay_buffer: ReplayBuffer,
                             world_model: WorldModel, agent: agents.ActorCriticAgent,
                             imagine_batch_size,
                             imagine_context_length, imagine_batch_length,
                             log_video, logger, global_step):
    '''
    Sample context from replay buffer, then imagine data with world model and agent
    '''
    world_model.eval()
    agent.eval()
    sample_obs, sample_action, sample_reward, sample_termination = replay_buffer.sample(
        imagine_batch_size, imagine_context_length, imagine=True)
    # sample_action = sample_action.to(torch.float32)
    if world_model.model == 'Transformer':
        latent, action, old_logits, context_latent, reward_hat, termination_hat = world_model.imagine_data(
            agent, sample_obs, sample_action,
            imagine_batch_size=imagine_batch_size,
            imagine_batch_length=imagine_batch_length,
            log_video=log_video,
            logger=logger, global_step=global_step
        )
    elif world_model.model == 'Mamba' or world_model.model == 'Mamba2':
         latent, action, old_logits, context_latent, reward_hat, termination_hat = world_model.imagine_data2(
            agent, sample_obs, sample_action,
            imagine_batch_size=imagine_batch_size,
            imagine_batch_length=imagine_batch_length,
            log_video=log_video,
            logger=logger, global_step=global_step
        )
    return latent, action, old_logits, context_latent, sample_reward, sample_termination, reward_hat, termination_hat

@profile
def joint_train_world_model_agent(config, logdir,
                                  replay_buffer: ReplayBuffer,
                                  world_model: WorldModel, agent: agents.ActorCriticAgent,
                                  logger):
    os.makedirs(f"{logdir}/ckpt", exist_ok=True)


    if config.BasicSettings.Env_name.startswith('ALE'):
        env = Atari(config.BasicSettings.Env_name, size=(config.BasicSettings.ImageSize,config.BasicSettings.ImageSize), seed=config.BasicSettings.Seed)
    elif config.BasicSettings.Env_name.startswith('memory'):
        env = MemoryMaze(config.BasicSettings.Env_name, size=(config.BasicSettings.ImageSize,config.BasicSettings.ImageSize), seed=config.BasicSettings.Seed)
    elif config.BasicSettings.Env_name.startswith('dm_control'):
        env = DMControl(config.BasicSettings.Env_name,
                        size=(config.BasicSettings.ImageSize, config.BasicSettings.ImageSize),
                        camera=getattr(config.BasicSettings, 'DMControlRenderCamera', 0),
                        seed=config.BasicSettings.Seed)
        config.BasicSettings.ActionSpaceType = 'continuous'
    else:
        assert ValueError(f'Unknown environment name: {config.BasicSettings.Env_name}')
    print("Current env: " + colorama.Fore.YELLOW + f"{config.BasicSettings.Env_name}" + colorama.Style.RESET_ALL)

    # Check if using DM Control
    is_dm_control = config.BasicSettings.Env_name.startswith('dm_control')

    atari_benchmark_df = None
    game_benchmark_df = None
    if not is_dm_control:
        atari_benchmark_df = pd.read_csv("atari_performance.csv", index_col='Task', usecols=lambda column: column in ['Task', 'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone', 'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack', 'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo', 'Krull', 'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner', 'Seaquest', 'UpNDown'])
        atari_pure_name = config.BasicSettings.Env_name.split('/')[-1].split('-')[0]
        game_benchmark_df = atari_benchmark_df.get(atari_pure_name)

    sum_reward = 0
    # print("Starting environment reset")
    current_ob, info = env.reset()
    # print("Environment reset complete")
    context_obs = deque(maxlen=config.JointTrainAgent.RealityContextLength)
    context_action = deque(maxlen=config.JointTrainAgent.RealityContextLength)

    # sample and train
    for total_steps in tqdm(range(config.JointTrainAgent.SampleMaxSteps // config.JointTrainAgent.NumEnvs), desc='Training'):
        # sample part >>>
        # print(f"Step {total_steps}: Starting action selection")
        # Print replay buffer state
        # print(f"Buffer length: {len(replay_buffer)}, WM ready: {replay_buffer.ready('world_model')}")
        if replay_buffer.ready('world_model'):
            world_model.eval()
            agent.eval()
            with torch.no_grad():
                if len(context_action) == 0:
                    action = env.action_space.sample()
                else:
                    context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1).to(world_model.device))
                    model_context_action = np.stack(list(context_action))
                    model_context_action = torch.Tensor(model_context_action).to(world_model.device)
                    if len(model_context_action.shape) == 1:  # Discrete actions
                        model_context_action = rearrange(model_context_action, "L -> 1 L")
                    else:  # Continuous actions
                        model_context_action = rearrange(model_context_action, "L D -> 1 L D")
                    if world_model.model == 'Transformer':
                        prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                    elif world_model.model == 'Mamba' or world_model.model == 'Mamba2':
                        prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                    action = agent.sample_as_env_action(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                        greedy=False
                    )[0]

            context_obs.append(rearrange(torch.Tensor(current_ob).to(world_model.device), "H W C -> 1 1 C H W")/255)
            context_action.append(action)
        else:
            action = env.action_space.sample()

        # print(f"Step {total_steps}: Action selected")
        # print(f"Step {total_steps}: Starting environment step")

        ob, reward, is_last, info = env.step(action)
        # print(f"Step {total_steps}: Finished environment step")
        replay_buffer.append(current_ob, action, reward, info['is_terminal'])
        # print(f"Step {total_steps}: Buffer appended")
        sum_reward += reward
        current_ob = ob

        if is_last:
            logger.log(f"episode/score", sum_reward, global_step=total_steps)
            logger.log(f"episode/length", info["episode_frame_number"], global_step=total_steps)
            # Only log normalized scores for Atari environments
            if not is_dm_control and game_benchmark_df is not None:
                logger.log(f"episode/normalised score", (sum_reward - game_benchmark_df['Random']) / (
                            game_benchmark_df['Human'] - game_benchmark_df['Random']), global_step=total_steps)
                for algorithm in game_benchmark_df.index[2:]:
                    denominator = game_benchmark_df[algorithm] - game_benchmark_df['Random']
                    if denominator != 0:
                        normalized_score = (sum_reward - game_benchmark_df['Random']) / denominator
                        logger.log(f"benchmark/normalised {algorithm} score", normalized_score, global_step=total_steps)

            sum_reward = 0
            ob, info = env.reset()
            context_obs.clear()
            context_action.clear()

        if replay_buffer.ready('world_model') and total_steps % (config.JointTrainAgent.TrainDynamicsEverySteps // config.JointTrainAgent.NumEnvs) == 0 and total_steps <= config.JointTrainAgent.FreezeWorldModelAfterSteps:
            train_world_model_step(
                replay_buffer=replay_buffer,
                world_model=world_model,
                batch_size=config.JointTrainAgent.BatchSize,
                batch_length=config.JointTrainAgent.BatchLength,
                logger=logger,
                epoch=config.JointTrainAgent.TrainDynamicsEpoch,
                global_step=total_steps
            )

        # print(f"Step {total_steps}: Trained world model")
        if replay_buffer.ready('behaviour') and total_steps % (config.JointTrainAgent.TrainAgentEverySteps // config.JointTrainAgent.NumEnvs) == 0 and total_steps <= config.JointTrainAgent.FreezeBehaviourAfterSteps:
            log_video = total_steps % (config.JointTrainAgent.SaveEverySteps // config.JointTrainAgent.NumEnvs) == 0
            # print(f"Step {total_steps}: Video logged")
            imagine_latent, agent_action, old_logits, context_latent, context_reward, context_termination, imagine_reward, imagine_termination = world_model_imagine_data(
                replay_buffer=replay_buffer,
                world_model=world_model,
                agent=agent,
                imagine_batch_size=config.JointTrainAgent.ImagineBatchSize,
                imagine_context_length=config.JointTrainAgent.ImagineContextLength,
                imagine_batch_length=config.JointTrainAgent.ImagineBatchLength,
                log_video=log_video,
                logger=logger,
                global_step=total_steps
            )
            print(f"Step {total_steps}: Imagination rollout")

            agent.update(
                latent=imagine_latent,
                action=agent_action,
                old_logits=old_logits,
                context_latent=context_latent,
                context_reward=context_reward,
                context_termination=context_termination,
                reward=imagine_reward,
                termination=imagine_termination,
                logger=logger,
                global_step=total_steps
            )
            print(f"Step {total_steps}: Trained agent")

        if config.Evaluate.DuringTraining and total_steps > 0 and total_steps % (config.Evaluate.EverySteps // config.JointTrainAgent.NumEnvs) == 0 and\
            replay_buffer.ready('world_model'):
            _ = eval_episodes(config, world_model, agent, logger, total_steps)
            print(f"Step {total_steps}: Evaluation Complete")
        if config.JointTrainAgent.SaveModels and total_steps % (config.JointTrainAgent.SaveEverySteps // config.JointTrainAgent.NumEnvs) == 0:
            print(colorama.Fore.GREEN + f"Saving model at total steps {total_steps}" + colorama.Style.RESET_ALL)
            torch.save(world_model.state_dict(), f"{logdir}/ckpt/world_model.pth")
            torch.save(agent.state_dict(), f"{logdir}/ckpt/agent.pth")



def build_world_model(conf, action_dim, device):
    return WorldModel(
        action_dim = action_dim,
        config = conf, 
        device = device
    ).cuda(device)


def build_agent(conf, action_dim, device):
    if conf.Models.Agent.Policy == 'AC':
        return agents.ActorCriticAgent(
            conf = conf,
            action_dim=action_dim,
            device = device
        ).cuda(device)
    elif conf.Models.Agent.Policy == 'PPO':
        return agents.PPOAgent(
            conf=conf,
            action_dim=action_dim,
            device = device
        ).cuda(device)        


class DotDict(dict):
    """Dictionary with dot notation access."""
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, item):
        try:
            value = self[item]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{item}'")
        if isinstance(value, dict):
            value = DotDict(value)
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def update_or_create(self, key_path, value):
        keys = key_path.split('.')
        d = self
        for key in keys[:-1]:
            if key not in d or not isinstance(d[key], dict):
                d[key] = DotDict()
            d = d[key]
        d[keys[-1]] = value

# Function to parse and update config from arguments
def parse_args_and_update_config(config, prefix=''):
    parser = argparse.ArgumentParser()

    # Map string dtype to torch dtype
    def dtype_mapper(dtype_str):
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16
        }
        return dtype_map[dtype_str]

    def add_arguments(config, prefix=''):
        for key, value in config.items():
            if isinstance(value, dict):
                add_arguments(value, prefix + key + '.')
            elif isinstance(value, bool):
                # Special handling for boolean arguments
                parser.add_argument(f'--{prefix}{key}', type=lambda x: x.lower() in ['true', '1', 'yes'], default=value)
            elif key == 'dtype':
                # Special handling for dtype arguments
                parser.add_argument(f'--{prefix}{key}', type=dtype_mapper, default=value)
            elif isinstance(value, (list, dict)):
                # Use a custom converter for list/dict-like arguments
                parser.add_argument(f'--{prefix}{key}', type=lambda x: ast.literal_eval(x), default=value)
            else:
                parser.add_argument(f'--{prefix}{key}', type=type(value), default=value)

    def update_dict(d, keys, value):
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    add_arguments(config, prefix)
    
    args = parser.parse_args()
    args_dict = vars(args)
    
    for arg_key, arg_value in args_dict.items():
        if arg_value is not None:
            keys = arg_key.split('.')
            update_dict(config, keys, arg_value)
    
    return config

def update_model_parameters(config, world_model, agent):
    config.update_or_create('Models.WorldModel.TotalParamNum', sum([p.numel() for p in world_model.parameters()]))
    print(f'World model total parameters: {sum([p.numel() for p in world_model.parameters()]):,}')
    
    config.update_or_create('Models.WorldModel.BackboneParamNum', sum([p.numel() for p in world_model.sequence_model.parameters()]))
    print(f'Dynamic model parameters: {sum([p.numel() for p in world_model.sequence_model.parameters()]):,}')
    
    config.update_or_create('Models.WorldModel.EncoderParamNum', sum([p.numel() for p in world_model.encoder.parameters()]))
    print(f'Encoder parameters: {sum([p.numel() for p in world_model.encoder.parameters()]):,}')
    
    config.update_or_create('Models.WorldModel.DecoderParamNum', sum([p.numel() for p in world_model.image_decoder.parameters()]))
    print(f'Decoder parameters: {sum([p.numel() for p in world_model.image_decoder.parameters()]):,}')
    
    config.update_or_create('Models.WorldModel.DiscretisationLayerParamNum', sum([p.numel() for p in world_model.dist_head.parameters()]))
    print(f'Discretisation layer parameters: {sum([p.numel() for p in world_model.dist_head.parameters()]):,}')
    
    config.update_or_create('Models.Agent.ActorParamNum', sum([p.numel() for p in agent.actor.parameters()]))
    print(f'Actor parameters: {sum([p.numel() for p in agent.actor.parameters()]):,}')
    
    config.update_or_create('Models.Agent.CriticParamNum', sum([p.numel() for p in agent.critic.parameters()]))
    print(f'Critic parameters: {sum([p.numel() for p in agent.critic.parameters()]):,}')

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings("ignore")

    with open('config_files/configure.yaml', 'r') as file:
        config = yaml.safe_load(file)

    config = parse_args_and_update_config(config)   
    config = DotDict(config)

    device = torch.device(config.BasicSettings.Device)
    # set seed
    seed_np_torch(seed=config.BasicSettings.Seed)

    
    # getting action_dim with dummy env
    if config.BasicSettings.Env_name.startswith('ALE'):
        dummy_env = Atari(config.BasicSettings.Env_name)
        action_dim = dummy_env.action_space.n
    elif config.BasicSettings.Env_name.startswith('memory'):
        dummy_env = MemoryMaze(config.BasicSettings.Env_name)
        action_dim = dummy_env.action_space.n
    elif config.BasicSettings.Env_name.startswith('dm_control'):
        dummy_env = DMControl(config.BasicSettings.Env_name)
        action_dim = dummy_env.action_space.shape[0]  # Continuous action
        # Add action_dim to config so ReplayBuffer can access it
        config.update_or_create('Models.Agent.action_dim', action_dim)
    else:
        assert ValueError(f'Unknown environment name: {config.BasicSettings.Env_name}')

    # build world model and agent
    world_model = build_world_model(config, action_dim, device=device)
    agent = build_agent(config, action_dim, device=device)
    update_model_parameters(config, world_model, agent)
    if (config.BasicSettings.Compile and os.name != "nt"):  # compilation is not supported on windows
        world_model = torch.compile(world_model)
        agent = torch.compile(agent)
    if config.BasicSettings.SavePath != 'None':
        print('Loading models')
        world_model.load_state_dict(torch.load(f"{config.BasicSettings.SavePath}/world_model.pth"))
        agent.load_state_dict(torch.load(f"{config.BasicSettings.SavePath}/agent.pth"))
   
    logger = WandbLogger(config=config, project=config.Wandb.Init.Project, mode=config.Wandb.Init.Mode)
    logdir = f"./saved_models/{config.n}/{config.BasicSettings.Env_name}/{logger.run.id}"

    # build replay buffer
    replay_buffer = ReplayBuffer(
        config,
        device=device
    )

    # train
    joint_train_world_model_agent(config, logdir, replay_buffer, world_model, agent, logger)

    logger.close()

