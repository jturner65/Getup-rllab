import numpy as np
from rllab.misc import tensor_utils
import time

#modified rollout function to handle broken sim- Added by JT
def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1, always_return_paths=False):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        #if broken, don't save data from this sim step, don't increment path length, consider done
        if env_info['brokenSim']:
            #if path_length == 0:#if broken with length 0, redo rollout
            print('Broken simulation : restart rollout')
            return rollout(env, agent, max_path_length=max_path_length, animated=animated, speedup=speedup, always_return_paths=always_return_paths)
            #if length > 0 then just save what we have so far but don't save broken rollout data
            #break            
        #   -or-
#        if env_info['brokenSim']: # if broken, toss entire rollout - this is dangerous - could be infinite recursive 
#            print('Broken simulation : restart rollout')
#            return rollout(env, agent, max_path_length=max_path_length, animated=animated, speedup=speedup, always_return_paths=always_return_paths)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )
    
    
    
#original rollout function
#def rollout_orig(env, agent, max_path_length=np.inf, animated=False, speedup=1,
#            always_return_paths=False):
#    observations = []
#    actions = []
#    rewards = []
#    agent_infos = []
#    env_infos = []
#    o = env.reset()
#    agent.reset()
#    path_length = 0
#    if animated:
#        env.render()
#    while path_length < max_path_length:
#        a, agent_info = agent.get_action(o)
#        next_o, r, d, env_info = env.step(a)
#        observations.append(env.observation_space.flatten(o))
#        rewards.append(r)
#        actions.append(env.action_space.flatten(a))
#        agent_infos.append(agent_info)
#        env_infos.append(env_info)
#        path_length += 1
#        if d:
#            break
#        o = next_o
#        if animated:
#            env.render()
#            timestep = 0.05
#            time.sleep(timestep / speedup)
#    if animated and not always_return_paths:
#        return
#
#    return dict(
#        observations=tensor_utils.stack_tensor_list(observations),
#        actions=tensor_utils.stack_tensor_list(actions),
#        rewards=tensor_utils.stack_tensor_list(rewards),
#        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
#        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
#    )    