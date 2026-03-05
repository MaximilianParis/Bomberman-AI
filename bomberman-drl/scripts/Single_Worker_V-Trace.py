
import heapq
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import distributions
import random
import gymnasium
from bomberman_rl import ScoreRewardWrapper
from argparsing import parse
from collections import deque
from Expert_Agent import Expert_Agent
from pathlib import Path
from Nets import BaseActor,BaseCritic
from State_Algorithms import Transform_State,find_rescue_route,compute_timestamp_dead_marker,compute_timestamp_dead_marker_with_extra_bomb,Compute_Bomb_Exlpoison_Radius,Compute_Bomb_Exlpoison_Radius_Local

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def _compute_returns(rewards,gamma):
    returns = []
    discounted_reward = 0
    for reward in reversed(rewards):
        discounted_reward = reward + discounted_reward * gamma
        returns.append(discounted_reward)
    return returns[::-1]


def random_sublist(lst, k):
        states, actions, log_probs, returns, rewards=lst
        n = len(states)
        if n <= k:
            return (states, actions, log_probs, returns, rewards)  # Sonderfall: Liste ist zu kurz, also ganze Liste zurueckgeben
    
        # j wird so gewaehlt, dass die Sub-Liste von Laenge k in die Liste passt
        j = random.randint(0, n - 1)
        if j + k > n:  # wenn zu weit rechts, nach links schieben
            j = n - k
        return (states[j:j+k], actions[j:j+k], log_probs[j:j+k], returns[j:j+k], rewards[j:j+k])

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = []
        self.maxlen=capacity
        
        
    def __len__(self):
        return len(self.memory)

    def add(self,states,actions,log_probs,rewards,unroll_length):
        
            returns = _compute_returns(rewards,0.99)
            key=random.random()
                           
            val=(states, actions, log_probs, returns, rewards)

            if self.maxlen>len(self.memory):
                heapq.heappush(self.memory,(key,val))
            elif self.maxlen==len(self.memory):
                heapq.heappushpop(self.memory,(key,val))

    

        
    def sample(self,batch_size,actor,critic,unroll_length):
        total_sample = 0
        t_states = []
        t_actions = []
        t_log_probs = []
        t_vs = []
        t_advantages = []

        # Shuffle the indices of the buffer
        idx = 0
        idx_arr = np.arange(len(self))
        np.random.shuffle(idx_arr)
        
        # Stack the experiences until it exceeds the batch size
        while total_sample < batch_size:
            # If there is no more candidates, break!
            if idx >= len(idx_arr):
                break

          
            key,experiences = self.memory[idx_arr[idx]]
            experiences=random_sublist(experiences,unroll_length)
            idx += 1

            states, actions, log_probs, returns, rewards = experiences

            total_sample += 1

           #nicht volle Epsioden nehmen
            with torch.no_grad():
                eval_mode(actor,critic)
                cur_values, cur_log_probs, _ = compute(
                    torch.Tensor(states).to(device),
                    torch.Tensor(actions).to(device),
                    actor,critic
                )
                train_mode(actor,critic)

            cur_values = cur_values.cpu().detach().numpy()
            cur_log_probs = cur_log_probs.cpu().detach().numpy()

            unclipped_rhos = np.exp(cur_log_probs - np.squeeze(np.array(log_probs)))
            rhos = np.clip(unclipped_rhos, 0.0, 1.0)
            cs = np.clip(unclipped_rhos, 0.0, 1.0)

            vs, advantages = vtrace(
                values=cur_values,
                returns=returns,
                rewards=rewards,
                gamma=0.99,
                rhos=rhos,
                cs=cs,
                lmbd=1
            )

            states = torch.Tensor(states).to(device)
            actions = torch.Tensor(actions).to(device)
            log_probs = torch.Tensor(log_probs).to(device)
            vs = torch.Tensor(vs).to(device)
            advantages = torch.Tensor(advantages).to(device)
            log_probs = torch.squeeze(log_probs)
            t_states.append(states)
            t_actions.append(actions)
            t_log_probs.append(log_probs)
            t_vs.append(vs)
            t_advantages.append(advantages)

        

      

        return t_states, t_actions, t_log_probs, t_vs, t_advantages


def loop(env,policy_net,citic,optimizer_policy_net,optimizer_citic, n_episodes=2000):
   
   
    cnt_steps=0
    cnt_epsiodes=0
    train_every_x_steps=20
    do_not_train_first_x_episodes=30

    for i in range(n_episodes):
        state, info = env.reset()
        terminated, truncated, quit = False, False, False
        
        cnt_epsiodes+=1
      
        while not (terminated or truncated):
            cnt_steps+=1
            transformed_state=Transform_State(state)
            ten_transformed_state_grid=torch.tensor(transformed_state[0],dtype=torch.float32).to(device)
            ten_transformed_state_rest=torch.tensor(transformed_state[1],dtype=torch.float32).to(device)

            ten_transformed_state_grid=ten_transformed_state_grid.unsqueeze(0)
            ten_transformed_state_rest=ten_transformed_state_rest.unsqueeze(0)

            net_action = policy_net.get_best_action(ten_transformed_state_grid,ten_transformed_state_rest)
            net_action=net_action.squeeze(0).item()
          
            new_state, _, terminated, truncated, info = env.step(net_action)
                     
           

            if cnt_epsiodes>=do_not_train_first_x_episodes and train_every_x_steps<=cnt_steps:
               train(policy_net,optimizer,events_replay_memory_list,events_replay_memory_list_mini_sample,batch_sizes,criterion)
               cnt_steps=0
               

            state = new_state 

        #print(cnt_epsiodes)
    torch.save(policy_net.state_dict(), Path(__file__).parent / "model_supervised.pt")    


def main(argv=None):
   
    args = parse(argv)
    replay_memories=[]
    policy_net = BaseActor(4096, 461,6).to(device)
    optimizer_policy_net = optim.AdamW(policy_net.parameters(), lr=3e-4, amsgrad=True)
     
    citic = BaseCritic(4096, 461,6).to(device)
    optimizer_citic = optim.AdamW(citic.parameters(), lr=3e-4, amsgrad=True)
    
    policy_net.load_state_dict(torch.load( Path(__file__).parent / "model_supervised.pt"))
    citic.load_state_dict(torch.load( Path(__file__).parent / "model_supervised_value_function.pt"))
         
    env = gymnasium.make("bomberman_rl/bomberman-v0", args=args)
  
    env = ScoreRewardWrapper(env)
   
    loop(env,policy_net,citic,optimizer_policy_net,optimizer_citic)





if __name__ == "__main__":
    main()
  