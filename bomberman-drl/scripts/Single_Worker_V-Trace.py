
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


def vtrace(values, returns, rewards, gamma, rhos, cs, lmbd):
    v_t_plus_1 = np.concatenate((values[1:], returns[-1:]))
    deltas = rhos * (rewards + gamma * v_t_plus_1 - values)
    vs_minus_v_xs = deque([deltas[-1]])
    for i in range(len(values) - 2, -1, -1):
        vs_minus_v_xs.appendleft(deltas[i] + gamma * lmbd * cs[i] * vs_minus_v_xs[0])

    vs = np.array(vs_minus_v_xs) + np.array(values)
    vs_t_plus_1 = np.concatenate((vs[1:], returns[-1:]))
    advantages = rewards + gamma * vs_t_plus_1 - values

    return vs, advantages

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = []
        self.maxlen=capacity
        
        
    def __len__(self):
        return len(self.memory)

    def add(self,states,actions,log_probs,rewards):
        
            returns = _compute_returns(rewards,0.99)
            key=random.random()
                           
            val=(states, actions, log_probs, returns, rewards)

            if self.maxlen>len(self.memory):
                heapq.heappush(self.memory,(key,val))
            elif self.maxlen==len(self.memory):
                heapq.heappushpop(self.memory,(key,val))

    

        
    def sample(self,batch_size,actor,critic,unroll_length):
        total_sample = 0
        t_states_grid = []
        t_states_rest = []
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

            states_grid=[st[0] for st in states]
            states_grid=np.array(states_grid)
            states_rest=[st[1] for st in states]
            states_rest=np.array(states_rest)

            t_states_grid_cur=torch.tensor(states_grid, dtype=torch.float32).to(device)
            t_states_rest_cur=torch.tensor(states_rest, dtype=torch.float32).to(device)
            t_actions_cur=torch.tensor(actions, dtype=torch.float32).to(device)
            t_log_probs_cur=torch.tensor(log_probs, dtype=torch.float32).to(device)
            

            total_sample += 1

           #nicht volle Epsioden nehmen
            with torch.no_grad():
                actor.eval()
                critic.eval()
                cur_values=critic(t_states_grid_cur,t_states_rest_cur)
                porbs_cur=actor.forward_with_softmax(t_states_grid_cur, t_states_rest_cur)
                cur_log_probs,_ = actor.eval_action(
                    porbs_cur,t_actions_cur
                )
                actor.train()
                critic.train()

            cur_values = cur_values.squeeze(-1).cpu().detach().numpy()
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

            vs=torch.tensor(vs, dtype=torch.float32).to(device)
            advantages=torch.tensor(advantages, dtype=torch.float32).to(device)
            
           
            t_states_grid.append(t_states_grid_cur)
            t_states_rest.append(t_states_rest_cur)
            t_actions.append(t_actions_cur)
            t_log_probs.append(t_log_probs_cur)
            t_vs.append(vs)
            t_advantages.append(advantages)

        

      

        return t_states_grid,t_states_rest, t_actions, t_log_probs, t_vs, t_advantages



def train(minibatch,clip,entropy_regularization,actor,actor_optim,critic,critic_optim,max_grad_norm):
        """
        Proximal Policy Optimization
        Pseudocode: https://spinningup.openai.com/en/latest/algorithms/ppo.html
        """
        states_grid, states_rest, actions, log_probs, vs, advantages = minibatch

        # Normalize advantages is a good idea?
        # https://costa.sh/blog-the-32-implementation-details-of-ppo.html
       
        states_grid=torch.cat(states_grid)
        states_rest=torch.cat(states_rest)
        actions=torch.cat(actions)
        log_probs=torch.cat(log_probs)
        vs=torch.cat(vs)
        advantages=torch.cat(advantages)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        

        for _ in range(1):

            cur_values=critic(states_grid,states_rest)
            cur_log_probs,entropy = actor.eval_action(
                    actor.forward_with_softmax(states_grid, states_rest),actions
                )
            
            #cur_log_probs=cur_log_probs.squeeze(-1)
            #entropy=entropy.squeeze(-1)
            cur_values=cur_values.squeeze(-1)
            # Compute the policy loss
            ratios = torch.exp(cur_log_probs - log_probs)
            surrogate_loss1 = ratios * advantages
            surrogate_loss2 = (
                torch.clamp(ratios, 1.0 - clip, 1.0 + clip) * advantages
            )
            actor_loss = torch.min(surrogate_loss1, surrogate_loss2)

            # Optimize the policy loss along with the entropy term to encourage exploration
            actor_loss = actor_loss + entropy_regularization * entropy
            actor_loss = actor_loss.mean()
            actor_loss = -actor_loss

            # MSE loss
            critic_loss = nn.MSELoss()
            critic_loss=critic_loss(cur_values, vs)

           
            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(), max_grad_norm
            )
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                critic.parameters(), max_grad_norm
            )
            critic_optim.step()

def loop(env,policy_net,critic,optimizer_policy_net,optimizer_critic, n_episodes=2000):
   
   
    cnt_steps=0
    cnt_epsiodes=0
    train_every_x_steps=20
    do_not_train_first_x_episodes=20
    replay_mem=ReplayMemory(2000)
    avg_return=0
    print_stats_every=30
    cnt_stats=0
    for i in range(n_episodes):
        state, info = env.reset()
        terminated, truncated, quit = False, False, False
        cnt_stats+=1
        cnt_epsiodes+=1
        states=[]
        actions=[]
        log_probs=[]
        rewards=[]
        cur_return=0
        while not (terminated or truncated):
            cnt_steps+=1
            transformed_state=Transform_State(state)
            ten_transformed_state_grid=torch.tensor(transformed_state[0],dtype=torch.float32).to(device)
            ten_transformed_state_rest=torch.tensor(transformed_state[1],dtype=torch.float32).to(device)

            ten_transformed_state_grid=ten_transformed_state_grid.unsqueeze(0)
            ten_transformed_state_rest=ten_transformed_state_rest.unsqueeze(0)

            net_action,prob = policy_net.get_action(ten_transformed_state_grid,ten_transformed_state_rest)
            net_action=net_action.squeeze(0).item()
            prob=prob.squeeze(0).item()
          
            new_state, _, terminated, truncated, info = env.step(net_action)

            states.append(transformed_state)
            actions.append(net_action)
            log_probs.append(prob)
            if(new_state is not None):
                rewards.append(new_state["self_info"]["score"]-state["self_info"]["score"])
                cur_return+=new_state["self_info"]["score"]-state["self_info"]["score"]
            else:
                rewards.append(-5)
                cur_return-=5
           

            if cnt_epsiodes>=do_not_train_first_x_episodes and train_every_x_steps<=cnt_steps:
               minibatch=replay_mem.sample(40,policy_net,critic,20)
               train(minibatch,0.2,0.0001,policy_net,optimizer_policy_net,critic,optimizer_critic,1000)
               cnt_steps=0
               
                
            state = new_state 

        replay_mem.add(states,actions,log_probs,rewards)
        avg_return+=cur_return
        if cnt_stats==print_stats_every:
            print(f"Episode: {i+1} AVG_Return: {avg_return/print_stats_every}")
            avg_return=0
            cnt_stats=0
        #print(cnt_epsiodes)
    torch.save(policy_net.state_dict(), Path(__file__).parent / "model_rl.pt")    
    torch.save(critic.state_dict(), Path(__file__).parent / "model_rl_value_function.pt")   


def main(argv=None):
   
    args = parse(argv)
    replay_memories=[]
    policy_net = BaseActor(4096, 461,6).to(device)
    optimizer_policy_net = optim.AdamW(policy_net.parameters(), lr=3e-4, amsgrad=True)
     
    critic = BaseCritic(4096, 461,6).to(device)
    optimizer_critic = optim.AdamW(critic.parameters(), lr=3e-4, amsgrad=True)
    
    policy_net.load_state_dict(torch.load( Path(__file__).parent / "model_supervised.pt"))
    critic.load_state_dict(torch.load( Path(__file__).parent / "model_supervised_value_function.pt"))
         
    env = gymnasium.make("bomberman_rl/bomberman-v0", args=args)
  
    env = ScoreRewardWrapper(env)
   
    loop(env,policy_net,critic,optimizer_policy_net,optimizer_critic)





if __name__ == "__main__":
    main()
  