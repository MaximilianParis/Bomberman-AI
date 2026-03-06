
from Nets import BaseCritic
import torch
import torch.optim as optim
import torch.nn as nn
import pickle
import numpy as np
from collections import deque
import random
from pathlib import Path
#from State_Algorithms import Transform_State


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def random_sublist(states, returns,rewards, k):
       
        n = len(states)
        if n <= k:
            return (states, returns, rewards)  # Sonderfall: Liste ist zu kurz, also ganze Liste zurueckgeben
    
        # j wird so gewaehlt, dass die Sub-Liste von Laenge k in die Liste passt
        j = random.randint(0, n - 1)
        if j + k > n:  # wenn zu weit rechts, nach links schieben
            j = n - k
        return (states[j:j+k], returns[j:j+k], rewards[j:j+k])

def vtrace_on_policy(values, returns, rewards, gamma=0.99):
    n = 7
    T = len(values)

    #values[T-1]=returns[T-1]
    vs = []
    if T <= n:
        G = 0.0
        gamma_k = 1.0
        for k in range(T):
            G += gamma_k * rewards[k]
            gamma_k *= gamma
    
        G += (gamma ** (T-1)) * values[T-1]
        vs.append(G)

        for s in range(1, T-1):
        
            G = (
                (G - rewards[s - 1]) / gamma
                )
        
            vs.append(G)

        return vs
    
    
    
    # --- 1) Ersten Return (s = 0) normal berechnen ---
    G = 0.0
    gamma_k = 1.0
    for k in range(n):
        G += gamma_k * rewards[k]
        gamma_k *= gamma
    
    G += (gamma ** n) * values[n]
    vs.append(G)
    
    # --- 2) Sliding Window Update ---
    gamma_n_minus_1 = gamma ** (n - 1)
    gamma_n = gamma ** n
    
    for s in range(1, T - n):
        
        G = (
            (G - rewards[s - 1]) / gamma
            + gamma_n_minus_1 * rewards[s + n - 1]
            + gamma_n * (values[s + n] - values[s + n - 1])
        )
        
        vs.append(G)

    for s in range(T - n, T-1):
        
        G = (
            (G - rewards[s - 1]) / gamma
            )
        
        vs.append(G)
    
    return vs

def sample(batch_size,critic,target_critic,unroll_length,epsiodes):
        total_sample = 0
        
        t_vs = []
        t_states_grid=[]
        t_states_rest=[]

        # Shuffle the indices of the buffer
        idx = 0
        idx_arr = np.arange(len(epsiodes))
        np.random.shuffle(idx_arr)
        
        # Stack the experiences until it exceeds the batch size
        while total_sample < batch_size:
            # If there is no more candidates, break!
            if idx >= len(idx_arr):
                break

            
            state_list = epsiodes[idx_arr[idx]][0]
            reward_list = epsiodes[idx_arr[idx]][1]
            return_list = epsiodes[idx_arr[idx]][2]
         
            experiences=random_sublist(state_list,return_list,reward_list,unroll_length)
            idx += 1

            states, returns,rewards = experiences

            total_sample += 1

            states_grid=[st[0] for st in states]
            states_grid=np.array(states_grid)
            states_rest=[st[1] for st in states]
            states_rest=np.array(states_rest)

            t_states_grid_cur=torch.tensor(states_grid, dtype=torch.float32).to(device)
            t_states_rest_cur=torch.tensor(states_rest, dtype=torch.float32).to(device)


            with torch.no_grad():
                critic.eval()
                cur_values=target_critic(t_states_grid_cur,t_states_rest_cur)
               
            critic.train()
           # cur_values=cur_values.squeeze(1)
            cur_values = cur_values.cpu().detach().numpy()
                        
            vs= vtrace_on_policy(cur_values, returns, rewards, gamma=0.99)

           
            if -5!=reward_list[-1]:
                vs.append(-5)
               
            else:
                t_states_grid_cur=torch.tensor(states_grid[0:len(states_grid)-1], dtype=torch.float32).to(device)
                t_states_rest_cur=torch.tensor(states_rest[0:len(states_rest)-1], dtype=torch.float32).to(device)

            vs=np.array(vs)     
            
            vs = torch.tensor(vs,dtype=torch.float32).to(device)
            #vs=vs.unsqueeze(0)   
                       
            

            t_vs.append(vs)
            t_states_grid.append(t_states_grid_cur)
            t_states_rest.append(t_states_rest_cur)
                  

        

        return t_states_grid,t_states_rest, t_vs



def update_target_net(critic,target_critic):
       
        tau=0.025
        critic_state_dict =critic.state_dict()
        target_critic_state_dict = target_critic.state_dict()
        for key in critic_state_dict:
            target_critic_state_dict[key] = critic_state_dict[key] *tau + target_critic_state_dict[key] * (1 - tau)
        target_critic.load_state_dict(target_critic_state_dict)

def train_supervised(critic,target_critic,optimizer,max_steps=20000):
  
   
    with open("Value_Function_Training_Data.pkl", "rb") as f:
         epsiodes_list= pickle.load(f)
    
    criterion=nn.MSELoss()
    cnt=0
    update_every=30
    average_loss=0.0
    average_explained_var=0.0
    cnt_stats=0
    print_stats_every=50
    for training_steps in range(0,max_steps):
        cnt+=1
        cnt_stats+=1
        t_states_grid,t_states_rest, t_vs=sample(20,critic,target_critic,20,epsiodes_list)

        t_states_grid=torch.cat(t_states_grid)
        t_states_rest=torch.cat(t_states_rest)
        t_vs=torch.cat(t_vs)
       # t_vs = (t_vs - t_vs.mean()) / (t_vs.std() + 1e-8)
        values = critic(t_states_grid,t_states_rest)

        loss = criterion(values, t_vs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            var_y = torch.var(t_vs)
            explained_var = 1 - torch.var(t_vs - values) / (var_y + 1e-8)

        #print(f"Epoch: {training_steps} Loss: {loss} Var: {explained_var}")
        average_loss+=loss.item()
        average_explained_var+=max(0,explained_var.item())

        if(cnt==update_every):
           cnt=0
           update_target_net(critic,target_critic)
           

        if(cnt_stats==print_stats_every):
           print(f"Epoch: {training_steps+1} Loss: {average_loss/print_stats_every} Var: {average_explained_var/print_stats_every}")
           average_loss=0
           average_explained_var=0
           cnt_stats=0



        

    torch.save(critic.state_dict(), Path(__file__).parent / "model_supervised_value_function.pt")




def main(argv=None):
   
    
    citic = BaseCritic(4096, 461,6).to(device)
    target_critic = BaseCritic(4096, 461,6).to(device)
    optimizer = optim.AdamW(citic.parameters(), lr=3e-4, amsgrad=True)
   
    train_supervised(citic,target_critic,optimizer)
  





if __name__ == "__main__":
    main()
  