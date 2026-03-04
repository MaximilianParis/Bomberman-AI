
from Nets import BaseCritic
import torch
import torch.optim as optim
import torch.nn as nn
import pickle
import numpy as np
from collections import deque
import random
#from State_Algorithms import Transform_State


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def random_sublist(states, log_probs, rewards, returns, k):
       
        n = len(states)
        if n <= k:
            return (states, log_probs, returns, rewards)  # Sonderfall: Liste ist zu kurz, also ganze Liste zurueckgeben
    
        # j wird so gewaehlt, dass die Sub-Liste von Laenge k in die Liste passt
        j = random.randint(0, n - 1)
        if j + k > n:  # wenn zu weit rechts, nach links schieben
            j = n - k
        return (states[j:j+k], log_probs[j:j+k], returns[j:j+k], rewards[j:j+k])

def vtrace_on_policy(values, returns, rewards, gamma=0.99):
    n = 6
    T = len(values)

    values[T-1]=returns[T-1]

    if T <= n:
        return []
    
    vs = []
    
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

    for s in range(T - n, T):
        
        G = (
            (G - rewards[s - 1]) / gamma
            )
        
        vs.append(G)
    
    return vs

def sample(batch_size,critic,unroll_length,epsiodes):
        total_sample = 0
        
        t_vs = []
        t_states=[]

        # Shuffle the indices of the buffer
        idx = 0
        idx_arr = np.arange(len(epsiodes[0]))
        np.random.shuffle(idx_arr)
        
        # Stack the experiences until it exceeds the batch size
        while total_sample < batch_size:
            # If there is no more candidates, break!
            if idx >= len(idx_arr):
                break

            experiences
            state_list = epsiodes[0][idx_arr[idx]]
            reward_list = epsiodes[1][idx_arr[idx]]
            return_list = epsiodes[2][idx_arr[idx]]
         
            experiences=random_sublist(state_list,reward_list,return_list,unroll_length)
            idx += 1

            states, log_probs, rewards, returns = experiences

            total_sample += 1

          
            with torch.no_grad():
                critic.eval()
                cur_values=critic(torch.tensor(states, dtype=torch.float32).to(device))
               
            critic.train()
            cur_values = cur_values.cpu().detach().numpy()
                        
            vs= vtrace_on_policy(cur_values, returns, rewards, gamma=0.99)
                      
            vs = torch.tensor(vs,dtype=torch.float32).to(device)
               
            states= torch.tensor(states,dtype=torch.float32).to(device)

            

            t_vs.append(vs)
            t_states.append(states)
                  

        

        return t_states, t_vs





def train_supervised(critic,optimizer,max_steps=2000):
  
   
    with open("Value_Function_Training_Data.pkl", "rb") as f:
         epsiodes_list= pickle.load(f)
    
    criterion=nn.MSELoss()

    for training_steps in range(0,max_steps):
       
        t_states,t_vs=sample(20,critic,20,epsiodes_list)

        t_states=torch.cat(t_states)
        t_vs=torch.cat(t_vs)
          
        values = critic(t_states)

        loss = criterion(values, t_vs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        

    #torch.save(policy_net.state_dict(), Path(__file__).parent / "model_supervised_training_by_file.pt")




def main(argv=None):
   
   
    citic = BaseCritic(2145, 6).to(device)
    optimizer = optim.AdamW(citic.parameters(), lr=1e-3, amsgrad=True)
   
    train_supervised(citic,optimizer)
  





if __name__ == "__main__":
    main()
  