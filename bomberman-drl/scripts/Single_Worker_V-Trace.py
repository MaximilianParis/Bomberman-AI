
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

STEPPED_IN_BOMB_NEG="STEPPED_IN_BOMB_NEG"

BOMB_ESCAPE_LENGTH_ONE_POS="BOMB_ESCAPE_LENGTH_ONE_POS"
BOMB_ESCAPE_LENGTH_ONE_NEG="BOMB_ESCAPE_LENGTH_ONE_NEG"

BOMB_ESCAPE_LENGTH_TWO_POS="BOMB_ESCAPE_LENGTH_TWO_POS"
BOMB_ESCAPE_LENGTH_TWO_NEG="BOMB_ESCAPE_LENGTH_TWO_NEG"

BOMB_ESCAPE_LENGTH_THREE_POS="BOMB_ESCAPE_LENGTH_THREE_POS"
BOMB_ESCAPE_LENGTH_THREE_NEG="BOMB_ESCAPE_LENGTH_THREE_NEG"

BOMB_ESCAPE_LENGTH_FOUR_POS="BOMB_ESCAPE_LENGTH_FOUR_POS"
BOMB_ESCAPE_LENGTH_FOUR_NEG="BOMB_ESCAPE_LENGTH_FOUR_NEG"

STEPPED_IN_BLOCKED_FIELD="STEPPED_IN_BLOCKED_FIELD"

STEPPED_IN_EXPLOSION="STEPPED_IN_EXPLOSION"

STEPPED_TOWARDS_TARGET_POS="STEPPED_TOWARDS_TARGET_POS"
STEPPED_TOWARDS_TARGET_NEG="STEPPED_TOWARDS_TARGET_NEG"

Target_IN_RANGE_POS="Target_IN_RANGE_POS"
Target_IN_RANGE_NEG="Target_IN_RANGE_NEG"

CAN_ESCAPE_OWN_BOMB_POS="CAN_ESCAPE_OWN_BOMB_POS"
CAN_ESCAPE_OWN_BOMB_NEG="CAN_ESCAPE_OWN_BOMB_NEG"

STEPPED_TOWARDS_COIN_POS="STEPPED_TOWARDS_COIN_POS"
STEPPED_TOWARDS_COIN_NEG="STEPPED_TOWARDS_COIN_NEG"

SHOULD_HAVE_PLANTED_BOMB="SHOULD_HAVE_PLANTED_BOMB"

PLANTED_WITHOUT_BOMBS_LEFT="PLANTED_WITHOUT_BOMBS_LEFT"



STEPPED_IN_BOMB_NEG="STEPPED_IN_BOMB_NEG"

CAN_ESCAPE_OWN_BOMB_POS="CAN_ESCAPE_OWN_BOMB_POS"


class Replay_Memory_Supervised(object):

    def __init__(self, capacity):
        self.memory_state =[]
        self.memory_action =[]
        self.capacity=capacity
        #self.batch_size=20

    def push(self, state, action):
        """Save a transition"""
        #entferne zu Alte Zustandaktionspaare wenn voll
        if self.capacity==len(self.memory_state):
            self.memory_state.pop(0)
            self.memory_action.pop(0)

        self.memory_state.append(state)
        self.memory_action.append(action)

    def sample(self,batch_size):
        
        indices = range(len(self.memory_state))
        sampled_indices = random.sample(indices, k=batch_size)
                    
        sampled_states = [self.memory_state[i] for i in sampled_indices]
        sampled_actions = [self.memory_action[i] for i in sampled_indices]

        return sampled_states,sampled_actions

    def __len__(self):
        return len(self.memory_state)


def configurate_batch_sizes(replay_memories,batch_sizes):
       
    #je nach Wichtigkeit der Kategorie wird pro Kategorie Anzahl an Zustandaktionspaaren berechnet die in den batch kommen
    print("")
    factor=replay_memories[BOMB_ESCAPE_LENGTH_ONE_NEG][1]/(replay_memories[BOMB_ESCAPE_LENGTH_ONE_NEG][1]+replay_memories[BOMB_ESCAPE_LENGTH_ONE_POS][1]+0.0000001)
    batch_sizes[0]=int(0.5*int((factor*70)+10))
    batch_sizes[1]=int(0.5*int((factor*70)+10))
   
    print(f"BOMB_ESCAPE_LENGTH_ONE_NEG:{replay_memories[BOMB_ESCAPE_LENGTH_ONE_NEG][1]}")
    print(f"BOMB_ESCAPE_LENGTH_ONE_POS:{replay_memories[BOMB_ESCAPE_LENGTH_ONE_POS][1]}")
    print(f"factor:{factor}")
    print(f"batch_sizes:{batch_sizes[0]}")
    print("")

    factor=replay_memories[BOMB_ESCAPE_LENGTH_TWO_NEG][1]/(replay_memories[BOMB_ESCAPE_LENGTH_TWO_NEG][1]+replay_memories[BOMB_ESCAPE_LENGTH_TWO_POS][1]+0.0000001)
    batch_sizes[2]=int(0.5*int((factor*60)+10))
    batch_sizes[3]=int(0.5*int((factor*60)+10))
   
    print(f"BOMB_ESCAPE_LENGTH_TWO_NEG:{replay_memories[BOMB_ESCAPE_LENGTH_TWO_NEG][1]}")
    print(f"BOMB_ESCAPE_LENGTH_TWO_POS:{replay_memories[BOMB_ESCAPE_LENGTH_TWO_POS][1]}")
    print(f"factor:{factor}")
    print(f"batch_sizes:{batch_sizes[2]}")
    print("")

    factor=replay_memories[BOMB_ESCAPE_LENGTH_THREE_NEG][1]/(replay_memories[BOMB_ESCAPE_LENGTH_THREE_NEG][1]+replay_memories[BOMB_ESCAPE_LENGTH_THREE_POS][1]+0.0000001)
    batch_sizes[4]=int(0.5*int((factor*50)+10))
    batch_sizes[5]=int(0.5*int((factor*50)+10))
    
    print(f"BOMB_ESCAPE_LENGTH_THREE_NEG:{replay_memories[BOMB_ESCAPE_LENGTH_THREE_NEG][1]}")
    print(f"BOMB_ESCAPE_LENGTH_THREE_POS:{replay_memories[BOMB_ESCAPE_LENGTH_THREE_POS][1]}")
    print(f"factor:{factor}")
    print(f"batch_sizes:{batch_sizes[4]}")
    print("")

    factor=replay_memories[BOMB_ESCAPE_LENGTH_FOUR_NEG][1]/(replay_memories[BOMB_ESCAPE_LENGTH_FOUR_NEG][1]+replay_memories[BOMB_ESCAPE_LENGTH_FOUR_POS][1]+0.0000001)
    batch_sizes[6]=int(0.5*int((factor*20)+10))
    batch_sizes[7]=int(0.5*int((factor*20)+10))
    
    print(f"BOMB_ESCAPE_LENGTH_FOUR_NEG:{replay_memories[BOMB_ESCAPE_LENGTH_FOUR_NEG][1]}")
    print(f"BOMB_ESCAPE_LENGTH_FOUR_POS:{replay_memories[BOMB_ESCAPE_LENGTH_FOUR_POS][1]}")
    print(f"factor:{factor}")
    print(f"batch_sizes:{batch_sizes[6]}")
    print("")
       
    batch_sizes[8]=int(0.5*int(replay_memories[STEPPED_IN_BLOCKED_FIELD][1]*0.5+10))
    
    print(f"STEPPED_IN_BLOCKED_FIELD:{replay_memories[STEPPED_IN_BLOCKED_FIELD][1]}")
    print(f"batch_sizes:{batch_sizes[8]}")
    print("")

    factor=replay_memories[STEPPED_TOWARDS_TARGET_NEG][1]/(replay_memories[STEPPED_TOWARDS_TARGET_NEG][1]+replay_memories[STEPPED_TOWARDS_TARGET_POS][1]+0.0000001)
    batch_sizes[9]=int(0.5*int((factor*120)+50))
    batch_sizes[10]=int(0.5*int((factor*120)+50))
   
    print(f"STEPPED_TOWARDS_TARGET_NEG:{replay_memories[STEPPED_TOWARDS_TARGET_NEG][1]}")
    print(f"STEPPED_TOWARDS_TARGET_POS:{replay_memories[STEPPED_TOWARDS_TARGET_POS][1]}")
    print(f"factor:{factor}")
    print(f"batch_sizes:{batch_sizes[9]}")
    print("")

    factor=replay_memories[STEPPED_TOWARDS_COIN_NEG][1]/(replay_memories[STEPPED_TOWARDS_COIN_NEG][1]+replay_memories[STEPPED_TOWARDS_COIN_POS][1]+0.0000001)
    batch_sizes[11]=int(0.5*int((factor*100)+40))
    batch_sizes[12]=int(0.5*int((factor*100)+40))
   
    print(f"STEPPED_TOWARDS_COIN_NEG:{replay_memories[STEPPED_TOWARDS_COIN_NEG][1]}")
    print(f"STEPPED_TOWARDS_COIN_POS:{replay_memories[STEPPED_TOWARDS_COIN_POS][1]}")
    print(f"factor:{factor}")
    print(f"batch_sizes:{batch_sizes[11]}")
    print("")

    factor=replay_memories[Target_IN_RANGE_NEG][1]/(replay_memories[Target_IN_RANGE_NEG][1]+replay_memories[Target_IN_RANGE_POS][1]+0.0000001)
    batch_sizes[13]=int(0.5*int((factor*80)+30))
    batch_sizes[14]=int(0.5*int((factor*80)+30))
   
    print(f"Target_IN_RANGE_NEG:{replay_memories[Target_IN_RANGE_NEG][1]}")
    print(f"Target_IN_RANGE_POS:{replay_memories[Target_IN_RANGE_POS][1]}")
    print(f"factor:{factor}")
    print(f"batch_sizes:{batch_sizes[13]}")
    print("")

    factor=replay_memories[CAN_ESCAPE_OWN_BOMB_NEG][1]/(replay_memories[CAN_ESCAPE_OWN_BOMB_NEG][1]+replay_memories[CAN_ESCAPE_OWN_BOMB_POS][1]+0.0000001)
    batch_sizes[15]=int(0.5*int((factor*80)+30))
    batch_sizes[16]=int(0.5*int((factor*80)+30))
   
    print(f"CAN_ESCAPE_OWN_BOMB_NEG:{replay_memories[CAN_ESCAPE_OWN_BOMB_NEG][1]}")
    print(f"CAN_ESCAPE_OWN_BOMB_POS:{replay_memories[CAN_ESCAPE_OWN_BOMB_POS][1]}")
    print(f"factor:{factor}")
    print(f"batch_sizes:{batch_sizes[15]}")
    print("")

    batch_sizes[17]=int(0.5*int(replay_memories[PLANTED_WITHOUT_BOMBS_LEFT][1]*0.5+10))
    
    print(f"PLANTED_WITHOUT_BOMBS_LEFT:{replay_memories[PLANTED_WITHOUT_BOMBS_LEFT][1]}")
    print(f"batch_sizes:{batch_sizes[17]}")
    print("")
   
    batch_sizes[18]=int(0.5*int(replay_memories[SHOULD_HAVE_PLANTED_BOMB][1]*0.5+20))
    

    print(f"SHOULD_HAVE_PLANTED_BOMB:{replay_memories[SHOULD_HAVE_PLANTED_BOMB][1]}")
    print(f"batch_sizes:{batch_sizes[18]}")
    print("")
    print("------------------------------------------------------------------------")


    replay_memories[BOMB_ESCAPE_LENGTH_ONE_NEG][1]=0
    replay_memories[BOMB_ESCAPE_LENGTH_ONE_POS][1]=0
    replay_memories[BOMB_ESCAPE_LENGTH_TWO_NEG][1]=0
    replay_memories[BOMB_ESCAPE_LENGTH_TWO_POS][1]=0
    replay_memories[BOMB_ESCAPE_LENGTH_THREE_NEG][1]=0
    replay_memories[BOMB_ESCAPE_LENGTH_THREE_POS][1]=0
    replay_memories[BOMB_ESCAPE_LENGTH_FOUR_NEG][1]=0
    replay_memories[BOMB_ESCAPE_LENGTH_FOUR_POS][1]=0
    replay_memories[STEPPED_IN_BLOCKED_FIELD][1]=0
    replay_memories[STEPPED_TOWARDS_TARGET_NEG][1]=0
    replay_memories[STEPPED_TOWARDS_TARGET_POS][1]=0
    replay_memories[Target_IN_RANGE_POS][1]=0
    replay_memories[Target_IN_RANGE_NEG][1]=0
    replay_memories[CAN_ESCAPE_OWN_BOMB_POS][1]=0
    replay_memories[CAN_ESCAPE_OWN_BOMB_NEG][1]=0
    replay_memories[PLANTED_WITHOUT_BOMBS_LEFT][1]=0
    replay_memories[SHOULD_HAVE_PLANTED_BOMB][1]=0
    replay_memories[STEPPED_TOWARDS_COIN_POS][1]=0
    replay_memories[STEPPED_TOWARDS_COIN_NEG][1]=0
                      


def evaluate_net_action(transfromed_state,state,net_action,expert_action,expert,events_replay_memory_list):
        bombs_old = state["bombs"]
        walls_old=state["walls"]
        crates_old=state["crates"]
        explosions_old=state["explosions"]
        position_old = state["self_info"]["position"]
        bombs_left=state["self_info"]["bombs_left"]
        for a in range(0,17):
            for b in range(0,17):
              if position_old[a,b]==1:
                  (x_old,y_old)=(a,b)
        
        memo={}
        limit=5
        timestamp_dead_marker=compute_timestamp_dead_marker(state["opponents_pos"],bombs_old,explosions_old,crates_old,walls_old)
        find_rescue_route(timestamp_dead_marker,x_old,y_old,0,memo,limit)

        #Agent ist in blockiertes Feld(Bombe,Wand,Explosion,Kiste) gelaufen?
        if timestamp_dead_marker[(1,x_old-1,y_old)]>=2 and net_action==3:
            events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][0].push(transfromed_state,expert_action)
            events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][1]= events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][1]+1
           
                
        elif timestamp_dead_marker[(1,x_old+1,y_old)]>=2 and net_action==1:
            events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][0].push(transfromed_state,expert_action)
            events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][1]= events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][1]+1
          
                
        elif timestamp_dead_marker[(1,x_old,y_old-1)]>=2 and net_action==2:
            events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][0].push(transfromed_state,expert_action)
            events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][1]= events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][1]+1
           
                
        elif timestamp_dead_marker[(1,x_old,y_old+1)]>=2 and net_action==0:
            events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][0].push(transfromed_state,expert_action)
            events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][1]= events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][1]+1
           


        if timestamp_dead_marker[(0,x_old,y_old)]==1 or timestamp_dead_marker[(0,x_old,y_old)]==4:

             #kategorisiere je nach Laenge des Wegs
            if memo[(x_old,y_old,0)]==1 and net_action==expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_ONE_POS][0].push(transfromed_state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_ONE_POS][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_ONE_POS][1]+1
                 


            elif memo[(x_old,y_old,0)]==1 and net_action!=expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_ONE_NEG][0].push(transfromed_state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_ONE_NEG][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_ONE_NEG][1]+1
               
                

            elif memo[(x_old,y_old,0)]==2 and net_action==expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_TWO_POS][0].push(transfromed_state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_TWO_POS][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_TWO_POS][1]+1
                


            elif memo[(x_old,y_old,0)]==2 and net_action!=expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_TWO_NEG][0].push(transfromed_state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_TWO_NEG][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_TWO_NEG][1]+1
                


            elif memo[(x_old,y_old,0)]==3 and net_action==expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_THREE_POS][0].push(transfromed_state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_THREE_POS][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_THREE_POS][1]+1
               


            elif memo[(x_old,y_old,0)]==3 and net_action!=expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_THREE_NEG][0].push(transfromed_state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_THREE_NEG][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_THREE_NEG][1]+1
              


            elif memo[(x_old,y_old,0)]==4 and net_action==expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_FOUR_POS][0].push(transfromed_state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_FOUR_POS][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_FOUR_POS][1]+1
              


            elif memo[(x_old,y_old,0)]==4 and net_action!=expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_FOUR_NEG][0].push(transfromed_state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_FOUR_NEG][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_FOUR_NEG][1]+1
              


        else:
       #target_x,target_y,distanz,coin?
            if expert.target_coin==True:

                if net_action==expert_action and expert.target_distanz<=15:
                     events_replay_memory_list[STEPPED_TOWARDS_COIN_POS][0].push(transfromed_state,expert_action)
                     events_replay_memory_list[STEPPED_TOWARDS_COIN_POS][1]= events_replay_memory_list[STEPPED_TOWARDS_COIN_POS][1]+1
                     
                elif net_action!=expert_action and expert.target_distanz<=15:
                     events_replay_memory_list[STEPPED_TOWARDS_COIN_NEG][0].push(transfromed_state,expert_action)
                     events_replay_memory_list[STEPPED_TOWARDS_COIN_NEG][1]= events_replay_memory_list[STEPPED_TOWARDS_COIN_NEG][1]+1
                    


            else:
                if expert_action==5:
                    if net_action==5:
                        events_replay_memory_list[Target_IN_RANGE_POS][0].push(transfromed_state,expert_action)
                        events_replay_memory_list[Target_IN_RANGE_POS][1]= events_replay_memory_list[Target_IN_RANGE_POS][1]+1
                        
                        events_replay_memory_list[CAN_ESCAPE_OWN_BOMB_POS][0].push(transfromed_state,expert_action)
                        events_replay_memory_list[CAN_ESCAPE_OWN_BOMB_POS][1]= events_replay_memory_list[CAN_ESCAPE_OWN_BOMB_POS][1]+1
                       
                    else:
                         events_replay_memory_list[SHOULD_HAVE_PLANTED_BOMB][0].push(transfromed_state,expert_action)
                         events_replay_memory_list[SHOULD_HAVE_PLANTED_BOMB][1]= events_replay_memory_list[SHOULD_HAVE_PLANTED_BOMB][1]+1
                        

                else:
                     if  net_action==expert_action and expert.target_distanz<=15:
                         events_replay_memory_list[STEPPED_TOWARDS_TARGET_POS][0].push(transfromed_state,expert_action)
                         events_replay_memory_list[STEPPED_TOWARDS_TARGET_POS][1]= events_replay_memory_list[STEPPED_TOWARDS_TARGET_POS][1]+1
                       
                     elif net_action!=expert_action and expert.target_distanz<=15:
                         events_replay_memory_list[STEPPED_TOWARDS_TARGET_NEG][0].push(transfromed_state,expert_action)
                         events_replay_memory_list[STEPPED_TOWARDS_TARGET_NEG][1]= events_replay_memory_list[STEPPED_TOWARDS_TARGET_NEG][1]+1
                       

                     if net_action==5:
                          timestamp_dead_marker=compute_timestamp_dead_marker_with_extra_bomb(state["opponents_pos"],bombs_old,explosions_old,crates_old,walls_old,x_old,y_old)
                          memo={}
                          limit=5
                          find_rescue_route(timestamp_dead_marker,x_old,y_old,0,memo,limit)
                          
                          if memo[(x_old,y_old,0)]>4:
                             events_replay_memory_list[CAN_ESCAPE_OWN_BOMB_NEG][0].push(transfromed_state,expert_action)
                             events_replay_memory_list[CAN_ESCAPE_OWN_BOMB_NEG][1]= events_replay_memory_list[CAN_ESCAPE_OWN_BOMB_NEG][1]+1
                             

                          Bomb_Exlpoison_Radius_Local=Compute_Bomb_Exlpoison_Radius_Local(walls_old,x_old,y_old)
                        
                          if (expert.target_x,expert.target_y) not in Bomb_Exlpoison_Radius_Local:
                             events_replay_memory_list[Target_IN_RANGE_NEG][0].push(transfromed_state,expert_action)
                             events_replay_memory_list[Target_IN_RANGE_NEG][1]= events_replay_memory_list[Target_IN_RANGE_NEG][1]+1
                            

                          if bombs_left==0:
                               events_replay_memory_list[PLANTED_WITHOUT_BOMBS_LEFT][0].push(transfromed_state,expert_action)
                               events_replay_memory_list[PLANTED_WITHOUT_BOMBS_LEFT][1]= events_replay_memory_list[PLANTED_WITHOUT_BOMBS_LEFT][1]+1
                              


def form_supervised_batch(replay_memories,batch_sizes):
     
   
    states_list_grid=[]
    states_list_rest=[]
    actions_list=[]
    cnt=0
    for replay_mem in replay_memories.items():
        #print(f"Kategorie:{replay_mem[0]} Batchgroesse:{batch_sizes[cnt]}")
      
        replay_mem=replay_mem[1][0]
        if(len(replay_mem)>0):
            sampled_states,sampled_actions = replay_mem.sample(min(int(batch_sizes[cnt]/2),len(replay_mem)))
                            

            sampled_states_grid=[state[0] for state in sampled_states]
            sampled_states_grid=np.array(sampled_states_grid)
            sampled_states_rest=[state[1] for state in sampled_states]
            sampled_states_rest=np.array(sampled_states_rest)
           
            states_list_grid.append(torch.tensor(sampled_states_grid, dtype=torch.float32).to(device))
            states_list_rest.append(torch.tensor(sampled_states_rest, dtype=torch.float32).to(device))
           
            actions_list.append(torch.tensor(sampled_actions, dtype=torch.long).to(device))
         
    return  states_list_grid,states_list_rest,actions_list 
       



def random_sublist(lst, k):
        states, actions, log_probs, rewards=lst
        n = len(states)
        if n <= k:
            return (states, actions, log_probs, rewards)  # Sonderfall: Liste ist zu kurz, also ganze Liste zurueckgeben
    
        # j wird so gewaehlt, dass die Sub-Liste von Laenge k in die Liste passt
        j = random.randint(0, n - 1)
        if j + k > n:  # wenn zu weit rechts, nach links schieben
            j = n - k
        return (states[j:j+k], actions[j:j+k], log_probs[j:j+k], rewards[j:j+k])


def vtrace(values, rewards, gamma, rhos, cs, lmbd):
    v_t_plus_1 = values[1:]
    deltas = rhos[:len(values)-1] * (rewards[:len(values)-1] + gamma * v_t_plus_1 - values[:len(values)-1])
    vs_minus_v_xs = deque([deltas[-1]])
    for i in range(len(v_t_plus_1) - 2, -1, -1):
        vs_minus_v_xs.appendleft(deltas[i] + gamma * lmbd * cs[i] * vs_minus_v_xs[0])

    vs = np.array(vs_minus_v_xs) + np.array(values[:len(values)-1])
    vs_t_plus_1 = vs[1:]
    advantages = rewards[:len(values)-2] + gamma * vs_t_plus_1 - values[:len(values)-2]

    return vs[:len(vs)-1], advantages

def vtrace_died(values, rewards, gamma, rhos, cs, lmbd):
    v_t_plus_1 = np.concatenate((values[1:], [0]))
    deltas = rhos * (rewards + gamma * v_t_plus_1 - values)
    vs_minus_v_xs = deque([deltas[-1]])
    for i in range(len(values) - 2, -1, -1):
        vs_minus_v_xs.appendleft(deltas[i] + gamma * lmbd * cs[i] * vs_minus_v_xs[0])

    vs = np.array(vs_minus_v_xs) + np.array(values)
    vs_t_plus_1 = np.concatenate((vs[1:], [0]))
    advantages = rewards + gamma * vs_t_plus_1 - values

    return vs, advantages

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = []
        self.maxlen=capacity
        
        
    def __len__(self):
        return len(self.memory)

    def add(self,states,actions,log_probs,rewards):
        
            #returns = _compute_returns(rewards,0.99)
            key=random.random()
                           
            val=(states, actions, log_probs, rewards)

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
            experiences=random_sublist(experiences,unroll_length+2)
            idx += 1

            states, actions, log_probs, rewards = experiences

            states_grid=[st[0] for st in states]
            states_grid=np.array(states_grid)
            states_rest=[st[1] for st in states]
            states_rest=np.array(states_rest)

            t_states_grid_cur=torch.tensor(states_grid, dtype=torch.float32).to(device)
            t_states_rest_cur=torch.tensor(states_rest, dtype=torch.float32).to(device)
            t_actions_cur=torch.tensor(actions, dtype=torch.float32).to(device)
               
            

            total_sample += 1

            
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
            #Agent gestorben?
            if rewards[-1]!=-5:
                               
                vs, advantages = vtrace(
                    values=cur_values,
                    
                    rewards=rewards,
                    gamma=0.99,
                    rhos=rhos,
                    cs=cs,
                    lmbd=1
                )
                           

                t_states_grid_cur=torch.tensor(states_grid[:unroll_length], dtype=torch.float32).to(device)
                t_states_rest_cur=torch.tensor(states_rest[:unroll_length], dtype=torch.float32).to(device)
                t_actions_cur=torch.tensor(actions[:unroll_length], dtype=torch.float32).to(device)
                t_log_probs_cur=torch.tensor(log_probs[:unroll_length], dtype=torch.float32).to(device)

            else:

                vs, advantages = vtrace_died(
                    values=cur_values,
                    
                    rewards=rewards,
                    gamma=0.99,
                    rhos=rhos,
                    cs=cs,
                    lmbd=1
                )
                                           
                t_log_probs_cur=torch.tensor(log_probs, dtype=torch.float32).to(device)



            vs=torch.tensor(vs, dtype=torch.float32).to(device)
            advantages=torch.tensor(advantages, dtype=torch.float32).to(device)

            t_states_grid.append(t_states_grid_cur)
            t_states_rest.append(t_states_rest_cur)
            t_actions.append(t_actions_cur)
            t_log_probs.append(t_log_probs_cur)
            t_vs.append(vs)
            t_advantages.append(advantages)

        

      

        return t_states_grid,t_states_rest, t_actions, t_log_probs, t_vs, t_advantages



def train(minibatch,minibatch_supervised,clip,entropy_regularization,actor,actor_optim,critic,critic_optim,max_grad_norm):
        """
        Proximal Policy Optimization
        Pseudocode: https://spinningup.openai.com/en/latest/algorithms/ppo.html
        """
        states_grid, states_rest, actions, log_probs, vs, advantages = minibatch

     
       
        states_grid=torch.cat(states_grid)
        states_rest=torch.cat(states_rest)
        actions=torch.cat(actions)
        log_probs=torch.cat(log_probs)
        vs=torch.cat(vs)
        advantages=torch.cat(advantages)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        supervised_criterion=nn.CrossEntropyLoss()
        states_list_grid_supervised,states_list_rest_supervised,actions_list_supervised =minibatch_supervised

        states_list_grid_supervised=torch.cat(states_list_grid_supervised)
        states_list_rest_supervised=torch.cat(states_list_rest_supervised)
        actions_list_supervised=torch.cat(actions_list_supervised)

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
            
            supervised_logits=actor(states_list_grid_supervised,states_list_rest_supervised)

            actor_loss = -actor_loss+0.02*supervised_criterion(supervised_logits,actions_list_supervised)

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

        with torch.no_grad():
            var_y = torch.var(vs)
            explained_var = 1 - torch.var(vs - cur_values) / (var_y + 1e-8)
                   
      
        return actor_loss.item(),critic_loss.item(),max(0,explained_var.item()),entropy.mean().item()

def loop(env,policy_net,critic,optimizer_policy_net,optimizer_critic, n_episodes=2000):
    expert=Expert_Agent()
    events_replay_memory_list=dict([(BOMB_ESCAPE_LENGTH_ONE_POS,[Replay_Memory_Supervised(1000),0]),(BOMB_ESCAPE_LENGTH_ONE_NEG,[Replay_Memory_Supervised(1000),0])
                            ,(BOMB_ESCAPE_LENGTH_TWO_POS,[Replay_Memory_Supervised(1000),0]),(BOMB_ESCAPE_LENGTH_TWO_NEG,[Replay_Memory_Supervised(1000),0]),(BOMB_ESCAPE_LENGTH_THREE_POS,[Replay_Memory_Supervised(1000),0])
                            ,(BOMB_ESCAPE_LENGTH_THREE_NEG,[Replay_Memory_Supervised(1000),0]),(BOMB_ESCAPE_LENGTH_FOUR_POS,[Replay_Memory_Supervised(500),0]),(BOMB_ESCAPE_LENGTH_FOUR_NEG,[Replay_Memory_Supervised(500),0])
                            ,(STEPPED_IN_BLOCKED_FIELD,[Replay_Memory_Supervised(1000),0]),(STEPPED_TOWARDS_TARGET_POS,[Replay_Memory_Supervised(4000),0])
                            ,(STEPPED_TOWARDS_TARGET_NEG,[Replay_Memory_Supervised(4000),0]),(STEPPED_TOWARDS_COIN_POS,[Replay_Memory_Supervised(3000),0]),(STEPPED_TOWARDS_COIN_NEG,[Replay_Memory_Supervised(3000),0])
                            ,(Target_IN_RANGE_NEG,[Replay_Memory_Supervised(1000),0]),(Target_IN_RANGE_POS,[Replay_Memory_Supervised(1000),0])
                            ,(CAN_ESCAPE_OWN_BOMB_NEG,[Replay_Memory_Supervised(1000),0]),(CAN_ESCAPE_OWN_BOMB_POS,[Replay_Memory_Supervised(1000),0])
                            ,(PLANTED_WITHOUT_BOMBS_LEFT,[Replay_Memory_Supervised(500),0]),(SHOULD_HAVE_PLANTED_BOMB,[Replay_Memory_Supervised(500),0])
                            ])
    cnt_steps=0
    cnt_epsiodes=0
    train_every_x_steps=20
    do_not_train_first_x_episodes=20
    replay_mem=ReplayMemory(700)
    avg_return=0
    avg_value_loss=0
    avg_explained_var=0
    avg_actor_loss=0
    avg_entropy=0
    cnt_trained_prior_to_print_stats=0
    print_stats_every=50
    cnt_stats=0
    entropy_regularization=0.0001
    entropy_decay=0.994
    batch_supervised_sizes=[10 for i in range(0,19)]
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

            with torch.no_grad():
                net_action,prob = policy_net.get_action(ten_transformed_state_grid,ten_transformed_state_rest)
                net_action=net_action.squeeze(0).item()
                prob=prob.squeeze(0).item()

            expert_action=action=expert.act(state)

            evaluate_net_action(transformed_state,state,net_action,expert_action,expert,events_replay_memory_list)

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
               minibatch=replay_mem.sample(40,policy_net,critic,30)
               minibatch_supervised=form_supervised_batch(events_replay_memory_list,batch_supervised_sizes)
               actor_loss,value_loss,explained_var,entropy=train(minibatch,minibatch_supervised,0.2,entropy_regularization,policy_net,optimizer_policy_net,critic,optimizer_critic,10000)
               cnt_steps=0
               cnt_trained_prior_to_print_stats+=1
               avg_value_loss+=value_loss
               avg_explained_var+=explained_var
               avg_actor_loss+=actor_loss
               avg_entropy+=entropy
               entropy_regularization*=entropy_decay
               
                
            state = new_state 

        replay_mem.add(states,actions,log_probs,rewards)
        avg_return+=cur_return
        if cnt_stats==print_stats_every:
            configurate_batch_sizes(events_replay_memory_list,batch_supervised_sizes)
            print(f"Episode: {i+1} AVG_Return: {avg_return/print_stats_every} AVG_value_loss: {avg_value_loss/cnt_trained_prior_to_print_stats} AVG_explained_var: {avg_explained_var/cnt_trained_prior_to_print_stats} AVG_actor_loss: {avg_actor_loss/cnt_trained_prior_to_print_stats} AVG_entropy: {avg_entropy/cnt_trained_prior_to_print_stats}")
            avg_value_loss=0
            avg_return=0
            avg_explained_var=0
            avg_actor_loss=0
            avg_entropy=0
            cnt_stats=0
            cnt_trained_prior_to_print_stats=0
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
  