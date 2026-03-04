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
from Nets import BaseActor
from State_Algorithms import Transform_State,find_rescue_route,compute_timestamp_dead_marker,compute_timestamp_dead_marker_with_extra_bomb,Compute_Bomb_Exlpoison_Radius,Compute_Bomb_Exlpoison_Radius_Local

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

event_liste=[

BOMB_ESCAPE_LENGTH_ONE_POS,
BOMB_ESCAPE_LENGTH_ONE_NEG,

BOMB_ESCAPE_LENGTH_TWO_POS,
BOMB_ESCAPE_LENGTH_TWO_NEG,

BOMB_ESCAPE_LENGTH_THREE_POS,
BOMB_ESCAPE_LENGTH_THREE_NEG,

BOMB_ESCAPE_LENGTH_FOUR_POS,
BOMB_ESCAPE_LENGTH_FOUR_NEG,

STEPPED_IN_BLOCKED_FIELD,

STEPPED_IN_EXPLOSION,

STEPPED_TOWARDS_TARGET_POS,
STEPPED_TOWARDS_TARGET_NEG,

Target_IN_RANGE_POS,
Target_IN_RANGE_NEG,

CAN_ESCAPE_OWN_BOMB_POS,


SHOULD_HAVE_PLANTED_BOMB,

PLANTED_WITHOUT_BOMBS_LEFT,

STEPPED_TOWARDS_COIN_POS,
STEPPED_TOWARDS_COIN_NEG,
]



class ReplayMemory(object):

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







device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)





def act_net(policy_net,state):
    out=policy_net(state)
    max_index=torch.argmax(out, dim=0).item()
    return max_index




def train(policy_net,optimizer,replay_memories,replay_memories_mini,batch_sizes,criterion,max_steps=1):
     
    for training_steps in range(0,max_steps):
       
        states_list_grid=[]
        states_list_rest=[]
        actions_list=[]
        cnt=0
        for replay_mem in replay_memories.items():
            #print(f"Kategorie:{replay_mem[0]} Batchgroesse:{batch_sizes[cnt]}")
            replay_mem_mini=replay_memories_mini[replay_mem[0]][0]
            replay_mem=replay_mem[1][0]
            if(len(replay_mem)>0):
                sampled_states,sampled_actions = replay_mem.sample(min(batch_sizes[cnt],len(replay_mem)))
                sampled_states_mini,sampled_actions_mini = replay_mem_mini.sample(min(batch_sizes[cnt],len(replay_mem_mini)))
                sampled_states.extend(sampled_states_mini)
                sampled_actions.extend(sampled_actions_mini)
                sampled_states=[Transform_State(state) for state in sampled_states]


                sampled_states_grid=[state[0] for state in sampled_states]
                sampled_states_grid=np.array(sampled_states_grid)
                sampled_states_rest=[state[1] for state in sampled_states]
                sampled_states_rest=np.array(sampled_states_rest)
           
                states_list_grid.append(torch.tensor(sampled_states_grid, dtype=torch.float32).to(device))
                states_list_rest.append(torch.tensor(sampled_states_rest, dtype=torch.float32).to(device))
           
                actions_list.append(torch.tensor(sampled_actions, dtype=torch.long).to(device))
            cnt+=1
                    
        policy_net.train()
        states_tensor_grid=torch.cat(states_list_grid)
        states_tensor_rest=torch.cat(states_list_rest)
        actions_tensor=torch.cat(actions_list)
        logits = policy_net(states_tensor_grid,states_tensor_rest)

        loss = criterion(logits, actions_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

            

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
                      




def evaluate_net_action(state,net_action,expert_action,expert,events_replay_memory_list,events_replay_memory_list_mini):
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
            events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][0].push(state,expert_action)
            events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][1]= events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][1]+1
            events_replay_memory_list_mini[STEPPED_IN_BLOCKED_FIELD][0].push(state,expert_action)
                
        elif timestamp_dead_marker[(1,x_old+1,y_old)]>=2 and net_action==1:
            events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][0].push(state,expert_action)
            events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][1]= events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][1]+1
            events_replay_memory_list_mini[STEPPED_IN_BLOCKED_FIELD][0].push(state,expert_action)
                
        elif timestamp_dead_marker[(1,x_old,y_old-1)]>=2 and net_action==2:
            events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][0].push(state,expert_action)
            events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][1]= events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][1]+1
            events_replay_memory_list_mini[STEPPED_IN_BLOCKED_FIELD][0].push(state,expert_action)
                
        elif timestamp_dead_marker[(1,x_old,y_old+1)]>=2 and net_action==0:
            events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][0].push(state,expert_action)
            events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][1]= events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][1]+1
            events_replay_memory_list_mini[STEPPED_IN_BLOCKED_FIELD][0].push(state,expert_action)


        if timestamp_dead_marker[(0,x_old,y_old)]==1 or timestamp_dead_marker[(0,x_old,y_old)]==4:

             #kategorisiere je nach Laenge des Wegs
            if memo[(x_old,y_old,0)]==1 and net_action==expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_ONE_POS][0].push(state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_ONE_POS][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_ONE_POS][1]+1
                 events_replay_memory_list_mini[BOMB_ESCAPE_LENGTH_ONE_POS][0].push(state,expert_action)


            elif memo[(x_old,y_old,0)]==1 and net_action!=expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_ONE_NEG][0].push(state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_ONE_NEG][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_ONE_NEG][1]+1
                 events_replay_memory_list_mini[BOMB_ESCAPE_LENGTH_ONE_NEG][0].push(state,expert_action)
                

            elif memo[(x_old,y_old,0)]==2 and net_action==expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_TWO_POS][0].push(state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_TWO_POS][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_TWO_POS][1]+1
                 events_replay_memory_list_mini[BOMB_ESCAPE_LENGTH_TWO_POS][0].push(state,expert_action)


            elif memo[(x_old,y_old,0)]==2 and net_action!=expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_TWO_NEG][0].push(state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_TWO_NEG][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_TWO_NEG][1]+1
                 events_replay_memory_list_mini[BOMB_ESCAPE_LENGTH_TWO_NEG][0].push(state,expert_action)


            elif memo[(x_old,y_old,0)]==3 and net_action==expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_THREE_POS][0].push(state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_THREE_POS][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_THREE_POS][1]+1
                 events_replay_memory_list_mini[BOMB_ESCAPE_LENGTH_THREE_POS][0].push(state,expert_action)


            elif memo[(x_old,y_old,0)]==3 and net_action!=expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_THREE_NEG][0].push(state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_THREE_NEG][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_THREE_NEG][1]+1
                 events_replay_memory_list_mini[BOMB_ESCAPE_LENGTH_THREE_NEG][0].push(state,expert_action)


            elif memo[(x_old,y_old,0)]==4 and net_action==expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_FOUR_POS][0].push(state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_FOUR_POS][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_FOUR_POS][1]+1
                 events_replay_memory_list_mini[BOMB_ESCAPE_LENGTH_FOUR_POS][0].push(state,expert_action)


            elif memo[(x_old,y_old,0)]==4 and net_action!=expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_FOUR_NEG][0].push(state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_FOUR_NEG][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_FOUR_NEG][1]+1
                 events_replay_memory_list_mini[BOMB_ESCAPE_LENGTH_FOUR_NEG][0].push(state,expert_action)


        else:
       #target_x,target_y,distanz,coin?
            if expert.target_coin==True:

                if net_action==expert_action and expert.target_distanz<=15:
                     events_replay_memory_list[STEPPED_TOWARDS_COIN_POS][0].push(state,expert_action)
                     events_replay_memory_list[STEPPED_TOWARDS_COIN_POS][1]= events_replay_memory_list[STEPPED_TOWARDS_COIN_POS][1]+1
                     events_replay_memory_list_mini[STEPPED_TOWARDS_COIN_POS][0].push(state,expert_action)
                elif net_action!=expert_action and expert.target_distanz<=15:
                     events_replay_memory_list[STEPPED_TOWARDS_COIN_NEG][0].push(state,expert_action)
                     events_replay_memory_list[STEPPED_TOWARDS_COIN_NEG][1]= events_replay_memory_list[STEPPED_TOWARDS_COIN_NEG][1]+1
                     events_replay_memory_list_mini[STEPPED_TOWARDS_COIN_NEG][0].push(state,expert_action)


            else:
                if expert_action==5:
                    if net_action==5:
                        events_replay_memory_list[Target_IN_RANGE_POS][0].push(state,expert_action)
                        events_replay_memory_list[Target_IN_RANGE_POS][1]= events_replay_memory_list[Target_IN_RANGE_POS][1]+1
                        events_replay_memory_list_mini[Target_IN_RANGE_POS][0].push(state,expert_action)
                        events_replay_memory_list[CAN_ESCAPE_OWN_BOMB_POS][0].push(state,expert_action)
                        events_replay_memory_list[CAN_ESCAPE_OWN_BOMB_POS][1]= events_replay_memory_list[CAN_ESCAPE_OWN_BOMB_POS][1]+1
                        events_replay_memory_list_mini[CAN_ESCAPE_OWN_BOMB_POS][0].push(state,expert_action)
                    else:
                         events_replay_memory_list[SHOULD_HAVE_PLANTED_BOMB][0].push(state,expert_action)
                         events_replay_memory_list[SHOULD_HAVE_PLANTED_BOMB][1]= events_replay_memory_list[SHOULD_HAVE_PLANTED_BOMB][1]+1
                         events_replay_memory_list_mini[SHOULD_HAVE_PLANTED_BOMB][0].push(state,expert_action)

                else:
                     if  net_action==expert_action and expert.target_distanz<=15:
                         events_replay_memory_list[STEPPED_TOWARDS_TARGET_POS][0].push(state,expert_action)
                         events_replay_memory_list[STEPPED_TOWARDS_TARGET_POS][1]= events_replay_memory_list[STEPPED_TOWARDS_TARGET_POS][1]+1
                         events_replay_memory_list_mini[STEPPED_TOWARDS_TARGET_POS][0].push(state,expert_action)
                     elif net_action!=expert_action and expert.target_distanz<=15:
                         events_replay_memory_list[STEPPED_TOWARDS_TARGET_NEG][0].push(state,expert_action)
                         events_replay_memory_list[STEPPED_TOWARDS_TARGET_NEG][1]= events_replay_memory_list[STEPPED_TOWARDS_TARGET_NEG][1]+1
                         events_replay_memory_list_mini[STEPPED_TOWARDS_TARGET_NEG][0].push(state,expert_action)

                     if net_action==5:
                          timestamp_dead_marker=compute_timestamp_dead_marker_with_extra_bomb(state["opponents_pos"],bombs_old,explosions_old,crates_old,walls_old,x_old,y_old)
                          memo={}
                          limit=5
                          find_rescue_route(timestamp_dead_marker,x_old,y_old,0,memo,limit)
                          
                          if memo[(x_old,y_old,0)]>4:
                             events_replay_memory_list[CAN_ESCAPE_OWN_BOMB_NEG][0].push(state,expert_action)
                             events_replay_memory_list[CAN_ESCAPE_OWN_BOMB_NEG][1]= events_replay_memory_list[CAN_ESCAPE_OWN_BOMB_NEG][1]+1
                             events_replay_memory_list_mini[CAN_ESCAPE_OWN_BOMB_NEG][0].push(state,expert_action)

                          Bomb_Exlpoison_Radius_Local=Compute_Bomb_Exlpoison_Radius_Local(walls_old,x_old,y_old)
                        
                          if (expert.target_x,expert.target_y) not in Bomb_Exlpoison_Radius_Local:
                             events_replay_memory_list[Target_IN_RANGE_NEG][0].push(state,expert_action)
                             events_replay_memory_list[Target_IN_RANGE_NEG][1]= events_replay_memory_list[Target_IN_RANGE_NEG][1]+1
                             events_replay_memory_list_mini[Target_IN_RANGE_NEG][0].push(state,expert_action)

                          if bombs_left==0:
                               events_replay_memory_list[PLANTED_WITHOUT_BOMBS_LEFT][0].push(state,expert_action)
                               events_replay_memory_list[PLANTED_WITHOUT_BOMBS_LEFT][1]= events_replay_memory_list[PLANTED_WITHOUT_BOMBS_LEFT][1]+1
                               events_replay_memory_list_mini[PLANTED_WITHOUT_BOMBS_LEFT][0].push(state,expert_action)


                        
                    
             



      
       
       
               

def loop(env,policy_net,optimizer,criterion, n_episodes=2000):
   
    expert=Expert_Agent()
    #supervised_events=[BOMB_ESCAPE_LENGTH_ONE_POS,BOMB_ESCAPE_LENGTH_TWO_POS,BOMB_ESCAPE_LENGTH_THREE_POS,BOMB_ESCAPE_LENGTH_FOUR_POS,STEPPED_TOWARDS_TARGET_POS,CAN_ESCAPE_OWN_BOMB_POS]
   # replay_memories=[ReplayMemory(3000),ReplayMemory(3000),ReplayMemory(3000),ReplayMemory(3000),ReplayMemory(3000),ReplayMemory(3000)]
   # event_liste=[BOMB_ESCAPE_LENGTH_ONE_POS,BOMB_ESCAPE_LENGTH_ONE_NEG,BOMB_ESCAPE_LENGTH_TWO_POS,BOMB_ESCAPE_LENGTH_TWO_NEG,BOMB_ESCAPE_LENGTH_THREE_POS,
#BOMB_ESCAPE_LENGTH_THREE_NEG,BOMB_ESCAPE_LENGTH_FOUR_POS,BOMB_ESCAPE_LENGTH_FOUR_NEG,STEPPED_IN_BLOCKED_FIELD,STEPPED_IN_EXPLOSION,STEPPED_TOWARDS_TARGET_POS,
#STEPPED_TOWARDS_TARGET_NEG,Target_IN_RANGE_NEG,Target_IN_RANGE_POS,CAN_ESCAPE_OWN_BOMB_POS,CAN_ESCAPE_OWN_BOMB_NEG,SHOULD_HAVE_PLANTED_BOMB,
#PLANTED_WITHOUT_BOMBS_LEFT]
    events_replay_memory_list=dict([(BOMB_ESCAPE_LENGTH_ONE_POS,[ReplayMemory(1000),0]),(BOMB_ESCAPE_LENGTH_ONE_NEG,[ReplayMemory(1000),0])
                            ,(BOMB_ESCAPE_LENGTH_TWO_POS,[ReplayMemory(1000),0]),(BOMB_ESCAPE_LENGTH_TWO_NEG,[ReplayMemory(1000),0]),(BOMB_ESCAPE_LENGTH_THREE_POS,[ReplayMemory(1000),0])
                            ,(BOMB_ESCAPE_LENGTH_THREE_NEG,[ReplayMemory(1000),0]),(BOMB_ESCAPE_LENGTH_FOUR_POS,[ReplayMemory(500),0]),(BOMB_ESCAPE_LENGTH_FOUR_NEG,[ReplayMemory(500),0])
                            ,(STEPPED_IN_BLOCKED_FIELD,[ReplayMemory(1000),0]),(STEPPED_TOWARDS_TARGET_POS,[ReplayMemory(4000),0])
                            ,(STEPPED_TOWARDS_TARGET_NEG,[ReplayMemory(4000),0]),(STEPPED_TOWARDS_COIN_POS,[ReplayMemory(3000),0]),(STEPPED_TOWARDS_COIN_NEG,[ReplayMemory(3000),0])
                            ,(Target_IN_RANGE_NEG,[ReplayMemory(1000),0]),(Target_IN_RANGE_POS,[ReplayMemory(1000),0])
                            ,(CAN_ESCAPE_OWN_BOMB_NEG,[ReplayMemory(1000),0]),(CAN_ESCAPE_OWN_BOMB_POS,[ReplayMemory(1000),0])
                            ,(PLANTED_WITHOUT_BOMBS_LEFT,[ReplayMemory(500),0]),(SHOULD_HAVE_PLANTED_BOMB,[ReplayMemory(500),0])
                            ])
    events_replay_memory_list_mini_push=dict([(BOMB_ESCAPE_LENGTH_ONE_POS,[ReplayMemory(1000),0]),(BOMB_ESCAPE_LENGTH_ONE_NEG,[ReplayMemory(1000),0])
                            ,(BOMB_ESCAPE_LENGTH_TWO_POS,[ReplayMemory(1000),0]),(BOMB_ESCAPE_LENGTH_TWO_NEG,[ReplayMemory(1000),0]),(BOMB_ESCAPE_LENGTH_THREE_POS,[ReplayMemory(1000),0])
                            ,(BOMB_ESCAPE_LENGTH_THREE_NEG,[ReplayMemory(1000),0]),(BOMB_ESCAPE_LENGTH_FOUR_POS,[ReplayMemory(500),0]),(BOMB_ESCAPE_LENGTH_FOUR_NEG,[ReplayMemory(500),0])
                            ,(STEPPED_IN_BLOCKED_FIELD,[ReplayMemory(1000),0]),(STEPPED_TOWARDS_TARGET_POS,[ReplayMemory(2000),0])
                            ,(STEPPED_TOWARDS_TARGET_NEG,[ReplayMemory(2000),0]),(STEPPED_TOWARDS_COIN_POS,[ReplayMemory(2000),0]),(STEPPED_TOWARDS_COIN_NEG,[ReplayMemory(2000),0])
                            ,(Target_IN_RANGE_NEG,[ReplayMemory(1000),0]),(Target_IN_RANGE_POS,[ReplayMemory(1000),0])
                            ,(CAN_ESCAPE_OWN_BOMB_NEG,[ReplayMemory(2000),0]),(CAN_ESCAPE_OWN_BOMB_POS,[ReplayMemory(2000),0])
                            ,(PLANTED_WITHOUT_BOMBS_LEFT,[ReplayMemory(500),0]),(SHOULD_HAVE_PLANTED_BOMB,[ReplayMemory(500),0])
                            ])

    events_replay_memory_list_mini_sample=dict([(BOMB_ESCAPE_LENGTH_ONE_POS,[ReplayMemory(1000),0]),(BOMB_ESCAPE_LENGTH_ONE_NEG,[ReplayMemory(1000),0])
                            ,(BOMB_ESCAPE_LENGTH_TWO_POS,[ReplayMemory(1000),0]),(BOMB_ESCAPE_LENGTH_TWO_NEG,[ReplayMemory(1000),0]),(BOMB_ESCAPE_LENGTH_THREE_POS,[ReplayMemory(1000),0])
                            ,(BOMB_ESCAPE_LENGTH_THREE_NEG,[ReplayMemory(1000),0]),(BOMB_ESCAPE_LENGTH_FOUR_POS,[ReplayMemory(500),0]),(BOMB_ESCAPE_LENGTH_FOUR_NEG,[ReplayMemory(500),0])
                            ,(STEPPED_IN_BLOCKED_FIELD,[ReplayMemory(1000),0]),(STEPPED_TOWARDS_TARGET_POS,[ReplayMemory(2000),0])
                            ,(STEPPED_TOWARDS_TARGET_NEG,[ReplayMemory(2000),0]),(STEPPED_TOWARDS_COIN_POS,[ReplayMemory(2000),0]),(STEPPED_TOWARDS_COIN_NEG,[ReplayMemory(2000),0])
                            ,(Target_IN_RANGE_NEG,[ReplayMemory(1000),0]),(Target_IN_RANGE_POS,[ReplayMemory(1000),0])
                            ,(CAN_ESCAPE_OWN_BOMB_NEG,[ReplayMemory(2000),0]),(CAN_ESCAPE_OWN_BOMB_POS,[ReplayMemory(2000),0])
                            ,(PLANTED_WITHOUT_BOMBS_LEFT,[ReplayMemory(500),0]),(SHOULD_HAVE_PLANTED_BOMB,[ReplayMemory(500),0])
                            ])
    batch_sizes=[20 for i in range(0,19)]
    cnt_steps=0
    epsiodes_to_reconfigurate_batch=40
    eps_decay=n_episodes/epsiodes_to_reconfigurate_batch
    eps_start=0.5
    eps_end=0
    steps=0
    epsilon=eps_end+eps_start*((eps_decay-steps)/(eps_decay))
    cnt_epsiodes=0
    first_episodes=0
    train_every_x_steps=20
   

    for i in range(n_episodes):
        state, info = env.reset()
        terminated, truncated, quit = False, False, False
        
        cnt_epsiodes+=1
        first_episodes+=1
        while not (terminated or truncated):
            cnt_steps+=1
            transformed_state=Transform_State(state)
            ten_transformed_state_grid=torch.tensor(transformed_state[0],dtype=torch.float32).to(device)
            ten_transformed_state_rest=torch.tensor(transformed_state[1],dtype=torch.float32).to(device)

            ten_transformed_state_grid=ten_transformed_state_grid.unsqueeze(0)
            ten_transformed_state_rest=ten_transformed_state_rest.unsqueeze(0)

            net_action = policy_net.get_best_action(ten_transformed_state_grid,ten_transformed_state_rest)
            net_action=net_action.squeeze(0).item()
            expert_action=action=expert.act(state)

            if random.random()>epsilon:
                action=net_action
           
            new_state, _, terminated, truncated, info = env.step(action)

            evaluate_net_action(state,action,expert_action,expert,events_replay_memory_list,events_replay_memory_list_mini_push)

            if cnt_epsiodes==epsiodes_to_reconfigurate_batch:
                steps+=1
                epsilon=eps_end+eps_start*((eps_decay-steps)/(eps_decay))
                print(f"Epsilon: {epsilon}")
                configurate_batch_sizes(events_replay_memory_list,batch_sizes)
                cnt_epsiodes=0
                events_replay_memory_list_mini_sample=events_replay_memory_list_mini_push
                events_replay_memory_list_mini_push=dict([(BOMB_ESCAPE_LENGTH_ONE_POS,[ReplayMemory(1000),0]),(BOMB_ESCAPE_LENGTH_ONE_NEG,[ReplayMemory(1000),0])
                            ,(BOMB_ESCAPE_LENGTH_TWO_POS,[ReplayMemory(1000),0]),(BOMB_ESCAPE_LENGTH_TWO_NEG,[ReplayMemory(1000),0]),(BOMB_ESCAPE_LENGTH_THREE_POS,[ReplayMemory(1000),0])
                            ,(BOMB_ESCAPE_LENGTH_THREE_NEG,[ReplayMemory(1000),0]),(BOMB_ESCAPE_LENGTH_FOUR_POS,[ReplayMemory(500),0]),(BOMB_ESCAPE_LENGTH_FOUR_NEG,[ReplayMemory(500),0])
                            ,(STEPPED_IN_BLOCKED_FIELD,[ReplayMemory(1000),0]),(STEPPED_TOWARDS_TARGET_POS,[ReplayMemory(2000),0])
                            ,(STEPPED_TOWARDS_TARGET_NEG,[ReplayMemory(2000),0]),(STEPPED_TOWARDS_COIN_POS,[ReplayMemory(2000),0]),(STEPPED_TOWARDS_COIN_NEG,[ReplayMemory(2000),0])
                            ,(Target_IN_RANGE_NEG,[ReplayMemory(1000),0]),(Target_IN_RANGE_POS,[ReplayMemory(1000),0])
                            ,(CAN_ESCAPE_OWN_BOMB_NEG,[ReplayMemory(2000),0]),(CAN_ESCAPE_OWN_BOMB_POS,[ReplayMemory(2000),0])
                            ,(PLANTED_WITHOUT_BOMBS_LEFT,[ReplayMemory(500),0]),(SHOULD_HAVE_PLANTED_BOMB,[ReplayMemory(500),0])
                            ])

            if first_episodes>=epsiodes_to_reconfigurate_batch and train_every_x_steps<=cnt_steps:
               train(policy_net,optimizer,events_replay_memory_list,events_replay_memory_list_mini_sample,batch_sizes,criterion)
               cnt_steps=0
               

            state = new_state 

        #print(cnt_epsiodes)
    torch.save(policy_net.state_dict(), Path(__file__).parent / "model_supervised.pt")    
                
  
def train_supervised(policy_net,optimizer,replay_memories,criterion,max_steps=1000):
    for i in range(0,6):
        with open("zustandaktionspaare_"+str(i)+".pkl", "rb") as f:
           # cur_memory=ReplayMemory(3000)
            #state_list,action_list=pickle.load(f)
            #cur_memory.memory_state=state_list
            #cur_memory.memory_action=action_list
            #replay_memories.append(cur_memory)
            replay_memories.append(pickle.load(f))
    supervised_events=[BOMB_ESCAPE_LENGTH_ONE_POS,BOMB_ESCAPE_LENGTH_TWO_POS,BOMB_ESCAPE_LENGTH_THREE_POS,BOMB_ESCAPE_LENGTH_FOUR_POS,STEPPED_TOWARDS_TARGET_POS,
                   CAN_ESCAPE_OWN_BOMB_POS]
    for training_steps in range(0,max_steps):
       
        states_list_grid=[]
        states_list_rest=[]
        actions_list=[]
        cnt=0
        for replay_mem in replay_memories:
            sampled_states,sampled_actions = replay_mem.sample()
            sampled_states=[Transform_State(state) for state in sampled_states]
            sampled_states_grid=[state[0] for state in sampled_states]
            sampled_states_rest=[state[1] for state in sampled_states]
           
            states_list_grid.append(torch.tensor(sampled_states_grid, dtype=torch.float32).to(device))
            states_list_rest.append(torch.tensor(sampled_states_rest, dtype=torch.float32).to(device))
            actions_list.append(torch.tensor(sampled_actions, dtype=torch.long).to(device))

           
            policy_net.eval()
            with torch.no_grad():
                preds = torch.argmax(policy_net(states_list_grid[cnt],states_list_rest[cnt]), dim=1)
                acc = (preds == actions_list[cnt]).float().mean()

            print(f"Kategorie: {supervised_events[cnt]}:  Acc={acc:.4f}")

            cnt+=1

                    
        policy_net.train()
        states_tensor_grid=torch.cat(states_list_grid)
        states_tensor_rest=torch.cat(states_list_rest)
        actions_tensor=torch.cat(actions_list)
        logits = policy_net(states_tensor_grid,states_tensor_rest)

        loss = criterion(logits, actions_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        policy_net.eval()
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == actions_tensor).float().mean()

        print(f"Epoch {training_steps}: Loss={loss.item():.4f}, Acc={acc:.4f}")

    torch.save(policy_net.state_dict(), Path(__file__).parent / "model_supervised_training_by_file.pt")

def main(argv=None):
   
    args = parse(argv)
    replay_memories=[]
    policy_net = BaseActor(4096, 461,6).to(device)
    optimizer = optim.AdamW(policy_net.parameters(), lr=1e-3, amsgrad=True)
    criterion = nn.CrossEntropyLoss()
    if args.init_supervised_with_file=='True':
        train_supervised(policy_net,optimizer,replay_memories,criterion)
    else:
        policy_net.load_state_dict(torch.load( Path(__file__).parent / "model_supervised_training_by_file.pt"))
        
    env = gymnasium.make("bomberman_rl/bomberman-v0", args=args)
  
    env = ScoreRewardWrapper(env)
    optimizer = optim.AdamW(policy_net.parameters(), lr=1e-3)
    loop(env,policy_net,optimizer,criterion)





if __name__ == "__main__":
    main()
  