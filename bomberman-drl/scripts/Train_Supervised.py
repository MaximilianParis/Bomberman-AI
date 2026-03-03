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



def find_rescue_route(timestamp_dead_marker,i,j,iteration=0,memo={},limit=10):
        if(((i,j,iteration) in memo)):
            return 
        memo[(i,j,iteration)]=iteration
      
        if(iteration>0 and timestamp_dead_marker[iteration,i,j]>=2):
            memo[(i,j,iteration)]=limit+1
           
        elif(timestamp_dead_marker[iteration,i,j]==1 or (iteration==0 and timestamp_dead_marker[iteration,i,j]==4)):
            memo[(i,j,iteration)]=limit+1
            if(iteration<limit):

                find_rescue_route(timestamp_dead_marker,i-1,j,iteration+1,memo,limit)
                memo[(i,j,iteration)]=min(memo[(i,j,iteration)],memo[(i-1,j,iteration+1)])
            

                find_rescue_route(timestamp_dead_marker,i+1,j,iteration+1,memo,limit)
                memo[(i,j,iteration)]=min(memo[(i,j,iteration)],memo[(i+1,j,iteration+1)])
           

                find_rescue_route(timestamp_dead_marker,i,j-1,iteration+1,memo,limit)
                memo[(i,j,iteration)]=min(memo[(i,j,iteration)],memo[(i,j-1,iteration+1)])
           

                find_rescue_route(timestamp_dead_marker,i,j+1,iteration+1,memo,limit)
                memo[(i,j,iteration)]=min(memo[(i,j,iteration)],memo[(i,j+1,iteration+1)])
           

                find_rescue_route(timestamp_dead_marker,i,j,iteration+1,memo,limit)
                memo[(i,j,iteration)]=min(memo[(i,j,iteration)],memo[(i,j,iteration+1)])
           

                if iteration==0 and memo[(i,j,iteration)]<limit+1 and memo[(i-1,j,iteration+1)]==memo[(i,j,iteration)]:
                    return 3

                if iteration==0 and memo[(i,j,iteration)]<limit+1 and memo[(i+1,j,iteration+1)]==memo[(i,j,iteration)]:
                    return 1


                if iteration==0 and memo[(i,j,iteration)]<limit+1 and memo[(i,j-1,iteration+1)]==memo[(i,j,iteration)]:
                    return 2

                if iteration==0 and memo[(i,j,iteration)]<limit+1 and memo[(i,j+1,iteration+1)]==memo[(i,j,iteration)]:
                    return 0
        else:
             return 4

        return 4

def Compute_Bomb_Exlpoison_Radius(bomb_exploison_radius,bombs,walls):
    bomb_xys = [(i, j) for i in range(0,17) for j in range(0,17) if bombs[i,j]>=1]
    for (i,j) in bomb_xys:
            bomb_exploison_radius[i,j]=bombs[i,j]
            for k in range(1,4):
                if(i-k>0 and walls[i-k,j]==0):
                  bomb_exploison_radius[i-k,j]=bombs[i,j]
                #Wand blockt Exlposion breche ab
                elif(i-k>0 and walls[i-k,j]==1):
                    break
            #Streifen oben
            for k in range(1,4): 
                if(j-k>0 and walls[i,j-k]==0):
                  bomb_exploison_radius[i,j-k]=bombs[i,j]
                #Wand blockt Exlposion breche ab
                elif(j-k>0 and walls[i,j-k]==1):
                    break

            #Streifen rechts
            for k in range(1,4):
                if(i+k<16 and walls[i+k,j]==0):
                    bomb_exploison_radius[i+k,j]=bombs[i,j]
                #Wand blockt Exlposion breche ab
                elif(i+k<16 and walls[i+k,j]==1):
                    break
            #Streifen unten
            for k in range(1,4): 
                if(j+k<16 and walls[i,j+k]==0):
                     bomb_exploison_radius[i,j+k]=bombs[i,j]
                #Wand blockt Exlposion breche ab
                elif(j+k<16 and walls[i,j+k]==1):
                    break




def Compute_Bomb_Exlpoison_Radius_Local(walls,i,j):
    bomb_blast = [(i, j)]
  
            
    for k in range(1,4):
        if(i-k>0 and walls[i-k,j]==0):
            bomb_blast.append((i-k,j))
        #Wand blockt Exlposion breche ab
        elif(i-k>0 and walls[i-k,j]==1):
            break
    #Streifen oben
    for k in range(1,4): 
        if(j-k>0 and walls[i,j-k]==0):
             bomb_blast.append((i,j-k))
        #Wand blockt Exlposion breche ab
        elif(j-k>0 and walls[i,j-k]==1):
            break

    #Streifen rechts
    for k in range(1,4):
        if(i+k<16 and walls[i+k,j]==0):
             bomb_blast.append((i+k,j))
        #Wand blockt Exlposion breche ab
        elif(i+k<16 and walls[i+k,j]==1):
            break
    #Streifen unten
    for k in range(1,4): 
        if(j+k<16 and walls[i,j+k]==0):
             bomb_blast.append((i,j+k))
        #Wand blockt Exlposion breche ab
        elif(j+k<16 and walls[i,j+k]==1):
            break

    return bomb_blast

def compute_timestamp_dead_marker_with_extra_bomb(opponents,bombs,explosions,crates,walls,cur_bomb_x,cur_bomb_y,time_limit_escape=6):
    
    limit_x=16
    limit_y=16
    #0 frei, 1 in Bombenradis, 2 Crate, 3 in Bombenradis und Crate, 4 Wand oder Explosion
    timestamp_dead_marker=np.zeros((time_limit_escape,limit_x+1,limit_y+1))

    #for i in range(0,17):
       #for j in range(0,17):
           #if explosions[i,j]==12:
           #    explosions[i,j]=2
               
           #elif explosions[i,j]==11:
           #    explosions[i,j]=1
               
           #else:
               #explosions[i,j]=0

    explosion_xy=[(i,j,explosions[i,j]-11) for i in range(0,limit_x+1) for j in range(0,limit_y+1) if explosions[i,j]>=12]
    bomb_xys = [(i, j,(4-bombs[i,j])) for i in range(0,17) for j in range(0,17) if bombs[i,j]>=1]
    bomb_xys.append((cur_bomb_x,cur_bomb_y,4))
      
    for (cur_x,cur_y,explosion_timer) in explosion_xy:
                      
        for t in range(0,explosion_timer):
            timestamp_dead_marker[t,cur_x,cur_y]=4
    #print(bombs)
    #print(bomb_xys)
    for (i, j,bomb_timer) in bomb_xys:
        bomb_blast=Compute_Bomb_Exlpoison_Radius_Local(walls,i,j)
        for (cur_x,cur_y) in bomb_blast:


            for t in range(bomb_timer,bomb_timer+2):
                timestamp_dead_marker[t,cur_x,cur_y]=4

            if cur_x==i and cur_y==j:
                 for t in range(0,bomb_timer):
                    timestamp_dead_marker[t,cur_x,cur_y]=4
            else:
                for t in range(0,bomb_timer):
                    timestamp_dead_marker[t,cur_x,cur_y]=max(1,timestamp_dead_marker[t,cur_x,cur_y])


    for i in range(0,limit_x+1):
        for j in range(0,limit_y+1):
            if walls[i,j]==1 or opponents[i,j]==1:
                    for t in range(0,time_limit_escape):
                        timestamp_dead_marker[t,i,j]=4
            elif crates[i,j]==1:
                for t in range(0,time_limit_escape):
                    if timestamp_dead_marker[t,i,j]==4:
                        break
                    elif timestamp_dead_marker[t,i,j]==1:
                        timestamp_dead_marker[t,i,j]=3
                    else:
                        timestamp_dead_marker[t,i,j]=2

    return timestamp_dead_marker


def compute_timestamp_dead_marker(opponents,bombs,explosions,crates,walls,time_limit_escape=6):
    
    limit_x=16
    limit_y=16
    #0 frei, 1 in Bombenradis, 2 Crate, 3 in Bombenradis und Crate, 4 Wand oder Explosion
    timestamp_dead_marker=np.zeros((time_limit_escape,limit_x+1,limit_y+1))
     

    explosion_xy=[(i,j,explosions[i,j]-10) for i in range(0,limit_x+1) for j in range(0,limit_y+1) if explosions[i,j]>=11]
    bomb_xys = [(i, j,(4-bombs[i,j])+1) for i in range(0,17) for j in range(0,17) if bombs[i,j]>=1]
      
      
    for (cur_x,cur_y,explosion_timer) in explosion_xy:
                      
        for t in range(0,explosion_timer):
            timestamp_dead_marker[t,cur_x,cur_y]=4
    #print(bombs)
    #print(bomb_xys)
    for (i, j,bomb_timer) in bomb_xys:
        bomb_blast=Compute_Bomb_Exlpoison_Radius_Local(walls,i,j)
        for (cur_x,cur_y) in bomb_blast:


            for t in range(bomb_timer,bomb_timer+2):
                timestamp_dead_marker[t,cur_x,cur_y]=4

            if cur_x==i and cur_y==j:
                 for t in range(0,bomb_timer):
                    timestamp_dead_marker[t,cur_x,cur_y]=4
            else:
                for t in range(0,bomb_timer):
                    timestamp_dead_marker[t,cur_x,cur_y]=max(1,timestamp_dead_marker[t,cur_x,cur_y])


    for i in range(0,limit_x+1):
        for j in range(0,limit_y+1):
          if walls[i,j]==1 or opponents[i,j]==1:
                    for t in range(0,time_limit_escape):
                        timestamp_dead_marker[t,i,j]=4
          elif crates[i,j]==1:
                for t in range(0,time_limit_escape):
                    if timestamp_dead_marker[t,i,j]==4:
                        break
                    elif timestamp_dead_marker[t,i,j]==1:
                        timestamp_dead_marker[t,i,j]=3
                    else:
                        timestamp_dead_marker[t,i,j]=2

    return timestamp_dead_marker

#frei alles 0
#wall 1
#Muenze 2
#Spieler 3
#Bombe 4
def Transform_State(state):

    opponents=state["opponents_pos"]
    bombs=state["bombs"]
    explosions=state["explosions"]
    crates=state["crates"]
    walls=state["walls"]
    position=state["self_info"]["position"]
    coins=state["coins"]

    state_representation_list=[]
    (x,y)=(0,0)
    limit_x=16
    limit_y=16
    for i in range(1,limit_x):
         for j in range(1,limit_y):
                if(position[i,j]==1):
                        (x,y)=(i,j)
 
    window_length=15
    for i in range(x - window_length, x + window_length+1):
           for j in range(y - window_length, y + window_length+1):
               distanz = abs(i - x) + abs(j - y)  
               if distanz < window_length:
                   if i>=0 and i<limit_x and j>=0 and j<limit_y:
                       if(walls[i,j]==1):
                           state_representation_list.append(1)
              
                       else:
                             state_representation_list.append(0)
                              
                   else:
                        state_representation_list.append(1)

  
    for i in range(x - window_length, x + window_length+1):
           for j in range(y - window_length, y + window_length+1):
               distanz = abs(i - x) + abs(j - y)  
               if distanz < window_length:
                   if i>=0 and i<limit_x and j>=0 and j<limit_y:
                       if(coins[i,j]==1):
                           state_representation_list.append(1)
              
                       else:
                             state_representation_list.append(0)
                              
                   else:
                        state_representation_list.append(1)


    for i in range(x - window_length, x + window_length+1):
           for j in range(y - window_length, y + window_length+1):
               distanz = abs(i - x) + abs(j - y)  
               if distanz < window_length:
                   if i>=0 and i<limit_x and j>=0 and j<limit_y:
                       if(opponents[i,j]==1):
                           state_representation_list.append(1)
              
                       else:
                             state_representation_list.append(0)
                              
                   else:
                        state_representation_list.append(1)

    for i in range(x - window_length, x + window_length+1):
           for j in range(y - window_length, y + window_length+1):
               distanz = abs(i - x) + abs(j - y)  
               if distanz < window_length:
                   if i>=0 and i<limit_x and j>=0 and j<limit_y:
                       if(crates[i,j]==1):
                           state_representation_list.append(1)
              
                       else:
                             state_representation_list.append(0)
                              
                   else:
                        state_representation_list.append(1)

    

   
    timestamp_dead_marker=compute_timestamp_dead_marker(opponents,bombs,explosions,crates,walls)
    time_limit_escape=6
    window_length_bombs=6
   
    for t in range(1,time_limit_escape):
        for pos1 in range(x - window_length_bombs+1, x + window_length_bombs):
            for pos2 in range(y - window_length_bombs+1, y + window_length_bombs):
                distanz = abs(pos1 - x) + abs(pos2 - y)  
                if distanz <= t:
                    if pos1>=0 and pos1<limit_x and pos2>=0 and pos2<limit_y:
                       if timestamp_dead_marker[t,pos1,pos2]==1:
                           state_representation_list.append(1)
                       else:
                            state_representation_list.append(0)
                    else:
                        state_representation_list.append(0)

    for t in range(1,time_limit_escape):
        for pos1 in range(x - window_length_bombs+1, x + window_length_bombs):
            for pos2 in range(y - window_length_bombs+1, y + window_length_bombs):
                distanz = abs(pos1 - x) + abs(pos2 - y)  
                if distanz <= t:
                    if pos1>=0 and pos1<limit_x and pos2>=0 and pos2<limit_y:
                        if timestamp_dead_marker[t,pos1,pos2]>1:
                            state_representation_list.append(1)
                        else:
                            state_representation_list.append(0)
                    else:
                        state_representation_list.append(0)


                        
    if timestamp_dead_marker[0,x,y]==1 or timestamp_dead_marker[0,x,y]==4:
         state_representation_list.append(1)
    else:
         state_representation_list.append(0)


    timestamp_dead_marker=compute_timestamp_dead_marker_with_extra_bomb(opponents,bombs,explosions,crates,walls,x,y)
     
    time_limit_escape=5
    window_length_bombs=5

    for t in range(1,time_limit_escape):
        for pos1 in range(x - window_length_bombs+1, x + window_length_bombs):
            for pos2 in range(y - window_length_bombs+1, y + window_length_bombs):
                distanz = abs(pos1 - x) + abs(pos2 - y)  
                if distanz <= t:
                    if pos1>=0 and pos1<limit_x and pos2>=0 and pos2<limit_y:
                       if timestamp_dead_marker[t,pos1,pos2]==1:
                           state_representation_list.append(1)
                       else:
                            state_representation_list.append(0)
                    else:
                        state_representation_list.append(0)

    for t in range(1,time_limit_escape):
        for pos1 in range(x - window_length_bombs+1, x + window_length_bombs):
            for pos2 in range(y - window_length_bombs+1, y + window_length_bombs):
                distanz = abs(pos1 - x) + abs(pos2 - y)  
                if distanz <= t:
                    if pos1>=0 and pos1<limit_x and pos2>=0 and pos2<limit_y:
                        if timestamp_dead_marker[t,pos1,pos2]>1:
                            state_representation_list.append(1)
                        else:
                            state_representation_list.append(0)
                    else:
                        state_representation_list.append(0)


    if timestamp_dead_marker[0,x,y]==1 or timestamp_dead_marker[0,x,y]==4:
         state_representation_list.append(1)
    else:
         state_representation_list.append(0)
  
    state_representation_list.append(state["self_info"]["bombs_left"])




    return state_representation_list



device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class BaseActor(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.gate = nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(state_size, int(state_size/2)), self.gate,
            nn.Linear(int(state_size/2), int(state_size/4)), self.gate,
            nn.Linear(int(state_size/4), action_size),
          
        )

    def forward(self, state):
        return self.net(state)

    def forward_with_softmax(self, state):
        return torch.softmax(self.net(state), dim=-1)
    
    def get_action(self, prob):
        dist = distributions.Categorical(prob)
        action = dist.sample()
        return action
    
    def get_best_action(self, prob):
        _, indices = prob.max(dim=1)
        return indices

    def eval_action(self, prob, action):
        dist = distributions.Categorical(prob)
        return {
            "log_prob": dist.log_prob(action),
            "entropy": dist.entropy()
        }


def compute(states, actions,actor,critic):
        logit = {"p": actor(states), "v": critic(states)}
        cur_values = logit["v"]
        x = actor.eval_action(logit["p"], actions.squeeze())
        cur_log_probs = x["log_prob"]
        entropy = x["entropy"]
        return cur_values.squeeze(), cur_log_probs.squeeze(), entropy

def act_net(policy_net,state):
    out=policy_net(state)
    max_index=torch.argmax(out, dim=0).item()
    return max_index




def train(policy_net,optimizer,replay_memories,batch_sizes,criterion,max_steps=1):
     
    for training_steps in range(0,max_steps):
       
        states_list=[]
        actions_list=[]
        cnt=0
        for replay_mem in replay_memories.items():
            replay_mem=replay_mem[1][0]
            if(len(replay_mem)>0):
                sampled_states,sampled_actions = replay_mem.sample(min(batch_sizes[cnt],len(replay_mem)))
                sampled_states=[Transform_State(state) for state in sampled_states]
           
                states_list.append(torch.tensor(sampled_states, dtype=torch.float32).to(device))
                actions_list.append(torch.tensor(sampled_actions, dtype=torch.long).to(device))
            cnt+=1
                    
        policy_net.train()
        states_tensor=torch.cat(states_list)
        actions_tensor=torch.cat(actions_list)
        logits = policy_net(states_tensor)

        loss = criterion(logits, actions_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

            

def configurate_batch_sizes(replay_memories,batch_sizes):
       
    #je nach Wichtigkeit der Kategorie wird pro Kategorie Anzahl an Zustandaktionspaaren berechnet die in den batch kommen
    print("")
    factor=replay_memories[BOMB_ESCAPE_LENGTH_ONE_NEG][1]/(replay_memories[BOMB_ESCAPE_LENGTH_ONE_NEG][1]+replay_memories[BOMB_ESCAPE_LENGTH_ONE_POS][1]+0.0000001)
    batch_sizes[0]=int((factor*70)+10)
    batch_sizes[1]=int((factor*70)+10)
   
    print(f"BOMB_ESCAPE_LENGTH_ONE_NEG:{replay_memories[BOMB_ESCAPE_LENGTH_ONE_NEG][1]}")
    print(f"BOMB_ESCAPE_LENGTH_ONE_POS:{replay_memories[BOMB_ESCAPE_LENGTH_ONE_POS][1]}")
    print(f"factor:{factor}")
    print(f"batch_sizes:{batch_sizes[0]}")
    print("")

    factor=replay_memories[BOMB_ESCAPE_LENGTH_TWO_NEG][1]/(replay_memories[BOMB_ESCAPE_LENGTH_TWO_NEG][1]+replay_memories[BOMB_ESCAPE_LENGTH_TWO_POS][1]+0.0000001)
    batch_sizes[2]=int((factor*60)+10)
    batch_sizes[3]=int((factor*60)+10)
   
    print(f"BOMB_ESCAPE_LENGTH_TWO_NEG:{replay_memories[BOMB_ESCAPE_LENGTH_TWO_NEG][1]}")
    print(f"BOMB_ESCAPE_LENGTH_TWO_POS:{replay_memories[BOMB_ESCAPE_LENGTH_TWO_POS][1]}")
    print(f"factor:{factor}")
    print(f"batch_sizes:{batch_sizes[2]}")
    print("")

    factor=replay_memories[BOMB_ESCAPE_LENGTH_THREE_NEG][1]/(replay_memories[BOMB_ESCAPE_LENGTH_THREE_NEG][1]+replay_memories[BOMB_ESCAPE_LENGTH_THREE_POS][1]+0.0000001)
    batch_sizes[4]=int((factor*50)+10)
    batch_sizes[5]=int((factor*50)+10)
    
    print(f"BOMB_ESCAPE_LENGTH_THREE_NEG:{replay_memories[BOMB_ESCAPE_LENGTH_THREE_NEG][1]}")
    print(f"BOMB_ESCAPE_LENGTH_THREE_POS:{replay_memories[BOMB_ESCAPE_LENGTH_THREE_POS][1]}")
    print(f"factor:{factor}")
    print(f"batch_sizes:{batch_sizes[4]}")
    print("")

    factor=replay_memories[BOMB_ESCAPE_LENGTH_FOUR_NEG][1]/(replay_memories[BOMB_ESCAPE_LENGTH_FOUR_NEG][1]+replay_memories[BOMB_ESCAPE_LENGTH_FOUR_POS][1]+0.0000001)
    batch_sizes[6]=int((factor*20)+10)
    batch_sizes[7]=int((factor*20)+10)
    
    print(f"BOMB_ESCAPE_LENGTH_FOUR_NEG:{replay_memories[BOMB_ESCAPE_LENGTH_FOUR_NEG][1]}")
    print(f"BOMB_ESCAPE_LENGTH_FOUR_POS:{replay_memories[BOMB_ESCAPE_LENGTH_FOUR_POS][1]}")
    print(f"factor:{factor}")
    print(f"batch_sizes:{batch_sizes[6]}")
    print("")
       
    batch_sizes[8]=int(replay_memories[STEPPED_IN_BLOCKED_FIELD][1]*0.5+10)
    
    print(f"STEPPED_IN_BLOCKED_FIELD:{replay_memories[STEPPED_IN_BLOCKED_FIELD][1]}")
    print(f"batch_sizes:{batch_sizes[8]}")
    print("")

    factor=replay_memories[STEPPED_TOWARDS_TARGET_NEG][1]/(replay_memories[STEPPED_TOWARDS_TARGET_NEG][1]+replay_memories[STEPPED_TOWARDS_TARGET_POS][1]+0.0000001)
    batch_sizes[9]=int((factor*120)+50)
    batch_sizes[10]=int((factor*120)+50)
   
    print(f"STEPPED_TOWARDS_TARGET_NEG:{replay_memories[STEPPED_TOWARDS_TARGET_NEG][1]}")
    print(f"STEPPED_TOWARDS_TARGET_POS:{replay_memories[STEPPED_TOWARDS_TARGET_POS][1]}")
    print(f"factor:{factor}")
    print(f"batch_sizes:{batch_sizes[9]}")
    print("")

    factor=replay_memories[STEPPED_TOWARDS_COIN_NEG][1]/(replay_memories[STEPPED_TOWARDS_COIN_NEG][1]+replay_memories[STEPPED_TOWARDS_COIN_POS][1]+0.0000001)
    batch_sizes[11]=int((factor*100)+40)
    batch_sizes[12]=int((factor*100)+40)
   
    print(f"STEPPED_TOWARDS_COIN_NEG:{replay_memories[STEPPED_TOWARDS_COIN_NEG][1]}")
    print(f"STEPPED_TOWARDS_COIN_POS:{replay_memories[STEPPED_TOWARDS_COIN_POS][1]}")
    print(f"factor:{factor}")
    print(f"batch_sizes:{batch_sizes[11]}")
    print("")

    factor=replay_memories[Target_IN_RANGE_NEG][1]/(replay_memories[Target_IN_RANGE_NEG][1]+replay_memories[Target_IN_RANGE_POS][1]+0.0000001)
    batch_sizes[13]=int((factor*80)+30)
    batch_sizes[14]=int((factor*80)+30)
   
    print(f"Target_IN_RANGE_NEG:{replay_memories[Target_IN_RANGE_NEG][1]}")
    print(f"Target_IN_RANGE_POS:{replay_memories[Target_IN_RANGE_POS][1]}")
    print(f"factor:{factor}")
    print(f"batch_sizes:{batch_sizes[13]}")
    print("")

    factor=replay_memories[CAN_ESCAPE_OWN_BOMB_NEG][1]/(replay_memories[CAN_ESCAPE_OWN_BOMB_NEG][1]+replay_memories[CAN_ESCAPE_OWN_BOMB_POS][1]+0.0000001)
    batch_sizes[15]=int((factor*80)+30)
    batch_sizes[16]=int((factor*80)+30)
   
    print(f"CAN_ESCAPE_OWN_BOMB_NEG:{replay_memories[CAN_ESCAPE_OWN_BOMB_NEG][1]}")
    print(f"CAN_ESCAPE_OWN_BOMB_POS:{replay_memories[CAN_ESCAPE_OWN_BOMB_POS][1]}")
    print(f"factor:{factor}")
    print(f"batch_sizes:{batch_sizes[15]}")
    print("")

    batch_sizes[17]=int(replay_memories[PLANTED_WITHOUT_BOMBS_LEFT][1]*0.5+10)
    
    print(f"PLANTED_WITHOUT_BOMBS_LEFT:{replay_memories[PLANTED_WITHOUT_BOMBS_LEFT][1]}")
    print(f"batch_sizes:{batch_sizes[17]}")
    print("")
   
    batch_sizes[18]=int(replay_memories[SHOULD_HAVE_PLANTED_BOMB][1]*0.5+20)
    

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
    replay_memories[Target_IN_RANGE_POS][1]=0
    replay_memories[CAN_ESCAPE_OWN_BOMB_POS][1]=0
    replay_memories[CAN_ESCAPE_OWN_BOMB_NEG][1]=0
    replay_memories[PLANTED_WITHOUT_BOMBS_LEFT][1]=0
    replay_memories[SHOULD_HAVE_PLANTED_BOMB][1]=0
    replay_memories[STEPPED_TOWARDS_COIN_POS][1]=0
    replay_memories[STEPPED_TOWARDS_COIN_NEG][1]=0
                      




def evaluate_net_action(state,net_action,expert_action,expert,events_replay_memory_list):
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
                
        elif timestamp_dead_marker[(1,x_old+1,y_old)]>=2 and net_action==1:
            events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][0].push(state,expert_action)
            events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][1]= events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][1]+1
                
        elif timestamp_dead_marker[(1,x_old,y_old-1)]>=2 and net_action==2:
            events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][0].push(state,expert_action)
            events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][1]= events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][1]+1
                
        elif timestamp_dead_marker[(1,x_old,y_old+1)]>=2 and net_action==0:
            events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][0].push(state,expert_action)
            events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][1]= events_replay_memory_list[STEPPED_IN_BLOCKED_FIELD][1]+1


        if timestamp_dead_marker[(0,x_old,y_old)]==1 or timestamp_dead_marker[(0,x_old,y_old)]==4:

             #kategorisiere je nach Laenge des Wegs
            if memo[(x_old,y_old,0)]==1 and net_action==expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_ONE_POS][0].push(state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_ONE_POS][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_ONE_POS][1]+1


            elif memo[(x_old,y_old,0)]==1 and net_action!=expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_ONE_NEG][0].push(state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_ONE_NEG][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_ONE_NEG][1]+1
                

            elif memo[(x_old,y_old,0)]==2 and net_action==expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_TWO_POS][0].push(state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_TWO_POS][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_TWO_POS][1]+1


            elif memo[(x_old,y_old,0)]==2 and net_action!=expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_TWO_NEG][0].push(state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_TWO_NEG][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_TWO_NEG][1]+1


            elif memo[(x_old,y_old,0)]==3 and net_action==expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_THREE_POS][0].push(state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_THREE_POS][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_THREE_POS][1]+1


            elif memo[(x_old,y_old,0)]==3 and net_action!=expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_THREE_NEG][0].push(state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_THREE_NEG][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_THREE_NEG][1]+1


            elif memo[(x_old,y_old,0)]==4 and net_action==expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_FOUR_POS][0].push(state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_FOUR_POS][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_FOUR_POS][1]+1


            elif memo[(x_old,y_old,0)]==4 and net_action!=expert_action:
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_FOUR_NEG][0].push(state,expert_action)
                 events_replay_memory_list[BOMB_ESCAPE_LENGTH_FOUR_NEG][1]= events_replay_memory_list[BOMB_ESCAPE_LENGTH_FOUR_NEG][1]+1


        else:
       #target_x,target_y,distanz,coin?
            if expert.target_coin==True:

                if net_action==expert_action:
                     events_replay_memory_list[STEPPED_TOWARDS_COIN_POS][0].push(state,expert_action)
                     events_replay_memory_list[STEPPED_TOWARDS_COIN_POS][1]= events_replay_memory_list[STEPPED_TOWARDS_COIN_POS][1]+1
                else:
                     events_replay_memory_list[STEPPED_TOWARDS_COIN_NEG][0].push(state,expert_action)
                     events_replay_memory_list[STEPPED_TOWARDS_COIN_NEG][1]= events_replay_memory_list[STEPPED_TOWARDS_COIN_NEG][1]+1


            else:
                if expert_action==5:
                    if net_action==5:
                        events_replay_memory_list[Target_IN_RANGE_POS][0].push(state,expert_action)
                        events_replay_memory_list[Target_IN_RANGE_POS][1]= events_replay_memory_list[Target_IN_RANGE_POS][1]+1
                        events_replay_memory_list[CAN_ESCAPE_OWN_BOMB_POS][0].push(state,expert_action)
                        events_replay_memory_list[CAN_ESCAPE_OWN_BOMB_POS][1]= events_replay_memory_list[CAN_ESCAPE_OWN_BOMB_POS][1]+1
                    else:
                         events_replay_memory_list[SHOULD_HAVE_PLANTED_BOMB][0].push(state,expert_action)
                         events_replay_memory_list[SHOULD_HAVE_PLANTED_BOMB][1]= events_replay_memory_list[SHOULD_HAVE_PLANTED_BOMB][1]+1

                else:
                     if net_action==expert_action:
                         events_replay_memory_list[STEPPED_TOWARDS_TARGET_POS][0].push(state,expert_action)
                         events_replay_memory_list[STEPPED_TOWARDS_TARGET_POS][1]= events_replay_memory_list[STEPPED_TOWARDS_TARGET_POS][1]+1
                     else:
                         events_replay_memory_list[STEPPED_TOWARDS_TARGET_NEG][0].push(state,expert_action)
                         events_replay_memory_list[STEPPED_TOWARDS_TARGET_NEG][1]= events_replay_memory_list[STEPPED_TOWARDS_TARGET_NEG][1]+1

                     if net_action==5:
                          timestamp_dead_marker=compute_timestamp_dead_marker_with_extra_bomb(state["opponents_pos"],bombs_old,explosions_old,crates_old,walls_old,x_old,y_old)
                          memo={}
                          limit=5
                          find_rescue_route(timestamp_dead_marker,x_old,y_old,0,memo,limit)
                          
                          if memo[(x_old,y_old,0)]>4:
                             events_replay_memory_list[CAN_ESCAPE_OWN_BOMB_NEG][0].push(state,expert_action)
                             events_replay_memory_list[CAN_ESCAPE_OWN_BOMB_NEG][1]= events_replay_memory_list[CAN_ESCAPE_OWN_BOMB_NEG][1]+1

                          Bomb_Exlpoison_Radius_Local=Compute_Bomb_Exlpoison_Radius_Local(walls_old,x_old,y_old)

                          if (expert.target_x,expert.target_y) not in Bomb_Exlpoison_Radius_Local:
                             events_replay_memory_list[Target_IN_RANGE_NEG][0].push(state,expert_action)
                             events_replay_memory_list[Target_IN_RANGE_NEG][1]= events_replay_memory_list[Target_IN_RANGE_NEG][1]+1

                          if bombs_left==0:
                               events_replay_memory_list[PLANTED_WITHOUT_BOMBS_LEFT][0].push(state,expert_action)
                               events_replay_memory_list[PLANTED_WITHOUT_BOMBS_LEFT][1]= events_replay_memory_list[PLANTED_WITHOUT_BOMBS_LEFT][1]+1


                        
                    
             



      
       
       
               

def loop(env,policy_net,optimizer,criterion, n_episodes=500):
   
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
                            ,(STEPPED_IN_BLOCKED_FIELD,[ReplayMemory(1000),0]),(STEPPED_TOWARDS_TARGET_POS,[ReplayMemory(2000),0])
                            ,(STEPPED_TOWARDS_TARGET_NEG,[ReplayMemory(2000),0]),(STEPPED_TOWARDS_COIN_POS,[ReplayMemory(2000),0]),(STEPPED_TOWARDS_COIN_NEG,[ReplayMemory(2000),0])
                            ,(Target_IN_RANGE_NEG,[ReplayMemory(1000),0]),(Target_IN_RANGE_POS,[ReplayMemory(1000),0])
                            ,(CAN_ESCAPE_OWN_BOMB_NEG,[ReplayMemory(2000),0]),(CAN_ESCAPE_OWN_BOMB_POS,[ReplayMemory(2000),0])
                            ,(PLANTED_WITHOUT_BOMBS_LEFT,[ReplayMemory(500),0]),(SHOULD_HAVE_PLANTED_BOMB,[ReplayMemory(500),0])
                            ])

    batch_sizes=[20 for i in range(0,19)]

    cnt_steps=0
    epsiodes_to_reconfigurate_batch=30
    epsilon=0.5
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
            transformed_state=torch.tensor(Transform_State(state),dtype=torch.float32).to(device)

            net_action = act_net(policy_net,transformed_state)

            expert_action=action=expert.act(state)

            if random.random()>epsilon:
                action=net_action
           
            new_state, _, terminated, truncated, info = env.step(action)

            evaluate_net_action(state,action,expert_action,expert,events_replay_memory_list)

            if cnt_epsiodes==epsiodes_to_reconfigurate_batch:
                configurate_batch_sizes(events_replay_memory_list,batch_sizes)
                cnt_epsiodes=0

            if first_episodes>=epsiodes_to_reconfigurate_batch and train_every_x_steps<=cnt_steps:
               train(policy_net,optimizer,events_replay_memory_list,batch_sizes,criterion)
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
       
        states_list=[]
        actions_list=[]
        cnt=0
        for replay_mem in replay_memories:
            sampled_states,sampled_actions = replay_mem.sample()
            sampled_states=[Transform_State(state) for state in sampled_states]
           
            states_list.append(torch.tensor(sampled_states, dtype=torch.float32).to(device))
            actions_list.append(torch.tensor(sampled_actions, dtype=torch.long).to(device))

           
            policy_net.eval()
            with torch.no_grad():
                preds = torch.argmax(policy_net(states_list[cnt]), dim=1)
                acc = (preds == actions_list[cnt]).float().mean()

            print(f"Kategorie: {supervised_events[cnt]}:  Acc={acc:.4f}")

            cnt+=1

                    
        policy_net.train()
        states_tensor=torch.cat(states_list)
        actions_tensor=torch.cat(actions_list)
        logits = policy_net(states_tensor)

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
    policy_net = BaseActor(2145, 6).to(device)
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
  