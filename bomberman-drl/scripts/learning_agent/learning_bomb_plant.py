# code is taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import math
import random
from pathlib import Path
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from bomberman_rl import ActionSpace

STEPPED_IN_BLOCKED_FIELD="STEPPED_IN_BLOCKED_FIELD"

STEPPED_IN_EXPLOSION="STEPPED_IN_EXPLOSION"

STEPPED_IN_BOMB_NEG="STEPPED_IN_BOMB_NEG"

STEPPED_TOWARDS_TARGET_POS="STEPPED_TOWARDS_TARGET_POS"
STEPPED_TOWARDS_TARGET_NEG="STEPPED_TOWARDS_TARGET_NEG"

Target_IN_RANGE_NEG="Target_IN_RANGE_NEG"
Target_IN_RANGE_POS="Target_IN_RANGE_POS"

CAN_ESCAPE_OWN_BOMB_POS="CAN_ESCAPE_OWN_BOMB_POS"
CAN_ESCAPE_OWN_BOMB_NEG="CAN_ESCAPE_OWN_BOMB_NEG"

SHOULD_HAVE_PLANTED_BOMB="SHOULD_HAVE_PLANTED_BOMB"

PLANTED_WITHOUT_BOMBS_LEFT="PLANTED_WITHOUT_BOMBS_LEFT"


Replay_Kategorie_List=[
    STEPPED_IN_BOMB_NEG,
    STEPPED_IN_EXPLOSION,
    STEPPED_TOWARDS_TARGET_POS,
    STEPPED_TOWARDS_TARGET_NEG,
    Target_IN_RANGE_NEG,
    Target_IN_RANGE_POS,
    CAN_ESCAPE_OWN_BOMB_POS,
    CAN_ESCAPE_OWN_BOMB_NEG,
    SHOULD_HAVE_PLANTED_BOMB,
    PLANTED_WITHOUT_BOMBS_LEFT,
STEPPED_IN_BLOCKED_FIELD]




# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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


def compute_timestamp_dead_marker(opponents,bombs,explosions,crates,walls,cur_bomb_x,cur_bomb_y,time_limit_escape=10):
    
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


def Transform_Timestamp_Dead_Marker(timestamp_dead_marker,position,time_limit_escape=5):
       
    limit_x=16
    limit_y=16
    (x,y)=(0,0)
    for i in range(1,16):
         for j in range(1,16):
                if(position[i,j]==1):
                        (x,y)=(i,j)

    if timestamp_dead_marker[0,x,y]==1 or timestamp_dead_marker[0,x,y]==4:
         transformed_state_liste=[1.0]
    else:
         transformed_state_liste=[0.0]

    #transformed_state_liste.extend([(position[pos1,pos2]*1.0) for pos1 in range(0,limit_x+1) for pos2 in range(0,limit_y+1) if abs(pos1-x)+abs(pos2-y)<=4])
    #for t in range(1,time_limit_escape):
        #for pos1 in range(0,limit_x+1) :
             #for pos2 in range(0,limit_y+1):
                 #if timestamp_dead_marker[t,pos1,pos2]==0:
                 #    transformed_state_liste.append(0.0)
                 #elif timestamp_dead_marker[t,pos1,pos2]==1:
                 #     transformed_state_liste.append(0.5)
                 #else:
                 #     transformed_state_liste.append(1.0)

    for t in range(1,time_limit_escape):
        for pos1 in range(x - 4, x + 5):
           for pos2 in range(y - 4, y + 5):
               distanz = abs(pos1 - x) + abs(pos2 - y)  
               if distanz <= t:
                   if pos1>=0 and pos1<limit_x and pos2>=0 and pos2<limit_y:
                       if timestamp_dead_marker[t,pos1,pos2]==1:
                          transformed_state_liste.append(1.0)
                       else:
                          transformed_state_liste.append(0.0)
                   else:
                        transformed_state_liste.append(0.0)

    for t in range(1,time_limit_escape):
        for pos1 in range(x - 4, x + 5):
           for pos2 in range(y - 4, y + 5):
               distanz = abs(pos1 - x) + abs(pos2 - y)  
               if distanz <= t:
                   if pos1>=0 and pos1<limit_x and pos2>=0 and pos2<limit_y:
                       if timestamp_dead_marker[t,pos1,pos2]>=2:
                          transformed_state_liste.append(1.0)
                       else:
                          transformed_state_liste.append(0.0)
                   else:
                        transformed_state_liste.append(1.0)

        
  
   
   
       #TODO GEGNER HINZUFUEGEN
    return transformed_state_liste


def Transform_State(state,x_target,y_target,target):
        walls = state["walls"]
        position = state["self_info"]["position"]
        bombs_left = state["self_info"]["bombs_left"]
        score= state["self_info"]["score"]
        opponents=state["opponents_pos"]
        crates=state["crates"]
        explosions=state["explosions"]
        coins=state["coins"]
        bombs = state["bombs"]
        bombs_left=state["self_info"]["bombs_left"]

       
       

       

        #berechnet Explosionradius von Bomben
        bomb_exploison_radius=np.zeros((17,17))
        Compute_Bomb_Exlpoison_Radius(bomb_exploison_radius,bombs,walls)

        limit_x=16
        limit_y=16

       
        #berechnet eigene Position
        (x_old,y_old)=(0,0)
        for i in range(1,16):
             for j in range(1,16):
                    if(position[i,j]==1):
                            (x_old,y_old)=(i,j)

        state_representation_list=[bombs_left]
        
        timestamp_dead_marker=compute_timestamp_dead_marker(state["opponents_pos"],state["bombs"],state["explosions"],state["crates"],state["walls"],x_old,y_old)
        #Informationen die das Netz braucht um zu bestimmen ob eine Flurcht bei legen
        #einer Bombe moeglich ist
        transformed_state=Transform_Timestamp_Dead_Marker(timestamp_dead_marker,state["self_info"]["position"])

        window_length=15
        #Fenster mit Feldern welche Abstand kleiner als window_length haben zu Agenten
        #Das hat bei uns tausend mal besser geklappt als dem Netz die Position als Wert zu geben.
        #Die Position ist immer eindeutig, Agent ist immer in der Mitte, so muss dem Netz die Position
        #nicht gesagt und antraeniert werden da diese per Konstruktion klar ist.

        #berechne 1. Staterepresentation fuer passierbare Felder bzw. unpassierbare Felder
        for i in range(x_old - window_length, x_old + window_length+1):
           for j in range(y_old - window_length, y_old + window_length+1):
               distanz = abs(i - x_old) + abs(j - y_old)  
               if distanz < window_length:
                   if i>=0 and i<limit_x and j>=0 and j<limit_y:
                       if(walls[i,j]==1 or (bombs[i,j]>=1 and distanz<=5) or crates[i,j]==1 or (explosions[i,j]==12 and distanz<=1) or bomb_exploison_radius[i,j]>=1):
                           state_representation_list.append(1.0)
              
                       else:
                             state_representation_list.append(0.0)
                              
                   else:
                        state_representation_list.append(1.0)


        #Position des Ziels wird bestimmt
        for i in range(x_old - window_length, x_old + window_length+1):
            for j in range(y_old - window_length, y_old + window_length+1):
               distanz = abs(i - x_old) + abs(j - y_old)  
               if distanz < window_length:
                   if i>=0 and i<limit_x and j>=0 and j<limit_y:
                       if((i,j)==(x_target,y_target)):
                           state_representation_list.append(1.0)
               
                       else:
                            state_representation_list.append(0.0)

                   else:
                        state_representation_list.append(0.0)


        

        state_representation_list.extend(transformed_state)
        state_representation=np.array(list(state_representation_list))
       

       
                 
               
                  

        return state_representation
       

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        if self.memory.maxlen==len(self.memory):
            self.memory.popleft()
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """ State approximation via Multi-Layer Perceptron """
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, int((2/3)*n_observations))
        self.layer2 = nn.Linear(int((2/3)*n_observations), int((2/3)*(2/3)*n_observations))
        self.layer3 = nn.Linear(int((2/3)*(2/3)*n_observations), int((2/3)*(2/3)*(2/3)*n_observations))
        self.layer4 = nn.Linear(int((2/3)*(2/3)*(2/3)*n_observations), n_actions)
        

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        """ Expects flattened state vector """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
    

class Tabular(nn.Module):
    """ State approximation via Multi-Layer Perceptron """
    def __init__(self, n_observations, n_actions):
        super(Tabular, self).__init__()
        self.layer1 = nn.Linear(n_observations, int((2/3)*n_observations))
        self.layer2 = nn.Linear(int((2/3)*n_observations), int((2/3)*(2/3)*n_observations))
        self.layer3 = nn.Linear(int((2/3)*(2/3)*n_observations), int((2/3)*(2/3)*(2/3)*n_observations))
        self.layer4 = nn.Linear(int((2/3)*(2/3)*(2/3)*n_observations), n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        """ Expects flattened state vector """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
    

class Model():
    def __init__(self, epsilon_greedy=True,load=False, path=Path(__file__).parent / "model_bomb_plant.pt"):
        self.initial=True
        self.gamma = 0.75 # self.gamma is the discount factor
        self.eps_start = 0.5 # self.eps_start is the starting value of epsilon
        self.eps_end = 0 # self.eps_end is the final value of epsilon
        self.eps_decay = 50 # self.eps_decay controls the rate of exponential decay of epsilon, higher means a slower decay
        self.tau = 0.05 # self.tau is the update rate of the target network
        self.lr = 1e-3 # self.lr is the learning rate of the ``AdamW`` optimizer
        self.gradient_clipping = 1000
        self.steps = 0
        self.path = path
        self.load = load # load trained model or init from scratch
        self.epsilon_greedy=epsilon_greedy
        self.epsilon=self.eps_end+self.eps_start*((self.eps_decay-self.steps)/(self.eps_decay))
        self.n_actions = 6
        self.memory = dict([(STEPPED_IN_BOMB_NEG,(ReplayMemory(5_000),300)),(STEPPED_IN_EXPLOSION,(ReplayMemory(5_000),300)),
                            (STEPPED_TOWARDS_TARGET_POS,(ReplayMemory(15000),1000)), (STEPPED_TOWARDS_TARGET_NEG,(ReplayMemory(15000),1000))
                            ,(STEPPED_IN_BLOCKED_FIELD,(ReplayMemory(7000),200)),(Target_IN_RANGE_NEG,(ReplayMemory(5000),400)),
                            (Target_IN_RANGE_POS,(ReplayMemory(5000),400)),(CAN_ESCAPE_OWN_BOMB_POS,(ReplayMemory(5000),400)),
                            (CAN_ESCAPE_OWN_BOMB_NEG,(ReplayMemory(5000),400)),(SHOULD_HAVE_PLANTED_BOMB,(ReplayMemory(5000),400)),
                            (PLANTED_WITHOUT_BOMBS_LEFT,(ReplayMemory(5000),200))]) 


        self.memory_state_action_pairs=dict([
                            (STEPPED_TOWARDS_TARGET_POS,([],[]))
                            ,
                            (Target_IN_RANGE_POS,([],[])),(CAN_ESCAPE_OWN_BOMB_POS,([],[])),
                         ]) 
       
        self.cnt_event=dict([(STEPPED_IN_BOMB_NEG,0),(STEPPED_IN_EXPLOSION,0),(STEPPED_TOWARDS_TARGET_POS,0),(STEPPED_TOWARDS_TARGET_NEG,0)
                            ,(STEPPED_IN_BLOCKED_FIELD,0),(Target_IN_RANGE_POS,0),(Target_IN_RANGE_NEG,0),(CAN_ESCAPE_OWN_BOMB_POS,0),(CAN_ESCAPE_OWN_BOMB_NEG,0)
                            ,(SHOULD_HAVE_PLANTED_BOMB,0),(PLANTED_WITHOUT_BOMBS_LEFT,0)]) 
        self.cnt=0
        self.policy_net = None

    def lazy_init(self, observation):
        # only on first observation can we lazy initialize as we have no upfront information on the environment
        self.n_observations = len(observation)
        self.policy_net = Tabular(self.n_observations, self.n_actions).to(device)
        if self.load:
            try:
                self.load_weights()
            except FileNotFoundError:
                pass
        self.target_net = Tabular(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)

    def act(self, state,x_target,y_target,target):

      

        transformed_state=Transform_State(state,x_target,y_target,target)

        if self.policy_net is None:
            self.lazy_init(transformed_state)
       
       
        transformed_state = torch.tensor(transformed_state, device=device, dtype=torch.float32).unsqueeze(0)
        
        sample = random.random()
        

        
       
       
       


        if (self.epsilon_greedy and sample >= self.epsilon) or (not self.epsilon_greedy):
        #if sample >= 1.5:
            self.policy_net.eval()
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(transformed_state).max(1).indices
        else:
           
            
            position = state["self_info"]["position"]
            walls_old=state["walls"]
            crates_old=state["crates"]
            coin_old =state["coins"]
            bombs_old=state["bombs"]
            explosions_old=state["explosions"]
            #berechne eigne Pos.
            (x_old,y_old)=(0,0)
            for i in range(1,16):
                    for j in range(1,16):
                        if(position[i,j]==1):
                            (x_old,y_old)=(i,j)

            bomb_exploison_radius=np.zeros((17,17))
            Compute_Bomb_Exlpoison_Radius(bomb_exploison_radius,bombs_old,walls_old)
            #berechnet min weg der durch keine Waende,Bomben,Explosionen und Kisten geht mit bfs.
            predecessor={}
            q=deque()
            q.append([x_old,y_old,0])
            path_list=[[x_old,y_old]]
            distanz=1e9
            
            while q:
                cur_pos=q.popleft()
                x=cur_pos[0]
                y=cur_pos[1]
                d=cur_pos[2]

                if x==x_target and y==y_target:
                    distanz=d             
                    break

                if (bomb_exploison_radius[x,y+1]==0 or d>4) and (explosions_old[x,y+1]!=12 or d>0) and walls_old[x,y+1]==0 and (target=="CRATE" or crates_old[x,y+1]==0) and [x,y+1] not in path_list:
                    q.append([x,y+1,d+1])
                    path_list.append([x,y+1])
                    predecessor[x,y+1]=(x,y)
                    
                if (bomb_exploison_radius[x,y-1]==0 or d>4) and (explosions_old[x,y-1]!=12 or d>0) and walls_old[x,y-1]==0 and (target=="CRATE" or crates_old[x,y-1]==0) and [x,y-1] not in path_list:
                    q.append([x,y-1,d+1])
                    path_list.append([x,y-1])
                    predecessor[x,y-1]=(x,y)

                if (bomb_exploison_radius[x+1,y]==0 or d>4) and (explosions_old[x+1,y]!=12 or d>0) and walls_old[x+1,y]==0 and (target=="CRATE" or crates_old[x+1,y]==0) and [x+1,y] not in path_list:
                    q.append([x+1,y,d+1])
                    path_list.append([x+1,y])
                    predecessor[x+1,y]=(x,y)

                if (bomb_exploison_radius[x-1,y]==0 or d>4) and (explosions_old[x-1,y]!=12 or d>0) and walls_old[x-1,y]==0 and (target=="CRATE" or crates_old[x-1,y]==0) and [x-1,y] not in path_list:
                    q.append([x-1,y,d+1])
                    path_list.append([x-1,y])
                    predecessor[x-1,y]=(x,y)

                

            if x_target==-1:
                action=4
            else:

                Bomb_Exlpoison_Radius_Local=Compute_Bomb_Exlpoison_Radius_Local(walls_old,x_old,y_old)
                timestamp_dead_marker=compute_timestamp_dead_marker(state["opponents_pos"],bombs_old,explosions_old,crates_old,walls_old,x_old,y_old)
                memo={}
                limit=5
                find_rescue_route(timestamp_dead_marker,x_old,y_old,0,memo,limit)
                bombs_left=state["self_info"]["bombs_left"]

               
                if distanz<=2 and (x_target,y_target) in Bomb_Exlpoison_Radius_Local and memo[(x_old,y_old,0)]<=limit and bombs_left>=1:
                     action=5
                     


                elif distanz>=2:
                    #backtracke Pfad von Agetn zu Muenze, ueber Feld der Muenze Vorganger bis zum Feld welches neben Agenten ist
                    x_cur=x_target
                    y_cur=y_target
                    while (x_cur,y_cur) != (x_old,y_old) and predecessor[x_cur,y_cur]!=(x_old,y_old):
                        (x_cur,y_cur)=predecessor[x_cur,y_cur]


                    #wo liegt das Feld? oben, rechts etc.
                    if (x_cur,y_cur) == (x_old-1,y_old):
                        action= 3

                    elif (x_cur,y_cur) == (x_old+1,y_old):
                         action= 1


                    elif (x_cur,y_cur) == (x_old,y_old-1):
                         action= 2

                    elif (x_cur,y_cur) == (x_old,y_old+1):
                         action= 0
                    else:
                        action=4

                else:
                    action=4

            return torch.tensor([action], device=device, dtype=torch.long)


    def optimize_incremental(self,cond):
        """
        One iteration of Q learning (Bellman optimality equation for Q values) on a random batch of past experience
        """
        if self.policy_net is not None:
            self.policy_net.train()
        
            transitions=[]
          # cond ist Bedingung in Main, alle x Epsisoden soll Epsilon verringert und Batchgroeße angepasst werden
          #in Abhaengigkeit der Performance des Agenten in den Kategorien
            if cond:
                self.initial=False
                self.steps += 1
                self.epsilon=self.eps_end+self.eps_start*((self.eps_decay-self.steps)/(self.eps_decay))


            

            for mem in self.memory.items():
            
               #fuelle Batch mit Transitionen bezueglich Haufigkeit und Wichtigkeit
                if (mem[0]==STEPPED_TOWARDS_TARGET_POS or mem[0]==STEPPED_TOWARDS_TARGET_NEG) and not self.initial and cond:
                    factor=self.cnt_event[STEPPED_TOWARDS_TARGET_NEG]/(self.cnt_event[STEPPED_TOWARDS_TARGET_POS]+self.cnt_event[STEPPED_TOWARDS_TARGET_NEG]+0.00000001)
                    mem=(mem[0],(mem[1][0],int(factor*2000)))
                    self.memory[mem[0]]=(mem[1][0],int(factor*2000))
                  
            
                elif (mem[0]==STEPPED_IN_EXPLOSION) and not self.initial and cond:
                     mem=(mem[0],(mem[1][0],self.cnt_event[STEPPED_IN_EXPLOSION]*10+50))
                     self.memory[mem[0]]=(mem[1][0],self.cnt_event[STEPPED_IN_EXPLOSION]*10+50)

                elif (mem[0]==STEPPED_IN_BLOCKED_FIELD) and not self.initial and cond:
                     mem=(mem[0],(mem[1][0],self.cnt_event[STEPPED_IN_BLOCKED_FIELD]+50))
                     self.memory[mem[0]]=(mem[1][0],self.cnt_event[STEPPED_IN_BLOCKED_FIELD]+50)

                elif (mem[0]==STEPPED_IN_BOMB_NEG) and not self.initial and cond:
                     mem=(mem[0],(mem[1][0],self.cnt_event[STEPPED_IN_BOMB_NEG]*4+50))
                     self.memory[mem[0]]=(mem[1][0],self.cnt_event[STEPPED_IN_BOMB_NEG]*4+50)

                elif (mem[0]==Target_IN_RANGE_POS or mem[0]==Target_IN_RANGE_NEG) and not self.initial and cond:
                    factor=self.cnt_event[Target_IN_RANGE_NEG]/(self.cnt_event[Target_IN_RANGE_POS]+self.cnt_event[Target_IN_RANGE_NEG]+0.00000001)
                    mem=(mem[0],(mem[1][0],int(factor*1000)+100))
                    self.memory[mem[0]]=(mem[1][0],int(factor*1000)+100)
                
                elif (mem[0]==CAN_ESCAPE_OWN_BOMB_POS or mem[0]==CAN_ESCAPE_OWN_BOMB_NEG) and not self.initial and cond:
                    factor=self.cnt_event[CAN_ESCAPE_OWN_BOMB_NEG]/(self.cnt_event[CAN_ESCAPE_OWN_BOMB_POS]+self.cnt_event[CAN_ESCAPE_OWN_BOMB_NEG]+0.00000001)
                    mem=(mem[0],(mem[1][0],int(factor*1000)+100))
                    self.memory[mem[0]]=(mem[1][0],int(factor*1000)+100)

                elif (mem[0]==SHOULD_HAVE_PLANTED_BOMB) and not self.initial and cond:
                     mem=(mem[0],(mem[1][0],self.cnt_event[SHOULD_HAVE_PLANTED_BOMB]*2+50))
                     self.memory[mem[0]]=(mem[1][0],self.cnt_event[SHOULD_HAVE_PLANTED_BOMB]*1+50)

                elif (mem[0]==PLANTED_WITHOUT_BOMBS_LEFT) and not self.initial and cond:
                     mem=(mem[0],(mem[1][0],self.cnt_event[PLANTED_WITHOUT_BOMBS_LEFT]*2+50))
                     self.memory[mem[0]]=(mem[1][0],self.cnt_event[PLANTED_WITHOUT_BOMBS_LEFT]*2+50)
            
                transitions.extend(mem[1][0].sample(mem[1][1]))
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.

            if cond:
                self.initial=False
                print("Bomb-Plant")
                print("Epsilon: ",self.epsilon)
                for event in self.cnt_event:
                     
                      print(event,self.cnt_event[event])
                      print(event,self.memory[event][1])
                      self.cnt_event[event]=0

            

            


            if transitions:
                
                batch = Transition(*zip(*transitions))

                # Compute a mask of non-final states and concatenate the batch elements
                # (a final state would've been the one after which simulation ended)
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=device, dtype=torch.bool)
                #non_final_next_states = torch.stack([s for s in batch.next_state
                 #                                           if s is not None]).to(device)
                state_batch = torch.stack(batch.state).to(device)
                action_batch = torch.stack(batch.action).to(device)
                reward_batch = torch.stack(batch.reward).to(device).squeeze(1)

                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                # columns of actions taken. These are the actions which would've been taken
                # for each batch state according to policy_net
                state_action_values = self.policy_net(state_batch).gather(1, action_batch)

                # Compute V(s_{t+1}) for all next states.
                # Expected values of actions for non_final_next_states are computed based
                # on the "older" target_net; selecting their best reward with max(1).values
                # This is merged based on the mask, such that we'll have either the expected
                # state value or 0 in case the state was final.
                #next_state_values = torch.zeros(state_batch.shape[0], device=device)
               # with torch.no_grad():
                #    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values # max Q values of next state

                # Compute the optimal Q values
                #expected_state_action_values = (next_state_values * self.gamma) + reward_batch # Bellman optimality equation
                expected_state_action_values =reward_batch
                # Compute Huber loss
                criterion = nn.SmoothL1Loss()
                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

                # Optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                # In-place gradient clipping
                torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.gradient_clipping)
                self.optimizer.step()

                self.update_target_net()
           

    def update_target_net(self):
        """
        Soft update of the target network's weights
        θ′ ← τ θ + (1 −τ )θ′
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def experience(self, old_state, action, new_state, reward,custom_events,target_x,target_y,target):
        """
        Save new experience
        """
        new_state=None
        
        old_state_transformed=Transform_State(old_state,target_x,target_y,target)
        if new_state is not None:
            new_state_transformed=Transform_State(new_state,target_x,target_y,target)
       
        if self.policy_net is None:
            self.lazy_init(old_state_transformed)
        for event in custom_events:
            if event in Replay_Kategorie_List:

                self.memory[event][0].push(
                    torch.tensor(old_state_transformed, dtype=torch.float32),
                    torch.tensor([action], device=device, dtype=torch.int64),
                    None if new_state is None else torch.tensor(new_state_transformed, dtype=torch.float32),
                    torch.tensor([reward], device=device, dtype=torch.float32))

                if event in [STEPPED_TOWARDS_TARGET_POS,CAN_ESCAPE_OWN_BOMB_POS,Target_IN_RANGE_POS]:
                    self.memory_state_action_pairs[event][0].append(old_state)
                    self.memory_state_action_pairs[event][1].append(action)

                self.cnt_event[event]=self.cnt_event[event]+1

    def save_weights(self):
        if self.policy_net is not None:
           torch.save(self.policy_net.state_dict(), self.path)

    def load_weights(self):
        self.policy_net.load_state_dict(torch.load(self.path))