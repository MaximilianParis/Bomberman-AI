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

STAYED_SAFE="STAYED_SAFE"
STEPPED_IN_BOMB_NEG="STEPPED_IN_BOMB_NEG"

STEPPED_IN_BLOCKED_FIELD="STEPPED_IN_BLOCKED_FIELD"

BOMB_ESCAPE_LENGTH_ONE_POS="BOMB_ESCAPE_LENGTH_ONE_POS"
BOMB_ESCAPE_LENGTH_ONE_NEG="BOMB_ESCAPE_LENGTH_ONE_NEG"

BOMB_ESCAPE_LENGTH_TWO_POS="BOMB_ESCAPE_LENGTH_TWO_POS"
BOMB_ESCAPE_LENGTH_TWO_NEG="BOMB_ESCAPE_LENGTH_TWO_NEG"

BOMB_ESCAPE_LENGTH_THREE_POS="BOMB_ESCAPE_LENGTH_THREE_POS"
BOMB_ESCAPE_LENGTH_THREE_NEG="BOMB_ESCAPE_LENGTH_THREE_NEG"

BOMB_ESCAPE_LENGTH_FOUR_POS="BOMB_ESCAPE_LENGTH_FOUR_POS"
BOMB_ESCAPE_LENGTH_FOUR_NEG="BOMB_ESCAPE_LENGTH_FOUR_NEG"



USED_SEMI_OPT_BOMB_ESCAPE_POS="USED_SEMI_OPT_BOMB_ESCAPE_POS"



Replay_Kategorie_List=[
    STEPPED_IN_BOMB_NEG,
    STAYED_SAFE,

BOMB_ESCAPE_LENGTH_ONE_POS,
BOMB_ESCAPE_LENGTH_ONE_NEG,

BOMB_ESCAPE_LENGTH_TWO_POS,
BOMB_ESCAPE_LENGTH_TWO_NEG,

BOMB_ESCAPE_LENGTH_THREE_POS,
BOMB_ESCAPE_LENGTH_THREE_NEG,

BOMB_ESCAPE_LENGTH_FOUR_POS,
BOMB_ESCAPE_LENGTH_FOUR_NEG,



USED_SEMI_OPT_BOMB_ESCAPE_POS,

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


def compute_timestamp_dead_marker(opponents,bombs,explosions,crates,walls,time_limit_escape=10):
    
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




def Transform_Timestamp_Dead_Marker(timestamp_dead_marker,position,time_limit_escape=5):
     #transformiert die Eingabe des DP Algorithmus fuer das neuronale Netz
    limit_x=16
    limit_y=16
   # berechne Position des Agenten
    (x,y)=(0,0)
    for i in range(1,16):
         for j in range(1,16):
                if(position[i,j]==1):
                        (x,y)=(i,j)
    #markiere ob Agent in Gefahr ist
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

   #fuer unseren DP Algo zum Fliehen muss man viel weniger Felder betrachten, erstellen zu viele.
   #hier nehmen wir uns nur die die der Algo wirklich braucht.
   #Uns ist das leider zu spaet aufgefallen, Aenderung des DP Algo ist zu aufwendig und riskant
   #bis zur Abgabe, ich bitte um Verstaendnis.
   #Tatsaechlich sind nur Felder die einen Abstand in der Manhattan-Metrik von kleiner gleich dem Zeitstempel t
   #wichtig die anderen kann man nicht erreichen, weil bis t man nur t moves machen kann.
   #Ebenso wird beim erstellen vom Zeitschrittmarkierer(timestamp_dead_marker) bei Bomben hoechstens 4 Zeitschritte,
   #als gefaerhlich markiert 0,1,2,3, sonst ist man sowieso tot.
   # man braucht ggf. noch den 4 Zeitschritte um zu pruefen ob man mit dem letzten move in ein freies Feld kommt.
   #Ausserdem haben wir noch die Status 2,3,4 im timestamp_dead_marker zusammengefasst.
   #So brauchen wir tatsaechlich nur 169 Bool Werte um zu entschieden ob wir fliehen koennen oder nicht.
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

        
  
   
    transformed_state=np.array(list(transformed_state_liste))

    return transformed_state
       

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        #entferne zu Alte Zustandaktionspaare wenn voll
        if self.memory.maxlen==len(self.memory):
            self.memory.popleft()
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self):
        return len(self.memory)


    

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
    def __init__(self,epsilon_greedy=True, load=False, path=Path(__file__).parent / "model_bomb_escape.pt"):
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
        self.n_actions = 5
        self.epsilon_greedy=epsilon_greedy
        self.epsilon=self.eps_end+self.eps_start*((self.eps_decay-self.steps)/(self.eps_decay))
        #ReplayMemory fuer die Kategorien woraus zufalig fuer den Batch gezoen wird, werden hier definiert
        #dazu wird groesse des Batches angegeben
        self.memory = dict([(STEPPED_IN_BOMB_NEG,(ReplayMemory(5_000),450)),(BOMB_ESCAPE_LENGTH_ONE_POS,(ReplayMemory(8_000),800)),(BOMB_ESCAPE_LENGTH_ONE_NEG,(ReplayMemory(8_000),800))
                            ,(BOMB_ESCAPE_LENGTH_TWO_POS,(ReplayMemory(6_000),500)),(BOMB_ESCAPE_LENGTH_TWO_NEG,(ReplayMemory(6_000),500)),(BOMB_ESCAPE_LENGTH_THREE_POS,(ReplayMemory(3_000),100))
                            ,(BOMB_ESCAPE_LENGTH_THREE_NEG,(ReplayMemory(3_000),100)),(BOMB_ESCAPE_LENGTH_FOUR_POS,(ReplayMemory(2_000),0)),(BOMB_ESCAPE_LENGTH_FOUR_NEG,(ReplayMemory(2_000),0))
                            ,(USED_SEMI_OPT_BOMB_ESCAPE_POS,(ReplayMemory(7000),0)),(STAYED_SAFE,(ReplayMemory(5000),400))
                            ,(STEPPED_IN_BLOCKED_FIELD,(ReplayMemory(10000),1000))]) 

        self.memory_state_action_pairs = dict([(BOMB_ESCAPE_LENGTH_ONE_POS,([],[]))
                            ,(BOMB_ESCAPE_LENGTH_TWO_POS,([],[])),(BOMB_ESCAPE_LENGTH_THREE_POS,([],[]))
                            ,(BOMB_ESCAPE_LENGTH_FOUR_POS,([],[]))]) 
       #zum Zaehlen der Haeufigkeit der Ereignisse
        self.cnt_event=dict([(STEPPED_IN_BOMB_NEG,0),(BOMB_ESCAPE_LENGTH_ONE_POS,0),(BOMB_ESCAPE_LENGTH_ONE_NEG,0)
                            ,(BOMB_ESCAPE_LENGTH_TWO_POS,0),(BOMB_ESCAPE_LENGTH_TWO_NEG,0),(BOMB_ESCAPE_LENGTH_THREE_POS,0)
                            ,(BOMB_ESCAPE_LENGTH_THREE_NEG,0),(BOMB_ESCAPE_LENGTH_FOUR_POS,0),(BOMB_ESCAPE_LENGTH_FOUR_NEG,0)
                            ,(USED_SEMI_OPT_BOMB_ESCAPE_POS,0),(STAYED_SAFE,0)
                            ,(STEPPED_IN_BLOCKED_FIELD,0)]) 
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

    def act(self, state):
        #berechne Eingabe fuer neuronales Netz
        timestamp_dead_marker=compute_timestamp_dead_marker(state["opponents_pos"],state["bombs"],state["explosions"],state["crates"],state["walls"])

        transformed_state=Transform_Timestamp_Dead_Marker(timestamp_dead_marker,state["self_info"]["position"])

        if self.policy_net is None:
            self.lazy_init(transformed_state)
       
       
        transformed_state = torch.tensor(transformed_state, device=device, dtype=torch.float32).unsqueeze(0)
        
        sample = random.random()
        

        
       
       
       

        #Epsilongreedy zum Umschalten zwischen Experte und Netz.
        #Mit Turnierseetings durch or
        if (self.epsilon_greedy and sample >= self.epsilon) or (not self.epsilon_greedy):
        
            self.policy_net.eval()
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(transformed_state).max(1).indices
        else:
           #Experte der Fluchtroute berechnet
            position = state["self_info"]["position"]
            
            (x,y)=(0,0)
            for i in range(1,16):
                    for j in range(1,16):
                        if(position[i,j]==1):
                            (x,y)=(i,j)
        
            
            action=find_rescue_route(timestamp_dead_marker,x,y,0,{})
                   
            return torch.tensor([action], device=device, dtype=torch.long)
        
    def optimize_incremental(self,cond):
        """
        One iteration of Q learning (Bellman optimality equation for Q values) on a random batch of past experience
        """
        if self.policy_net is not None:
            self.policy_net.train()
        
            transitions=[]
      
            if cond:
                self.initial=False
                #nur bei Cond Epsilon aktualisieren
                self.steps += 1
                self.epsilon=self.eps_end+self.eps_start*((self.eps_decay-self.steps)/(self.eps_decay))


            for mem in self.memory.items():
            
                #je nach Wichtigkeit der Kategorie wird pro Kategorie Anzahl an Zustandaktionspaaren berechnet die in den batch kommen
                if (mem[0]==BOMB_ESCAPE_LENGTH_ONE_POS or mem[0]==BOMB_ESCAPE_LENGTH_ONE_NEG) and not self.initial and cond:
                    factor=self.cnt_event[BOMB_ESCAPE_LENGTH_ONE_NEG]/(self.cnt_event[BOMB_ESCAPE_LENGTH_ONE_POS]+self.cnt_event[BOMB_ESCAPE_LENGTH_ONE_NEG]+0.0000001)
                    mem=(mem[0],(mem[1][0],int(factor*2000)+100))
                    self.memory[mem[0]]=(mem[1][0],int(factor*2000)+100)
               

                elif (mem[0]==BOMB_ESCAPE_LENGTH_TWO_POS or mem[0]==BOMB_ESCAPE_LENGTH_TWO_NEG) and not self.initial and cond:
                     factor=self.cnt_event[BOMB_ESCAPE_LENGTH_TWO_NEG]/(self.cnt_event[BOMB_ESCAPE_LENGTH_TWO_POS]+self.cnt_event[BOMB_ESCAPE_LENGTH_TWO_NEG]+0.0000001)
                     mem=(mem[0],(mem[1][0],int(factor*1000)+100))
                     self.memory[mem[0]]=(mem[1][0],int(factor*1000)+100)
                     #print(2,factor)           

                elif (mem[0]==BOMB_ESCAPE_LENGTH_THREE_POS or mem[0]==BOMB_ESCAPE_LENGTH_THREE_NEG) and not self.initial and cond:
                     factor=self.cnt_event[BOMB_ESCAPE_LENGTH_THREE_NEG]/(self.cnt_event[BOMB_ESCAPE_LENGTH_THREE_POS]+self.cnt_event[BOMB_ESCAPE_LENGTH_THREE_NEG]+0.0000001)
                     mem=(mem[0],(mem[1][0],int(factor*200)+50))
                     self.memory[mem[0]]=(mem[1][0],int(factor*600)+50)

                elif (mem[0]==BOMB_ESCAPE_LENGTH_FOUR_POS or mem[0]==BOMB_ESCAPE_LENGTH_FOUR_NEG) and not self.initial and cond:
                     factor=self.cnt_event[BOMB_ESCAPE_LENGTH_FOUR_NEG]/(self.cnt_event[BOMB_ESCAPE_LENGTH_FOUR_POS]+self.cnt_event[BOMB_ESCAPE_LENGTH_FOUR_NEG]+0.0000001)
                     mem=(mem[0],(mem[1][0],int(factor*50)+25))
                     self.memory[mem[0]]=(mem[1][0],int(factor*200)+25)

                elif (mem[0]==STEPPED_IN_BLOCKED_FIELD) and not self.initial and cond:
                     mem=(mem[0],(mem[1][0],self.cnt_event[STEPPED_IN_BLOCKED_FIELD]+100))
                     self.memory[mem[0]]=(mem[1][0],self.cnt_event[STEPPED_IN_BLOCKED_FIELD]+75)

                elif (mem[0]==STEPPED_IN_BOMB_NEG) and not self.initial and cond:
                     mem=(mem[0],(mem[1][0],self.cnt_event[STEPPED_IN_BOMB_NEG]*2+100))
                     self.memory[mem[0]]=(mem[1][0],self.cnt_event[STEPPED_IN_BOMB_NEG]*2+75)

                elif (mem[0]==STAYED_SAFE) and not self.initial and cond:
                     mem=(mem[0],(mem[1][0],50))
                     self.memory[mem[0]]=(mem[1][0],50)



               #batch wird vergoessert
                transitions.extend(mem[1][0].sample(mem[1][1]))
           

            #Statistik bezueglich Performance der letzten X Epsioden wird hier ausgegeben
            if cond:
                print("Bomb-Escape")
                print("Epsilon: ",self.epsilon)
                self.initial=False
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
               # next_state_values = torch.zeros(state_batch.shape[0], device=device)
                #with torch.no_grad():
                  #  next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values # max Q values of next state

                # Compute the optimal Q values
                #expected_state_action_values = (next_state_values * self.gamma) + reward_batch # Bellman optimality equation
                expected_state_action_values=reward_batch
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

    def experience(self, old_state, action, new_state, reward,custom_events):
        """
        Save new experience
        """
        #provisorische Aenderung um
        new_state=None
        old_state_transformed=Transform_Timestamp_Dead_Marker(compute_timestamp_dead_marker(old_state["opponents_pos"],old_state["bombs"],old_state["explosions"],old_state["crates"],old_state["walls"]),old_state["self_info"]["position"])
        if new_state is not None:
            new_state_transformed=Transform_Timestamp_Dead_Marker(compute_timestamp_dead_marker(new_state["opponents_pos"],new_state["bombs"],new_state["explosions"],new_state["crates"],new_state["walls"]),new_state["self_info"]["position"])
       
        if self.policy_net is None:
            self.lazy_init(old_state_transformed)
        for event in custom_events:
            if event in Replay_Kategorie_List:

                self.memory[event][0].push(
                    torch.tensor(old_state_transformed, dtype=torch.float32),
                    torch.tensor([action], device=device, dtype=torch.int64),
                    None if new_state is None else torch.tensor(new_state_transformed, dtype=torch.float32),
                    torch.tensor([reward], device=device, dtype=torch.float32))

               
                if event in [BOMB_ESCAPE_LENGTH_ONE_POS,BOMB_ESCAPE_LENGTH_TWO_POS,BOMB_ESCAPE_LENGTH_THREE_POS,BOMB_ESCAPE_LENGTH_FOUR_POS]:
                    self.memory_state_action_pairs[event][0].append(old_state)
                    self.memory_state_action_pairs[event][1].append(action)
              
                self.cnt_event[event]=self.cnt_event[event]+1

    def save_weights(self):
        if self.policy_net is not None:
           torch.save(self.policy_net.state_dict(), self.path)

    def load_weights(self):
        self.policy_net.load_state_dict(torch.load(self.path))