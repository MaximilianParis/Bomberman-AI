from gymnasium.spaces import flatten
from bomberman_rl import LearningAgent, events as e
import numpy as np
import sys
import pickle
from collections import deque
from .learning_coin_collect import Model

# Custom events

STEPPED_IN_BLOCKED_FIELD="STEPPED_IN_BLOCKED_FIELD"

STEPPED_IN_EXPLOSION="STEPPED_IN_EXPLOSION"

STEPPED_IN_BOMB_NEG="STEPPED_IN_BOMB_NEG"

STEPPED_TOWARDS_COIN_POS="STEPPED_TOWARDS_COIN_POS"
STEPPED_TOWARDS_COIN_NEG="STEPPED_TOWARDS_COIN_NEG"

COLLECTED_COIN="COLLECTED_COIN"


                       




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


def compute_timestamp_dead_marker(bombs,explosions,crates,walls,time_limit_escape=10):
    
    limit_x=16
    limit_y=16
    #0 frei, 1 in Bombenradis, 2 Crate, 3 in Bombenradis und Crate, 4 Wand oder Explosion
    timestamp_dead_marker=np.zeros((time_limit_escape,limit_x+1,limit_y+1))

    #for i in range(0,17):
       #for j in range(0,17):
         #  if explosions[i,j]==12:
          #     explosions[i,j]=2
               
          # elif explosions[i,j]==11:
          #     explosions[i,j]=1
               
          # else:
           #    explosions[i,j]=0

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
            if walls[i,j]==1:
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



class Agent_Coin_Collect(LearningAgent):
    """
    Sticking to the ``LearningAgent`` interface is optional.
    It enables your agent to **learn** as proper part of the environment (``/src/bomberman_rl/envs/agent_code/<agent>``) in order to enable Self-Play.
    The example training loop in main.py supports this interface as well by calling the respective callbacks.
    (Demonstration only - do not inherit)
    """
    def __init__(self,load,train):
        self.load=load
        self.train=train
        self.setup()
        self.setup_training()
       

    def setup(self):
        """
        Before episode. Use this to setup action related state that is required to act on the environment.
        """
        self.q_learning = Model(self.train,self.load)

    def act(self, state, **kwargs) -> int:
        """
        Before step. Return action based on state.
        :param state: The state of the environment.
        """
        #np.set_printoptions(threshold=sys.maxsize)
        #3 (i,j)->(i-1,j) links
        #1 (i,j)->(i+1,j) rechts
        #2 (i,j)->(i,j-1) unten
        #0 (i,j)->(i,j+1) oben
        #position = state["self_info"]["position"]
        #for i in range(1,16):
        #    for j in range(1,16):
        #       if(position[i,j]==1):
         #          print(i,j)
        #transformed_state= Transform_State(state)
        #return 2
       
        #walls = state["walls"]
        #position = state["self_info"]["position"]
        #opponents=state["opponents_pos"]
        #crates=state["crates"]
        #explosions=state["explosions"]
        #bombs = state["bombs"]
        #(x,y)=(0,0)
        #for i in range(1,16):
         #       for j in range(1,16):
        #            if(position[i,j]==1):
         #               (x,y)=(i,j)
        #opponents_pos=[(i, j) for i in range(0,17) for j in range(0,17) if opponents[i,j]==1]
        #bomb_exploison_radius=np.zeros((17,17))
        #Compute_Bomb_Exlpoison_Radius(bomb_exploison_radius,bombs,walls)
        #if(bomb_exploison_radius[x,y]>=1):
            #timestamp_dead_marker=compute_timestamp_dead_marker(bombs,explosions,crates,walls)
            #print(timestamp_dead_marker)
            #action=find_rescue_route(timestamp_dead_marker,x,y,0,{},10)
           
        #else:
        #    action= 4
            
        #print(f"Action {action} Position: {x}, {y}")
        
        #return action
        return self.q_learning.act(state)[0].item()

    def setup_training(self):
        """
        Before episode (optional). Use this to setup additional learning related state e.g. a replay memory, hyper parameters etc.
        """
        pass

    def game_events_occurred(
        self,
        old_state,
        self_action,
        new_state,
        events,
        cond
    ):
        """
        After step in environment (optional). Use this e.g. for model training.

        :param old_state: Old state of the environment.
        :param self_action: Performed action.
        :param new_state: New state of the environment.
        :param events: Events that occurred during step. These might be used for Reward Shaping.
        """
        reward=0
        if cond:
            custom_events = self._custom_events(old_state,self_action, new_state)
            reward = self._shape_reward(events + custom_events)
            self.q_learning.experience(old_state=old_state, action=self_action, new_state=new_state, reward=reward,custom_events=custom_events)
        if self.train:
            self.q_learning.optimize_incremental(False)

        return reward


    def end_of_round(self,cond):
        """
        After episode ended (optional). Use this e.g. for model training and saving.
        """
        if self.train:
            self.q_learning.optimize_incremental(cond)
            self.q_learning.save_weights() # save model in case this was last round


    def save_supervised_state_action_pairs_coin_collect(self):
        return self.q_learning.memory_state_action_pairs.items()


    def _custom_events(self, old_state,action, new_state):
        """
        Just an idea to demonstrate that you are not solely bound to official events for reward shaping
        """
        custom_events = []
        
        bombs_old = old_state["bombs"]
        walls_old=old_state["walls"]
        crates_old=old_state["crates"]
        explosions_old=old_state["explosions"]
        position_old = old_state["self_info"]["position"]
        coin_old =old_state["coins"]

        bomb_exploison_radius=np.zeros((17,17))
        Compute_Bomb_Exlpoison_Radius(bomb_exploison_radius,bombs_old,walls_old)

        for a in range(0,17):
            for b in range(0,17):
              if position_old[a,b]==1:
                  (x_old,y_old)=(a,b)
        
               
       #Agent soll fuer Bombenposition nicht traeniert werden, macht anderer Agent
        if bomb_exploison_radius[x_old,y_old]==0:

            #checkt welcher Move getaeigt wurde und dann ob der Agent in ein blockieres Feld bzw. Bombe, Explosion getreten ist
            if (walls_old[x_old-1,y_old]==1 or bombs_old[x_old-1,y_old]>=1 or crates_old[x_old-1,y_old]==1)  and action==3:
                    custom_events.append(STEPPED_IN_BLOCKED_FIELD)

            elif bomb_exploison_radius[x_old-1,y_old]>=1 and  action==3:
                    custom_events.append(STEPPED_IN_BOMB_NEG)

            elif explosions_old[x_old-1,y_old]==12 and new_state is None and action==3:
                    custom_events.append(STEPPED_IN_EXPLOSION)


                
            elif (walls_old[x_old+1,y_old]==1 or bombs_old[x_old+1,y_old]>=1 or crates_old[x_old+1,y_old]==1)  and action==1:
                custom_events.append(STEPPED_IN_BLOCKED_FIELD)
                        
            elif bomb_exploison_radius[x_old+1,y_old]>=1 and action==1:
                    custom_events.append(STEPPED_IN_BOMB_NEG)
                   
            elif explosions_old[x_old+1,y_old]==12 and new_state is None and action==1:
                    custom_events.append(STEPPED_IN_EXPLOSION)


                
            elif (walls_old[x_old,y_old-1]==1 or bombs_old[x_old,y_old-1]>=1 or crates_old[x_old,y_old-1]==1) and action==2:
                custom_events.append(STEPPED_IN_BLOCKED_FIELD)
            
            elif bomb_exploison_radius[x_old,y_old-1]>=1 and action==2:
                    custom_events.append(STEPPED_IN_BOMB_NEG)
                            
            elif explosions_old[x_old,y_old-1]==12 and new_state is None and action==2:
                    custom_events.append(STEPPED_IN_EXPLOSION)


                
            elif (walls_old[x_old,y_old+1]==1 or bombs_old[x_old,y_old+1]>=1 or crates_old[x_old,y_old+1]==1) and action==0:
                custom_events.append(STEPPED_IN_BLOCKED_FIELD)
            
            elif bomb_exploison_radius[x_old,y_old+1]>=1 and action==0:
                    custom_events.append(STEPPED_IN_BOMB_NEG)
              
            elif explosions_old[x_old,y_old+1]==12 and new_state is None and action==0:
                    custom_events.append(STEPPED_IN_EXPLOSION)


        
       
            if new_state is not None:
           
            
           
           
            
                position_new = new_state["self_info"]["position"]

           
                #berchnet neue Position, die nur dann ungleich der aus new_state wenn ein Gegner bestimmte Aktion
                #getaetigt hat und das wollen wir nicht beruecksichtigen, weil wir eben direct policy learning haben
                if action==3 and STEPPED_IN_BLOCKED_FIELD not in custom_events:
                       x_new=x_old-1
                       y_new=y_old

                       
                elif action==1 and STEPPED_IN_BLOCKED_FIELD not in custom_events:
                       x_new=x_old+1
                       y_new=y_old

                       
                elif action==2 and STEPPED_IN_BLOCKED_FIELD not in custom_events:
                       x_new=x_old
                       y_new=y_old-1

                       
                elif action==0 and STEPPED_IN_BLOCKED_FIELD not in custom_events:
                       x_new=x_old
                       y_new=y_old+1
                else:
                     x_new=x_old
                     y_new=y_old

           
                if coin_old[x_new,y_new]==1:
                        custom_events.append(COLLECTED_COIN)

            
            
           
               #berechnet min Distanz zur Muenze von alter Pos
                q=deque()
                q.append([x_old,y_old,0])
                path_list=[[x_old,y_old]]
                distanz_old=1e9
                predecessor={}
                x_coin=-1
                y_coin=-1
                while q:
                    cur_pos=q.popleft()
                    x=cur_pos[0]
                    y=cur_pos[1]
                    d=cur_pos[2]

                    if coin_old[x,y]==1:
                        x_coin=x
                        y_coin=y
                        distanz_old=d
                        break

                    if (bomb_exploison_radius[x,y+1]==0 or d>4) and (explosions_old[x,y+1]!=12 or d>0) and walls_old[x,y+1]==0 and (crates_old[x,y+1]==0) and [x,y+1] not in path_list:
                        q.append([x,y+1,d+1])
                        path_list.append([x,y+1])
                        predecessor[x,y+1]=(x,y)
                       
                    
                    if (bomb_exploison_radius[x,y-1]==0 or d>4) and (explosions_old[x,y-1]!=12 or d>0) and walls_old[x,y-1]==0 and (crates_old[x,y-1]==0) and [x,y-1] not in path_list:
                        q.append([x,y-1,d+1])
                        path_list.append([x,y-1])
                        predecessor[x,y-1]=(x,y)

                    if (bomb_exploison_radius[x+1,y]==0 or d>4) and (explosions_old[x+1,y]!=12 or d>0) and walls_old[x+1,y]==0 and (crates_old[x+1,y]==0) and [x+1,y] not in path_list:
                        q.append([x+1,y,d+1])
                        path_list.append([x+1,y])
                        predecessor[x+1,y]=(x,y)

                    if (bomb_exploison_radius[x-1,y]==0 or d>4) and (explosions_old[x-1,y]!=12 or d>0) and walls_old[x-1,y]==0 and (crates_old[x-1,y]==0) and [x-1,y] not in path_list:
                        q.append([x-1,y,d+1])
                        path_list.append([x-1,y])
                        predecessor[x-1,y]=(x,y)

                #Ziel nicht zu nah und Weg ex. ?
                if distanz_old>0 and distanz_old!=1e9:
                      #berechne Aktion bezueglich Weg ueber backtracking
                    x_cur=x_coin
                    y_cur=y_coin
                    while (x_cur,y_cur) != (x_old,y_old) and predecessor[x_cur,y_cur]!=(x_old,y_old):
                        (x_cur,y_cur)=predecessor[x_cur,y_cur]

                    possible_good_actions=[]
                    if (x_cur,y_cur) == (x_old-1,y_old):
                            possible_good_actions.append(3)

                    elif (x_cur,y_cur) == (x_old+1,y_old):
                            possible_good_actions.append(1)


                    elif (x_cur,y_cur) == (x_old,y_old-1):
                            possible_good_actions.append(2)

                    elif (x_cur,y_cur) == (x_old,y_old+1):
                            possible_good_actions.append(0)
                    else:
                        possible_good_actions.append(4)
                        #berechnet Weg ohne gerade berechnete Aktion zu benutzen
                    q=deque()
                    q.append([x_old,y_old,0])
                    path_list=[[x_old,y_old],[x_cur,y_cur]]
                    distanz_new=1e9
                    predecessor={}
                    while q:
                        cur_pos=q.popleft()
                        x=cur_pos[0]
                        y=cur_pos[1]
                        d=cur_pos[2]

                        if (x,y)==(x_coin,y_coin):
                        
                            distanz_new=d
                            break

                        if (bomb_exploison_radius[x,y+1]==0 or d>4) and (explosions_old[x,y+1]!=12 or d>0) and walls_old[x,y+1]==0 and (crates_old[x,y+1]==0) and [x,y+1] not in path_list:
                            q.append([x,y+1,d+1])
                            path_list.append([x,y+1])
                            predecessor[x,y+1]=(x,y)
                           
                       
                    
                        if (bomb_exploison_radius[x,y-1]==0 or d>4) and (explosions_old[x,y-1]!=12 or d>0) and walls_old[x,y-1]==0 and (crates_old[x,y-1]==0) and [x,y-1] not in path_list:
                            q.append([x,y-1,d+1])
                            path_list.append([x,y-1])
                            predecessor[x,y-1]=(x,y)
                        

                        if (bomb_exploison_radius[x+1,y]==0 or d>4) and (explosions_old[x+1,y]!=12 or d>0) and walls_old[x+1,y]==0 and (crates_old[x+1,y]==0) and [x+1,y] not in path_list:
                            q.append([x+1,y,d+1])
                            path_list.append([x+1,y])
                            predecessor[x+1,y]=(x,y)
                       

                        if (bomb_exploison_radius[x-1,y]==0 or d>4) and (explosions_old[x-1,y]!=12 or d>0) and walls_old[x-1,y]==0 and (crates_old[x-1,y]==0) and [x-1,y] not in path_list:
                            q.append([x-1,y,d+1])
                            path_list.append([x-1,y])
                            predecessor[x-1,y]=(x,y)
                      

                      


                             #Distanz ist gleich, dann berechne die andere gute Aktion
                    if distanz_new == distanz_old and distanz_new!=1e9:
                        x_cur=x_coin
                        y_cur=y_coin
                        while (x_cur,y_cur) != (x_old,y_old) and predecessor[x_cur,y_cur]!=(x_old,y_old):
                            (x_cur,y_cur)=predecessor[x_cur,y_cur]
                                
                        if (x_cur,y_cur) == (x_old-1,y_old):
                                possible_good_actions.append(3)

                        elif (x_cur,y_cur) == (x_old+1,y_old):
                                possible_good_actions.append(1)


                        elif (x_cur,y_cur) == (x_old,y_old-1):
                                possible_good_actions.append(2)

                        elif (x_cur,y_cur) == (x_old,y_old+1):
                                possible_good_actions.append(0)
                        else:
                            possible_good_actions.append(4)

                            #Agent hat passende Aktion gewaehlt um Weg zu verkuerzen?
                    if action in possible_good_actions:
                            custom_events.append(STEPPED_TOWARDS_COIN_POS)
                
                    else :                                                 
                            custom_events.append(STEPPED_TOWARDS_COIN_NEG)
               
           

        
             

             

        return custom_events

    def _shape_reward(self, events: list[str]) -> float:
        """
        Shape rewards here instead of in an Environment Wrapper in order to be more flexible (e.g. use this agent as proper component of the environment where no environment wrappers are possible)
        """
        reward_mapping = {
            
           
            
            STEPPED_IN_BLOCKED_FIELD:-10,
            STEPPED_IN_BOMB_NEG: -10,
            STEPPED_IN_EXPLOSION:-10,
            STEPPED_TOWARDS_COIN_POS:1,
            STEPPED_TOWARDS_COIN_NEG:-1,
            COLLECTED_COIN:5,
           
           
           
        }
         
       
        return sum([reward_mapping.get(event, 0) for event in events])
       
      