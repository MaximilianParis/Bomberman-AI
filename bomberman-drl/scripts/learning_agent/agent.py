from gymnasium.spaces import flatten
from bomberman_rl import LearningAgent, events as e
import numpy as np
import sys
import pickle
from collections import deque
from .agent_bomb_plant import Agent_Bomb_Plant
from .agent_bomb_escape import Agent_Bomb_Escape
from .agent_coin_collect import Agent_Coin_Collect




BOMB_ESCAPE_LENGTH_ONE_POS="BOMB_ESCAPE_LENGTH_ONE_POS"


BOMB_ESCAPE_LENGTH_TWO_POS="BOMB_ESCAPE_LENGTH_TWO_POS"


BOMB_ESCAPE_LENGTH_THREE_POS="BOMB_ESCAPE_LENGTH_THREE_POS"


BOMB_ESCAPE_LENGTH_FOUR_POS="BOMB_ESCAPE_LENGTH_FOUR_POS"

STEPPED_TOWARDS_TARGET_POS="STEPPED_TOWARDS_TARGET_POS"

Target_IN_RANGE_POS="Target_IN_RANGE_POS"

CAN_ESCAPE_OWN_BOMB_POS="CAN_ESCAPE_OWN_BOMB_POS"

STEPPED_TOWARDS_COIN_POS="STEPPED_TOWARDS_COIN_POS"

supervised_events=[BOMB_ESCAPE_LENGTH_ONE_POS,BOMB_ESCAPE_LENGTH_TWO_POS,BOMB_ESCAPE_LENGTH_THREE_POS,BOMB_ESCAPE_LENGTH_FOUR_POS,STEPPED_TOWARDS_TARGET_POS,
                   CAN_ESCAPE_OWN_BOMB_POS,STEPPED_TOWARDS_COIN_POS,Target_IN_RANGE_POS]






# Custom events

scenarios=["classic_no_train","classic_train","bomb-escape","bomb-plant","coin-collect"]
#berechnet kuerzeste Fluchtroute mittels DP
def find_rescue_route(timestamp_dead_marker,i,j,iteration=0,memo={},limit=10):
    #Zustand bereits besucht?
        if(((i,j,iteration) in memo)):
            return 
        #merke erstmal Anzahl FLuchtschritte
        memo[(i,j,iteration)]=iteration
       #prueft ob Agent beim erreichen des Feldes stirbt
        if(iteration>0 and timestamp_dead_marker[iteration,i,j]>=2):
            memo[(i,j,iteration)]=limit+1
        #prueft ob Agent in Gefahr ist wenn dieser das Feld erreicht
        elif(timestamp_dead_marker[iteration,i,j]==1 or (iteration==0 and timestamp_dead_marker[iteration,i,j]==4)):
           #damit min klappt
            memo[(i,j,iteration)]=limit+1
            if(iteration<limit):
                #betrachte benachbarte Felder, pruefe ob Fluchtroute ueber diese moeglich ist
                find_rescue_route(timestamp_dead_marker,i-1,j,iteration+1,memo,limit)
                #aktualisiere Fluchtroute bezueglich Laenge
                memo[(i,j,iteration)]=min(memo[(i,j,iteration)],memo[(i-1,j,iteration+1)])
            

                find_rescue_route(timestamp_dead_marker,i+1,j,iteration+1,memo,limit)
                memo[(i,j,iteration)]=min(memo[(i,j,iteration)],memo[(i+1,j,iteration+1)])
           

                find_rescue_route(timestamp_dead_marker,i,j-1,iteration+1,memo,limit)
                memo[(i,j,iteration)]=min(memo[(i,j,iteration)],memo[(i,j-1,iteration+1)])
           

                find_rescue_route(timestamp_dead_marker,i,j+1,iteration+1,memo,limit)
                memo[(i,j,iteration)]=min(memo[(i,j,iteration)],memo[(i,j+1,iteration+1)])
           

                find_rescue_route(timestamp_dead_marker,i,j,iteration+1,memo,limit)
                memo[(i,j,iteration)]=min(memo[(i,j,iteration)],memo[(i,j,iteration+1)])
           
                #berechne Aktion bezueglich optimaler FLuchtroute fuer ersten Aufruf
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
            



    #berechnet Explosionsradius einer Bombe, Mittelpunkt der Bombe ist (i,j)
def Compute_Bomb_Exlpoison_Radius_Local(walls,i,j):
    bomb_blast = [(i, j)]
  
     #Streifen links
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

   #berechnet Explosionsradius von Bomben
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


def compute_timestamp_dead_marker(opponents,bombs,explosions,crates,walls):
    time_limit_escape=10
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

    #wandle in brauchbare Daten um
    explosion_xy=[(i,j,explosions[i,j]-10) for i in range(0,limit_x+1) for j in range(0,limit_y+1) if explosions[i,j]>=11]
    bomb_xys = [(i, j,(4-bombs[i,j])+1) for i in range(0,17) for j in range(0,17) if bombs[i,j]>=1]
      
      #markiere Zeitschritte wo Agent durch Explosion stirbt
    for (cur_x,cur_y,explosion_timer) in explosion_xy:
                      
        for t in range(0,explosion_timer):
            timestamp_dead_marker[t,cur_x,cur_y]=4
    #print(bombs)
    #print(bomb_xys)

      #markiere Zeitschritte wo Agent durch Bombe in Gefahr ist und nachfolgenden Explosion stirbt
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

    #markiere Waende und Gegner als Todespunkte, gleiches fuer Crates beachten aber das diese auch noch
    #verschwinden koennen nach einer Explosion
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

class Agent(LearningAgent):
    """
    Sticking to the ``LearningAgent`` interface is optional.
    It enables your agent to **learn** as proper part of the environment (``/src/bomberman_rl/envs/agent_code/<agent>``) in order to enable Self-Play.
    The example training loop in main.py supports this interface as well by calling the respective callbacks.
    (Demonstration only - do not inherit)
    """
    def __init__(self,num=0):
        #waehle Szenario
        self.cur_scenario=scenarios[num]
        self.setup()
        self.setup_training()
        

    def setup(self):
        """
        Before episode. Use this to setup action related state that is required to act on the environment.
        """
        #Gewichte Laden oder nicht
        self.load=False
        #zaehlt Episoden
        self.cnt=0
        #erstellt Agenten je nach Szenario
        if self.cur_scenario=="classic_train":
            #nach self.limit_agent_bomb_escape Episoden wird beim replay Memory die Anzahl an Zustandaktionspaaren
            #jeder Kategrie nach Perfomrance die in den batch kommt aktualisiert
             self.limit_agent_bomb_escape=20
             self.limit_agent_bomb_plant=20
             self.limit_agent_coin_collect=20
             #erstelle Agenten
             self.agent_bomb_escape= Agent_Bomb_Escape(self.load,True)
             self.agent_bomb_plant = Agent_Bomb_Plant(self.load,True)
             self.agent_coin_collect=Agent_Coin_Collect(self.load,True)

        elif self.cur_scenario=="classic_no_train":
             self.limit_agent_bomb_escape=20
             self.limit_agent_bomb_plant=20
             self.limit_agent_coin_collect=20
             self.agent_bomb_escape= Agent_Bomb_Escape(True,False)
             self.agent_bomb_plant = Agent_Bomb_Plant(True,False)
             self.agent_coin_collect=Agent_Coin_Collect(True,False)

        elif self.cur_scenario=="bomb-escape":
             self.limit_agent_bomb_escape=100
             self.agent_bomb_escape= Agent_Bomb_Escape(self.load,True)

        elif self.cur_scenario=="bomb-plant":
            self.limit_agent_bomb_plant=20
            self.agent_bomb_plant = Agent_Bomb_Plant(self.load,True)

        elif self.cur_scenario=="coin-collect":
            self.limit_agent_coin_collect=20
            self.agent_coin_collect=Agent_Coin_Collect(self.load,True)

        
     
       
    def act(self, state, **kwargs) -> int:
        #waehle passendes Act fuer jeweiliges Szenario
        if self.cur_scenario=="classic_train":
            return self.act_classic(state)

        elif self.cur_scenario=="classic_no_train":
            return self.act_classic(state)

        elif self.cur_scenario=="bomb-escape":
            self.last_active="bomb-escape"
            return self.agent_bomb_escape.act(state)

        elif self.cur_scenario=="bomb-plant":
            
            return self.act_bomb_plant(state)

        elif self.cur_scenario=="coin-collect":
            self.last_active="coin_collect"   
            return self.agent_coin_collect.act(state)

        #Act fuer Bombenleger Szenario
    def act_bomb_plant(self, state, **kwargs) -> int:
        """
        Before step. Return action based on state.
        :param state: The state of the environment.
        """

        walls = state["walls"]
        position = state["self_info"]["position"]
        bombs_left = state["self_info"]["bombs_left"]
        score= state["self_info"]["score"]
        opponents=state["opponents_pos"]
        crates=state["crates"]
        explosions=state["explosions"]
        coins=state["coins"]
        bombs = state["bombs"]
                         

        bomb_exploison_radius=np.zeros((17,17))
        Compute_Bomb_Exlpoison_Radius(bomb_exploison_radius,bombs,walls)
        #cnt_opponents=0

       # for a in range(0,17):
        #    for b in range(0,17):
        #      if opponents[a,b]==1:
         #         cnt_opponents+=1

       # cnt_crates=0

       # for a in range(0,17):
        #    for b in range(0,17):
        #      if crates[a,b]==1:
        #          cnt_crates+=1

        for a in range(0,17):
            for b in range(0,17):
              if position[a,b]==1:
                  (x_old,y_old)=(a,b)

         #falls in Bombenradius lass Experten  entkommen
        if bomb_exploison_radius[x_old,y_old]>=1:
            self.last_active="bomb-escape"
            timestamp_dead_marker=compute_timestamp_dead_marker(state["opponents_pos"],bombs,explosions,crates,walls)
            memo={}
            limit=5
            return find_rescue_route(timestamp_dead_marker,x_old,y_old,0,memo,limit)
        else:

           #berechne kuerzesten Weg zu Gegner und Kiste
            q=deque()
            q.append([x_old,y_old,0])
            path_list=[[x_old,y_old]]
            distanz_crate=0
            x_crate=-1
            y_crate=-1
            while q:
                cur_pos=q.popleft()
                x=cur_pos[0]
                y=cur_pos[1]
                d=cur_pos[2]

                if crates[x,y]==1:
                    x_crate=x
                    y_crate=y
                    distanz_crate=d
                    break
                #falls Feld frei ist
                #Bombe ist bei d>4 nicht mehr da deswegen egal, gleiches hier fuer Explosion
                if (bomb_exploison_radius[x,y+1]==0 or d>4) and (explosions[x,y+1]!=12 or d>0) and walls[x,y+1]==0 and [x,y+1] not in path_list:
                    q.append([x,y+1,d+1])
                    path_list.append([x,y+1])
                    
                    
                if (bomb_exploison_radius[x,y-1]==0 or d>4) and (explosions[x,y-1]!=12 or d>0) and walls[x,y-1]==0 and [x,y-1] not in path_list:
                    q.append([x,y-1,d+1])
                    path_list.append([x,y-1])
                    

                if (bomb_exploison_radius[x+1,y]==0 or d>4) and (explosions[x+1,y]!=12 or d>0) and walls[x+1,y]==0 and [x+1,y] not in path_list:
                    q.append([x+1,y,d+1])
                    path_list.append([x+1,y])
                    

                if (bomb_exploison_radius[x-1,y]==0 or d>4) and (explosions[x-1,y]!=12 or d>0) and walls[x-1,y]==0 and [x-1,y] not in path_list:
                    q.append([x-1,y,d+1])
                    path_list.append([x-1,y])
                    

                

            q=deque()
            q.append([x_old,y_old,0])
            path_list=[[x_old,y_old]]
            distanz_enemy=1e9
            x_enemy=-1
            y_enemy=-1
            while q:
                cur_pos=q.popleft()
                x=cur_pos[0]
                y=cur_pos[1]
                d=cur_pos[2]

                
                if (bomb_exploison_radius[x,y+1]==0 or d>4) and (explosions[x,y+1]!=12 or d>0) and walls[x,y+1]==0 and crates[x,y+1]==0 and [x,y+1] not in path_list:
                    q.append([x,y+1,d+1])
                    path_list.append([x,y+1])
                    
                    
                if (bomb_exploison_radius[x,y-1]==0 or d>4) and (explosions[x,y-1]!=12 or d>0) and walls[x,y-1]==0 and crates[x,y-1]==0 and [x,y-1] not in path_list:
                    q.append([x,y-1,d+1])
                    path_list.append([x,y-1])
                    

                if (bomb_exploison_radius[x+1,y]==0 or d>4) and (explosions[x+1,y]!=12 or d>0) and walls[x+1,y]==0 and crates[x+1,y]==0 and [x+1,y] not in path_list:
                    q.append([x+1,y,d+1])
                    path_list.append([x+1,y])
                    

                if (bomb_exploison_radius[x-1,y]==0 or d>4) and (explosions[x-1,y]!=12 or d>0) and walls[x-1,y]==0 and crates[x-1,y]==0 and [x-1,y] not in path_list:
                    q.append([x-1,y,d+1])
                    path_list.append([x-1,y])
                    

                if opponents[x,y]==1:
                    x_enemy=x
                    y_enemy=y
                    distanz_enemy=d
                    break

            #schalte je nach Distanz zwischen Agenten
            if distanz_enemy<=distanz_crate+8:
                self.target_x=x_enemy
                self.target_y=y_enemy
                self.target="ENEMY"
                self.last_active="bomb-plant"
                return self.agent_bomb_plant.act(state,x_enemy,y_enemy,"ENEMY")
            elif distanz_enemy>distanz_crate+8:
                self.target_x=x_crate
                self.target_y=y_crate
                self.target="CRATE"
                self.last_active="bomb-plant"
                return self.agent_bomb_plant.act(state,x_crate,y_crate,"CRATE")
            else:
                 return 4


    def act_classic(self, state, **kwargs) -> int:
        """
        Before step. Return action based on state.
        :param state: The state of the environment.
        """

        walls = state["walls"]
        position = state["self_info"]["position"]
        bombs_left = state["self_info"]["bombs_left"]
        score= state["self_info"]["score"]
        opponents=state["opponents_pos"]
        crates=state["crates"]
        explosions=state["explosions"]
        coins=state["coins"]
        bombs = state["bombs"]
                         

        bomb_exploison_radius=np.zeros((17,17))
        Compute_Bomb_Exlpoison_Radius(bomb_exploison_radius,bombs,walls)
        

        for a in range(0,17):
            for b in range(0,17):
              if position[a,b]==1:
                  (x_old,y_old)=(a,b)


        if bomb_exploison_radius[x_old,y_old]>=1:
            self.last_active="bomb-escape"
            return self.agent_bomb_escape.act(state)
        else:

           #berechne kuerzesten Weg zu Kiste,Muenze,Gegner mittels bfs
            q=deque()
            q.append([x_old,y_old,0])
            path_list=[[x_old,y_old]]
            distanz_crate=1e9
            x_crate=-1
            y_crate=-1
            while q:
                cur_pos=q.popleft()
                x=cur_pos[0]
                y=cur_pos[1]
                d=cur_pos[2]

                if crates[x,y]==1:
                    x_crate=x
                    y_crate=y
                    distanz_crate=d
                    break

                if (bomb_exploison_radius[x,y+1]==0 or d>4) and (explosions[x,y+1]!=12 or d>0) and walls[x,y+1]==0 and [x,y+1] not in path_list:
                    q.append([x,y+1,d+1])
                    path_list.append([x,y+1])
                    
                    
                if (bomb_exploison_radius[x,y-1]==0 or d>4) and (explosions[x,y-1]!=12 or d>0) and walls[x,y-1]==0 and [x,y-1] not in path_list:
                    q.append([x,y-1,d+1])
                    path_list.append([x,y-1])
                    

                if (bomb_exploison_radius[x+1,y]==0 or d>4) and (explosions[x+1,y]!=12 or d>0) and walls[x+1,y]==0 and [x+1,y] not in path_list:
                    q.append([x+1,y,d+1])
                    path_list.append([x+1,y])
                    

                if (bomb_exploison_radius[x-1,y]==0 or d>4) and (explosions[x-1,y]!=12 or d>0) and walls[x-1,y]==0 and [x-1,y] not in path_list:
                    q.append([x-1,y,d+1])
                    path_list.append([x-1,y])
                    

                

            q=deque()
            q.append([x_old,y_old,0])
            path_list=[[x_old,y_old]]
            distanz_enemy=1e9
            x_enemy=-1
            y_enemy=-1
            while q:
                cur_pos=q.popleft()
                x=cur_pos[0]
                y=cur_pos[1]
                d=cur_pos[2]

                if (bomb_exploison_radius[x,y+1]==0 or d>4)  and (explosions[x,y+1]!=12 or d>0) and walls[x,y+1]==0 and crates[x,y+1]==0 and [x,y+1] not in path_list:
                    q.append([x,y+1,d+1])
                    path_list.append([x,y+1])
                    
                    
                if (bomb_exploison_radius[x,y-1]==0 or d>4) and (explosions[x,y-1]!=12 or d>0) and walls[x,y-1]==0 and crates[x,y-1]==0 and [x,y-1] not in path_list:
                    q.append([x,y-1,d+1])
                    path_list.append([x,y-1])
                    

                if (bomb_exploison_radius[x+1,y]==0 or d>4) and (explosions[x+1,y]!=12 or d>0) and walls[x+1,y]==0 and crates[x+1,y]==0 and [x+1,y] not in path_list:
                    q.append([x+1,y,d+1])
                    path_list.append([x+1,y])
                    

                if (bomb_exploison_radius[x-1,y]==0 or d>4) and (explosions[x-1,y]!=12 or d>0) and walls[x-1,y]==0 and crates[x-1,y]==0 and [x-1,y] not in path_list:
                    q.append([x-1,y,d+1])
                    path_list.append([x-1,y])
                    

                if opponents[x,y]==1:
                    x_enemy=x
                    y_enemy=y
                    distanz_enemy=d
                    break

            q=deque()
            q.append([x_old,y_old,0])
            path_list=[[x_old,y_old]]
            distanz_coin=1e9
            x_coin=-1
            y_coin=-1
            while q:
                cur_pos=q.popleft()
                x=cur_pos[0]
                y=cur_pos[1]
                d=cur_pos[2]

                if (bomb_exploison_radius[x,y+1]==0 or d>4) and (explosions[x,y+1]!=12 or d>0) and walls[x,y+1]==0 and crates[x,y+1]==0 and [x,y+1] not in path_list:
                    q.append([x,y+1,d+1])
                    path_list.append([x,y+1])
                    
                    
                if (bomb_exploison_radius[x,y-1]==0 or d>4) and (explosions[x,y-1]!=12 or d>0) and walls[x,y-1]==0 and crates[x,y-1]==0 and [x,y-1] not in path_list:
                    q.append([x,y-1,d+1])
                    path_list.append([x,y-1])
                    

                if (bomb_exploison_radius[x+1,y]==0 or d>4) and (explosions[x+1,y]!=12 or d>0) and walls[x+1,y]==0 and crates[x+1,y]==0 and [x+1,y] not in path_list:
                    q.append([x+1,y,d+1])
                    path_list.append([x+1,y])
                    

                if (bomb_exploison_radius[x-1,y]==0 or d>4) and (explosions[x-1,y]!=12 or d>0) and walls[x-1,y]==0 and crates[x-1,y]==0 and [x-1,y] not in path_list:
                    q.append([x-1,y,d+1])
                    path_list.append([x-1,y])
                    

                if coins[x,y]==1:
                    x_coin=x
                    y_coin=y
                    distanz_coin=d
                    break

               
       
                   #schalte je nach Distanz zwischen Agenten
            if distanz_coin<=5:
                self.last_active="coin_collect"   
                return self.agent_coin_collect.act(state)
            elif distanz_coin<=distanz_enemy and distanz_coin<=distanz_crate+8:
                  self.last_active="coin_collect"   
                  return self.agent_coin_collect.act(state)
             
            elif distanz_enemy<=distanz_crate+8:
                self.last_active="bomb-plant"
                self.target_x=x_enemy
                self.target_y=y_enemy
                #print("ENEMY: ", self.target_x, self.target_y)
                self.target="ENEMY"
                return self.agent_bomb_plant.act(state,x_enemy,y_enemy,"ENEMY")
            elif distanz_crate!=1e9:
                self.last_active="bomb-plant"
                self.target_x=x_crate
                self.target_y=y_crate
                self.target="CRATE"
                #print("CRATE: ", self.target_x, self.target_y)
                return self.agent_bomb_plant.act(state,x_crate,y_crate,"CRATE")
            elif distanz_coin!=1e9:
                 self.last_active="coin_collect"   
                 return self.agent_coin_collect.act(state)
            elif distanz_enemy!=1e9:
                self.last_active="bomb-plant"
                self.target_x=x_enemy
                self.target_y=y_enemy
                self.target="ENEMY"
                #print("ENEMY: ", self.target_x, self.target_y)
                return self.agent_bomb_plant.act(state,x_enemy,y_enemy,"ENEMY")
            else:
                 return 4


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
        events
    ):
        """
        After step in environment (optional). Use this e.g. for model training.

        :param old_state: Old state of the environment.
        :param self_action: Performed action.
        :param new_state: New state of the environment.
        :param events: Events that occurred during step. These might be used for Reward Shaping.
        """
        #schalte zwischen Szenarien um und lass die jeweilige Funktion wissen ob diese zuletzt dran war
        reward=0
        if self.cur_scenario=="classic_train":
            
             reward+= self.agent_bomb_escape.game_events_occurred(old_state,self_action,new_state,events,self.last_active=="bomb-escape")
     
             reward+= self.agent_bomb_plant.game_events_occurred(old_state,self_action,new_state,events,self.target_x,self.target_y,self.target,self.last_active=="bomb-plant")
                
             reward+=self.agent_coin_collect.game_events_occurred(old_state,self_action,new_state,events,self.last_active=="coin_collect")

        elif self.cur_scenario=="classic_no_train":
            # pass
           
             reward+= self.agent_bomb_escape.game_events_occurred(old_state,self_action,new_state,events,self.last_active=="bomb-escape")
     
             reward+= self.agent_bomb_plant.game_events_occurred(old_state,self_action,new_state,events,self.target_x,self.target_y,self.target,self.last_active=="bomb-plant")
                
             reward+=self.agent_coin_collect.game_events_occurred(old_state,self_action,new_state,events,self.last_active=="coin_collect")

        elif self.cur_scenario=="bomb-escape":
           
               reward+= self.agent_bomb_escape.game_events_occurred(old_state,self_action,new_state,events,self.last_active=="bomb-escape")

        elif self.cur_scenario=="bomb-plant":
           
             reward+= self.agent_bomb_plant.game_events_occurred(old_state,self_action,new_state,events,self.target_x,self.target_y,self.target,self.last_active=="bomb-plant")

        elif self.cur_scenario=="coin-collect":
          
             reward+=self.agent_coin_collect.game_events_occurred(old_state,self_action,new_state,events,self.last_active=="coin_collect")
      
        
             
        return reward


    def end_of_round(self):
        self.cnt+=1
        """
        After episode ended (optional). Use this e.g. for model training and saving.
        """
        #schalte zwischen Szenarien um und lass die jeweilige Funktion wissen ob Batchgroesse angepasst werden soll
        if self.cur_scenario=="classic_train":
            
             self.agent_bomb_escape.end_of_round((self.cnt%self.limit_agent_bomb_escape)==0)
             self.agent_bomb_plant.end_of_round((self.cnt%self.limit_agent_bomb_plant)==0)
             self.agent_coin_collect.end_of_round((self.cnt%self.limit_agent_coin_collect)==0)

        elif self.cur_scenario=="classic_no_train":
           pass
            # self.agent_bomb_escape.end_of_round((self.cnt%self.limit_agent_bomb_escape)==0)
             #self.agent_bomb_plant.end_of_round((self.cnt%self.limit_agent_bomb_plant)==0)
             #self.agent_coin_collect.end_of_round((self.cnt%self.limit_agent_coin_collect)==0)

        elif self.cur_scenario=="bomb-escape":
           
             self.agent_bomb_escape.end_of_round((self.cnt%self.limit_agent_bomb_escape)==0)

        elif self.cur_scenario=="bomb-plant":
           
           self.agent_bomb_plant.end_of_round((self.cnt%self.limit_agent_bomb_plant)==0)

        elif self.cur_scenario=="coin-collect":
          
           self.agent_coin_collect.end_of_round((self.cnt%self.limit_agent_coin_collect)==0)

    def save_supervised_state_action_pairs(self):
        items_bomb_escape=self.agent_bomb_escape.save_supervised_state_action_pairs_bomb_escape()

        items_bomb_plant=self.agent_bomb_plant.save_supervised_state_action_pairs_bomb_plant()

        items_coin_collect=self.agent_coin_collect.save_supervised_state_action_pairs_coin_collect()
                             

        for cur in items_bomb_escape:
            with open("zustandaktionspaare_"+cur[0]+".pkl", "wb") as f:
                 pickle.dump(cur[1][:3000], f)

        for cur in items_bomb_plant:
            with open("zustandaktionspaare_"+cur[0]+".pkl", "wb") as f:
                 pickle.dump(cur[1][:3000], f)


        for cur in items_coin_collect:
           with open("zustandaktionspaare_"+cur[0]+".pkl", "wb") as f:
                 pickle.dump(cur[1][:3000], f)

      




   

    
       
      