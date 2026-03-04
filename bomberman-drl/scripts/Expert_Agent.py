import numpy as np
import random
from collections import deque
import pickle



def find_rescue_route(timestamp_dead_marker,i,j,iteration=0,memo={},limit=6):
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
          if walls[i,j]==1:
                    for t in range(0,time_limit_escape):
                        timestamp_dead_marker[t,i,j]=4
          elif opponents[i,j]==1:
              for t in range(0,time_limit_escape):
                        timestamp_dead_marker[t,i,j]=5
          elif crates[i,j]==1:
                for t in range(0,time_limit_escape):
                    if timestamp_dead_marker[t,i,j]==4:
                        break
                    elif timestamp_dead_marker[t,i,j]==1:
                        timestamp_dead_marker[t,i,j]=3
                    else:
                        timestamp_dead_marker[t,i,j]=2

    return timestamp_dead_marker

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory_state =[]
        self.memory_action =[]
        self.capacity=capacity
        self.batch_size=20

    def push(self, state, action):
        """Save a transition"""
        #entferne zu Alte Zustandaktionspaare wenn voll
        if self.capacity==len(self.memory_state):
            self.memory_state.pop(0)
            self.memory_action.pop(0)

        self.memory_state.append(state)
        self.memory_action.append(action)

    def sample(self):
        indices = range(len(self.memory_state))
        sampled_indices = random.sample(indices, k=self.batch_size)
                    
        sampled_states = [self.memory_state[i] for i in sampled_indices]
        sampled_actions = [self.memory_action[i] for i in sampled_indices]

        return sampled_states,sampled_actions

    def __len__(self):
        return len(self.memory_state)

class Expert_Agent:

    def __init__(self):
        #[BOMB_ESCAPE_LENGTH_ONE_POS,BOMB_ESCAPE_LENGTH_TWO_POS,BOMB_ESCAPE_LENGTH_THREE_POS,BOMB_ESCAPE_LENGTH_FOUR_POS,STEPPED_TOWARDS_TARGET_POS,CAN_ESCAPE_OWN_BOMB_POS,Target_IN_RANGE_POS]
        self.replay_memories=[ReplayMemory(3000),ReplayMemory(3000),ReplayMemory(3000),ReplayMemory(3000),ReplayMemory(3000),ReplayMemory(3000)]



    def write_experience_to_file(self):
        for cur in range(0,len(self.replay_memories)):
            with open("zustandaktionspaare_"+str(cur)+".pkl", "wb") as f:
                 pickle.dump(self.replay_memories[cur], f)

    def back_track(self,x_old,y_old,x_target,y_target,predecessor):
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

        return action

    def plant_bomb(self,state,x_old,y_old,x_target,y_target,distanz,predecessor):
        walls = state["walls"]
        position = state["self_info"]["position"]
        bombs_left = state["self_info"]["bombs_left"]
        score= state["self_info"]["score"]
        opponents=state["opponents_pos"]
        crates=state["crates"]
        explosions=state["explosions"]
        coins=state["coins"]
        bombs = state["bombs"]

        if x_target==-1:
            action=4
        else:

            Bomb_Exlpoison_Radius_Local=Compute_Bomb_Exlpoison_Radius_Local(walls,x_old,y_old)
            timestamp_dead_marker=compute_timestamp_dead_marker_with_extra_bomb(state["opponents_pos"],bombs,explosions,crates,walls,x_old,y_old)
            memo={}
            limit=5
            find_rescue_route(timestamp_dead_marker,x_old,y_old,0,memo,limit)
            bombs_left=state["self_info"]["bombs_left"]
            self.target_distanz=abs(x_old-x_target)+abs(y_old-y_target)
               
            if distanz<=2 and (x_target,y_target) in Bomb_Exlpoison_Radius_Local and memo[(x_old,y_old,0)]<=limit and bombs_left>=1:
                    action=5
                    self.replay_memories[5].push(state,action)
                   
                     


            elif distanz>=2:
                action=self.back_track(x_old,y_old,x_target,y_target,predecessor)
                if(self.target_distanz<=15):
                    self.replay_memories[4].push(state,action)

            else:
                action=4

        return action



    def collect_coin(self,state,x_old,y_old,x_target,y_target,distanz,predecessor):
        self.target_coin=True
        walls = state["walls"]
        position = state["self_info"]["position"]
        bombs_left = state["self_info"]["bombs_left"]
        score= state["self_info"]["score"]
        opponents=state["opponents_pos"]
        crates=state["crates"]
        explosions=state["explosions"]
        coins=state["coins"]
        bombs = state["bombs"]
        self.target_distanz=abs(x_old-x_target)+abs(y_old-y_target)
        if x_target==-1:
            action=4
        else:
             
            if distanz>=1:
                action=self.back_track(x_old,y_old,x_target,y_target,predecessor)
                if(self.target_distanz<=15):
                    self.replay_memories[4].push(state,action)
            else:
                action=4

        return action

    def act(self,state):
            self.target_coin=False
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

            timestamp_dead_marker=compute_timestamp_dead_marker(opponents,bombs,explosions,crates,walls)
            if bomb_exploison_radius[x_old,y_old]>=1:
                 memo={}
                 
                 action=find_rescue_route(timestamp_dead_marker,x_old,y_old,0,memo)
            
                 if   memo[(x_old,y_old,0)]==1:
                        self.replay_memories[0].push(state,action)
                 elif memo[(x_old,y_old,0)]==2:
                        self.replay_memories[1].push(state,action)
                         
                 elif memo[(x_old,y_old,0)]==3:
                        self.replay_memories[2].push(state,action)
                         
                 elif memo[(x_old,y_old,0)]==4:
                        self.replay_memories[3].push(state,action)
                # else:
                    # print("dead")

                 return action
            else:

               #berechne kuerzesten Weg zu Kiste,Muenze,Gegner mittels bfs
                predecessor_crate={}
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

                    if (d>4 or timestamp_dead_marker[d+1,x,y+1]==0 or timestamp_dead_marker[d+1,x,y+1]==2) and walls[x,y+1]==0 and [x,y+1] not in path_list:
                        q.append([x,y+1,d+1])
                        path_list.append([x,y+1])
                        predecessor_crate[x,y+1]=(x,y)
                    
                    
                    if (d>4 or timestamp_dead_marker[d+1,x,y-1]==0 or timestamp_dead_marker[d+1,x,y-1]==2) and walls[x,y-1]==0 and [x,y-1] not in path_list:
                        q.append([x,y-1,d+1])
                        path_list.append([x,y-1])
                        predecessor_crate[x,y-1]=(x,y)
                    

                    if (d>4 or timestamp_dead_marker[d+1,x+1,y]==0 or timestamp_dead_marker[d+1,x+1,y]==2) and walls[x+1,y]==0 and [x+1,y] not in path_list:
                        q.append([x+1,y,d+1])
                        path_list.append([x+1,y])
                        predecessor_crate[x+1,y]=(x,y)
                    

                    if (d>4 or timestamp_dead_marker[d+1,x-1,y]==0 or timestamp_dead_marker[d+1,x-1,y]==2) and walls[x-1,y]==0 and [x-1,y] not in path_list:
                        q.append([x-1,y,d+1])
                        path_list.append([x-1,y])
                        predecessor_crate[x-1,y]=(x,y)
                    

                
                predecessor_enemy={}
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

                    if (d>4 or timestamp_dead_marker[d+1,x,y+1]==0 or timestamp_dead_marker[d+1,x,y+1]==5) and walls[x,y+1]==0 and crates[x,y+1]==0 and [x,y+1] not in path_list:
                        q.append([x,y+1,d+1])
                        path_list.append([x,y+1])
                        predecessor_enemy[x,y+1]=(x,y)
                    
                    
                    if (d>4 or timestamp_dead_marker[d+1,x,y-1]==0 or timestamp_dead_marker[d+1,x,y-1]==5) and walls[x,y-1]==0 and crates[x,y-1]==0 and [x,y-1] not in path_list:
                        q.append([x,y-1,d+1])
                        path_list.append([x,y-1])
                        predecessor_enemy[x,y-1]=(x,y)
                    

                    if (d>4 or timestamp_dead_marker[d+1,x+1,y]==0 or timestamp_dead_marker[d+1,x+1,y]==5) and walls[x+1,y]==0 and crates[x+1,y]==0 and [x+1,y] not in path_list:
                        q.append([x+1,y,d+1])
                        path_list.append([x+1,y])
                        predecessor_enemy[x+1,y]=(x,y)
                    

                    if (d>4 or timestamp_dead_marker[d+1,x-1,y]==0 or timestamp_dead_marker[d+1,x-1,y]==5) and walls[x-1,y]==0 and crates[x-1,y]==0 and [x-1,y] not in path_list:
                        q.append([x-1,y,d+1])
                        path_list.append([x-1,y])
                        predecessor_enemy[x-1,y]=(x,y)
                    

                    if opponents[x,y]==1:
                        x_enemy=x
                        y_enemy=y
                        distanz_enemy=d
                        break



                predecessor_coin={}
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

                    if (d>4 or timestamp_dead_marker[d+1,x,y+1]==0)  and walls[x,y+1]==0 and crates[x,y+1]==0 and [x,y+1] not in path_list:
                        q.append([x,y+1,d+1])
                        path_list.append([x,y+1])
                        predecessor_coin[x,y+1]=(x,y)
                    
                    
                    if (d>4 or timestamp_dead_marker[d+1,x,y-1]==0)  and walls[x,y-1]==0 and crates[x,y-1]==0 and [x,y-1] not in path_list:
                        q.append([x,y-1,d+1])
                        path_list.append([x,y-1])
                        predecessor_coin[x,y-1]=(x,y)
                    

                    if (d>4 or timestamp_dead_marker[d+1,x+1,y]==0)  and walls[x+1,y]==0 and crates[x+1,y]==0 and [x+1,y] not in path_list:
                        q.append([x+1,y,d+1])
                        path_list.append([x+1,y])
                        predecessor_coin[x+1,y]=(x,y)
                    

                    if (d>4 or timestamp_dead_marker[d+1,x-1,y]==0) and walls[x-1,y]==0 and crates[x-1,y]==0 and [x-1,y] not in path_list:
                        q.append([x-1,y,d+1])
                        path_list.append([x-1,y])
                        predecessor_coin[x-1,y]=(x,y)
                    

                    if coins[x,y]==1:
                        x_coin=x
                        y_coin=y
                        distanz_coin=d
                        break

               
       
                      
                if distanz_coin<=5:
                      self.target_x,self.target_y=x_coin,y_coin
                      return self.collect_coin(state,x_old,y_old,x_coin,y_coin,distanz_coin,predecessor_coin)

                elif distanz_coin<=distanz_enemy and distanz_coin!=1e9:
                      self.target_x,self.target_y=x_coin,y_coin
                      return self.collect_coin(state,x_old,y_old,x_coin,y_coin,distanz_coin,predecessor_coin)
             
                elif distanz_enemy<distanz_coin:
                      self.target_x,self.target_y=x_enemy,y_enemy
                      return self.plant_bomb(state,x_old,y_old,x_enemy,y_enemy,distanz_enemy,predecessor_enemy)

                elif distanz_crate!=1e9:
                      self.target_x,self.target_y=x_crate,y_crate
                      return self.plant_bomb(state,x_old,y_old,x_crate,y_crate,distanz_crate,predecessor_crate)

                elif distanz_coin!=1e9:
                      self.target_x,self.target_y=x_coin,y_coin
                      return self.collect_coin(state,x_old,y_old,x_coin,y_coin,distanz_coin,predecessor_coin)

                elif distanz_enemy!=1e9:
                      self.target_x,self.target_y=x_enemy,y_enemy
                      return self.plant_bomb(state,x_old,y_old,x_enemy,y_enemy,distanz_enemy,predecessor_enemy)

                else:
                      return 4