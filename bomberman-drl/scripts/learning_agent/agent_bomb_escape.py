from gymnasium.spaces import flatten
from bomberman_rl import LearningAgent, events as e
import numpy as np
import sys
import pickle
from .learning_bomb_escape import Model

# Custom events

STEPPED_IN_BLOCKED_FIELD="STEPPED_IN_BLOCKED_FIELD"

ESCAPED_BOMB= "ESCAPED_BOMB"
STEPPED_IN_BOMB="STEPPED_IN_BOMB"
USED_OPT_BOMB_ESCAPE="USED_OPT_BOMB_ESCAPE"

WRONG_BOMB_ESCAPE="WRONG_BOMB_ESCAPE"

STEPPED_IN_BOMB_NEG="STEPPED_IN_BOMB_NEG"

BOMB_ESCAPE_LENGTH_ONE_POS="BOMB_ESCAPE_LENGTH_ONE_POS"
BOMB_ESCAPE_LENGTH_ONE_NEG="BOMB_ESCAPE_LENGTH_ONE_NEG"

BOMB_ESCAPE_LENGTH_TWO_POS="BOMB_ESCAPE_LENGTH_TWO_POS"
BOMB_ESCAPE_LENGTH_TWO_NEG="BOMB_ESCAPE_LENGTH_TWO_NEG"

BOMB_ESCAPE_LENGTH_THREE_POS="BOMB_ESCAPE_LENGTH_THREE_POS"
BOMB_ESCAPE_LENGTH_THREE_NEG="BOMB_ESCAPE_LENGTH_THREE_NEG"

BOMB_ESCAPE_LENGTH_FOUR_POS="BOMB_ESCAPE_LENGTH_FOUR_POS"
BOMB_ESCAPE_LENGTH_FOUR_NEG="BOMB_ESCAPE_LENGTH_FOUR_NEG"




STAYED_SAFE="STAYED_SAFE"




                       
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



class Agent_Bomb_Escape(LearningAgent):
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



    def save_supervised_state_action_pairs_bomb_escape(self):
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
        for a in range(0,17):
            for b in range(0,17):
              if position_old[a,b]==1:
                  (x_old,y_old)=(a,b)
        
        memo={}
        limit=10
        timestamp_dead_marker=compute_timestamp_dead_marker(old_state["opponents_pos"],bombs_old,explosions_old,crates_old,walls_old)
        find_rescue_route(timestamp_dead_marker,x_old,y_old,0,memo,limit)

        #Agent ist in blockiertes Feld(Bombe,Wand,Explsosion,Kiste) gelaufen?
        if timestamp_dead_marker[(1,x_old-1,y_old)]>=2 and action==3:
                custom_events.append(STEPPED_IN_BLOCKED_FIELD)
                
        elif timestamp_dead_marker[(1,x_old+1,y_old)]>=2 and action==1:
            custom_events.append(STEPPED_IN_BLOCKED_FIELD)
                
        elif timestamp_dead_marker[(1,x_old,y_old-1)]>=2 and action==2:
            custom_events.append(STEPPED_IN_BLOCKED_FIELD)
                
        elif timestamp_dead_marker[(1,x_old,y_old+1)]>=2 and action==0:
            custom_events.append(STEPPED_IN_BLOCKED_FIELD)
       
        if new_state is not None:
           
            bombs_new = new_state["bombs"]
            walls_new=new_state["walls"]
            position_old = old_state["self_info"]["position"]
            position_new = new_state["self_info"]["position"]

           

            for a in range(0,17):
                 for b in range(0,17):
                    if position_new[a,b]==1:
                        (x_new,y_new)=(a,b)


            bomb_exlpoison_radius_old=np.zeros((17,17))
            bomb_exlpoison_radius_new=np.zeros((17,17))

            Compute_Bomb_Exlpoison_Radius(bomb_exlpoison_radius_old,bombs_old,walls_old)
            Compute_Bomb_Exlpoison_Radius(bomb_exlpoison_radius_new,bombs_new,walls_new)

            #Agent konnte fliehen?
            if bomb_exlpoison_radius_old[x_old,y_old]>=1 and bomb_exlpoison_radius_new[x_new,y_new]==0:
                custom_events.append(ESCAPED_BOMB)
            #Agent ist unnoetigerweise in Bomberadius gelaufen?
            elif bomb_exlpoison_radius_old[x_old,y_old]==0 and bomb_exlpoison_radius_new[x_new,y_new]>=1 and bomb_exlpoison_radius_old[x_new,y_new]>=1:
                custom_events.append(STEPPED_IN_BOMB)
                custom_events.append(STEPPED_IN_BOMB_NEG)
            #Agent war nicht in Gefahr und hat gewartet um sicher zu bleiben?
            elif bomb_exlpoison_radius_old[x_old,y_old]==0 and action == 4:
                custom_events.append(STAYED_SAFE)
            #Agent ist in Gefahr?
            if bomb_exlpoison_radius_old[x_old,y_old]>=1:
                
                #Agent hat Aktion bezueglich kuerzesten Weg raus genommen?
                if memo[(x_new,y_new,1)]==memo[(x_old,y_old,0)]:
                    custom_events.append(USED_OPT_BOMB_ESCAPE)
                    #kategorisiere je nach Laenge des Wegs
                    if memo[(x_old,y_old,0)]==1:
                          custom_events.append(BOMB_ESCAPE_LENGTH_ONE_POS)
                    elif memo[(x_old,y_old,0)]==2:
                         custom_events.append(BOMB_ESCAPE_LENGTH_TWO_POS)
                         
                    elif memo[(x_old,y_old,0)]==3:
                         custom_events.append(BOMB_ESCAPE_LENGTH_THREE_POS)
                         
                    elif memo[(x_old,y_old,0)]==4:
                         custom_events.append(BOMB_ESCAPE_LENGTH_FOUR_POS)
                         
                   


               
                 #Agent hat nicht kuerzesten Weg genommen
                else:
                    custom_events.append(WRONG_BOMB_ESCAPE)
                       #kategorisiere je nach Laenge des Wegs, 1 kann nicht sein new_state ist nicht None
                    if memo[(x_old,y_old,0)]==2:
                         custom_events.append(BOMB_ESCAPE_LENGTH_TWO_NEG)
                         
                    elif memo[(x_old,y_old,0)]==3:
                         custom_events.append(BOMB_ESCAPE_LENGTH_THREE_NEG)
                         
                    elif memo[(x_old,y_old,0)]==4:
                         custom_events.append(BOMB_ESCAPE_LENGTH_FOUR_NEG)
                         
                  


                         #3 (i,j)->(i-1,j) links
        #1 (i,j)->(i+1,j) rechts
        #2 (i,j)->(i,j-1) unten
        #0 (i,j)->(i,j+1) oben
        #STEPPED_IN_BLOCKED_FIELD
            
           #Agent hat nicht kuerzesten Weg genommen
        else:
             
              #kategorisiere je nach Laenge des Wegs
             if memo[(x_old,y_old,0)]==1:
                    custom_events.append(BOMB_ESCAPE_LENGTH_ONE_NEG)
                    custom_events.append(WRONG_BOMB_ESCAPE)

             elif memo[(x_old,y_old,0)]==2:
                    custom_events.append(BOMB_ESCAPE_LENGTH_TWO_NEG)
                    custom_events.append(WRONG_BOMB_ESCAPE)
                         
             elif memo[(x_old,y_old,0)]==3:
                    custom_events.append(BOMB_ESCAPE_LENGTH_THREE_NEG)
                    custom_events.append(WRONG_BOMB_ESCAPE)
                         
             elif memo[(x_old,y_old,0)]==4:
                    custom_events.append(BOMB_ESCAPE_LENGTH_FOUR_NEG)
                    custom_events.append(WRONG_BOMB_ESCAPE)
                         
            

        return custom_events

    def _shape_reward(self, events: list[str]) -> float:
        """
        Shape rewards here instead of in an Environment Wrapper in order to be more flexible (e.g. use this agent as proper component of the environment where no environment wrappers are possible)
        """
        reward_mapping = {
            
            #e.COIN_COLLECTED: 3,
            USED_OPT_BOMB_ESCAPE:1,
            STAYED_SAFE:1,
            #USED_SEMI_OPT_BOMB_ESCAPE:0.1,
            WRONG_BOMB_ESCAPE:-10,
           STEPPED_IN_BLOCKED_FIELD:-10,
            STEPPED_IN_BOMB: -10,
            #e.KILLED_OPPONENT: 5,
            #e.CRATE_DESTROYED: 0.2,
            ESCAPED_BOMB: 3,
             #e.GOT_KILLED: -5,
            #e.KILLED_SELF: -5,
        }

       
        return sum([reward_mapping.get(event, 0) for event in events])
       
      