import numpy as np



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

    grid = np.zeros((4,31,31))
    grid[0]=np.ones((31,31))
    window_length=15
    offset_x=x - window_length
    offset_y=y - window_length

    
    for i in range(x - window_length, x + window_length+1):
           for j in range(y - window_length, y + window_length+1):
               distanz = abs(i - x) + abs(j - y)  
              # if distanz < window_length:
               if i>=0 and i<limit_x and j>=0 and j<limit_y:
                  if(walls[i,j]==0):
                       grid[0][i-offset_x][j-offset_y]=0

                  if(coins[i,j]==1):
                       grid[1][i-offset_x][j-offset_y]=1
                       
                  if(opponents[i,j]==1):
                       grid[2][i-offset_x][j-offset_y]=1
                       
                  if(crates[i,j]==1):
                       grid[3][i-offset_x][j-offset_y]=1
                          # state_representation_list.append(1)
              
                       #else:
                             #state_representation_list.append(0)
                              
                  # else:
                        #state_representation_list.append(1)


    

   
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




    return [grid,state_representation_list]