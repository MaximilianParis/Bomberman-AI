import gymnasium
import numpy as np
from bomberman_rl import ScoreRewardWrapper
from argparsing import parse
from Nets import BaseActor
from State_Algorithms import Transform_State
import torch
from pathlib import Path
import pickle



device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def _compute_returns(rewards,gamma):
    returns = []
    discounted_reward = 0
    for reward in reversed(rewards):
        discounted_reward = reward + discounted_reward * gamma
        returns.append(discounted_reward)
    return returns[::-1]

def loop(env, policy_net, args, n_episodes=1000):
   
    episodes_list=[]
    policy_net.eval()
    for i in range(n_episodes):
        state, info = env.reset()
        terminated, truncated, quit = False, False, False
        states_list=[]
        reward_list=[]
        while not (terminated or truncated):

            transformed_state=Transform_State(state)
            with torch.no_grad():
                 prob=policy_net.forward_with_softmax(transformed_state)

            action=policy_net.get_best_action(prob)

            states_list.append(transformed_state)
           
            if(new_state is not None):
                reward_list.append(new_state["self_info"]["score"]-state["self_info"]["score"])
            else:
                reward_list.append(-5)

                      
            new_state, _, terminated, truncated, info = env.step(action)
                        
            state = new_state 
        print(i)
        episodes_list.append([states_list,reward_list,_compute_returns(reward_list,0.99)])

           
      
  
    with open("Value_Function_Training_Data.pkl", "wb") as f:
                 pickle.dump(episodes_list, f)

    env.close()




def main(argv=None):
    args = parse(argv)
    env = gymnasium.make("bomberman_rl/bomberman-v0", args=args)
    env = ScoreRewardWrapper(env)

    policy_net = BaseActor(2145, 6).to(device)

    policy_net.load_state_dict(torch.load( Path(__file__).parent / "model_supervised.pt"))
    
    loop(env, policy_net, args)


if __name__ == "__main__":
    main()