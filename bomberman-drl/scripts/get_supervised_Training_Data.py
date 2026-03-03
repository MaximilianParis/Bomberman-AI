import time
import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

from bomberman_rl import ScoreRewardWrapper, RestrictedKeysWrapper, FlattenWrapper

from argparsing import parse
from bomberman_rl.envs.agent_code.interface import LearningAgent

from Expert_Agent import Expert_Agent as Agent

class DummyAgent:
    def setup(self):
        pass

    def act(self, *args, **kwargs):
        return None

def loop(env, agent, args, n_episodes=100):
   

   
    for i in range(n_episodes):
        state, info = env.reset()
        terminated, truncated, quit = False, False, False
        while not (terminated or truncated):
           
            action, quit = agent.act(state), env.unwrapped.get_user_quit()

            new_state, _, terminated, truncated, info = env.step(action)
                        
            state = new_state 
        print(i)


           
    agent.write_experience_to_file()        
  
  

    env.close()


def provideAgent(passive: bool):
    if passive:
        return DummyAgent()
    else:
        agent = Agent()
        return agent

def main(argv=None):
    args = parse(argv)
    env = gymnasium.make("bomberman_rl/bomberman-v0", args=args)

  
    env = ScoreRewardWrapper(env)

    if args.video:
        env = RecordVideo(env, video_folder=args.video, name_prefix=args.match_name)

    agent = provideAgent(passive=args.passive)
    if agent is None and not args.passive:
        raise AssertionError("Either provide an agent or run in passive mode by providing the command line argument --passive")
    
    loop(env, agent, args)


if __name__ == "__main__":
    main()