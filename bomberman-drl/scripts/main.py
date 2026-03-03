import time
import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

from bomberman_rl import ScoreRewardWrapper, RestrictedKeysWrapper, FlattenWrapper

from argparsing import parse
from bomberman_rl.envs.agent_code.interface import LearningAgent

from learning_agent.agent import Agent

class DummyAgent:
    def setup(self):
        pass

    def act(self, *args, **kwargs):
        return None

def loop(env, agent, args, n_episodes=1100):
    cnt=1
    limit=20
    number_of_average_rewards=int(n_episodes/limit)
    reward_average_limit_list=np.zeros(number_of_average_rewards,dtype=float)
    survived_timestamps_average_limit_list=np.zeros(number_of_average_rewards,dtype=float)
    score_average_limit_list=np.zeros(number_of_average_rewards,dtype=float)
    cur_avg_reward=0
    cur_avg_score=0
    epsiode_it=0
    cur_reward=0
    cur_score=0
    timestamp=0
    survived_timestamps_avg=0
    old_state1=None
    old_state2=None
    for i in range(n_episodes):
        state, info = env.reset()
        terminated, truncated, quit = False, False, False
        while not (terminated or truncated):
            if args.user_play:
                action, quit = env.unwrapped.get_user_action()
                while action is None and not quit:
                    time.sleep(0.1)  # wait for user action or quit
                    action, quit = env.unwrapped.get_user_action()
            else:
                action, quit = agent.act(state), env.unwrapped.get_user_quit()

            if quit:
                env.close()
                return None
            else:
                new_state, _, terminated, truncated, info = env.step(action)
                if args.train:
                 
                    reward=agent.game_events_occurred(state, action, new_state, info["events"])
                    cur_reward+=reward
                    #print(reward)
                if not terminated:
                    old_state1=state
                    old_state2=new_state
                timestamp+=1
                if terminated:
                    
                    if new_state is None:
                        cur_score=state["self_info"]["score"]-5
                    else:
                        cur_score=new_state["self_info"]["score"]
                state = new_state 
              
                
        cur_avg_score=cur_avg_score+1/cnt*(cur_score-cur_avg_score)
        cur_score=0
       
        survived_timestamps_avg=survived_timestamps_avg+1/cnt*(timestamp-survived_timestamps_avg)
        timestamp=0

        cur_avg_reward=cur_avg_reward+1/cnt*(cur_reward-cur_avg_reward)
        cur_reward=0
        cnt+=1
        #print(cnt)
        if cnt>=limit+1:
            reward_average_limit_list[epsiode_it]=cur_avg_reward
            survived_timestamps_average_limit_list[epsiode_it]=survived_timestamps_avg
            score_average_limit_list[epsiode_it]=cur_avg_score
            print(f"{cur_avg_score=}")
            print(f"{cur_avg_reward=}")
            print(f"{survived_timestamps_avg=}")
            cur_avg_reward=0
            survived_timestamps_avg=0
            cur_avg_score=0
            cnt=1
            epsiode_it+=1
            print(epsiode_it)
       
        if args.train:
            agent.end_of_round()

   #hier plotten

    plt.figure(figsize=(10, 6))
    plt.plot(survived_timestamps_average_limit_list, label="Durchschnittliche Lebenszeit in Schritten pro 20 Episoden")
    plt.xlabel("Episode (x20)")
    plt.ylabel("Durchschnittliche Lebenszeit in Schritten")
    plt.title("Verlauf der durchschnittliche Lebenszeit in Schritten")
    plt.legend()
    plt.grid()
    plt.show()

    print(survived_timestamps_average_limit_list)


    plt.figure(figsize=(10, 6))
    plt.plot(reward_average_limit_list, label="Durchschnittliche Belohnung pro 20 Episoden")
    plt.xlabel("Episode (x20)")
    plt.ylabel("Durchschnittliche Belohnung")
    plt.title("Verlauf der durchschnittlichen Belohnung")
    plt.legend()
    plt.grid()
    plt.show()

    print(reward_average_limit_list)


    plt.figure(figsize=(10, 6))
    plt.plot(score_average_limit_list, label="Durchschnittlicher Score pro 20 Episoden")
    plt.xlabel("Episode (x20)")
    plt.ylabel("Durchschnittlicher Score")
    plt.title("Verlauf der durchschnittlichen Score")
    plt.legend()
    plt.grid()
    plt.show()

    print(score_average_limit_list)

    if not args.no_gui:
        quit = env.unwrapped.get_user_quit()
        while not quit:
            time.sleep(0.5) # wait for quit
            quit = env.unwrapped.get_user_quit()

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

    # Notice that you can not use wrappers in the tournament!
    # However, you might wanna use this example interface to kickstart your experiments
    env = ScoreRewardWrapper(env)
    #env = RestrictedKeysWrapper(env, keys=["self_pos"])
    #env = FlattenWrapper(env)
    if args.video:
        env = RecordVideo(env, video_folder=args.video, name_prefix=args.match_name)

    agent = provideAgent(passive=args.passive)
    if agent is None and not args.passive:
        raise AssertionError("Either provide an agent or run in passive mode by providing the command line argument --passive")
    if args.train:
        agent.setup_training()

    loop(env, agent, args)


if __name__ == "__main__":
    main()