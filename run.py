from Env_modified import Env
from RL_brain_modified import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(13000):
        print('episode .....................................', episode)
        with open('result.txt', 'a') as res:
            res.write('episode .................................' + str(episode) + '\n')
        total_reward = 0
        # initial observation
        init = True
        env = Env()
        episode_step = 0
        while True:

            if init is True:
                observation, reward, done, phase = env.reset(RL)
                total_reward += reward
                init = False

            # RL choose action based on observation
            action = RL.choose_action(observation, env.get_evn_time())
            # RL take action and get next observation and reward
            observation_, reward, done, phase = env.step(action)

            total_reward += reward
            last_reward = reward
            RL.store_transition(observation, action, reward, done, phase, observation_)
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_
            episode_step += 1
            # break while loop when end of this episode，所有的奖励，能量，都要减去最后一次的
            if done:
                print('total reward         ', total_reward - last_reward)
                with open('result.txt', 'a') as res:
                    res.write('总奖励:    ' + str(total_reward - last_reward) + '\n')

                with open('reward.txt', 'a') as rwa:
                    rwa.write(str(episode) + ',' + str(total_reward - last_reward) + '\n')

                with open('energy.txt', 'a') as ene:
                    ene.write(str(episode) + ',' + str(
                        env.mc_move_energy_consumption - env.last_time_mc_move_energy_consumption) + ','
                              + str(env.mc_charging_energy_consumption - env.last_time_mc_charging_energy_consumption)
                              + '\n')
                break
            step += 1

    # end of game
    print('game over')


if __name__ == "__main__":
    # game
    RL = DeepQNetwork(learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    run_maze()