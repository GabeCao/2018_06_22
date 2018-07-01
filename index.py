import math
def get_probability(t, times):
    return 1 - math.exp(-times * t)

if __name__ == '__main__':
    # env = Env()
    # observation, reward, done, phase = env.reset([30,2])
    # print(reward)
    # observation_, reward, _, _ = env.step([27,4])
    # print(reward)
    t = 5 / 20
    print(get_probability(3 * t, 4))