from sklearn.preprocessing import LabelBinarizer
import math
from Hotspot import Hotspot
from Point import Point
import numpy as np


class Env:
    def __init__(self):
        # 当前环境state
        self.state = []
        # mc移动花费的时间
        self.move_time = 0
        # 一个回合最大的时间，用秒来表示，早上八点到晚上10点，十四个小时，总共 14 * 3600 秒的时间
        # 如果self.get_evn_time() 得到的时间大于这个时间，则表示该回合结束
        self.one_episode_time = 14 * 3600
        # sensor 和 mc的能量信息
        self.sensors_mobile_charger = {}
        # 初始化所有的sensor,mc 的能量信息
        self.set_sensors_mobile_charger()
        # 初始化self.sensors_mobile_charger 和 self.sensors
        # self.set_sensors_mobile_charger()
        # 对剩余寿命进行独热编码
        self.rl = ['Greater than the threshold value, 0', 'Smaller than the threshold value, 1', 'dead, -1']
        self.rl_label_binarizer = LabelBinarizer()
        self.rl_one_hot_encoded = self.rl_label_binarizer.fit_transform(self.rl)
        # 对是否属于hotspot 独热编码
        self.belong = ['1', '0']
        self.belong_label_binarizer = LabelBinarizer()
        self.belong_one_hot_encoded = self.belong_label_binarizer.fit_transform(self.belong)
        # 获得所有的hotspot
        self.hotspots = []
        # 初始化hotspots
        self.set_hotspots()
        # 记录当前时刻所在的hotspot，在环境初始化的时候设置为base_station
        self.current_hotspot = self.hotspots[0]

        # mc移动速度
        self.speed = 5
        # mc 移动消耗的能量
        self.mc_move_energy_consumption = 0
        # mc 给sensor充电消耗的能量
        self.mc_charging_energy_consumption = 0
        # 充电惩罚值
        self.charging_penalty = -1
        # 最近一次mc 移动和给sensor充电花费的能量，因为在charging trajectory 中，最后一次不能算作在 trajectory中
        self.last_time_mc_move_energy_consumption = 0
        self.last_time_mc_charging_energy_consumption = 0

    def set_sensors_mobile_charger(self):
        # [0.7 * 6 * 1000, 0.6, 0, True]  依次代表：上一次充电后的剩余能量，能量消耗的速率，上一次充电的时间，
        # 是否已经死掉(计算reward的惩罚值时候使用，避免将一个sensor计算死掉了多次)，
        # 最后一个标志位，表示senor在该hotpot，还没有被充过电，如果已经充过了为True，避免被多次充电
        self.sensors_mobile_charger['0'] = [0.7 * 6 * 1000, 0.5, 0, True, False]
        self.sensors_mobile_charger['1'] = [0.3 * 6 * 1000, 0.3, 0, True, False]
        self.sensors_mobile_charger['2'] = [0.9 * 6 * 1000, 0.5, 0, True, False]
        self.sensors_mobile_charger['3'] = [0.5 * 6 * 1000, 0.3, 0, True, False]
        self.sensors_mobile_charger['4'] = [0.2 * 6 * 1000, 0.2, 0, True, False]
        self.sensors_mobile_charger['5'] = [0.4 * 6 * 1000, 0.3, 0, True, False]
        self.sensors_mobile_charger['6'] = [1 * 6 * 1000, 0.6, 0, True, False]
        self.sensors_mobile_charger['7'] = [0.3 * 6 * 1000, 0.5, 0, True, False]
        self.sensors_mobile_charger['8'] = [1 * 6 * 1000, 0.3, 0, True, False]
        self.sensors_mobile_charger['9'] = [0.9 * 6 * 1000, 0.2, 0, True, False]
        self.sensors_mobile_charger['10'] = [0.8 * 6 * 1000, 0.2, 0, True, False]
        self.sensors_mobile_charger['11'] = [0.5 * 6 * 1000, 0.4, 0, True, False]
        self.sensors_mobile_charger['12'] = [0.4 * 6 * 1000, 0.2, 0, True, False]
        self.sensors_mobile_charger['13'] = [0.6 * 6 * 1000, 0.2, 0, True, False]
        self.sensors_mobile_charger['14'] = [0.3 * 6 * 1000, 0.2, 0, True, False]
        self.sensors_mobile_charger['15'] = [0.9 * 6 * 1000, 0.6, 0, True, False]
        self.sensors_mobile_charger['16'] = [0.8 * 6 * 1000, 0.4, 0, True, False]
        self.sensors_mobile_charger['MC'] = [1000 * 1000, 50]

    def set_hotspots(self):
        # 这是编号为0 的hotspot，也就是base_stattion,位于整个充电范围中心
        base_station = Hotspot((116.333 - 116.318) * 85000 / 2, (40.012 - 39.997) * 110000 / 2, 0)
        self.hotspots.append(base_station)
        # 读取hotspot.txt 的文件，获取所有的hotspot，放入self.hotspots中
        path = 'hotspot.txt'
        with open(path) as file:
            for line in file:
                data = line.strip().split(',')
                hotspot = Hotspot(float(data[0]), float(data[1]), int(data[2]))
                self.hotspots.append(hotspot)

    # 根据hotspot 的编号，在self.hotspots 中找到对应的hotpot
    def find_hotspot_by_num(self, num):
        for hotspot in self.hotspots:
            if hotspot.get_num() == num:
                return hotspot

    def initial_is_charged(self):
        for key, value in self.sensors_mobile_charger.items():
            if key != 'MC':
                value[4] = False

    # 传入一个action, 得到下一个state，reward，和 done(是否回合结束)的信息
    def step(self, action):
        reward = 0
        # action 的 表示形如 43,1 表示到43 号hotspot 等待1个t
        # action = action.split(',')
        # 得到hotspot 的编号
        # hotspot_num = int(action[0])
        # 得到等待时间
        # staying_time = int(action[1])

        # action 是ndarray 一维 形如[23 4 .. .. ..] 表示在第23个hotspot，等待4个t，后面的点表示这个action的独热编码，
        # 当前方法用不上
        hotspot_num = action[0]
        staying_time = action[1]

        with open('result.txt', 'a') as res:
            res.write('选择的hotspot:  ' + str(hotspot_num) + '     等待时间:    ' + str(staying_time) + '\n')

        # 得到下一个hotspot
        hotspot = self.find_hotspot_by_num(hotspot_num)
        # 当前hotspot 和 下一个hotspot间的距离,得到移动花费的时间，添加到self.move_time 里
        distance = hotspot.get_distance_between_hotspot(self.current_hotspot)
        time = distance / self.speed
        # self.move_time += time

        for key, value in self.sensors_mobile_charger.items():
            if key == 'MC':
                break
            sensor_energy_after_last_time_charging = value[0]
            # 当前sensor 电量消耗的速率
            sensor_consumption_ratio = value[1]
            # 上一次的充电时间
            previous_charging_time = value[2]
            # 当前sensor 的剩余电量
            evn_time = self.get_evn_time()
            sensor_reserved_energy = sensor_energy_after_last_time_charging - \
                                     (evn_time - previous_charging_time) * sensor_consumption_ratio
            # 如果剩余能量小于0，且value[3] is True，则sensor第一次死亡，reward 减去惩罚值
            if (sensor_reserved_energy < 0) and (value[3] is True):
                value[3] = False
                reward += self.charging_penalty
                with open('result.txt', 'a') as res:
                    res.write('sensor  ' + key + '  死了  ' + '\n')
                print('sensor   ' + key + '  死了  ')
        # 更新self.current_hotspot 为 action 中选择的 hotspot
        self.current_hotspot = hotspot
        # 更新mc 移动消耗的能量
        self.mc_move_energy_consumption += self.sensors_mobile_charger['MC'][1] * distance
        self.last_time_mc_move_energy_consumption = self.sensors_mobile_charger['MC'][1] * distance
        # 更新mc的剩余能量，减去移动消耗的能量
        self.sensors_mobile_charger['MC'][0] = self.sensors_mobile_charger['MC'][0] \
                                               - self.sensors_mobile_charger['MC'][1] * distance

        # 获取当前时间段,下面的path中使用，加8是因为从8点开始
        start_wait_seconds = self.get_evn_time()
        hour = int(start_wait_seconds / 1200) + 1
        # 将在hotspot_num 等待的时间 添加到state中的CS
        self.state[2 * (hotspot_num - 1)] += 1
        self.state[2 * (hotspot_num - 1) + 1] += staying_time
        # mc 结束等待后环境的时间
        self.move_time += time
        end_wait_seconds = self.get_evn_time()
        # 在一次执行action 的过程中，sensor只能被充电一次
        self.initial_is_charged()

        path = 'hotspot中sensor的访问情况/' + str(hour) + '时间段/' + str(hotspot_num) + '.txt'
        # 读取文件，得到在当前时间段，hotspot_num 的访问情况，用字典保存。key: sensor 编号；value: 访问次数
        hotspot_num_sensor_arrived_times = {}
        with open(path) as f:
            for line in f:
                data = line.strip().split(',')
                hotspot_num_sensor_arrived_times[data[0]] = data[1]
        # 一共17个sensor，现在更新每个sensor 的信息

        for i in range(17):
            # 取出当前sensor 访问 hotspot_num 的次数,如果大于 0, 则belong = 1, 表示属于，sensor 可能会到达这个hotspot 。
            # 否则0，表示不属于，sensor 不可能会到达这个hotspot
            times = int(hotspot_num_sensor_arrived_times[str(i)])
            # times == 0，表示该sensor不属于该hotspot，不可能到达该hotspot充电
            if times == 0:
                belong = str(0)
                # 取出sensor
                sensor = self.sensors_mobile_charger[str(i)]
                # 上一次充电后的电量
                sensor_energy_after_last_time_charging = sensor[0]
                # 当前sensor 电量消耗的速率
                sensor_consumption_ratio = sensor[1]
                # 上一次的充电时间
                previous_charging_time = sensor[2]
                # 在mc等待了action 中的等待时间以后，sensor 的剩余电量
                sensor_reserved_energy = sensor_energy_after_last_time_charging - \
                                         (end_wait_seconds - previous_charging_time) * sensor_consumption_ratio
                # 当前sensor 的剩余寿命
                rl = sensor_reserved_energy / sensor_consumption_ratio

                # 如果剩余寿命大于两个小时
                if rl >= 2 * 3600:
                    # 得到 大于阈值的 独热编码,转换成list,然后更新state 中的能量状态
                    rl_one_hot_encoded = self.rl_label_binarizer.transform(['Greater than the threshold value, 0']).tolist()[0]
                    start = 84 + i * 4
                    end = start + 2
                    rl_i = 0
                    while start <= end:
                        self.state[start] = rl_one_hot_encoded[rl_i]
                        rl_i += 1
                        start += 1
                    # 更新是否属于 state中 的 是否属于hotspot 的信息
                    # 通过belong 找到对应的 独热编码，转换成list,因为这个独热编码只有一位，所以取第一位得到结果
                    belong_one_hot_encoded = self.belong_label_binarizer.transform([belong]).tolist()[0][0]
                    self.state[start] = belong_one_hot_encoded
                elif 0 < rl < 2 * 3600:
                    # 更新state中 的剩余寿命信息的状态
                    # 得到 小于阈值的 独热编码,转换成list,然后更新state 中的状态
                    rl_one_hot_encoded = self.rl_label_binarizer.transform(['Smaller than the threshold value, 1']).tolist()[0]
                    start = 84 + i * 4
                    end = start + 2
                    rl_i = 0
                    while start <= end:
                        a = rl_one_hot_encoded[rl_i]
                        self.state[start] = a
                        rl_i += 1
                        start += 1
                    # 更新是否属于 state中 的 是否属于hotspot 的信息
                    # 通过belong 找到对应的 独热编码，转换成list,因为这个独热编码只有一位，所以取第一位得到结果
                    belong_one_hot_encoded = self.belong_label_binarizer.transform([belong]).tolist()[0][0]
                    self.state[start] = belong_one_hot_encoded
                else:
                    # 更新state中 的剩余寿命信息的状态
                    # 得到 死掉的 独热编码,转换成list,然后更新state 中的状态
                    rl_one_hot_encoded = self.rl_label_binarizer.transform(['dead, -1']) \
                        .tolist()[0]
                    start = 84 + i * 4
                    end = start + 2
                    rl_i = 0
                    while start <= end:
                        a = rl_one_hot_encoded[rl_i]
                        self.state[start] = a
                        rl_i += 1
                        start += 1
                    # 更新是否属于 state中 的 是否属于hotspot 的信息
                    # 通过belong 找到对应的 独热编码，转换成list,因为这个独热编码只有一位，所以取第一位得到结果
                    belong_one_hot_encoded = self.belong_label_binarizer.transform([belong]).tolist()[0][0]
                    self.state[start] = belong_one_hot_encoded
                    if sensor[3]:
                        sensor[3] = False
                        reward += self.charging_penalty
                        with open('result.txt', 'a') as res:
                            res.write('sensor   ' + str(i) + '  死了  ' + '\n')

            else:
                # times不等于0 的情况下，sensor可能会过来
                belong = str(1)
                # 读取第i 个sensor 的轨迹点信息
                sensor_path = 'sensor数据五秒/' + str(i) + '.txt'
                with open(sensor_path) as f:
                    for point_line in f:
                        # 检查当前sensor 是否在该hotspot 已经被充过电了，如果是，跳出循环
                        sensor_is_charged = self.sensors_mobile_charger[str(i)]
                        if sensor_is_charged[4] is True:
                            break

                        data = point_line.strip().split(',')
                        point_time = self.str_to_seconds(data[2])
                        point = Point(float(data[0]), float(data[1]), data[2])

                        # 如果第 i 个sensor的轨迹点的时间 小于end_wait_seconds且大于start_wait_seconds，
                        # 同时轨迹点和hotspot 的距离小于60，则到达该hotspot
                        if (start_wait_seconds <= point_time <= end_wait_seconds) and (point.get_distance_between_point_and_hotspot(self.current_hotspot) < 60):
                            # 取出sensor
                            sensor = self.sensors_mobile_charger[str(i)]
                            # 上一次充电后的电量
                            sensor_energy_after_last_time_charging = sensor[0]
                            # 当前sensor 电量消耗的速率
                            sensor_consumption_ratio = sensor[1]
                            # 上一次的充电时间
                            previous_charging_time = sensor[2]
                            # 当前sensor 的剩余电量
                            sensor_reserved_energy = sensor_energy_after_last_time_charging - \
                                                     (point_time - previous_charging_time) * sensor_consumption_ratio
                            # 当前sensor 的剩余寿命
                            rl = sensor_reserved_energy / sensor_consumption_ratio
                            # 如果剩余寿命大于两个小时
                            if rl >= 2 * 3600:
                                reward += 0
                            # 如果剩余寿命在0 到 两个小时
                            elif 0 < rl < 2 * 3600:
                                # 更新mc 的sensor充的电量
                                self.mc_charging_energy_consumption += 6 * 1000 - sensor_reserved_energy
                                self.last_time_mc_charging_energy_consumption = 6 * 1000 - sensor_reserved_energy
                                # mc 给该sensor充电， 充电后更新剩余能量
                                self.sensors_mobile_charger['MC'][0] = self.sensors_mobile_charger['MC'][0] \
                                                                       - (6 * 1000 - sensor_reserved_energy)
                                # 设置sensor 充电后的剩余能量 是满能量
                                sensor[0] = 6 * 1000
                                # 更新被充电的时间
                                sensor[2] = point_time
                                # sensor 被充电了
                                sensor[4] = True
                                # 加上得到的奖励,需要先将 rl 的单位先转化成小时
                                rl = rl / 3600
                                reward += math.exp(-rl)
                            else:
                                # 如果是第一次死
                                if sensor[3] is True:
                                    sensor[3] = False
                                    reward += self.charging_penalty
                                    with open('result.txt', 'a') as res:
                                        res.write('sensor   ' + str(i) + '  死了  ' + '\n')

                # 取出sensor
                sensor = self.sensors_mobile_charger[str(i)]
                # 上一次充电后的电量
                sensor_energy_after_last_time_charging = sensor[0]
                # 当前sensor 电量消耗的速率
                sensor_consumption_ratio = sensor[1]
                # 上一次的充电时间
                previous_charging_time = sensor[2]
                # 在mc等待了action 中的等待时间以后，sensor 的剩余电量
                sensor_reserved_energy = sensor_energy_after_last_time_charging - \
                                         (end_wait_seconds - previous_charging_time) * sensor_consumption_ratio
                # 当前sensor 的剩余寿命
                rl = sensor_reserved_energy / sensor_consumption_ratio

                # 如果剩余寿命大于两个小时
                if rl >= 2 * 3600:
                    # 得到 大于阈值的 独热编码,转换成list,然后更新state 中的能量状态
                    rl_one_hot_encoded = self.rl_label_binarizer.transform(['Greater than the threshold value, 0']).tolist()[0]
                    start = 84 + i * 4
                    end = start + 2
                    rl_i = 0
                    while start <= end:
                        self.state[start] = rl_one_hot_encoded[rl_i]
                        rl_i += 1
                        start += 1
                        # 更新是否属于 state中 的 是否属于hotspot 的信息
                        # 通过belong 找到对应的 独热编码，转换成list,因为这个独热编码只有一位，所以取第一位得到结果
                    belong_one_hot_encoded = self.belong_label_binarizer.transform([belong]).tolist()[0][0]
                    self.state[start] = belong_one_hot_encoded
                elif 0 < rl < 2 * 3600:
                    # 更新state中 的剩余寿命信息的状态
                    # 得到 小于阈值的 独热编码,转换成list,然后更新state 中的状态
                    rl_one_hot_encoded = \
                        self.rl_label_binarizer.transform(['Smaller than the threshold value, 1']).tolist()[0]
                    start = 84 + i * 4
                    end = start + 2
                    rl_i = 0
                    while start <= end:
                        a = rl_one_hot_encoded[rl_i]
                        self.state[start] = a
                        rl_i += 1
                        start += 1
                    # 更新是否属于 state中 的 是否属于hotspot 的信息
                    # 通过belong 找到对应的 独热编码，转换成list,因为这个独热编码只有一位，所以取第一位得到结果
                    belong_one_hot_encoded = self.belong_label_binarizer.transform([belong]).tolist()[0][0]
                    self.state[start] = belong_one_hot_encoded
                else:
                    # 更新state中 的剩余寿命信息的状态
                    # 得到 死掉的 独热编码,转换成list,然后更新state 中的状态
                    rl_one_hot_encoded = self.rl_label_binarizer.transform(['dead, -1']) \
                        .tolist()[0]
                    start = 84 + i * 4
                    end = start + 2
                    rl_i = 0
                    while start <= end:
                        a = rl_one_hot_encoded[rl_i]
                        self.state[start] = a
                        rl_i += 1
                        start += 1
                    # 更新是否属于 state中 的 是否属于hotspot 的信息
                    # 通过belong 找到对应的 独热编码，转换成list,因为这个独热编码只有一位，所以取第一位得到结果
                    belong_one_hot_encoded = self.belong_label_binarizer.transform([belong]).tolist()[0][0]
                    self.state[start] = belong_one_hot_encoded
                    if sensor[3]:
                        sensor[3] = False
                        reward += self.charging_penalty
                        with open('result.txt', 'a') as res:
                            res.write('sensor   ' + str(i) + '  死了  ' + '\n')

        phase = int(end_wait_seconds / 1200) + 1
        # mc 给到达的sensor 充电后，如果能量为负或者 self.get_evn_time() > self.one_episode_time，则回合结束，反之继续
        if self.sensors_mobile_charger['MC'][0] <= 0 or self.get_evn_time() > self.one_episode_time:
            with open('result.txt', 'a') as res:
                res.write('回合结束:    ' + 'mc 移动消耗的电量     '
                          + str(self.mc_move_energy_consumption - self.last_time_mc_move_energy_consumption)
                          + '       mc 给sensor充电消耗的电量   '
                          + str(self.mc_charging_energy_consumption - self.last_time_mc_charging_energy_consumption)
                          + '\n')
            done = True
        else:
            done = False

        observation = np.array(self.state)
        return observation, reward, done, phase

    # 初始化整个环境
    def reset(self, RL):
        # 前面0~83 都初始化为 0。记录CS的信息，每个hotspot占两位
        for i in range(42 * 2):
            self.state.append(0)
        # 84位开始记录sensor的信息,每一个sensor需要4位，17个sensor，共68位
        for i in range(42 * 2, 42 * 2 + 68):
                self.state.append(0)
        # 得到一个随机的8点时间段的action,例如 43,1 表示到43 号hotspot 等待1个t
        # print(len(self.state))
        action = RL.random_action()
        self.current_hotspot = self.hotspots[0]

        state__, reward_, done_, phase = self.step(action)
        return state__, reward_, done_, phase

    # 传入时间字符串，如：09：02：03，转化成与 08:00:00 间的秒数差
    def str_to_seconds(self, input_str):
        data = input_str.split(':')
        hour = int(data[0]) - 8
        minute = int(data[1])
        second = int(data[2])
        return hour * 3600 + minute * 60 + second

    # 获得当前环境的秒
    def get_evn_time(self):
        total_t = 0
        for i in range(42 * 2):
            if i % 2 == 1:
                total_t += self.state[i]
        total_time = total_t * 5 * 60 + self.move_time
        return total_time
