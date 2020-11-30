import numpy as np
import os
import scipy as sp
from environment import random_walk_2d, taxi
from Q_learning import Q_learning
import matplotlib.pyplot as plt
import random

class BVFT_discrete(object):
    def __init__(self, data, gamma, rmax, rmin, resolution=1e-2):
        self.data = data
        self.n = len(data)
        self.rmax = rmax
        self.gamma = gamma
        self.vmax = rmax / (1.0 - gamma)
        self.vmin = rmin / (1.0 - gamma)
        self.res = resolution
        self.qs = None
        self.qs_discrete = None
        self.q_size = 0


    def discretize(self, Q):
        q_out = np.array([Q[t[0], t[1]] for t in self.data])
        inds = np.digitize(q_out, np.linspace(self.vmin, self.vmax, int((self.vmax - self.vmin) / self.res) + 1), right=True)
        dic = {}
        for i, ind in enumerate(inds):
            if ind not in dic:
                dic[ind] = i
            else:
                if isinstance(dic[ind], int):
                    dic[ind] = [dic[ind]]
                dic[ind].append(i)
        return q_out, inds, dic


    def get_groups(self, q1_discretized, q2_discretized):
        q1_out, q1_inds, q1_dic = q1_discretized
        q2_out, q2_inds, q2_dic = q2_discretized
        groups = []
        for key in q1_dic:
            if isinstance(q1_dic[key], list):
                q1_list = q1_dic[key]
                set1 = set(q1_list)
                for p1 in q1_list:
                    if p1 in set1 and isinstance(q2_dic[q2_inds[p1]], list):
                        set2 = set(q2_dic[q2_inds[p1]])
                        intersect = set1.intersection(set2)
                        set1 = set1.difference(intersect)
                        if len(intersect) > 1:
                            groups.append(list(intersect))
        return groups

    def compute_Tf(self, Q1, groups):
        r = np.array([self.data[i][2] for i in range(self.n)])
        vfsp = [0 if self.data[i][3] is None else self.gamma * np.max(Q1[self.data[i][3]]) for i in range(self.n)]
        Tf = r + vfsp
        for group in groups:
            Tf[group] = np.sum(Tf[group]) / len(group)
        return Tf

    def compute_loss(self, q1_discretized, Tf):
        q1_out, q1_inds, q1_dic = q1_discretized
        diff = q1_out - Tf

        s_a_dic = {}
        for i in range(self.n):
            sa = (self.data[i][0], self.data[i][1])
            if sa in s_a_dic:
                s_a_dic[sa] += diff[i]
            else:
                s_a_dic[sa] = diff[i]
        return np.sqrt(np.sum(np.array(list(s_a_dic.values()))**2))

    def discretize_qs(self, qs):
        self.qs = qs
        self.qs_discrete = [self.discretize(q) for q in qs]
        self.q_size = len(qs)

    def get_loss(self, q1, q2):
        groups = self.get_groups(self.qs_discrete[q1], self.qs_discrete[q2])
        Tf1 = self.compute_Tf(self.qs[q1], groups)
        Tf2 = self.compute_Tf(self.qs[q2], groups)
        l1 = self.compute_loss(self.qs_discrete[q1], Tf1)
        l2 = self.compute_loss(self.qs_discrete[q2], Tf2)

        return l1, l2

    def get_loss_1(self, q1):
        groups = self.get_groups(self.qs_discrete[q1], self.qs_discrete[q1])
        Tf1 = self.compute_Tf(self.qs[q1], groups)
        l1 = self.compute_loss(self.qs_discrete[q1], Tf1)
        return l1

    def run_BVFT(self, Qs):
        self.discretize_qs(Qs)

        loss_matrix = np.zeros((self.q_size, self.q_size))
        for i in range(self.q_size):
            for j in range(i, self.q_size):
                if i == j:
                    loss_matrix[i, i] = self.get_loss_1(i)
                    print("e(Q" + str(i) + "; Q" + str(j) + ") = " + str(loss_matrix[i, i]))
                else:
                    l1, l2 = self.get_loss(i, j)
                    loss_matrix[i, j] = l1
                    loss_matrix[j, i] = l2
                    print("e(Q" + str(i) + "; Q" + str(j) + ") = " + str(loss_matrix[i, j]))
                    print("e(Q" + str(j) + "; Q" + str(i) + ") = " + str(loss_matrix[j, i]))
        q_ranks = np.argsort(np.max(loss_matrix, axis=1))
        return loss_matrix, q_ranks

def plot_bars():
    gamma = 0.99
    rmax = 20.0
    rmin = -1.0

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = []
    for i in range(25, 27):
        data.append(np.load(dir_path + '/taxi-d/d{}.npy'.format(i)))
    data = np.concatenate(data, axis=0)
    qs = []
    for i in range(70, 100):
        qs.append(np.load(dir_path + '/taxi-q/q{}.npy'.format(i)))
    avg_reward = (np.load(dir_path + '/taxi-q/rewards1.npy'))

    b = BVFT_discrete(data[:20], gamma, rmax, rmin, resolution=1e-3)
    ranks = b.run_BVFT(qs)[1]
    y_val = [avg_reward[i] for i in ranks]
    plt.bar([i for i in range(30)], y_val)
    plt.ylabel("rollout estimates")
    plt.xlabel("rank")
    plt.show()


def plot_bars1():
    gamma = 0.99
    rmax = 20.0
    rmin = -1.0
    k = 15

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = []
    for i in range(80, 82):
        data.append(np.load(dir_path + '/taxi-d/d{}.npy'.format(i)))
    data = np.concatenate(data, axis=0)
    # models = [i for i in range(30)] + [i for i in range(40, 100)]
    models = [i for i in range(40, 70)]
    ids = random.sample(models, k)

    qs = []
    avg_rewards = []

    r = np.load(dir_path + '/taxi-q/rewards.npy')
    for id in ids:
        qs.append(np.load(dir_path + '/taxi-q/q{}.npy'.format(id)))
        avg_rewards.append(r[id])
    np.random.shuffle(data)
    b = BVFT_discrete(data[:5000], gamma, rmax, rmin, resolution=1e-3)
    ranks = b.run_BVFT(qs)[1]
    y_val = [avg_rewards[i]+1 for i in ranks]
    plt.bar([i for i in range(k)], y_val)
    plt.ylabel("rollout estimates")
    plt.xlabel("rank")
    plt.show()

def roll_out(state_num, env, policy, num_trajectory, truncate_size):
    SASR = []
    total_reward = 0.0
    frequency = np.zeros(state_num)
    for i_trajectory in range(num_trajectory):
        state = env.reset()
        sasr = []
        for i_t in range(truncate_size):
            # env.render()
            p_action = policy[state, :]
            action = np.random.choice(p_action.shape[0], 1, p=p_action)[0]
            next_state, reward = env.step(action)

            sasr.append((state, action, next_state, reward))
            frequency[state] += 1
            total_reward += reward
            # print env.state_decoding(state)
            # a = input()

            state = next_state
        SASR.append(sasr)
    return SASR, frequency, total_reward / (num_trajectory * truncate_size)



def rank_correlation():
    pass


def group_roll_outs():
    length = 5
    env = taxi(length)
    n_state = env.n_state
    n_action = env.n_action
    num_trajectory = 200
    truncate_size = 200


    dir_path = os.path.dirname(os.path.realpath(__file__))
    rewards = np.load(dir_path + '/taxi-q/rewards.npy')
    for i in range(0, 30):
        print(i)
        agent = Q_learning(n_state, n_action, 0.005, 0.99)
        agent.Q = np.load(dir_path + '/taxi-q/q{}.npy'.format(i))
        SAS, f, avr_reward = roll_out(n_state, env, agent.get_pi(2.0), num_trajectory, truncate_size)
        rewards[i] = avr_reward
        np.save(dir_path + '/taxi-q/rewards.npy', rewards)

if __name__ == '__main__':
    # test_data = [(0, 0, 1.0, 1), (0, 1, 1.0, 2),
    #              (1, 0, 0.0, None), (1, 1, 0.0, None),
    #              (2, 0, 1.0, None), (2, 1, 1.0, None),
    #              (3, 0, 1.0, 4), (3, 1, 1.0, 4)]
    # gamma = 0.9
    # rmax = 1.0

    # Q1 = np.array([[1.0, 1.9], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    # Q2 = np.array([[7.0, 1.9], [0.0, 0.0], [1.0, 1.0], [7.0, 7.0], [10.0, 10.0]])
    # b = BVFT_discrete(test_data, gamma, rmax)
    # print(b.run_BVFT([Q1, Q2]))
    # plot_bars1()

    plot_bars1()

