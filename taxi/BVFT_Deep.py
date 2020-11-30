import numpy as np
import os

from keras.models import Model, load_model

class BVFT_deep(object):
    def __init__(self, data, gamma, rmax, rmin, resolution=1e-2):
        self.states = data[0]
        self.ar = data[1]
        self.n = len(self.states)
        self.rmax = rmax
        self.gamma = gamma
        self.vmax = rmax / (1.0 - gamma)
        self.vmin = rmin / (1.0 - gamma)
        self.res = resolution
        self.qs = None
        self.qs_discrete = None
        self.q_size = 0


    def discretize(self, Q):
        q_out = np.array([Q.predict(self.states[t][0])[0, int(self.ar[t][0])] for t in range(self.n)])
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
        r = np.array([self.ar[i][1] for i in range(self.n)])
        vfsp = [0 if self.states[i][1] is None else self.gamma * np.max(Q1.predict(self.states[i][1])) for i in range(self.n)]
        Tf = r + vfsp
        for group in groups:
            Tf[group] = np.sum(Tf[group]) / len(group)
        return Tf

    def compute_loss(self, q1_discretized, Tf):
        q1_out, q1_inds, q1_dic = q1_discretized
        diff = q1_out - Tf

        # s_a_dic = {}
        # for i in range(self.n):
        #     sa = (self.states[i][0], int(self.ar[i][0]))
        #     if sa in s_a_dic:
        #         s_a_dic[sa] += diff[i]
        #     else:
        #         s_a_dic[sa] = diff[i]
        return np.sqrt(np.sum(diff**2))

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

def plot_ranks():
    gamma = 0.95
    rmax = 1.0
    rmin = 0.0

    dir_path = os.path.dirname(os.path.realpath(__file__))
    states = []
    ar = []
    for i in range(0, 5):
        states.append(np.load(dir_path + '/cartpole-d/d-state{}.npy'.format(i), allow_pickle=True))
        ar.append(np.load(dir_path + '/cartpole-d/d-ar{}.npy'.format(i)))
    states = np.concatenate(states, axis=0)[:100]
    ar = np.concatenate(ar, axis=0)[:100]
    qs = []
    for i in range(5):
        qs.append(load_model("cartpole-dqn{}.h5".format(i)))

    b = BVFT_deep([states, ar], gamma, rmax, rmin, resolution=1e-3)
    print(b.run_BVFT(qs))

if __name__ == '__main__':
    plot_ranks()