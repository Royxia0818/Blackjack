# 将该文件改名为Group_Policy_X.py（X是你的组号）
# 文件夹名字"Group_X"中的X也需要修改
from StudentCode.BlackJackBattleEnv import BasePolicy
import numpy as np
import matplotlib.pyplot as plt

class Policy_11:  # X改为你的组号
    def __init__(self) -> None:
        self.pq = np.load("results/org/nplayerQ.npy")
        self.dq = np.load("results/org/ndealerQ.npy")

    def act_player(self, obs):
        sum_hand, card, have_A = obs
        if sum_hand >= 21:
            return 0
        elif sum_hand <= 11:
            return 1
        elif have_A and sum_hand>=18:
            return 0
        else:
            if have_A:
                return self.pq[1, 1, card, sum_hand] > self.pq[0, 1, card, sum_hand]
            else:
                return self.pq[1, 0, card, sum_hand] > self.pq[0, 0, card, sum_hand]

    def act_dealer(self, obs):
        sum_hand, card, have_A = obs
        if sum_hand >= 21:
            return 0
        elif sum_hand <= 11:
            return 1
        else:
            if have_A:
                return self.dq[1, 1, card, sum_hand] > self.dq[0, 1, card, sum_hand]
            else:
                return self.dq[1, 0, card, sum_hand] > self.dq[0, 0, card, sum_hand]
            

class Policy_10:  # X改为你的组号
    def __init__(self) -> None:
        self.pq = np.load("results/xzh/nplayerQ.npy")
        self.dq = np.load("results/xzh/ndealerQ.npy")

    def act_player(self, obs):
        sum_hand, card, have_A = obs
        if sum_hand >= 21:
            return 0
        elif sum_hand <= 11:
            return 1
        elif have_A and sum_hand>=18:
            return 0
        else:
            if have_A:
                return self.pq[1, 1, card, sum_hand] > self.pq[0, 1, card, sum_hand]
            else:
                return self.pq[1, 0, card, sum_hand] > self.pq[0, 0, card, sum_hand]

    def act_dealer(self, obs):
        sum_hand, card, have_A = obs
        if sum_hand >= 21:
            return 0
        elif sum_hand <= 11:
            return 1
        else:
            if have_A:
                return self.dq[1, 1, card, sum_hand] > self.dq[0, 1, card, sum_hand]
            else:
                return self.dq[1, 0, card, sum_hand] > self.dq[0, 0, card, sum_hand]
            
            
