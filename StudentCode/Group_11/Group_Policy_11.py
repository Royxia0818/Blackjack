# 将该文件改名为Group_Policy_X.py（X是你的组号）
# 文件夹名字"Group_X"中的X也需要修改
from StudentCode.BlackJackBattleEnv import BasePolicy
import numpy as np
import matplotlib.pyplot as plt

'''
class Policy_11:  # X改为你的组号
    def __init__(self) -> None:
        self.pq = np.load("playerQ.npy")
        self.dq = np.load("dealerQ.npy")
        print("p")
        print(self.pq[1]-self.pq[0])
        print("q")
        print(self.dq[1]-self.dq[0])

    def act_player(self, obs):
        sum_hand, dealer_card, have_A = obs
        if sum_hand >= 21:
            return 0
        elif sum_hand <= 11:
            return 1
        else:
            if have_A and dealer_card == 1:
                return self.pq[1, 3, sum_hand] > self.pq[0, 3, sum_hand]
            elif dealer_card == 1:
                return self.pq[1, 2, sum_hand] > self.pq[0, 2, sum_hand]
            elif have_A:
                return self.pq[1, 1, sum_hand] > self.pq[0, 1, sum_hand]
            else:
                return self.pq[1, 0, sum_hand] > self.pq[0, 0, sum_hand]

    def act_dealer(self, obs):
        sum_hand, player_card, have_A = obs
        if sum_hand >= 21:
            return 0
        elif sum_hand <= 11:
            return 1
        else:
            if have_A and player_card == 1:
                return self.dq[1, 3, sum_hand] > self.dq[0, 3, sum_hand]
            elif player_card == 1:
                return self.dq[1, 2, sum_hand] > self.dq[0, 2, sum_hand]
            elif have_A:
                return self.dq[1, 1, sum_hand] > self.dq[0, 1, sum_hand]
            else:
                return self.dq[1, 0, sum_hand] > self.dq[0, 0, sum_hand]
'''


class Policy_11:  # X改为你的组号
    def __init__(self) -> None:
        self.pq = np.load("nplayerQ.npy")
        self.dq = np.load("ndealerQ.npy")

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
