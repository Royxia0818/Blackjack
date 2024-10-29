import numpy as np
import random
from alive_progress import alive_bar
alpha = 0.5


class Player:
    def __init__(self) -> None:
        self.q = np.zeros((2, 2, 11, 22), dtype=float)
        self.a = np.zeros((2, 2, 11, 22), dtype=float)
        self.c = np.zeros((2, 2, 11, 22), dtype=int)

    def updateq(self, obs, action, G):
        if obs[0] <= 11:
            return
        sum_hand, dealer_card, have_A = obs
        ha = int(have_A)
        
        self.q[action, ha, dealer_card, sum_hand] += 0.001 * \
            (G-self.q[action, ha, dealer_card, sum_hand])

    def update_fromseq(self, seq, rseq):
        G = rseq[-1]
        for obs, act in seq:
            self.updateq(obs, act, G)

    def act(self, obs):
        sum_hand, dealer_card, have_A = obs
        if sum_hand >= 21:
            return 0
        elif sum_hand <= 11:
            return 1
        else:
            if have_A:
                return int(self.q[1, 1, dealer_card, sum_hand] > self.q[0, 1, dealer_card, sum_hand])
            else:
                return int(self.q[1, 0, dealer_card, sum_hand] > self.q[0, 0, dealer_card, sum_hand])


def usable_ace(hand):
    return 1 in hand and sum(hand) <= 11


def sum_hand(hand):
    return sum(hand) + 10 if usable_ace(hand) else sum(hand)


def is_bust(hand):
    return sum_hand(hand) > 21


def get_obs(hands, op_hand):
    obs = (sum_hand(hands), op_hand, usable_ace(hands))
    return obs


cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
cr = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


def train_one_epoch(player: Player, dealer: Player,  num_rounds):
    pwin = 0
    draw = 0
    plos = 0
    with alive_bar(num_rounds) as bar:
        for _ in range(num_rounds):
            bar()
            finish = False
            player_hands = [random.choice(cards), random.choice(cards)]
            pactionseq = []
            pr = []
            dealer_hands = [random.choice(cards), random.choice(cards)]
            dactionseq = []
            dr = []
            obs_player = get_obs(player_hands, dealer_hands[0])
            paction = player.act(obs_player)
            pactionseq.append((obs_player, paction))
            while paction == 1:
                pcard = random.choice(cards)
                player_hands.append(pcard)
                if obs_player[0]+pcard > 21:
                    finish = True
                    plos += 1
                    pr.append(-1)
                    player.update_fromseq(pactionseq, pr)
                    break
                pr.append(0)
                obs_player = get_obs(player_hands, dealer_hands[0])
                paction = player.act(obs_player)
                pactionseq.append((obs_player, paction))
            if finish:
                continue

            obs_dealer = get_obs(dealer_hands, player_hands[0])
            daction = dealer.act(obs_dealer)
            dactionseq.append((obs_dealer, daction))
            while daction == 1:
                dcard = random.choice(cards)
                dealer_hands.append(dcard)
                if obs_dealer[0]+dcard > 21:
                    finish = True
                    pwin += 1
                    pr.append(1)
                    dr.append(-1)
                    dealer.update_fromseq(dactionseq, dr)
                    player.update_fromseq(pactionseq, pr)
                    break
                dr.append(0)
                obs_dealer = get_obs(dealer_hands, player_hands[0])
                daction = dealer.act(obs_dealer)
                dactionseq.append((obs_dealer, daction))
            if finish:
                continue
            pp = sum_hand(player_hands)
            dp = sum_hand(dealer_hands)
            if pp > dp:
                pr.append(1)
                dr.append(-1)
                pwin += 1
            elif pp == dp:
                pr.append(0)
                dr.append(0)
                draw += 1
            else:
                pr.append(-1)
                dr.append(1)
                plos += 1
            player.update_fromseq(pactionseq, pr)
            dealer.update_fromseq(dactionseq, dr)
    return pwin, draw, plos


def main():
    player = Player()
    dealer = Player()
    epoch = 100
    for e in range(epoch):
        oldpq = player.q.copy()
        olddq = dealer.q.copy()
        w, d, l = train_one_epoch(player,  dealer,  100000)
        print(
            f'epoch: {e}\t{w=}\t{d=}\t{l=}\tdeltaq:{np.linalg.norm(oldpq-player.q):.2f}\t{np.linalg.norm(olddq-dealer.q):.2f}')

    np.save("results/xzh/ndealerQ", dealer.q)
    np.save("results/xzh/nplayerQ", player.q)


if __name__ == '__main__':
    main()
