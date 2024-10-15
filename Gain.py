import numpy as np
import random
from alive_progress import alive_bar
gamma = 0.9
alpha = 0.5


class Player:
    def __init__(self) -> None:
        self.q = np.zeros((2, 4, 22), dtype=float)

    def updateq(self, obs, action, reward):
        sum_hand, dealer_card, have_A = obs
        if sum_hand >= 21:
            return
        if action == 1:
            if have_A and dealer_card == 1:
                possible_state = [
                    sum_hand+_ for _ in range(1, 11) if sum_hand+_ <= 21]
                maxq = np.max(self.q[:, 3, possible_state])
                not_have_A_possible_state = [
                    sum_hand+_-10 for _ in range(1, 11) if sum_hand+_ > 21]
                nmaxq = np.max(self.q[:, 2, not_have_A_possible_state])
                if nmaxq > maxq:
                    maxq = nmaxq
                self.q[1, 3, sum_hand] += alpha * \
                    (reward+gamma*maxq-self.q[1, 3, sum_hand])
            elif dealer_card == 1:
                possible_state = [
                    sum_hand+_ for _ in range(1, 11) if sum_hand+_ <= 21]
                maxq = np.max(self.q[:, 2, possible_state])
                if sum_hand+11 <= 21:
                    maxq_A = np.max(self.q[:, 3, sum_hand+11])
                    if maxq < maxq_A:
                        maxq = maxq_A
                self.q[1, 2, sum_hand] += alpha * \
                    (reward+gamma*maxq-self.q[1, 2, sum_hand])
            elif have_A:
                possible_state = [
                    sum_hand+_ for _ in range(1, 11) if sum_hand+_ <= 21]
                maxq = np.max(self.q[:, 1, possible_state])
                not_have_A_possible_state = [
                    sum_hand+_-10 for _ in range(1, 11) if sum_hand+_ > 21]
                nmaxq = np.max(self.q[:, 0, not_have_A_possible_state])
                if nmaxq > maxq:
                    maxq = nmaxq
                self.q[1, 1, sum_hand] += alpha * \
                    (reward+gamma*maxq-self.q[1, 1, sum_hand])
            else:
                possible_state = [
                    sum_hand+_ for _ in range(1, 11) if sum_hand+_ <= 21]
                maxq = np.max(self.q[:, 0, possible_state])
                if sum_hand+11 <= 21:
                    maxq_A = np.max(self.q[:, 1, sum_hand+11])
                    if maxq < maxq_A:
                        maxq = maxq_A
                self.q[1, 0, sum_hand] += alpha * \
                    (reward+gamma*maxq-self.q[1, 0, sum_hand])
        else:
            if have_A and dealer_card == 1:
                self.q[0, 3, sum_hand] += alpha * \
                    (reward-self.q[0, 3, sum_hand])
            elif dealer_card == 1:
                self.q[0, 2, sum_hand] += alpha * \
                    (reward-self.q[0, 2, sum_hand])
            elif have_A:
                self.q[0, 1, sum_hand] += alpha * \
                    (reward-self.q[0, 1, sum_hand])
            else:
                self.q[0, 0, sum_hand] += alpha * \
                    (reward-self.q[0, 0, sum_hand])

    def act(self, obs):
        sum_hand, dealer_card, have_A = obs
        if sum_hand >= 21:
            return 0
        elif sum_hand <= 11:
            return 1
        else:
            if have_A and dealer_card == 1:
                return self.q[1, 3, sum_hand] > self.q[0, 3, sum_hand]
            elif dealer_card == 1:
                return self.q[1, 2, sum_hand] > self.q[0, 2, sum_hand]
            elif have_A:
                return self.q[1, 1, sum_hand] > self.q[0, 1, sum_hand]
            else:
                return self.q[1, 0, sum_hand] > self.q[0, 0, sum_hand]
    
    def update_fromseq(self, seq, reward):
        for obs, act in seq:
            self.updateq(obs, act, reward)


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
            dealer_hands = [random.choice(cards), random.choice(cards)]
            dactionseq = []
            obs_player = get_obs(player_hands, dealer_hands[0])
            paction = player.act(obs_player)
            pactionseq.append((obs_player, paction))
            while paction == 1:  # 玩家不停抽牌
                pcard = random.choice(cards)
                player_hands.append(pcard)
                if obs_player[0]+pcard > 21:
                    finish = True
                    player.update_fromseq(pactionseq, -1)
                    plos += 1
                    break
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
                    dealer.update_fromseq(dactionseq, -1)
                    pwin += 1
                    break
                obs_dealer = get_obs(dealer_hands, player_hands[0])
                daction = dealer.act(obs_dealer)
                dactionseq.append((obs_dealer, daction))
            if finish:
                continue
            pp = sum_hand(player_hands)
            dp = sum_hand(dealer_hands)
            if pp > dp:
                player.update_fromseq(pactionseq, 1)
                dealer.update_fromseq(dactionseq, -1)
                pwin += 1
            elif pp == dp:
                player.update_fromseq(pactionseq, 0)
                dealer.update_fromseq(dactionseq, 0)
                draw += 1
            else:
                player.update_fromseq(pactionseq, -1)
                dealer.update_fromseq(dactionseq, 1)
                plos += 1
    return pwin, draw, plos


def main():
    player = Player()
    dealer = Player()
    epoch = 25
    for e in range(epoch):
        oldpq = player.q.copy()
        olddq = dealer.q.copy()
        w, d, l = train_one_epoch(player,  dealer,  100000)
        print(
            f'epoch: {e}\t{w=}\t{d=}\t{l=}\tdeltaq:{np.linalg.norm(oldpq-player.q):.2f}\t{np.linalg.norm(olddq-dealer.q):.2f}')

    print(f'p_dealer: \n{dealer.q[1]-dealer.q[0]}')
    np.save("dealerQ", dealer.q)
    print(f'p_player: \n{player.q[1]-dealer.q[0]}')
    np.save("playerQ", player.q)


if __name__ == '__main__':
    main()
