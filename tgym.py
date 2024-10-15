import matplotlib
import numpy as np
import gym
import sys
import random
from collections import defaultdict
from matplotlib import pyplot as plt
import seaborn as sns
from alive_progress import alive_bar


# 编写三维画图函数
def plot_3D(X, Y, Z, xlabel, ylabel, zlabel, title):
    fig = plt.figure(figsize=(20, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.rainbow, vmin=-1.0, vmax=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.view_init(ax.elev, -120)
    ax.set_facecolor("white")
    fig.colorbar(surf)
    return fig


# 首次访问MC预测算法
def MCES(env, num_episodes):
    """
    env         : 问题环境
    num_episodes: 幕数量
    return      : 返回状态价值函数与最优策略
    """

    # 初始化策略(任何状态下都不要牌)
    policy = defaultdict(int)
    # 初始化回报和
    r_sum = defaultdict(float)
    # 初始化访问次数
    r_count = defaultdict(float)
    # 初始化状态价值函数
    r_v = defaultdict(float)

    # 对各幕循环迭代
    with alive_bar(num_episodes) as bar:
        for each_episode in range(num_episodes):
            bar()
            # 输出迭代过程
            sys.stdout.flush()

            # 初始化空列表记录幕过程
            episode = []
            # 初始化环境
            state,info = env.reset()
            # 选择试探性的初始状态动作
            action = random.randint(0, 1)

            # 生成（采样）幕
            done = False
            while not done:
                # 驱动环境的物理引擎得到下一个状态、回报以及该幕是否结束标志
                next_state, reward, done, _, info = env.step(action)
                # 对幕进行采样并记录
                episode.append((state, action, reward))
                # 更新状态
                state = next_state
                # 根据当前状态获得策略下的下一动作
                action = policy[state]

            # 对生成的单幕内进行倒序迭代更新状态价值矩阵
            G = 0
            episode_len = len(episode)
            episode.reverse()
            for seq, data in enumerate(episode):
                # 记录当前状态
                state_visit = data[0]
                action = data[1]
                # 累加计算期望回报
                G += data[2]
                # 若状态第一次出现在该幕中则进行价值和策略更新
                if seq != episode_len - 1:
                    if data[0] in episode[seq+1:][0]:
                        continue
                r_sum[(state_visit, action)] += G
                r_count[(state_visit, action)] += 1
                r_v[(state_visit, action)] = r_sum[(state_visit, action)] / \
                    r_count[(state_visit, action)]
                if r_v[(state_visit, action)] < r_v[(state_visit, 1-action)]:
                    policy[state_visit] = 1 - action
    return policy, r_v


# 处理价值矩阵方便后续绘图
def process_q_for_draw(q, policy, ace):
    """
    v     : 状态价值函数
    ace   : 是否有可用A
    return: 返回处理好的三个坐标轴
    """
    # 根据动作价值函数到处最优状态价值函数
    v = defaultdict(float)
    for state in policy.keys():
        v[state] = q[(state, policy[state])]
    # 生成网格点
    x_range = np.arange(12, 22)
    y_range = np.arange(1, 11)
    X, Y = np.meshgrid(x_range, y_range)

    # 根据是否有可用的A选择绘制不同的3D图
    if ace:
        Z = np.apply_along_axis(
            lambda _: v[(_[0], _[1], True)], 2, np.dstack([X, Y]))
    else:
        Z = np.apply_along_axis(
            lambda _: v[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    return X, Y, Z


# 处理策略方便后续绘图
def process_policy_for_draw(policy, ace):
    """
    policy:输入策略
    ace   :是否有可用A
    return:以二维数组形式返回
    """
    policy_list = np.zeros((10, 10))
    # 将字典形式换为列表，方便后续作图
    if ace:
        for playerscores in range(12, 22):
            for dealercard in range(1, 11):
                policy_list[playerscores - 12][dealercard -
                                               1] = policy[(playerscores, dealercard, 1)]
    else:
        for playerscores in range(12, 22):
            for dealercard in range(1, 11):
                policy_list[playerscores - 12][dealercard -
                                               1] = policy[(playerscores, dealercard, 0)]
    return policy_list


# 主函数
if __name__ == '__main__':
    # 从gym库中调用Blackjack-v1环境
    env = gym.make("Blackjack-v1")
    # 对策略进行评估（预测）
    policy, q = MCES(env, num_episodes=5000000)
    print(policy)
    # 绘制最优策略矩阵热力图
    # 准备画布大小，并准备多个子图
    _, axes = plt.subplots(1, 2, figsize=(40, 20))
    # 调整子图的间距，wspace=0.1为水平间距，hspace=0.2为垂直间距
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    # 这里将子图形成一个1*2的列表
    axes = axes.flatten()
    # 有可用ACE下的最优策略
    fig = sns.heatmap(np.flipud(process_policy_for_draw(
        policy, 1)), cmap="Wistia", ax=axes[0])
    fig.set_ylabel('Player Sum', fontsize=20)
    fig.set_yticks(list(reversed(range(10))))
    fig.set_xlabel('Dealer Open Card', fontsize=20)
    fig.set_xticks(range(10))
    fig.set_title('Usable Ace', fontsize=20)
    # 无可用ACE下的最优策略
    fig = sns.heatmap(np.flipud(process_policy_for_draw(
        policy, 0)), cmap="Wistia", ax=axes[-1])
    fig.set_ylabel('Player Sum', fontsize=20)
    fig.set_yticks(list(reversed(range(10))))
    fig.set_xlabel('Dealer Open Card', fontsize=20)
    fig.set_xticks(range(10))
    fig.set_title('NO Usable Ace', fontsize=20)
    plt.show()
    plt.savefig("./Optimal Policy.jpg")

    # 3D绘图-状态价值矩阵
    X, Y, Z = process_q_for_draw(q, policy, ace=True)
    fig = plot_3D(X, Y, Z, xlabel="Player Sum",
                  ylabel="Dealer Open Card", zlabel="Value", title="Usable Ace")
    fig.show()
    fig.savefig("./Usable_Ace.jpg")
    X, Y, Z = process_q_for_draw(q, policy,  ace=False)
    fig = plot_3D(X, Y, Z, xlabel="Player Sum",
                  ylabel="Dealer Open Card", zlabel="Value", title="No Usable Ace")
    fig.show()
    fig.savefig("./NO_Usable_Ace.jpg")
