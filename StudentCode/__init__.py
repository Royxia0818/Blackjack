from StudentCode.BlackJackBattleEnv import SimplePolicy

from StudentCode.Group_11.Group_Policy_11 import Policy_11 as P11  # 可以参考上面的group_test
Group11 = P11()

agents = [Group11, SimplePolicy(17), SimplePolicy(
    16), SimplePolicy(15), SimplePolicy(18)]
agent_names = ['Group11', "Agent17", "Agent16", "Agent15", "Agent18"]

assert len(agents) == len(agent_names), '智能体和组名个数不匹配'
