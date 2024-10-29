from StudentCode.BlackJackBattleEnv import SimplePolicy

from StudentCode.Group_11.Group_Policy_11 import Policy_11 as P11
from StudentCode.Group_11.Group_Policy_11 import Policy_10 as P10

Group11 = P11()
Group10 = P10()
agents = [Group11, Group10, SimplePolicy(17), SimplePolicy(
    16), SimplePolicy(15), SimplePolicy(18)]
agent_names = ['Group11', "Group10", "Agent17", "Agent16", "Agent15", "Agent18"]

assert len(agents) == len(agent_names), '智能体和组名个数不匹配'
