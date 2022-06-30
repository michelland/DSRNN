policy_factory = dict()
def none_policy():
    return None

from crowd_nav.policy.orca import ORCA
from crowd_nav.policy.social_force import SOCIAL_FORCE
from crowd_nav.policy.srnn import SRNN
from crowd_nav.policy.srnn2 import SRNN2
from crowd_nav.policy.srnn3 import SRNN3
from crowd_nav.policy.srnn4 import SRNN4

policy_factory['orca'] = ORCA
policy_factory['none'] = none_policy
policy_factory['social_force'] = SOCIAL_FORCE
policy_factory['srnn'] = SRNN
policy_factory['srnn2'] = SRNN2
policy_factory['srnn3'] = SRNN3
policy_factory['srnn4'] = SRNN4

