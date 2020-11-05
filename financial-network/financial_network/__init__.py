import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Financial_network-v0',
    entry_point='financial_network.envs:Financial_Network_Env',
    )

