from isaaclab.utils import configclass

from cyclotron.tasks.locomotion.agents.rsl_rl_ppo_cfg import Asimov1PPORunnerCfg


@configclass
class Asimov1StandIdealPPORunnerCfg(Asimov1PPORunnerCfg):
    # A new experiment name: runs land in logs/rsl_rl/asimov1_stand_ideal/.
    experiment_name = "asimov1_stand_ideal"
    run_name = "ppo"
    # Standing is a smaller problem than walking; this is a starting point, not a tuned value.
    max_iterations = 1500
    # A checkpoint every 50 iterations, to watch training progress afterwards.
    save_interval = 50


@configclass
class Asimov1StandIdealNoPosePPORunnerCfg(Asimov1StandIdealPPORunnerCfg):
    experiment_name = "asimov1_stand_ideal_no_pose"


@configclass
class Asimov1StandIdealNoActionRatePPORunnerCfg(Asimov1StandIdealPPORunnerCfg):
    experiment_name = "asimov1_stand_ideal_no_action_rate"


@configclass
class Asimov1StandIdealNoFallEndPPORunnerCfg(Asimov1StandIdealPPORunnerCfg):
    experiment_name = "asimov1_stand_ideal_no_fall_end"
