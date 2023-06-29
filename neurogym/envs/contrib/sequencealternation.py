#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from gymnasium import spaces
import neurogym as ngym


class SequenceAlternation(ngym.TrialEnv):
    """Sequence alternation matching task.

    The agent is rewarded when it selects the correct alternating sequence
    of elements as generated by the computer.
    opponent_type: Type of opponent. (def: 'mean_action', str)

    Args:
        sequence_length: Length of the sequence to be matched (int)
        element_space: Space of the elements in the sequence (gym.Space)
    """

    metadata = {
        "paper_link": "https://doi.org/10.1523/jneurosci.2901-04.2005",
        "paper_name": """Sequential-context-dependent hippocampal activity is
        not necessary to learn sequences with repeated elements""",
        "tags": ["sequence"],
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 4,
    }

    def __init__(
        self,
        dt=100,
        rewards=None,
        timing=None,
        num_zones=8,
        sequence_length=8,
        cued_epoch_periodicity=3,
        element_space=None,
        render_mode=None,
    ):
        super().__init__(dt=dt)
        if timing is not None:
            print(
                "Warning: Sequence-Alternation task does not require"
                + " timing variable."
            )

        self.num_zones = num_zones
        self.num_epoch = 0
        self.sequence_length = sequence_length  # expects even number
        self.cued_epoch_periodicity = cued_epoch_periodicity
        self.element_space = element_space or spaces.Discrete(num_zones)

        # Rewards
        self.rewards = {"correct": +1.0, "fail": 0.0}
        if rewards:
            self.rewards.update(rewards)

        self.action_space = self.element_space
        self.observation_space = spaces.MultiBinary(n=num_zones)

        self.sequence = self._generate_sequence()
        self.ob = np.zeros((1, self.element_space.n), dtype=np.bool)

        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _generate_sequence(self):
        numel = self.sequence_length // 2 + 1
        sequence = np.random.choice(self.element_space.n, size=numel, replace=False)
        sequence = np.insert(
            sequence, -1, values=sequence[: self.sequence_length - numel]
        )
        return sequence

    def set_groundtruth(self, value, period=None, where=None):
        """Set groundtruth value."""
        if not self._gt_built:
            self._init_gt()

        if where is not None:
            # TODO: Only works for Discrete action_space, make it work for Box
            value = self.action_space.name[where][value]
        if isinstance(period, str):
            self.gt[self.start_ind[period]: self.end_ind[period]] = value
        elif period is None:
            self.gt[:] = value
        else:
            for p in period:
                self.set_groundtruth(value, p)

    def view_groundtruth(self, period):
        """View observation of an period."""
        if not self._gt_built:
            self._init_gt()
        return self.gt[self.start_ind[period]:self.end_ind[period]]

    def in_period(self, period, t=None):
        """Check if current time or time t is in period"""
        if t is None:
            t = self.t  # Default
        return self.start_t[period] <= t < self.end_t[period]

    @property
    def gt_now(self):
        return self.gt[self.t_ind]

    def _new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Trial (trials are variable number of steps long)
        # ---------------------------------------------------------------------
        # determine the transitions
        self.seq_step = self.current_step = self.cumulative_reward = 0
        ground_truth = self.sequence[self.seq_step]
        trial = dict(seq_step=self.seq_step, current_step=self.current_step,
                     ground_truth=ground_truth)


        if (self.num_epoch / self.cued_epoch_periodicity) % 2 == 0:
            self.ob[0, ground_truth] = True  # cue light on

        self.gt = ground_truth

        info = {
            "new_trial": False,
            "gt": ground_truth,
            "cumulative_reward": self.cumulative_reward,
            "performance": self.performance,
        }

        self.set_groundtruth(ground_truth)

        return trial, info

    def _step(self, action):
        trial = self.trial
        obs = self.ob[0]
        reward = self.rewards["fail"]

        seq_step_cyc = trial["seq_step"] % self.sequence_length
        ground_truth = self.sequence[seq_step_cyc]
        if action == ground_truth:
            reward = self.rewards["correct"]
            self.cumulative_reward += 1
            obs[action] = False  # cue light off
            trial["seq_step"] += 1
            # if trial ended, next cue light on will be handled in _new_trial()
            if trial["seq_step"] == self.sequence_length: self.num_epoch += 1
            if (self.num_epoch / self.cued_epoch_periodicity) % 2 == 0:
                obs[self.sequence[trial["seq_step"] % self.sequence_length]] = True  # next cue light on

        self.current_step += 1
        self.performance = self.cumulative_reward / self.current_step
        done = trial["seq_step"] > self.sequence_length

        info = {
            "new_trial": done,
            "gt": ground_truth,
            "cumulative_reward": self.cumulative_reward,
            "performance": self.performance,
        }

        trial["current_step"] = self.current_step

        return obs, reward, done, False, info


if __name__ == "__main__":
    env = SequenceAlternation()
    ngym.utils.plot_env(env, num_steps=100)  # , def_act=0)
