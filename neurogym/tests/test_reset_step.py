from packaging import version

import gym
import numpy as np

import neurogym as ngym
from neurogym.utils.scheduler import RandomSchedule
from neurogym.wrappers import ScheduleEnvs


# In gym 0.24.0, env_checker calls reset() when the env is created => no error if env.step() before env.reset() but it
# doens't mean that ScheduleEnvs properly reset all its env, so disable env_checker to test that
# TODO: already disable env_checker in ngym.make for now so don't have to do it here
# if version.parse(gym.__version__) >= version.parse('0.24.0'):
#     disable_env_checker = True
# else:
#     disable_env_checker = False
disable_env_checker = False


def make_env(name, **kwargs):
    if disable_env_checker:
        return ngym.make(name, disable_env_checker=True, **kwargs)
    else:
        # cannot add the arg disable_env_checker to gym.make in versions lower than 0.24
        return ngym.make(name, **kwargs)


class CstObTrialWrapper(ngym.TrialWrapper):
    def __init__(self, env, cst_ob):
        super().__init__(env)
        self.cst_ob = cst_ob

    def new_trial(self, **kwargs):
        trial = self.env.new_trial(**kwargs)
        self.ob = np.repeat(self.cst_ob[None, :], self.ob.shape[0], axis=0)
        return trial

    # modifying new_trial is not enough to modify the ob returned by step()
    def step(self, action):
        _, reward, done, info = self.env.step(action)
        new_ob = self.ob[self.t_ind]
        return new_ob, reward, done, info


def _setup_env(cst_ob):
    env = make_env(ngym.all_envs()[0])
    env = CstObTrialWrapper(env, cst_ob)
    return env


def test_wrapper_new_trial():
    """
    Test that the ob returned by new_trial takes the wrapper correctly into account
    """
    cst_ob = np.random.random(10)
    env = _setup_env(cst_ob)
    env.new_trial()
    ob = env.ob[0]
    assert ob.shape == cst_ob.shape, "Got shape {} but expected shape {}".format(ob.shape, cst_ob.shape)
    assert np.all(ob == cst_ob)


def test_wrapper_reset():
    """
    Test that the ob returned by reset takes the wrapper correctly into account.
    """
    cst_ob = np.random.random(10)
    env = _setup_env(cst_ob)
    ob = env.reset()

    assert ob.shape == cst_ob.shape, "Got shape {} but expected shape {}".format(ob.shape, cst_ob.shape)
    assert np.all(ob == cst_ob)


def test_wrapper_step():
    """
    Test that the ob returned by step takes the wrapper correctly into account.
    """
    cst_ob = np.random.random(10)
    env = _setup_env(cst_ob)
    env.reset()
    ob, _, _, _ = env.step(env.action_space.sample())
    assert ob.shape == cst_ob.shape, "Got shape {} but expected shape {}".format(ob.shape, cst_ob.shape)
    assert np.all(ob == cst_ob)


def test_reset_with_scheduler():
    """
    Test that ScheduleEnvs.reset() resets all the environments in its list envs, which is required before being able to
    call step() (enforced by the gym wrapper OrderEnforcing).
    """
    tasks = ngym.get_collection('yang19')
    envs = [make_env(task) for task in tasks]
    schedule = RandomSchedule(len(envs))
    env = ScheduleEnvs(envs, schedule=schedule, env_input=True)

    env.reset()
    env.step(env.action_space.sample())


def test_schedule_envs():
    tasks = ngym.get_collection('yang19')
    envs = [make_env(task) for task in tasks]
    for i, env in enumerate(envs):
        envs[i] = CstObTrialWrapper(env, np.array([i]))

    schedule = RandomSchedule(len(envs))
    env = ScheduleEnvs(envs, schedule=schedule, env_input=True)
    env.reset()
    for _ in range(5):
        env.new_trial()
        assert np.all([ob == env.i_env for ob in env.ob])
        # test rule input
        assert env.i_env == np.argmax(env.unwrapped.ob[0, -len(envs):])
