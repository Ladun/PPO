import gymnasium as gym

# from stable_baselines3.common.atari_wrappers import (  # isort:skip
#     ClipRewardEnv,      # Clip the reward to {+1, 0, -1} by its sign.
#     EpisodicLifeEnv,    # Make end-of-life == end-of-episode, but only reset on true game over. Done by DeepMind for the DQN and co. since it helps value estimation.
#     FireResetEnv,       # Take action on reset for environments that are fixed until firing.
#     MaxAndSkipEnv,      # Return only every ``skip``-th frame (frameskipping) and return the max between the two last frames.
#     NoopResetEnv,       # Sample initial states by taking random number of no-ops on reset. No-op is assumed to be action 0.
# )

# def create_atari_env(env_name, video_store_path=None,
#                stack_frame=4,
#                frameskip=4,
#                resized_img=(84,84),
#                noop_max=30,
#                gray_scale=True,
#                episodic_life=False,
#                reward_clip=False,
#                repeat_action_probability=0.0):
#     env = gym.make(env_name, render_mode="rgb_array", frameskip=1, repeat_action_probability=repeat_action_probability)
#     env = gym.wrappers.RecordEpisodeStatistics(env)
#     if video_store_path:
#         env = gym.wrappers.RecordVideo(env, video_store_path)
#     if noop_max > 0:
#         env = NoopResetEnv(env, noop_max=noop_max)
#     if frameskip > 1:
#         env = MaxAndSkipEnv(env, skip=frameskip)
#     if episodic_life:
#         env = EpisodicLifeEnv(env)
#     if "FIRE" in env.unwrapped.get_action_meanings():
#         env = FireResetEnv(env)
#     if reward_clip:
#         env = ClipRewardEnv(env)
#     if resized_img is not None:
#         env = gym.wrappers.ResizeObservation(env, resized_img)
#     if gray_scale:
#         env = gym.wrappers.GrayScaleObservation(env)
#     if stack_frame > 1:
#         env = gym.wrappers.FrameStack(env, stack_frame)
#     return env

def create_mujoco_env(env_name, video_store_path=None):
    env = gym.make(env_name, render_mode='rgb_array')
    if video_store_path:
        env = gym.wrappers.RecordVideo(env, video_store_path)
    return env