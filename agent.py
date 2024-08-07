
import os
import glob
import numpy as np
import logging
import shutil
import math
from datetime import datetime
from PIL import Image
from collections import deque

from omegaconf import OmegaConf
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import PPOMemory
from model import ActorCritic, Discriminator
from scheduler import WarmupLinearSchedule
from utils.general import (
    set_seed, get_rng_state, set_rng_state,
    pretty_config, get_cur_time_code,
    TimerManager, get_config, get_device
)
from utils.stuff import RewardScaler, ObservationNormalizer
from utils.expert import get_expert_data




logger = logging.getLogger(__name__)

class PPOAgent:
    def __init__(self, config, for_eval=False):

        self.config = config
        self.device = get_device(config.device)
        
        set_seed(self.config.seed)
        rng_state, _ = gym.utils.seeding.np_random(self.config.seed)
        self.env_rng_state = rng_state
        
        # -------- Define models --------
        self.network = ActorCritic(config, self.device).to(self.device)        
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            **self.config.network.optimizer
        )

        if 'gail' in self.config:
            self.disc = Discriminator(config)
            self.disc_optimizer  = torch.optim.Adam([
                {'params': self.disc.parameters(),
                **config.gail.optimizer}
            ])
            
            if not for_eval:
                self.expert_data = get_expert_data(config)

        if self.config.train.scheduler:
            self.scheduler = WarmupLinearSchedule(optimizer=self.optimizer,
                                                warmup_steps=0,
                                                max_steps=self.config.train.total_timesteps // (self.config.train.max_episode_len * self.config.env.num_envs))
            if 'gail' in self.config:
                self.disc_scheduler = WarmupLinearSchedule(optimizer=self.disc_optimizer,
                                                        warmup_steps=0, 
                                                        max_steps=self.config.train.total_timesteps // (self.config.train.max_episode_len * self.config.env.num_envs))
                    
        # [EXPERIMENT] - reward scaler: r / rs.std()
        if self.config.train.reward_scaler:
            self.reward_scaler = RewardScaler(self.config.env.num_envs, gamma=self.config.train.gamma)

        # [EXPERIMENT] - observation scaler: (ob - ob.mean()) / (ob.std())
        if self.config.train.observation_normalizer:
            sp = (config.env.state_dim, ) if isinstance(config.env.state_dim, int) else list(config.env.state_dim)
            self.obs_normalizer = ObservationNormalizer(self.config.env.num_envs, sp)

        self.timer_manager  = TimerManager()
        self.writer         = None
        self.memory         = None
        self.timesteps      = 0

        logger.info("----------- Config -----------")
        pretty_config(config, logger=logger)
        logger.info(f"Device: {self.device}")
    

    def save(self, postfix, envs=None):
        '''
        ckpt_root
            exp_name
                config.yaml
                checkpoints
                    1
                    2
                    ...
                
        '''

        ckpt_path = os.path.join(self.config.experiment_path, "checkpoints")
        if os.path.exists(ckpt_path):
            # In order to save only the maximum number of checkpoints as max_save_store,
            # checkpoints exceeding that number are deleted. (exclude 'best')
            current_ckpt = [f for f in os.listdir(ckpt_path) if f.startswith('timesteps')]
            current_ckpt.sort(key=lambda x: int(x[9:]))
            # Delete exceeded checkpoints
            if self.config.train.max_ckpt_count > 0 and self.config.train.max_ckpt_count <= len(current_ckpt):
                for ckpt in current_ckpt[:len(current_ckpt) - self.config.train.max_ckpt_count - 1]:
                    shutil.rmtree(os.path.join(self.config.experiment_path, "checkpoints", ckpt), ignore_errors=True)


        # Save configuration file
        os.makedirs(self.config.experiment_path, exist_ok=True)
        with open(os.path.join(self.config.experiment_path, "config.yaml"), 'w') as fp:
            OmegaConf.save(config=self.config, f=fp)

        # postfix is ​​a variable for storing each episode or the best model
        ckpt_path = os.path.join(ckpt_path, postfix)
        os.makedirs(ckpt_path, exist_ok=True)
        
        # save model and optimizers
        torch.save(self.network.state_dict(), os.path.join(ckpt_path, "network.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
        if 'gail' in self.config:
            torch.save(self.disc.state_dict(), os.path.join(ckpt_path, "discriminator.pt"))
            torch.save(self.disc_optimizer.state_dict(), os.path.join(ckpt_path, "disc_optimizer.pt"))
        if self.config.train.scheduler:
            torch.save(self.scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))
            if 'gail' in self.config:
                torch.save(self.disc_scheduler.state_dict(), os.path.join(ckpt_path, "disc_scheduler.pt"))

        if self.config.train.reward_scaler:
            self.reward_scaler.save(ckpt_path)
        if self.config.train.observation_normalizer:
            self.obs_normalizer.save(ckpt_path)

        # save random state
        torch.save(get_rng_state(), os.path.join(ckpt_path, 'rng_state.ckpt'))
        if envs:
            torch.save(envs.np_random, os.path.join(ckpt_path, 'env_rng_state.ckpt'))

        with open(os.path.join(ckpt_path, "appendix"), "w") as f:
            f.write(f"{self.timesteps}\n")

    @classmethod
    def load(cls, experiment_path, postfix, resume=True):

        config = get_config(os.path.join(experiment_path, "config.yaml"))
        if config.env.is_continuous:
            config.network.action_std_init = config.network.min_action_std
        ppo_algo = PPOAgent(config, for_eval=True)
        
        # Create a variable to indicate which path the model will be read from
        ckpt_path = os.path.join(experiment_path, "checkpoints", postfix)
        print(f"Load pretrained model from {ckpt_path}")

        ppo_algo.network.load_state_dict(torch.load(os.path.join(ckpt_path, "network.pt")))
        ppo_algo.optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, "optimizer.pt")))
        if 'gail' in ppo_algo.config:
            ppo_algo.disc.load_state_dict(torch.load(os.path.join(ckpt_path, "discriminator.pt")))
            ppo_algo.disc_optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, "disc_optimizer.pt")))
        if ppo_algo.config.train.scheduler:
            ppo_algo.scheduler.load_state_dict(torch.load(os.path.join(ckpt_path, "scheduler.pt")))
            if 'gail' in ppo_algo.config:
                ppo_algo.disc_scheduler.load_state_dict(torch.load(os.path.join(ckpt_path, "disc_scheduler.pt")))
        
        if ppo_algo.config.train.reward_scaler:
            ppo_algo.reward_scaler.load(ckpt_path)
        if ppo_algo.config.train.observation_normalizer:
            ppo_algo.obs_normalizer.load(ckpt_path)

        # load random state
        set_rng_state(torch.load(os.path.join(ckpt_path, 'rng_state.ckpt'), map_location='cpu'))

        with open(os.path.join(ckpt_path, "appendix"), "r") as f:
            lines = f.readlines()

        if resume:
            ppo_algo.timesteps = int(lines[0])
            if os.path.exists(os.path.join(ckpt_path, 'env_rng_state.ckpt')):
                ppo_algo.env_rng_state = torch.load(os.path.join(ckpt_path, 'env_rng_state.ckpt'), map_location='cpu')

        return ppo_algo

    def prepare_data(self, data):       
        s        = data['state'].float()
        a        = data['action']
        logp     = data['logprob'].float()
        v_target = data['v_target'].float()
        adv      = data['advant'].float()
        v        = data['value'].float()

        # # normalize advant a.k.a atarg
        # adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        return s, a, logp, adv, v_target, v

    def optimize_gail(self, data):

        def gail_iter(batch_size, given_data, num_for_loop=None):
            '''
            
            given_data: (state_tensor, action_tensor)
            '''
            # Simple mini-batch spliter
            ob, ac = given_data
            total_size = len(ob)
            indices = np.arange(total_size)
            if num_for_loop and len(indices) < num_for_loop:
                # Adjusting the expert's data to match the number of data points of the learner
                indices = np.append(indices, np.random.randint(0, 
                                                               total_size, 
                                                               num_for_loop - len(indices)))
                total_size = len(indices)
            np.random.shuffle(indices)
            n_batches = total_size // batch_size
            for nb in range(n_batches):
                ind = indices[batch_size * nb : batch_size * (nb + 1)]
                yield ob[ind], ac[ind]
        
        loss_fn = nn.BCELoss()
        discriminator_losses = []
        learner_accuracies = []
        expert_accuracies = []
        
        if self.config.gail.batch_size == -1:
            self.config.gail.batch_size = max(len(data[0]), len(self.expert_data[0]))

        learner_iter = gail_iter(self.config.gail.batch_size, data[0:2])
        expert_iter = gail_iter(self.config.gail.batch_size, self.expert_data, num_for_loop=len(data[0]))

        self.disc.train()
        for _ in range(self.config.gail.epoch):
            for ob, ac in learner_iter:
                expert_ob, expert_ac = next(expert_iter)

                learner_prob = self.disc(ob, ac)
                expert_prob = self.disc(expert_ob, expert_ac)

                learner_loss = loss_fn(learner_prob, torch.ones_like(learner_prob))
                expert_loss = loss_fn(expert_prob, torch.zeros_like(expert_prob))
            
                # maximize E_learner [ log(D(s,a))] + E_expert [ log(1 - D(s,a))]
                loss = learner_loss + expert_loss
                discriminator_losses.append(loss.item())

                self.disc_optimizer.zero_grad()
                loss.backward()
                self.disc_optimizer.step()

                learner_acc = ((learner_prob >= 0.5).float().mean().item())
                expert_acc = ((expert_prob < 0.5).float().mean().item())

                learner_accuracies.append(learner_acc)
                expert_accuracies.append(expert_acc)

        avg_d_loss = np.mean(discriminator_losses)
        avg_learner_accuracy = np.mean(learner_accuracies)
        avg_expert_accuracy = np.mean(expert_accuracies)

        self.writer.add_scalar("train_gail/discrim_loss", avg_d_loss, self.timesteps)
        self.writer.add_scalar("train_gail/learner_accuracy", avg_learner_accuracy, self.timesteps)
        self.writer.add_scalar("train_gail/expert_accuracy", avg_expert_accuracy, self.timesteps)
                                                        

    def optimize_ppo(self, data):

        def ppo_iter(batch_size, given_data):
            # Simple mini-batch spliter

            ob, ac, oldpas, adv, tdlamret, old_v = given_data
            total_size = len(ob)
            indices = np.arange(total_size)
            np.random.shuffle(indices)
            n_batches = total_size // batch_size
            for nb in range(n_batches):
                ind = indices[batch_size * nb : batch_size * (nb + 1)]
                yield ob[ind], ac[ind], oldpas[ind], adv[ind], tdlamret[ind], old_v[ind]


        # -------- PPO Training Loop --------

        self.network.train()
        for _ in range(self.config.train.ppo.optim_epochs):
            data_loader = ppo_iter(self.config.train.ppo.batch_size, data)

            policy_losses   = []
            entropy_losses  = []
            value_losses    = []
            total_losses    = []

            with self.timer_manager.get_timer("\t\tone_epoch"):
                for batch in data_loader:
                    ob, ac, old_logp, adv, vtarg, old_v = batch
                    adv = (adv - adv.mean()) / (adv.std() + 1e-7)

                    # -------- Loss calculate --------

                    # --- policy loss
                    _, cur_logp, cur_ent, cur_v = self.network(ob, action=ac)
                    cur_v = cur_v.reshape(-1)

                    ratio = torch.exp(cur_logp - old_logp)
                    surr1 = ratio * adv

                    if self.config.train.ppo.loss_type == "clip":
                        # clipped loss
                        clipped_ratio = torch.clamp(ratio, 1. - self.config.train.ppo.eps_clip, 1. + self.config.train.ppo.eps_clip)
                        surr2 = clipped_ratio * adv

                        policy_surr = torch.min(surr1, surr2)
                        
                    elif self.config.train.ppo.loss_type == "kl":
                        # kl-divergence loss
                        policy_surr = surr1 - 0.01 * torch.exp(old_logp) * (old_logp - cur_logp)
                    else:
                        # simple ratio loss
                        policy_surr = surr1
                    
                    policy_surr = -policy_surr.mean()

                    # --- entropy loss

                    policy_ent = -cur_ent.mean()

                    # --- value loss

                    if self.config.train.ppo.value_clipping:
                        cur_v_clipped = old_v + (cur_v - old_v).clamp(-self.config.train.ppo.eps_clip, self.config.train.ppo.eps_clip)
                        vloss1 = (cur_v - vtarg) ** 2 # F.smooth_l1_loss(cur_v, vtarg, reduction='none')
                        vloss2 = (cur_v_clipped - vtarg) ** 2 # F.smooth_l1_loss(cur_v_clipped, vtarg, reduction='none')
                        vf_loss = torch.max(vloss1, vloss2)
                    else:
                        vf_loss = (cur_v - vtarg) ** 2 #F.smooth_l1_loss(cur_v, vtarg, reduction='none')
                        
                    vf_loss = 0.5 * vf_loss.mean()

                    # -------- Backward process --------

                    c1 = self.config.train.ppo.coef_value_function
                    c2 = self.config.train.ppo.coef_entropy_penalty

                    total_loss = policy_surr + c2 * policy_ent + c1 * vf_loss

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    if self.config.train.clipping_gradient:
                        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                    self.optimizer.step()

                    # ---------- Record training loss data ----------

                    policy_losses.append(policy_surr.item())
                    entropy_losses.append(policy_ent.item())
                    value_losses.append(vf_loss.item())
                    total_losses.append(total_loss.item())

            avg_policy_loss = np.mean(policy_losses)
            avg_entropy_loss = np.mean(entropy_losses)
            avg_value_loss = np.mean(value_losses)
            avg_total_loss = np.mean(total_losses)

            self.writer.add_scalar("train/policy_loss", avg_policy_loss, self.timesteps)
            self.writer.add_scalar("train/entropy_loss", avg_entropy_loss, self.timesteps)
            self.writer.add_scalar("train/value_loss", avg_value_loss, self.timesteps)
            self.writer.add_scalar("train/total_loss", avg_total_loss, self.timesteps)              

    def optimize(self, data):
        with self.timer_manager.get_timer("\tprepare_data"):
            data = self.prepare_data(data)

        if 'gail' in self.config:
            with self.timer_manager.get_timer("\toptimize_gail"):
                self.optimize_gail(data)

        with self.timer_manager.get_timer("\toptimize_ppo"):
            self.optimize_ppo(data)


    def step(self, envs, exp_name=None):

        # Set random state for reproducibility
        envs.np_random = self.env_rng_state        

        # -------- Initialize --------

        start_time = datetime.now().replace(microsecond=0)
        
        # Create an experiment directory to record training data
        self.config.experiment_name = f"exp{get_cur_time_code()}" if exp_name is None else exp_name
        self.config.experiment_path = os.path.join(self.config.checkpoint_path, self.config.experiment_name)

        # If an existing experiment has the same name, add a number to the end of the path.
        while os.path.exists(self.config.experiment_path):
            exp_name  = self.config.experiment_path[len(self.config.checkpoint_path) + 1:]
            exp_split = exp_name.split("_")

            try:
                exp_num  = int(exp_split[-1]) + 1
                exp_name = f"{'_'.join(exp_split[:max(1, len(exp_split) - 1)])}_{str(exp_num)}"
            except:
                exp_name = f"{exp_name}_0"

            self.config.experiment_name = exp_name
            self.config.experiment_path = os.path.join(self.config.checkpoint_path, self.config.experiment_name)
        os.makedirs(self.config.experiment_path, exist_ok=True)
        logger.addHandler( logging.FileHandler(os.path.join(self.config.experiment_path, f"running_train_log.log")))

        # For logging training state
        writer_path     = os.path.join( self.config.experiment_path, 'runs')
        self.writer     = SummaryWriter(writer_path)
        # Queue to record learning data,
        # [0] is a value to prevent errors caused by missing data.
        reward_queue    = deque([0], maxlen=self.config.train.average_interval)
        duration_queue  = deque([0], maxlen=self.config.train.average_interval)
        if 'gail' in self.config:
            irl_score_queue     = deque([0], maxlen=self.config.train.average_interval)

        episodic_reward = np.zeros(self.config.env.num_envs)
        duration        = np.zeros(self.config.env.num_envs)
        if 'gail' in self.config:
            irl_episodic_reward = np.zeros(self.config.env.num_envs)
        best_score      = -1e9

        # make rollout buffer
        self.memory = PPOMemory(
            gamma=self.config.train.gamma,
            tau=self.config.train.tau,
            device=self.device
        )        
          
        # for continuous action space
        if self.config.env.is_continuous:
            next_action_std_decay_step = self.config.network.action_std_decay_freq

        '''
        Environment symbol's information
        ===========  ==========================  ==================
        Symbol       Shape                       Type
        ===========  ==========================  ==================
        state        (num_envs, (obs_space))     numpy.ndarray
        reward       (num_envs,)                 numpy.ndarray
        term         (num_envs,)                 numpy.ndarray
        done         (num_envs,)                 numpy.ndarray
        ===========  ==========================  ==================
        '''
        state, _  = envs.reset()
        done = np.zeros(self.config.env.num_envs)

        # -------- Training Loop --------
        
        print(f"================ Start training ================")
        print(f"========= Exp name: {self.config.experiment_name} ==========")
        while self.timesteps < self.config.train.total_timesteps:

            with self.timer_manager.get_timer("Total"):
                with self.timer_manager.get_timer("Collect Trajectory"):
                    for t in range(0, self.config.train.max_episode_len ):

                        # ------------- Collect Trajectories -------------

                        '''
                        Actor-Critic symbol's information
                        ===========  ==========================  ==================
                        Symbol       Shape                       Type
                        ===========  ==========================  ==================
                        action       (num_envs,)                 torch.Tensor
                        logprobs     (num_envs,)                 torch.Tensor
                        ent          (num_envs,)                 torch.Tensor
                        values       (num_envs, 1)               torch.Tensor
                        ===========  ==========================  ==================
                        '''

                            
                        with torch.no_grad():
                            if self.config.train.observation_normalizer:
                                state = self.obs_normalizer(state)
                            state = torch.from_numpy(state).to(self.device, dtype=torch.float)
                            action, logprobs, _, values = self.network(state)
                            values = values.flatten() # reshape shape of the value to (num_envs,)
                        if self.config.env.is_continuous:
                            next_state, reward, terminated, truncated, _ = envs.step(np.clip(action.cpu().numpy(), envs.action_space.low, envs.action_space.high))
                        else:
                            next_state, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
                        self.timesteps += self.config.env.num_envs

                        # update episodic_reward
                        episodic_reward += reward
                        duration += 1

                        if 'gail' in self.config:
                            with torch.no_grad():
                                irl_reward = self.disc.get_irl_reward(state, action).cpu().detach().squeeze(-1).numpy()
                                irl_episodic_reward += irl_reward
                                
                                alpha = min(1, self.timesteps / self.config.train.total_timesteps)
                                reward = alpha * irl_reward + (1 - alpha) * reward

                        if self.config.train.reward_scaler:
                            reward = self.reward_scaler(reward, terminated + truncated)
                            
                        # add experience to the memory                    
                        self.memory.store(
                            state=state,
                            action=action,
                            reward=reward,
                            done=done,
                            value=values,
                            logprob=logprobs
                        )
                        done = terminated + truncated

                        for idx, d in enumerate(done):
                            if d:
                                reward_queue.append(episodic_reward[idx])
                                duration_queue.append(duration[idx])                            
                                if 'gail' in self.config:
                                    irl_score_queue.append(irl_episodic_reward[idx])

                                episodic_reward[idx] = 0
                                duration[idx] = 0                    
                                if 'gail' in self.config:
                                    irl_episodic_reward[idx] = 0
                        
                        # update state
                        state = next_state                        
                    
                # ------------- Calculate gae for optimizing-------------

                # Estimate next state value for gae
                with torch.no_grad():
                    if self.config.train.observation_normalizer:
                        next_state = self.obs_normalizer(next_state)
                    _, _, _, next_value = self.network(torch.Tensor(next_state).to(self.device))
                    next_value = next_value.flatten()

                # update gae & tdlamret
                # Optimize
                with self.timer_manager.get_timer("Optimize"):
                    with self.timer_manager.get_timer("Calculate gae"):
                        data = self.memory.compute_gae_and_get(next_value, done)
                    self.optimize(data)

                # action std decaying
                if self.config.env.is_continuous:
                    while self.timesteps > next_action_std_decay_step:
                        next_action_std_decay_step += self.config.network.action_std_decay_freq
                        self.network.action_decay(
                            self.config.network.action_std_decay_rate,
                            self.config.network.min_action_std
                        )

                # scheduling learning rate
                if self.config.train.scheduler:
                    self.scheduler.step()
                    if 'gail' in self.config:
                        self.disc_scheduler.step()

            # ------------- Logging training state -------------

            avg_score       = np.round(np.mean(reward_queue), 4)
            std_score       = np.round(np.std(reward_queue), 4)
            avg_duration    = np.round(np.mean(duration_queue), 4)

            # Writting for tensorboard
            self.writer.add_scalar("train/score", avg_score, self.timesteps)
            self.writer.add_scalar("train/duration", avg_duration, self.timesteps)
            if 'gail' in self.config:
                avg_irl_score = np.mean(irl_score_queue)
                self.writer.add_scalar("train_gail/irl_score", avg_irl_score, self.timesteps)
            if self.config.train.scheduler:
                for idx, lr in enumerate(self.scheduler.get_lr()):
                    self.writer.add_scalar(f"train/learning_rate{idx}", lr, self.timesteps)
                if 'gail' in self.config:
                    for idx, lr in enumerate(self.disc_scheduler.get_lr()):
                        self.writer.add_scalar(f"train_gail/learning_rate{idx}", lr, self.timesteps)

            # Printing for console
            remaining_num_of_optimize = int(math.ceil((self.config.train.total_timesteps - self.timesteps) /
                                                      (self.config.env.num_envs * self.config.train.max_episode_len)))
            remaining_training_time_min = int(self.timer_manager.get_timer('Total').get() * remaining_num_of_optimize // 60)
            remaining_training_time_sec = int(self.timer_manager.get_timer('Total').get() * remaining_num_of_optimize % 60)
            logger.info(f"[{datetime.now().replace(microsecond=0) - start_time}] {self.timesteps}/{self.config.train.total_timesteps} - score: {avg_score} +-{std_score} \t duration: {avg_duration}")
            for k, v in self.timer_manager.timers.items():
                logger.info(f"\t\t {k} time: {v.get()} sec")
            logger.info(f"\t\t Estimated training time remaining: {remaining_training_time_min} min {remaining_training_time_sec} sec")

            # Save best model
            if avg_score >= best_score:
                self.save(f'best', envs)
                best_score = avg_score

            self.save(f"timesteps{self.timesteps}", envs)

        envs.close()
        self.save('last')
        return best_score

    
    def play(self, env, max_ep_len, num_episodes=10):  

        rewards = []
        durations = []

        for episode in range(num_episodes):

            episodic_reward = 0
            duration = 0
            state, _ = env.reset()

            for t in range(max_ep_len):

                # ------------- Collect Trajectories -------------

                with torch.no_grad():
                    if self.config.train.observation_normalizer:
                        state = self.obs_normalizer(state, update=False)
                    action, _, _, _ = self.network(torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.float))
                    
                if self.config.env.is_continuous:
                    next_state, reward, terminated, truncated, info = env.step(np.clip(action.cpu().numpy().squeeze(0), 
                                                                                       env.action_space.low, 
                                                                                       env.action_space.high))
                else:
                    next_state, reward, terminated, truncated, info = env.step(action.cpu().numpy().squeeze(0))

                episodic_reward += reward
                duration += 1

                done = terminated + truncated
                if done:
                    break

                # update state
                state = next_state

            rewards.append(episodic_reward)
            durations.append(duration)
            logger.info(f"Episode {episode}: score - {episodic_reward} duration - {t}")

        avg_reward = np.mean(rewards)
        avg_duration = np.mean(durations)
        logger.info(f"Average score {avg_reward}, duration {avg_duration} on {num_episodes} games")               
        env.close()

