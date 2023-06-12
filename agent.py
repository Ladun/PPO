
import os
import glob
import numpy as np
from datetime import datetime
from PIL import Image
from collections import deque

from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from env import TrainEnvironment
from replay_buffer import PPOMemory
from model import Actor, Critic
from utils.general import set_seed, pretty_config
from utils.stuff import RewardScaler, ObservationScaler

class PPOAgent:
    def __init__(self, config, device):

        self.config = config
        self.device = device

        set_seed(config.seed)

        # -------- Define models --------

        self.actor      = Actor(config, device)
        self.critic     = Critic(config)
        self.optimizer  = torch.optim.Adam([
            {'params': self.actor.parameters(),
             **config.actor.optimizer},
            {'params': self.critic.parameters(),
             **config.critic.optimizer}
        ])

        if self.config.train.scheduler:
            self.scheduler  = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda epoch: 1 - epoch / self.config.train.max_episodes
            )
        # [EXPERIMENT] - reward scaler: r / rs.std()
        if self.config.train.reward_scaler:
            self.reward_scaler = RewardScaler(gamma=self.config.train.gamma)

        # [EXPERIMENT] - observation scaler: (ob - ob.mean()) / (ob.std())
        # if self.config.train_observation_scaler:
        #     self.obs_scaler = ObservationScaler()

        self.writer = None
        self.memory = None
        self.episode = 0

        print("----------- Config -----------")
        pretty_config(config)

    def get_experiments_base_path(self):
        return os.path.join(self.config.checkpoint_path, 'experiments', self.config.env.env_name, self.config.exp_name)

    def save(self, postfix):
        '''
        ckpt_root
            exp_name
                config.yaml
                checkpoints
                    1
                    2
                    ...
                
        '''
        # saving things
        #   actor, critic, optimizer
        #   config file
        base_path = self.get_experiments_base_path()
        os.makedirs(base_path, exist_ok=True)
        with open(os.path.join(base_path, "config.yaml"), 'w') as fp:
            OmegaConf.save(config=self.config, f=fp)

        # save model and optimizers
        ckpt_path = os.path.join(base_path, "checkpoints", postfix)
        os.makedirs(ckpt_path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(ckpt_path, "actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(ckpt_path, "critic.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
        if self.config.train.scheduler:
            torch.save(self.scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))

        with open(os.path.join(ckpt_path, "appendix"), "w") as f:
            f.write(f"{self.episode}\n")


    def load(self, postfix):
        base_path = self.get_experiments_base_path()

        ckpt_path = os.path.join(base_path, "checkpoints", postfix)
        self.actor.load_state_dict(torch.load(os.path.join(ckpt_path, "actor.pt")))
        self.critic.load_state_dict(torch.load(os.path.join(ckpt_path, "critic.pt")))
        self.optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, "optimizer.pt")))
        if self.config.train.scheduler:
            self.scheduler.load_state_dict(torch.load(os.path.join(ckpt_path, "scheduler.pt")))

    def prepare_data(self, data):       
        s        = torch.cat(data['states'], dim=0).float()
        a        = torch.cat(data['actions'], dim=0)
        logp     = torch.tensor(data['logpas']).float()
        tdlamret = torch.tensor(data['tdlamret']).float()
        adv      = torch.tensor(data['advants']).float()
        v        = torch.tensor(data['values']).float()

        # normalize advant a.k.a atarg
        adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        return s, a, logp, adv, tdlamret, v

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


        # -------- Initialize --------

        policy_losses   = []
        entropy_losses  = []
        value_losses    = []
        total_losses    = []

        # -------- PPO Training Loop --------

        for _ in range(self.config.train.ppo.optim_epochs):
            data_loader = ppo_iter(self.config.train.ppo.batch_size, data)

            for batch in data_loader:
                ob, ac, old_logp, adv, vtarg, old_v = batch

                # -------- Loss calculate --------

                # --- policy loss
                _, cur_logp, cur_ent = self.actor(ob, action=ac)
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

                # --- value loss
                cur_v = self.critic(ob)
                cur_v = cur_v.reshape(-1)

                if self.config.train.ppo.value_clipping:
                    cur_v_clipped = old_v + (cur_v - old_v).clamp(-self.config.train.ppo.eps_clip, self.config.train.ppo.eps_clip)
                    vloss1 = F.smooth_l1_loss(cur_v, vtarg)
                    vloss2 = F.smooth_l1_loss(cur_v_clipped, vtarg)
                    vf_loss = torch.max(vloss1, vloss2)
                else:
                    vf_loss =  F.smooth_l1_loss(cur_v, vtarg)

                # --- entropy loss
                policy_ent = -cur_ent.mean()

                # -------- Backward process --------

                c1 = self.config.train.ppo.coef_value_function
                c2 = self.config.train.ppo.coef_entropy_penalty

                self.optimizer.zero_grad()
                policy_loss = policy_surr + c2 * policy_ent
                value_loss = c1 * vf_loss

                total_loss = policy_loss + value_loss
                total_loss.backward()

                if self.config.train.clipping_gradient:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)

                self.optimizer.step()

                policy_losses.append(policy_surr.item())
                entropy_losses.append(policy_ent.item())
                value_losses.append(vf_loss.item())
                total_losses.append(total_loss.item())

            avg_policy_loss = np.mean(policy_losses)
            avg_entropy_loss = np.mean(entropy_losses)
            avg_value_loss = np.mean(value_losses)
            avg_total_loss = np.mean(total_losses)

            self.writer.add_scalar("train/policy_loss", avg_policy_loss, self.episode)
            self.writer.add_scalar("train/entropy_loss", avg_entropy_loss, self.episode)
            self.writer.add_scalar("train/value_loss", avg_value_loss, self.episode)
            self.writer.add_scalar("train/total_loss", avg_total_loss, self.episode)                 


    def optimize(self):
        data = self.prepare_data(self.memory.get())

        self.optimize_ppo(data)


    def step(self):

        # -------- Initialize --------

        start_time = datetime.now().replace(microsecond=0)

        # For logging training state
        writer_path     = os.path.join(self.get_experiments_base_path(), 'runs')
        self.writer     = SummaryWriter(writer_path)
        # log data store queue
        score_queue     = deque(maxlen=self.config.train.average_interval)
        length_queue    = deque(maxlen=self.config.train.average_interval)

        best_score = 0

        # init env  
        env = TrainEnvironment(
            env_name=self.config.env.env_name,
            is_continuous=self.config.env.is_continuous,
            seed=self.config.seed
        )
        env.init()

        # make rollout buffer
        self.memory = PPOMemory(
            gamma=self.config.train.gamma,
            tau=self.config.train.tau
        )

        # -------- Training Loop --------
        
        for episode in range(1, self.config.train.max_episodes + 1):
            self.episode = episode
            episode_score = 0

            # shape
            # state: (obserbation_size, )
            state, _ = env.reset()

            for t in range(1, self.config.train.max_episode_len + 1):

                # ------------- Collect Trajectories -------------

                # step action
                # discrete: 
                #   state         , action, logprobs, ent
                #   (1, state_dim),   (1,),     (1,), (1,)
                with torch.no_grad():
                    state = torch.tensor(state).unsqueeze(0).float()
                    action, logprobs, _ = self.actor(state)
                    values = self.critic(state).squeeze(1)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # update episode_score
                episode_score += reward

                if self.config.train.reward_scaler:
                    reward = self.reward_scaler(reward, update=True)
                    
                # add experience to the memory
                self.memory.store(
                    s=state,
                    a=action,
                    r=reward, 
                    v=values.item(), 
                    lp=logprobs.item()
                )
                
                # ------------- Checking for optimize -------------
                
                # done or timeout or memory full
                # done => v = 0
                # timeout or memory full => v = critic(next_state)
                # update gae & return in the memory!!
                timeout = t == self.config.train.max_episode_len
                time_to_optimize = len(self.memory) == self.config.train.ppo.memory_size
                if done or timeout or time_to_optimize:
                    if done:
                        # cuz the game is over, value of the next state is 0
                        v = 0
                    else:
                        # if not, estimate it with the critic
                        next_state_tensor = torch.tensor(next_state).unsqueeze(0).float() # bsz = 1
                        with torch.no_grad():
                            next_value_tensor = self.critic(next_state_tensor).squeeze(1)
                        v = next_value_tensor.item()

                    # update gae & tdlamret
                    self.memory.finish_path(v)
                    
                # if memory is full, optimize PPO
                if time_to_optimize:
                    self.optimize()

                if done:
                    score_queue.append(episode_score)
                    length_queue.append(t)
                    break
                
                # update state
                state = next_state

            # scheduling learning rate
            if self.config.train.scheduler:
                self.scheduler.step()

            # ------------- Logging training state -------------

            avg_score       = np.mean(score_queue)
            std_score       = np.std(score_queue)
            avg_duration    = np.mean(length_queue)

            # Writting for tensorboard
            self.writer.add_scalar("train/score", avg_score, self.episode)
            self.writer.add_scalar("train/duration", avg_score, self.episode)
            if self.config.train.scheduler:
                for idx, lr in enumerate(self.scheduler.get_lr()):
                    self.writer.add_scalar(f"train/learning_rate{idx}", lr, self.episode)
            # Printing for console
            if self.episode % self.config.train.log_interval == 0:
                print(f"[Info {datetime.now().replace(microsecond=0) - start_time}] {self.episode} - score: {avg_score} +-{std_score} \t duration: {avg_duration}")

            # Save best model
            if avg_score >= best_score and self.episode >= self.config.train.max_episodes * 0.1:
                # print(f"[Info] found best model at episode: {self.episode}")
                self.save(f'best')
                best_score = avg_score

            if self.episode % self.config.train.save_interval == 0:
                self.save(str(self.episode))

        env.close()
        self.save('last')
        return best_score

    def make_gif_from_images(self, total_timesteps = 300, step = 10, frame_duration = 150):

        # Make gif directories
        gif_path = os.path.join(self.get_experiments_base_path(), "gif_path")
        os.makedirs(gif_path, exist_ok=True)        
        gif_path = gif_path + "/gif.gif"
        
        image_path = os.path.join(self.get_experiments_base_path(), "render_images")
        image_path = image_path + '/*.jpg'

        # Get images
        img_paths = sorted(glob.glob(image_path))
        img_paths = img_paths[:total_timesteps]
        img_paths = img_paths[::step]
        
        print("total frames in gif : ", len(img_paths))
        print("total duration of gif : " + str(round(len(img_paths) * frame_duration / 1000, 2)) + " seconds")
            
        # save gif
        img, *imgs = [Image.open(f) for f in img_paths]
        img.save(fp=gif_path, format='GIF', append_images=imgs, save_all=True, optimize=True, duration=frame_duration, loop=0)

    
    def play(self, num_episodes=1, max_ep_len=100, use_rendering=False):  

        if use_rendering:
            image_path = os.path.join(self.get_experiments_base_path(), "render_images")
            os.makedirs(image_path, exist_ok=True)
            
        env = TrainEnvironment(
            env_name=self.config.env.env_name,
            is_continuous=self.config.env.is_continuous,
            seed=self.config.seed
        )
        env.init(render_mode='rgb_array')

        scores = []

        for episode in range(num_episodes):

            episode_score = 0

            # shape
            # state: (obserbation_size, )
            state, _ = env.reset()

            for t in range(1, max_ep_len + 1):
                
                with torch.no_grad():
                    state = torch.tensor(state).unsqueeze(0).float()
                    action, _, _ = self.actor(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # update episode_score
                episode_score += reward   

                # rendering 
                if use_rendering and episode == num_episodes - 1:
                    # render
                    img = Image.fromarray(env.env.render())
                    # save
                    img.save(image_path + '/' + str(t).zfill(6) + '.jpg')             

                # update state
                state = next_state

                # game over condition
                if done:
                    scores.append(episode_score)
                    break
            print(f"Episode {episode}: score - {episode_score}")

        avg_score = np.mean(scores)
        print(f"Average score {avg_score} on {num_episodes} games")               
        env.close()

        if use_rendering:
            self.make_gif_from_images()
