# PPO
Minimal implementation of Proximal Policy Optimization (PPO) in PyTorch
- support discrete and continuous action space 
    - In continuous action space, we use the constance std for sampling.
- utils to plot learning graphs in tensorboard

## Update
- 2023-09-09 
    - Update "Generative Adversarial Imitation Learning(GAIL)"

# Train
Find or make a config file and run the following command.
```
python main.py --config=configs/Ant-v4.yaml 
               --exp_name=test
               --train
```

## Make expert dataset for gail

```
python make_expert_dataset.py --experiment_path=checkpoints/Ant/test
                              --load_postfix=last
                              --minimum_score=5000
                              --n_episode=30
```

# How to play
```
python main.py --experiment_path=checkpoints/Ant/test
               --eval
               --eval_n_episode=50
               --load_postfix=last
               --video_path=videos/Ant
```
- load_path: pretrained model prefix(ex/ number of episode, 'best' or 'last') to play

# Result

## Ant-v4
<table style="border: 2px;">
    <tr>
        <th>Environment</th>
        <th>Performance Chart</th>
        <th>Evaluation Video</th>
    </tr>
    <tr>
        <th>Ant-v4</th>
        <th><img src="https://github.com/Ladun/PPO/raw/master/plots/ant.png" alt="Ant-v4 Performance" width="500"/> </th>
<th>


https://github.com/user-attachments/assets/c087dfec-924a-43bf-a5f0-9f6ffcf5a4bd


</th>
    </tr>
    <tr>
        <th>Ant-v4<br/>(GAIL)</th>
        <th><img src="https://github.com/Ladun/PPO/raw/master/plots/ant_gail.png" alt="Ant-v4 Performance" width="500"/> </th>
<th>
  

https://github.com/user-attachments/assets/76c0cddd-cdad-45ae-a9c2-4d518ac3e655

       
</th>
    </tr>
    <tr>
        <th>Reacher-v4</th>
        <th><img src="https://github.com/Ladun/PPO/raw/master/plots/reacher.png" alt="Ant-v4 Performance" width="500"/> </th>
<th>
    

https://github.com/user-attachments/assets/13a58082-d4cc-483d-bab9-e330404ca8f7


</th>
    </tr>
    <tr>
        <th>HalfCheetah-v4</th>
        <th><img src="https://github.com/Ladun/PPO/raw/master/plots/cheetah.png" alt="Ant-v4 Performance" width="500"/> </th>
<th>
    

https://github.com/user-attachments/assets/39f5b3b6-7d50-4d3c-8333-08081b21a671


</th>
    </tr>
</table>


# Reference
- IMPLEMENTATION MATTERS IN DEEP POLICY GRADIENTS: A CASE STUDY ON PPO AND TRPO
- https://github.com/junkwhinger/PPO_PyTorch
- https://github.com/nikhilbarhate99/PPO-PyTorch
