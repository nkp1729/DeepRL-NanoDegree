#### Udacity Deep Reinforcement Learning Nanodegree
### Project 3: Multi-Agent Collaboration & Competition
#  Tennis
# Neilkunal Panchal


## Objective
The purpose of this project is to train an RL agent in self play in a competitive setting.

Multi Agent systems pose a challenge since the environment consists of other agents. Since the environment is non-stationary, traditional methods do not seem to work well from a stability point of view.


## The Environment
The environment used here for a 2 player Tennis game is given by [Unity Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis).

Here, two agents use rackets to hit a ball over a net. Each episode gives the winning side 0.1 pts of they win a point.

- observation space 8 dimensions:
    - ball position and velocity
    - racket position and velocity
- action space 2 dimensions
- number of agents: 2


The game is won or solved if a consecutive reward of > 0.5 is gained for more than 100 consecutive episodes.

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.



## Methodology
The hyper parameters were chosen by trial and error.
The training is not stable and many attempt were taken to reach the final solution.
If one has available computational resources hyper-parameter searches are recommended.



## Learning algorithm

In this Tennis Game there are multiple agents and a continuous action space. The value based methods of Q learning are designed for descrete action spaces and hence wont be applicable here.

Policy based approaches work by directly optimising a stochastic policy instead of through a value function by proxy. Policy gradient methods are well suited for continuous action spaces. The method used here is DDPG [DDPG](https://arxiv.org/pdf/1509.02971.pdf). This is a policy gradient approach which is a  model-free, off-policy actor-critic algorithm. A Q function much like DQN is learned in order to reduce the variance of the gradient estimator, where the Q funtion is the baseline.




#### MADDPG - Multi-Agent Deep Deterministic Policy Gradient

We extend the DDPG for the multi agent setting based on the paper [DDPG](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)


The approach we use has a decentralised actor and a centralised critic. Here both agents contribute to the same replay buffer to for the critic.



For continuous action spaces exploration is considered by sampling from an policy distribution. Here noise is added to the paramtererised policy for exploration. The noise distribution is the [Ornstein-Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process). This is useful for time correlated noise.

This process contributed multiple hyperparamters which were quite important to understand for training here.

The paramters of the OU-process are:
- mu:  mean,
- theta: mean reversion factor
- sigma: volatility

sigma values of 0.1 -0.3 seemed to perform quite well. Large values of sigma were found to not increase the moving average rewards behond 0.2

Another hyper parameter is the epsilon which controls the scheduling of exploration. Here the coefficient of EPS_START is chosen to be 5 to encourage early exploration.




## the chosen hyperparameters.


The final noise parameters were set as follows:

```python
BUFFER_SIZE = int(1e6)  
BATCH_SIZE = 128        
LR_ACTOR = 1e-3        
LR_CRITIC = 1e-3     
WEIGHT_DECAY = 0       
LEARN_EVERY = 1        
LEARN_NUM = 8        
GAMMA = 0.96       
TAU = 8e-3              
OU_SIGMA = 0.25      
OU_THETA = 0.15    
EPS_START = 5.0      
EPS_EP_END = 300        
EPS_FINAL = 0
```


##### &nbsp;

## Results
For the above hyper parameters the environment was solved

The chart below shows the training curves.

The best moving average was 0.7 and episodic reward was 5.3

[alt text][https://github.com/nkp1729/DeepRL-NanoDegree/p3_collab-compet/results.jpg]

## Futher Work
- Training is very unstable. The noise parameters should have thier seeds set. The hyper parameters can be better set using random search or grid search methods

- Further network architectures can be investigated for better performance

- BatchNorm or its equivilent for MLPs can be used to improve stability.

- Investigate better exploration strategies
