### Evaluation Reports
#### Neilkunal Panchal

#### Algorithm Model

The models of the DQN agent is in the 'model.py' file

### DQN Background

The algorithm used to solve this network is Deep Q learning [DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). In this algorithm a *Q-function* representing the action value function.

The algorithm used to solve this network is Deep Q learning [DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). In this algorithm a *Q-function* representing the action value function $Q(s,a) = \mathrm{E}(\sum_t^{T}{\gamma^t R_t})$ is approximated using a neural network.

An agent represented by a policy $\pi(s_t)$ interacts with an environment $E$ recieving a state $s_t$ and a reward $r_t$. In DQN the action is given by a greedy policy for a given state $s_t$ where $a = \pi(s_t) = \textit{argmax}_{a}{Q(s,a)}$. This policy is *$\epsilon$- greedy* meaning that as an exploration strategy a greedy action is chosen with probability $\epsilon$ and a random action is chosen with probability $1- \epsilon $.

An experience replay buffer $(s_t, a_t, s_{t+1}, r_t )$ is used to collect state action pairs along with the corresponding reward, and transitioned state. by interacting with the environment with an epsilon greedy policy a replay buffer is collected. From here the approximate $\mathit{Q}$ function is trained using Bellman's equation $Q^{*}_{i+1}(s,a) = \mathrm{E}_{s'}[r_t + Q^{*}_i(s',a')]$. The expession inside the expectation $r_t + Q^{*}_i(s',a')$ is called the target $y_t$. The idea behind the DQN algorithm is optimize the approximation of the $Q$ function by minimising $Q^{*}_{i+1}(s,a) - r_t - Q^{*}_i(s',a')$ through minibatch samples from the experience replay buffer. Finally for stability the $Q$ function in the target slowly changes during updates. the update is determined by the following relationship $Q_{target}(s,a) = \tau Q_{learned}(s,a) + (1-\tau )Q_{target}(s,a)$$





Here two models are presents. the first model is a feed forward neural network in the form of:
``Input -> 64 Layer -> Relu Activation -> 64 Layer -> Relu -> 64 Layer -> Action``

For the visual Neural network:
``Input -> Conv 3D Layer -> BatchNorm -> Relu Activation ->
Conv 3D Layer -> BatchNorm -> Relu Activation ->
 Conv 3D Layer -> Relu -> 64 Layer -> Relu -> 64 Layer -> Action``

#### Hyperparameters Used

```python
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
```

##### Discussion on Hyperparameters.
- The Buffer size for the experience replay is fixed to be of length 1e5. The tradeoff here is between memory, however if the size is too short then the buffer may not contain enough high reward samples.
- The Batch size was appropriate for the amount of RAM in the GPU available (8GB for a GTX1080). A smaller batch size may train faster but would have less variance for SGD.
- The discount rate gamma determined how much the agent prefers instant rewards compared to distant rewards. A value of *0.99* is almost undiscounted.
- The value of tau ensures a slow continuous updating of the target. The value chosen above was sufficient for training.
- The learning rate of 5e-4 was sufficient to solve the agent. A lower learning rate is found to have much slower training. A larger learning rate may make training unstable.
#### DQN Visual hyperparameters Used

```python
BUFFER_SIZE = int(5e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.95           # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 6        # how often to update the network
```
##### Discussion on Hyperparameters.

The hyper parameters here were similar to the on visual network.

- A larger replay buffer was found to improve performance.
- A discount rate of 0.95 was found to improve training. The intuition here is that the agent would have more incentive to maximise the current reward more.



![Reward Plots](./DQN-TrainingError.jpg)


![Trained Agent][./DQN.gif]

### Ideas for Further work

Ideas for further work will include investigating Double DQN.
The performance can further be improved by using a priorotised experience replay. The hyper-parameters can be optimised by using a grid search
