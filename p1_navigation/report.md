### Evaluation Reports
#### Neilkunal Panchal

#### Algorithm Model

The models of the DQN agent is in the 'model.py' file

Here two models are presents. the first model is a feed forward neural network in the form of:
``Input -> 64 Layer -> Relu Activation -> 64 Layer -> Relu -> 64 Layer -> Action``

For the visual Neural network:
``Input -> Conv 3D Layer -> BatchNorm -> Relu Activation ->
Conv 3D Layer -> BatchNorm -> Relu Activation ->
 Conv 3D Layer -> Relu -> 64 Layer -> Relu -> 64 Layer -> Action``


![Reward Plots](./DQN-TrainingError.jpg)


![Trained Agent][./DQN.gif]

### Ideas for Further work

Ideas for further work will include investigating Double DQN.
The performance can further be improved by using a priorotised experience replay. The hyper-parameters can be optimised by using a grid search
