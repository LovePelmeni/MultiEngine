# Schema of the TrainerConfig class.
This document outlines core attributes, which can be tweaked
to adjust model's behavour during training phase.

Target class: `src.training.trainers.TrainerConfig`

## Optimizer configuration.
Configuration parameters for training optimization algorithm.

```
"name" -> (str). name of the optimization algorithm.
```
```
"learning_rate" -> (float). learning rate for training classifier.
```
```
"weight_decay" -> (float). weight decay for decreasing weights.
```
```
"nesterov" -> typing.Optional[bool] = False - use nesterov optimization approach.
```

## LR Scheduler configuration
Core parameters of the LR Scheduler algorithm.
```
"name" -> (str). name of the LR Scheduler.
```
```
"verbose" -> (str). print out logs during training.
```
```
"gamma" -> (float). decreasing factor for reducing learning rate.
```
```
"total_iters" -> (int). total number of epochs, should match to number of training epochs.
```