# Rethink-BiasVariance-Tradeoff
Rethinking Bias-Variance Trade-off for Generalization of Neural Networks

This is the code for the paper "Rethinking Bias-Variance Trade-off for Generalization of Neural Networks".

## Prerequisites
* Python
* Pytorch (1.3.1)
* CUDA
* numpy


## How to train models on different datasets?
There are 4 folders, ```cifar10```, ```cifar100```, ```fmnist```, and ```mnist```. First ```cd``` into the directory, and run
```python
python train.py --trial 2 --arch resnet34 --width 10 --num-epoch 500 --lr-decay --outdir part1
```
### Arguments:
* ```trial```: how many splits, i.e., if ```trial=2``` on ```cifar10```, then the trainig sample size is ```50000/2 = 25000```
* ```arch```: network architecture
* ```width```: width of the network
* ```num-epoch```: how many epochs for training
* ```lr-decay```: after how many epochs to decay the learning rate
* ```outdir```: specify the name of the folder for saving logs and checkpoints

### Log file:
The results (including bias and variance) will be save in ```'log_width{}.txt'.format(args.width) ```, in the folder ```'{}_{}_trial{}_mse{}'.format(args.dataset, args.arch, args.trial, args.outdir)```. 

The log file includes the following,

| trial | train loss  | train acc | test loss | test acc | bias | variance |
| --------------------- | ------------- | ------------| ------------ |--------------- |-------- | ------- | 



## Reference
For technical details and full experimental results, please check [the paper](https://todo).
```
@article{todo, 
	author = {todo}, 
	title = {Rethinking Bias-Variance Trade-off for Generalization of Neural Networks}, 
	journal = {todo},
	year = {2020}
}
```
