# ICPP 2023 Artifact
Artifacts needed to reproduce the results of the paper ***Adaptive Worker Group Management For Mitigating Straggler Problem In Distributed Deep Learning Systems*** inside a computer cluster.

## Requirements
* OS: Ubuntu 18.04.4 LTS (x86_64)

|Library|Version|
|:---:|:---:|
|GCC|7.5.0|
|CMake|3.22.3|
|Libc|glibc-2.9|
|Python|3.7.0|
|NVIDIA Driver|450.102.04|
|CUDA|11.1|
|NCCL|2.10.3|
|cuDNN|8.1.0|
|PyTorch|1.10.2+cu111|
|Torchvision|0.11.3+cu111|
|Horovod|0.25.0|
|MPICH|3.3a2|

## Preparation
[Download CIFAR-10 Dataset.](https://www.cs.toronto.edu/~kriz/cifar.html)<br/>
This will take a few minutes.<br/><br/>
Download CIFAR-10 Dataset and Horovod in `/data`<br/><br/>
Move `icpp-artifact` inside `/data/horovod/examples/pytorch/`
```
mv <path to icpp-artifact> /data/horovod/examples/pytorch/
```
Make directory `signal` and `straggler`
* `signal` is where normal worker process set job saves signal text file to notify straggler job to save current model and be ready to synchronize
* `straggler` is where straggler job saves current model when notified by normal worker process job to stop training
```
cd /data/horovod/examples/pytorch/icpp-artifact
mkdir signal
mkdir straggler
```

## Experiments
* All experiments are performed on nodes configured with 4 GPUs each.
* Number of nodes: 1~4

### Experiments to produce Figure 1
* Without slowdown worker
```bash
# 4 workers
tensorboard --logdir=/data/horovod/examples/pytorch/icpp-artifact/logs --port=6011 & horovodrun -np 4 -H localhost:4 python3 pytorch_cifar10_resnet18.py

# 8 workers
tensorboard --logdir=/data/horovod/examples/pytorch/icpp-artifact/logs --port=6012 & horovodrun -np 8 -H <private IP of node1>:4,<private IP of node2>:4 python3 pytorch_cifar10_resnet18.py

# 16 workers
tensorboard --logdir=/data/horovod/examples/pytorch/icpp-artifact/logs --port=6013 & horovodrun -np 16 -H <private IP of node1>:4,<private IP of node2>:4,<private IP of node3>:4,<private IP of node4>:4 python3 pytorch_cifar10_resnet18.py
```
* With one 5x slowdown worker
```bash
# 4 workers
tensorboard --logdir=/data/horovod/examples/pytorch/icpp-artifact/logs --port=6014 & horovodrun -np 4 -H localhost:4 python3 pytorch_cifar10_resnet18_01.py

# 8 workers
tensorboard --logdir=/data/horovod/examples/pytorch/icpp-artifact/logs --port=6015 & horovodrun -np 8 -H <private IP of node1>:4,<private IP of node2>:4 python3 pytorch_cifar10_resnet18_01.py

# 16 workers
tensorboard --logdir=/data/horovod/examples/pytorch/icpp-artifact/logs --port=6016 & horovodrun -np 16 -H <private IP of node1>:4,<private IP of node2>:4,<private IP of node3>:4,<private IP of node4>:4 python3 pytorch_cifar10_resnet18_01.py
```

### Experiments to produce Figure 2~5
* With two 5x slowdown workers (total 4 workers)
```bash
# AllReduce
tensorboard --logdir=/data/horovod/examples/pytorch/icpp-artifact/logs --port=6017 & horovodrun -np 4 -H localhost:4 python3 pytorch_cifar10_resnet18_02.py

# straggler-drop
tensorboard --logdir=/data/horovod/examples/pytorch/icpp-artifact/logs --port=6018 & horovodrun -np 4 -H localhost:4 python3 pytorch_cifar10_resnet18_straggler_drop_02.py

# The proposed method
tensorboard --logdir=/data/horovod/examples/pytorch/icpp-artifact/logs --port=6019 & horovodrun -np 4 -H localhost:4 python3 pytorch_cifar10_resnet18_proposed_02.py
```

### Experiments to produce Figure 6
* With one 5x slowdown workers (total 4 workers)
```bash
# AllReduce
tensorboard --logdir=/data/horovod/examples/pytorch/icpp-artifact/logs --port=6020 & horovodrun -np 4 -H localhost:4 python3 pytorch_cifar10_resnet18_01.py

# straggler-drop
tensorboard --logdir=/data/horovod/examples/pytorch/icpp-artifact/logs --port=6021 & horovodrun -np 4 -H localhost:4 python3 pytorch_cifar10_resnet18_straggler_drop_01.py

# The proposed method
tensorboard --logdir=/data/horovod/examples/pytorch/icpp-artifact/logs --port=6022 & horovodrun -np 4 -H localhost:4 python3 pytorch_cifar10_resnet18_proposed_01.py
```

## Figures
[See `icpp2023-figs.ipynb`.](https://github.com/ledzep0830/icpp-artifact/blob/main/icpp2023-figs.ipynb)
