import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
import horovod.torch as hvd
import os
import math
from tqdm import tqdm
from mpi4py import MPI
import time
import subprocess

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=os.path.expanduser('/data'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('/data'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

# PROPOSED: additional arguments
parser.add_argument('--threshold-level', type=float, default=3.0,
                    help='what times bigger should threshold value be compared to normal computation time (default: 3.0)')
parser.add_argument('--sleep-time', type=float, default=0.2116,
                    help='sleep time for stragglers (default: 0.2116)')
parser.add_argument('--checkpoint-straggler-format', default="./straggler/checkpoint-{epoch}-{rank}.pth.tar",
                    help='straggler checkpoint file format')
parser.add_argument('--iteration-straggler-format', default="./straggler/iteration-{epoch}-{rank}.txt",
                    help='straggler iteration file format')

def train(epoch):
    torch.autograd.set_detect_anomaly(True)
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    
    # PROPOSED: for recording computation time for each worker for the first 5 iterations 
    # PROPOSED: for saving threshold value and the number of iterations executed for this epoch 
    # PROPOSED: for sending and receiving computation time and threshold value between workers 
    # PROPOSED: for checking if the worker is a straggler
    if hvd.rank() == 0:
        computation_time_record_dict = dict()
    threshold_this_epoch = 0
    iteration_num_per_worker = 0
    comm = MPI.COMM_WORLD
    is_straggler = 0
    warning_counter = 0
    # PROPOSED: for checking the number of stragglers 
    # PROPOSED: for representing the slowdown degree of stragglers 
    num_straggler = 0
    slowdown_level = 0

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            adjust_learning_rate(epoch, batch_idx)

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                # PROPOSED: increase iteration number 
                iteration_num_per_worker += 1
                # PROPOSED: record computation start time 
                start_time = time.time()

                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                output = model(data_batch)
                train_accuracy.update(accuracy(output, target_batch))
                loss = F.cross_entropy(output, target_batch)
                train_loss.update(loss)
                # Average gradients among sub-batches
                # PROPOSED: gradient averaging only between normal workers               
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()
                
                # PROPOSED: record computation time with start time and end time 
                # PROPOSED: send computation time to the root worker 
                end_time = time.time()
                computation_time = end_time - start_time

                # PROPOSED: THIS IS WHERE YOU MAKE STRAGGLER 
                # PROPOSED: CHANGE THIS PART IF YOU HAVE CUSTOM STRAGGLER REQUIREMENTS 
                # PROPOSED: THIS CODE BLOCK HAS BEEN MOVED SO THAT NORMAL WORKERS' COMPUTATION IS NOT AFFECTED BY SLEEP 
                if hvd.rank() % 2 == 1 and iteration_num_per_worker <= 10:
                    # time.sleep(args.sleep_time)
                    computation_time += args.sleep_time

                if iteration_num_per_worker <= 5:
                    if hvd.rank() == 0:
                        computation_time_record_dict[0] = computation_time
                        for i in range(1, hvd.size(), 1):
                            computation_time_record_dict[i] = comm.recv(source=i, tag=11)
                        # PROPOSED: at this point, threshold_this_epoch represents computation time of normal worker 
                        # PROPOSED: it is set as threshold value in the next block
                        threshold_this_epoch += 0.2 * min(computation_time_record_dict.values())
                    else:
                        comm.send(computation_time, dest=0, tag=11)
                
                # PROPOSED: send threshold value to non root workers 
                if iteration_num_per_worker == 5:
                    if hvd.rank() == 0:
                        # PROPOSED: set threshold value (230409)
                        # print('\nAVERAGE MIN COMPUTATION_TIME IS ', threshold_this_epoch)
                        threshold_this_epoch = args.threshold_level * threshold_this_epoch
                        for i in range(1, hvd.size(), 1):
                            comm.send(threshold_this_epoch, dest=i, tag=11)
                    else:
                        threshold_this_epoch = comm.recv(source=0, tag=11)
                    if computation_time > threshold_this_epoch:
                        warning_counter += 1

                # PROPOSED: straggler detection 
                # PROPOSED: only for normal workers 
                if iteration_num_per_worker > 5 and is_straggler == 0:
                    if computation_time > threshold_this_epoch:
                        warning_counter += 1
                        if warning_counter == 5:
                            # PROPOSED: the first time this worker is classified as straggler in this epoch 
                            # PROPOSED: this value will change to 2 shortly after updating num_straggler value 
                            is_straggler = 1
                            # print('\nSTRAGGLER DETECTED: ITERATION NUM ', batch_idx)
                            # PROPOSED: calculate slowdown_level 
                            slowdown_level = math.ceil(computation_time / (threshold_this_epoch / args.threshold_level))
                            # PROPOSED: straggler subprocess start 
                            
                            # PROPOSED: changed the value of '--sleep-time' from 'args.sleep_time' to 'threshold_this_epoch' 
                            straggler_cmd = 'python3 pytorch_cifar10_resnet18_proposed_straggler.py --rank {} --sleep-time {} --resume-from-epoch {} --total-iteration-num {}'.format(
                                    hvd.local_rank(), args.sleep_time, epoch, float(len(train_loader)))
                           
                            straggler_subprocess = subprocess.Popen(straggler_cmd, shell=True)                       
                            
                    else:
                        if warning_counter > 0:
                            warning_counter -= 1

                # PROPOSED: updating number of stragglers 
                if hvd.rank() > 0:
                    comm.send(is_straggler, dest=0, tag=12)
                    num_straggler = comm.recv(source=0, tag=13)
                if hvd.rank() == 0:
                    num_new_straggler = 0
                    for i in range(1, hvd.size(), 1):
                        check_if_new_straggler = comm.recv(source=i, tag=12)
                        if check_if_new_straggler == 1:
                            num_new_straggler += check_if_new_straggler
                    num_straggler += num_new_straggler
                    for i in range(1, hvd.size(), 1):
                        comm.send(num_straggler, dest=i, tag=13)

                if is_straggler == 1:
                    is_straggler = 2

                # PROPOSED: notify straggler to finish job and save current model 
                if batch_idx == len(train_loader) - slowdown_level and is_straggler == 2:
                    with open("./signal/epoch-{epoch}-normal-ready.txt".format(epoch=epoch+1), "w") as f:
                        f.write("normal worker set almost finished job")

                # PROPOSED: changed position of model averaging part 
                if batch_idx == len(train_loader):
                    if is_straggler == 2 and hvd.rank() > hvd.local_rank():
                        is_straggler = 3

                    if hvd.rank() > 0:
                        comm.send(is_straggler, dest=0, tag=14)
                        
                        if is_straggler > 1:
                            straggler_subprocess.kill()

                        if is_straggler == 3:
                            f1 = open(args.iteration_straggler_format.format(epoch=epoch+1, rank=hvd.local_rank()), 'r')
                            # PROPOSED: weight variable below denotes the ratio of epoch executed by this straggler
                            tmp_weight = f1.read()
                            f1.close()
                            weight = float(tmp_weight)
                            comm_send(weight, dest=0, tag=15)
            
                            checkpoint = torch.load(args.checkpoint_straggler_format.format(epoch=epoch+1, rank=hvd.local_rank()))
                            comm_send(checkpoint['model'], dest=0, tag=16)

                    if hvd.rank() == 0:
                        divide_factor = 1.0
                        tmp_final_state_dict = model.state_dict()
                        for i in range(1, hvd.size(), 1):
                            tmp_is_straggler = comm.recv(source=i, tag=14)
                            if tmp_is_straggler == 2:
                                f2 = open(args.iteration_straggler_format.format(epoch=epoch+1, rank=i), 'r')
                                tmp_weight = f2.read()
                                f2.close()
                                weight = float(tmp_weight)
                                divide_factor += weight

                                checkpoint = torch.load(args.checkpoint_straggler_format.format(epoch=epoch+1, rank=i))
                                tmp_final_state_dict = merge_dictionaries(tmp_final_state_dict, checkpoint['model'], divide_factor)
                            if tmp_is_straggler == 3:
                                tmp_weight = comm.recv(source=i, tag=15)
                                divide_factor += tmp_weight

                                checkpoint = comm.recv(source=i, tag=16)
                                tmp_final_state_dict = merge_dictionaries(tmp_final_state_dict, checkpoint['model'], divide_factor)
                        # PROPOSED: make final model and broadcast to other workers
                        final_state_dict = {key: value / divide_factor for key, value in tmp_final_state_dict.items()}
                        model.load_state_dict(final_state_dict)
                    # hvd.broadcast_parameters(model.state_dict(), root_rank=0)
                    # hvd.broadcast_optimizer_state(optimizer, root_rank=0)


            # Gradient is applied across all ranks
            optimizer.step()
            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})
            t.update(1)

            # PROPOSED: sleep is applied here so that it does not affect recorded computation_time of normal workers 
            if iteration_num_per_worker < 10:
                time.sleep(args.sleep_time)

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)

def merge_dictionaries(dict1, dict2, divide_factor):
    merged_dictionary = {}

    for key in dict1:
        if key in dict2:
            new_value = dict1[key] + dict2[key]
        else:
            new_value = dict1[key] * divide_factor

        merged_dictionary[key] = new_value

    return merged_dictionary

        
def validate(epoch):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * args.batches_per_allreduce * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    hvd.init()
    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break

    # Horovod: broadcast resume_from_epoch from rank 0 (which will have
    # checkpoints) to other ranks.
    resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                      name='resume_from_epoch').item()

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    # Horovod: write TensorBoard logs on first worker.
    log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    train_dataset = \
        datasets.CIFAR10(root=args.train_dir, train=True,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                      std=[0.2023, 0.1994, 0.2010])
                             ]))
    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=allreduce_batch_size,
        sampler=train_sampler, **kwargs)

    val_dataset = \
        datasets.CIFAR10(root=args.val_dir, train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                      std=[0.2023, 0.1994, 0.2010])
                             ]))
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                             sampler=val_sampler, **kwargs)


    # Set up standard ResNet-18 model.
    model = models.resnet18()

    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = args.batches_per_allreduce * hvd.local_size()

    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(),
                          lr=(args.base_lr *
                              lr_scaler),
                          momentum=args.momentum, weight_decay=args.wd)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression,
        backward_passes_per_step=args.batches_per_allreduce,
        op=hvd.Adasum if args.use_adasum else hvd.Average,
        gradient_predivide_factor=args.gradient_predivide_factor)

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights to other workers.
    if resume_from_epoch > 0 and hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # PROPOSED: broadcast parameters & optimizer state.
    for epoch in range(resume_from_epoch, args.epochs):
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        train(epoch)
        validate(epoch)
        save_checkpoint(epoch)
