import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
import os
import math
from tqdm import tqdm
import time

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
parser.add_argument('--sleep-time', type=float, default=0.09,
                    help='sleep time for stragglers (default: 0.09)')
parser.add_argument('--rank', type=int, default=1,
                    help='this corresponds to hvd.local_rank() of this straggler')
parser.add_argument('--resume-from-epoch', type=int, default=0,
                    help='resume from this epoch')
parser.add_argument('--checkpoint-straggler-format', default="./straggler/checkpoint-{epoch}-{rank}.pth.tar",
                    help='straggler checkpoint file format')
parser.add_argument('--signal-format', default="./signal/epoch-{epoch}-normal-ready.txt",
                    help='signal file format')
parser.add_argument('--total-iteration-num', type=float,
                    help='total iteration number per epoch of main horovod job')
parser.add_argument('--iteration-straggler-format', default="./straggler/iteration-{epoch}-{rank}.txt",
                    help='straggler iteration file format')


def train(epoch):
    model.train()
   
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    for batch_idx, (data, target) in enumerate(train_loader):
        adjust_learning_rate(epoch, batch_idx)

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        # Split data into sub-batches of size batch_size
        for i in range(0, len(data), args.batch_size):
            data_batch = data[i:i + args.batch_size]
            target_batch = target[i:i + args.batch_size]
            output = model(data_batch)
            train_accuracy.update(accuracy(output, target_batch))
            loss = F.cross_entropy(output, target_batch)
            train_loss.update(loss)
            # Average gradients among sub-batches
            loss.div_(math.ceil(float(len(data)) / args.batch_size))
            loss.backward()
        # PROPOSED: sleeps as much as designated sleep_time to implement straggler
        time.sleep(args.sleep_time)
        # Gradient is applied across all ranks
        optimizer.step()

        if os.path.exists(args.signal_format.format(epoch=epoch+1)):
            # PROPOSED: calculate progress rate to hand over to main horovod job 
            # PROPOSED: convert progree rate to string in order to save in txt file
            cur_iteration = '{}'.format(float(batch_idx) / float(len(train_loader)))
            save_straggler_checkpoint_and_iteration(epoch, model, cur_iteration)
            return


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = epoch / float(args.warmup_epochs)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * args.batches_per_allreduce * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_straggler_checkpoint_and_iteration(epoch, model, iteration):
    filepath_checkpoint = args.checkpoint_straggler_format.format(epoch=epoch + 1, rank=args.rank)
    # PROPOSED: calculate weighted state dict value before division
    tmp_state_dict = model.state_dict()
    weight = float(iteration)
    weighted_state_dict = {key: value * weight for key, value in tmp_state_dict.items()}

    state = {
        'model': weighted_state_dict,
        # 'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filepath_checkpoint)
    filepath_iteration = args.iteration_straggler_format.format(epoch=epoch + 1, rank=args.rank)
    with open(filepath_iteration, "w") as f:
        f.write(iteration)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += val.detach().cpu()
        # self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        # PROPOSED: args.rank equals hvd.local_rank() (230409)
        torch.cuda.set_device(args.rank)
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True


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
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=allreduce_batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, **kwargs)


    # Set up standard ResNet-18 model.
    model = models.resnet18()

    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    lr_scaler = args.batches_per_allreduce if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        
    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(),
                          lr=(args.base_lr *
                              lr_scaler),
                          momentum=args.momentum, weight_decay=args.wd)
    
    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights to other workers.
    if args.resume_from_epoch > 0:
        filepath = args.checkpoint_format.format(epoch=args.resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])


    train(args.resume_from_epoch)
