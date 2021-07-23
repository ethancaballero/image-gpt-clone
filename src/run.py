
"""
import pytorch_lightning as pl
import pytorch_lightning.logging
#"""
import argparse

from module import ImageGPT

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import wandb


def train(args, model, device, train_loader, test_loader, optimizer, epoch, iters):
    model.train()
    for batch_idx, (data, cond_ll, ll) in enumerate(train_loader):
        #import pdb; pdb.set_trace()
        #"""
        if iters % args.iters_per_eval == 0:
            test(args, model, device, test_loader, epoch, iters)
        #"""
        model.train()
        iters += 1
        data = data.long()
        data = data.to(device)

        data = data.permute(1, 0).contiguous()
        #data = data.permute(0, 3, 1, 2)
        optimizer.zero_grad()
        logits = model(data)
        #loss = F.nll_loss(output, target)
        #loss = F.nll_loss(output, target.argmax(-1))
        #import pdb; pdb.set_trace()
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), data.view(-1))
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    return iters


    """
    if args.classify:
        # classification
        # stop early for best validation accuracy for finetuning
        early_stopping = pl.callbacks.EarlyStopping(monitor="val_acc", patience=3)
        checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_acc")
        trainer = pl.Trainer(
            max_steps=args.steps,
            gpus=args.gpus,
            early_stopping_callback=early_stopping,
            checkpoint_callback=checkpoint,
            logger=logger,
        )

    else:
        # pretraining
        checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss")
        trainer = pl.Trainer(
            max_steps=args.steps,
            gpus=args.gpus,
            checkpoint_callback=checkpoint,
            logger=logger,
        )

    trainer.fit(model)
    """

"""
def test(args):
    trainer = pl.Trainer(gpus=args.gpus)
    model = ImageGPT.load_from_checkpoint(args.checkpoint)
    model.prepare_data()
    trainer.test(model)
#"""

def test(args, model, device, test_loader, epoch, iters):
    model.eval()
    test_loss = 0
    correct = 0
    losses = []
    corrects = []
    lls = []
    kl_divs = []
    with torch.no_grad():
        for (data, cond_ll, ll) in test_loader:
            data = data.long()
            data, cond_ll, ll = data.to(device), cond_ll.to(device), ll

            data = data.permute(1, 0).contiguous()
            #data = data.permute(0, 3, 1, 2)
            logits = model(data)
            #target = target.argmax(-1)
            logprobs_model = torch.nn.functional.log_softmax(logits, -1)
            kl_div = torch.nn.functional.kl_div(logprobs_model.permute(1, 0, 2), cond_ll, log_target=True, reduction='none')
            kl_div = kl_div.sum(-1).mean(-1)
            kl_divs.append(kl_div)
            #import pdb; pdb.set_trace()
            #torch.nn.functional.kl_div()
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), data.view(-1), reduction='none')
            loss = loss.mean(-1)
            test_loss += loss.sum().item()  # sum up batch loss
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()
            lls.append(ll)
            losses.append(loss)

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: ', test_loss)

        ll_cat = torch.cat(lls).float().cpu()
        kl_div_cat = torch.cat(kl_divs).float().cpu()
        #correct_cat = torch.cat(corrects).float().cpu()
        sort_idxs = ll_cat.float().sort(descending=True)[1]
        ll_cat, kl_div_cat = ll_cat[sort_idxs], kl_div_cat[sort_idxs]

        markersize = 2.0
        figure, axis = plt.subplots(1, 2)

        axis[0].plot(np.arange(0, len(ll_cat)), ll_cat.cpu().numpy(), '.', markersize=markersize, color='k')
        axis[1].plot(np.arange(0, len(ll_cat)), kl_div_cat.cpu().numpy(), '.', markersize=markersize, color='k')

        axis[0].set_ylabel("log-likelihoods")
        axis[1].set_ylabel("kl_divs")

        log_dict = {}
        log_dict.update({'eval_loss': test_loss,
        })
        log_dict.update({"charts__ll/"+str(epoch): figure})
        #wandb.log(log_dict, step=epoch)
        wandb.log(log_dict, step=iters*args.batch_size)
        

class ImageDataset(Dataset):
    """
    wrap up the pytorch CIFAR-10 dataset into our own, which will convert images into sequences of integers
    """
    
    def __init__(self, pt_dataset, perm=None):
        self.pt_dataset = pt_dataset
        
    def __len__(self):
        return len(self.pt_dataset)

    def __getitem__(self, idx):
        x, y, z = self.pt_dataset[idx]
        #x, y = (np.array(x)), (np.array(y))
        x, y, z = x.float(), y.float(), z.float()
        return x, y, z

class _Dataset(object):

    def __init__(self, x, cond_ll, ll=None, dataset_size=None, transform=None, in_mem=True):
        self.in_mem = in_mem
        self.dataset = torch.load(x).cpu()
        self.cond_ll = None
        self.lls = None
        if cond_ll != None:
            self.cond_ll = torch.load(cond_ll).cpu()
        if ll != None:
            self.lls = torch.load(ll).cpu()
        #if in_mem: self.dataset = self.dataset.float().div(255)
        self.transform = transform

        #import pdb; pdb.set_trace()

        if dataset_size != None:
            self.dataset = self.dataset[:dataset_size]
            self.cond_ll = self.cond_ll[:dataset_size]
            self.lls = self.lls[:dataset_size]


    def __len__(self):
        return self.dataset.size(0)

    """
    @property
    def ndim(self):
        return self.dataset.size(1)
    #"""

    def __getitem__(self, index):
        x = self.dataset[index]
        cond_ll = 0
        z = 0
        if self.cond_ll != None:
            cond_ll = self.cond_ll[index]
        if self.lls != None:
            z = self.lls[index]
        #if not self.in_mem: x = x.float().div(255)
        #x = self.transform(x) if self.transform is not None else x
        return x, cond_ll, z

class TeacherFolder(_Dataset):
    TRAIN_LOC = 'data/TeacherFolder/train_32x32.pth'
    TEST_LOC = 'data/TeacherFolder/valid_32x32.pth'

    def __init__(self, x, cond_ll, ll, dataset_size, train=True, transform=None):
        return super(TeacherFolder, self).__init__(x, cond_ll, ll, dataset_size, transform)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = ImageGPT.add_model_specific_args(parser)

    parser.add_argument("--train_x", default="data/train_x.npy")
    parser.add_argument("--train_y", default="data/train_y.npy")
    parser.add_argument("--test_x", default="data/test_x.npy")
    parser.add_argument("--test_y", default="data/test_y.npy")

    parser.add_argument("--gpus", default="0")

    parser.add_argument('--train_set_size', type=int, default=100000000000000)
    parser.add_argument('--test_set_size', type=int, default=10000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--epochs', type=int, default=1000000000, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--iters-per-eval', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--wandb_tag', type=str, default="scaling_kl")
    parser.add_argument('--wandb_project', type=str, default="scaling_kl")
    """
    subparsers = parser.add_subparsers()
    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--pretrained", type=str, default=None)
    parser_train.add_argument("-n", "--name", type=str, required=True)
    parser_train.set_defaults(func=train)

    parser_test = subparsers.add_parser("test")
    parser_test.add_argument("checkpoint", type=str)
    parser_test.set_defaults(func=test)
    """

    args = parser.parse_args()
    #args.func(args)

    direct = '/home/mila/c/caballero/research/scaling_outer/scaling_laws/minGPT'
    #direct = '/home/ethancab/research/scaling_breadth/teacher_data'
    #direct = '/scratch/ethancab/teacher_data/teacher_data'

    direct = '/Users/ethancaballero/research/scaling_breadth/conditional_kl/image-gpt-clone'
    direct = '/home/mila/c/caballero/research/image-gpt-clone'
    train_data = TeacherFolder(x=direct+'/_data/imgs.pt', cond_ll=direct+'/_data/cond_ll.pt', ll=direct+'/_data/lls.pt', dataset_size=args.train_set_size, train=True, transform=None)
    test_data = TeacherFolder(x=direct+'/_data/imgs.pt', cond_ll=direct+'/_data/cond_ll.pt', ll=direct+'/_data/lls.pt', dataset_size=args.test_set_size, train=False, transform=None)

    train_dataset = ImageDataset(train_data)
    test_dataset = ImageDataset(test_data)

    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=True,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

    test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True,
                                batch_size=args.test_batch_size,
                                num_workers=args.num_workers)

    """
    if args.pretrained is not None:
        model = ImageGPT.load_from_checkpoint(args.pretrained)
        # potentially modify model for classification
        model.hparams = args
    else:
        model = ImageGPT(args)
    #"""

    model = ImageGPT(args)    

    #logger = pl.logging.TensorBoardLogger("logs", name=args.name)

    #use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        ])

    model = model.to(device)
    #optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    params = sum(p.numel() for p in model.parameters())
    print("params: ", params)

    #"""
    wandb.init(project=args.wandb_project, reinit=True, tags=[args.wandb_tag])
    wandb.config.update(args)
    wandb.config.update({"params": params})
    #"""

    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    iters = 0
    epoch = 1
    for epoch in range(1, args.epochs + 1):
        #test(model, device, test_loader, epoch-1)
        iters = train(args, model, device, train_loader, test_loader, optimizer, epoch-1, iters)
        #scheduler.step()
    