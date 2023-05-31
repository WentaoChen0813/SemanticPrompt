import os
import argparse
import numpy as np
import random
import torch
import torch.utils.data
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import visformer
from data.dataloader import EpisodeSampler, RepeatSampler
from data.dataset import DatasetWithTextLabel
from data.randaugment import RandAugmentMC
from utils import mean_confidence_interval


def main(args):
    # checkpoint and tensorboard dir
    args.tensorboard_dir = 'tensorboard/'+args.dataset+'/'+args.model+'/'+args.exp + '/'
    args.checkpoint_dir = 'checkpoint/'+args.dataset+'/'+args.model+'/'+args.exp + '/'
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.logger = SummaryWriter(args.tensorboard_dir)

    # prepare training and testing dataloader
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    train_aug = transforms.Compose([transforms.Resize(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    norm])
    if args.aug:
        train_aug = transforms.Compose([transforms.RandomResizedCrop(args.image_size),
                                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        norm])
    if args.rand_aug:
        train_aug = transforms.Compose([transforms.RandomResizedCrop(args.image_size),
                                        RandAugmentMC(2, 10, args.image_size),
                                        transforms.ToTensor(),
                                        norm])
    test_aug = transforms.Compose([transforms.Resize(int(args.image_size * 1.1)),
                                   transforms.CenterCrop(args.image_size),
                                   transforms.ToTensor(),
                                   norm])

    train_dataset = DatasetWithTextLabel(args.dataset, train_aug, split='train')
    if args.repeat_aug:
        repeat_sampler = RepeatSampler(train_dataset, batch_size=128, repeat=2)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=repeat_sampler, num_workers=8)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=128, num_workers=8)
    num_classes = len(train_dataset.dataset.classes)
    args.num_classes = num_classes

    test_dataset = DatasetWithTextLabel(args.dataset, test_aug, split=args.split)
    episode_sampler = EpisodeSampler(test_dataset.dataset.targets, args.episodes, args.way, args.shot + 15)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=episode_sampler, num_workers=8)

    # build model
    if args.model == 'visformer-t':
        student = visformer.visformer_tiny(num_classes=num_classes)
    elif args.model == 'visformer-t-84':
        student = visformer.visformer_tiny_84(num_classes=num_classes)
    else:
        raise ValueError(f'unknown model: {args.model}')

    student = student.cuda(args.gpu)

    if args.optim == 'adam':
        optim = torch.optim.Adam(student.parameters(), lr=args.lr)
    elif args.optim == 'adamw':
        optim = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=5e-2)
    else:
        raise ValueError(f'unknown optim: {args.optim}')

    scheduler = None
    if args.cosine_annealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.annealing_period)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        student.load_state_dict(checkpoint['state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    if args.test:
        test(student, test_loader, start_epoch, args)
        return 0

    for epoch in range(start_epoch, args.epochs):
        train(student, train_loader, optim, scheduler, epoch, args)

        if (epoch+1) % args.test_freq == 0:
            test(student, test_loader, epoch, args)

            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': student.state_dict(),
                'optimizer': optim.state_dict(),
            }
            torch.save(checkpoint, args.checkpoint_dir+f'checkpoint_epoch_{epoch+1:03d}.pth')

        if args.cosine_annealing and (epoch+1) % args.annealing_period == 0:
            lr = args.lr * args.gamma**int((epoch+1)/args.annealing_period)
            if args.optim == 'adam':
                optim = torch.optim.Adam(student.parameters(), lr=lr)
            elif args.optim == 'adamw':
                optim = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=5e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.annealing_period)


def train(student, train_loader, optim, scheduler, epoch, args):
    student.train()
    losses = 0.
    accs = 0.
    for idx, episode in enumerate(train_loader):
        image = episode[0].cuda(args.gpu)  # way * (shot+15)
        labels = episode[1].cuda(args.gpu)

        logit, features = student(image)
        loss = F.cross_entropy(logit, labels)
        losses += loss.item()
        _, pred = logit.max(-1)
        accs += labels.eq(pred).sum().float().item() / labels.shape[0]

        optim.zero_grad()
        loss.backward()
        optim.step()

        if idx % args.print_step == 0 or idx == len(train_loader) - 1:
            print_string = f'Train epoch: {epoch}, step: {idx:3d}, loss: {losses/(idx+1):.4f}, acc: {accs*100/(idx+1):.2f}'
            print(print_string)
    args.logger.add_scalar('train/loss', losses/len(train_loader), epoch)
    args.logger.add_scalar('train/acc', accs/len(train_loader), epoch)

    if scheduler is not None:
        args.logger.add_scalar('train/lr', float(scheduler.get_last_lr()[0]), epoch)
        scheduler.step()


def test(student, test_loader, epoch, args):
    student.eval()
    accs = []
    with torch.no_grad():
        for episode in test_loader:
            image = episode[0].cuda(args.gpu)  # way * (shot+15)
            labels = torch.arange(args.way).unsqueeze(-1).repeat(1, 15).view(-1).cuda(args.gpu)

            _, im_features = student(image)
            im_features = im_features.view(args.way, args.shot + 15, -1)
            sup_im_features, que_im_features = im_features[:, :args.shot], im_features[:, args.shot:]

            sup_im_features = sup_im_features.mean(dim=1)
            que_im_features = que_im_features.contiguous().view(args.way * 15, -1)

            sim = F.normalize(que_im_features, dim=-1) @ F.normalize(sup_im_features, dim=-1).t()
            _, pred = sim.max(-1)
            acc = labels.eq(pred).sum().float().item() / labels.shape[0]
            accs.append(acc)

    m, h = mean_confidence_interval(accs)
    print(f'Test epoch: {epoch}, test acc: {m*100:.2f}+-{h*100:.2f}')
    args.logger.add_scalar('test/acc', m*100, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='debug')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100'])
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--image_size', type=int, default=224, choices=[224, 84])
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--rand_aug', action='store_true')
    parser.add_argument('--repeat_aug', action='store_true')
    parser.add_argument('--model', type=str, default='visformer-t', choices=['visformer-t', 'visformer-t-84'])
    parser.add_argument('--optim', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--cosine_annealing', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--annealing_period', type=int, default=100)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--episodes', type=int, default=600)
    parser.add_argument('--print_step', type=int, default=100)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_freq', type=int, default=10)

    args = parser.parse_args()
    if args.seed >= 0:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    main(args)