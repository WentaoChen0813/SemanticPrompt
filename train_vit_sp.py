import os
import argparse
import numpy as np
import random
import torch
import torch.utils.data
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import clip
from sentence_transformers import SentenceTransformer
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import visformer
from data.dataloader import EpisodeSampler, MultiTrans
from data.dataset import DatasetWithTextLabel
from data.randaugment import RandAugmentMC
from utils import mean_confidence_interval


def main(args):
    # checkpoint and tensorboard dir
    args.tensorboard_dir = 'tensorboard/' + args.dataset + '/' + args.model + '/' + args.exp + '/'
    args.checkpoint_dir = 'checkpoint/' + args.dataset + '/' + args.model + '/' + args.exp + '/'
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
    if args.aug_support > 1:
        aug = transforms.Compose([transforms.RandomResizedCrop(args.image_size),
                                  # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  norm])
        test_aug = MultiTrans([test_aug] + [aug]*(args.aug_support-1))

    train_dataset = DatasetWithTextLabel(args.dataset, train_aug, split='train')
    n_episodes = args.train_episodes
    args.train_way = args.way if args.train_way == -1 else args.train_way
    if n_episodes == -1:
        n_episodes = int(len(train_dataset) / (args.train_way * (args.shot + 15)))
    episode_sampler = EpisodeSampler(train_dataset.dataset.targets,
                                     n_episodes,
                                     args.train_way,
                                     args.shot + 15, fix_seed=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=episode_sampler, num_workers=8)
    num_classes = len(train_dataset.dataset.classes)

    test_dataset = DatasetWithTextLabel(args.dataset, test_aug, split=args.split)
    episode_sampler = EpisodeSampler(test_dataset.dataset.targets, args.episodes, args.way, args.shot + 15)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=episode_sampler, num_workers=6)

    if args.nlp_model == 'clip':
        teacher, _ = clip.load("ViT-B/32", device='cuda:' + str(args.gpu))
        text_dim = 512
        # set the max text length
        if args.text_length != -1:
            teacher.context_length = args.text_length
            teacher.positional_embedding.data = teacher.positional_embedding.data[:args.text_length]
            for layer in teacher.transformer.resblocks:
                layer.attn_mask.data = layer.attn_mask.data[:args.text_length, :args.text_length]
    elif args.nlp_model == 'mpnet':
        teacher = SentenceTransformer('all-mpnet-base-v2', device=f'cuda:{args.gpu}')
        text_dim = 768
    elif args.nlp_model == 'glove':
        teacher = SentenceTransformer('average_word_embeddings_glove.6B.300d', device=f'cuda:{args.gpu}')
        text_dim = 300
    else:
        raise ValueError(f'unknown nlp_model: {args.nlp_model}')
    train_text = get_text_feature(teacher, train_dataset, args)
    test_text = get_text_feature(teacher, test_dataset, args)
    if args.eqnorm:
        if args.nlp_model in ['mpnet', 'glove']:
            # the bert features have been normalized to unit length. use the avg norm of clip text features
            avg_length = 9.
        else:
            avg_length = (train_text ** 2).sum(-1).sqrt().mean().item()
        train_text = F.normalize(train_text, dim=-1) * avg_length
        test_text = F.normalize(test_text, dim=-1) * avg_length

    if args.model == 'visformer-t':
        student = visformer.visformer_tiny(num_classes=num_classes)
    elif args.model == 'visformer-t-84':
        student = visformer.visformer_tiny_84(num_classes=num_classes)
    else:
        raise ValueError(f'unknown model: {args.model}')

    feature_dim = 384
    if 2 <= args.stage < 3:
        feature_dim = 192
    if args.projector == 'linear':
        student.t2i = torch.nn.Linear(text_dim, feature_dim, bias=False)
    elif args.projector == 'mlp':
        student.t2i = torch.nn.Sequential(torch.nn.Linear(text_dim, text_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(text_dim, feature_dim, bias=False))
    elif args.projector == 'mlp3':
        student.t2i = torch.nn.Sequential(torch.nn.Linear(text_dim, text_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(text_dim, text_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(text_dim, feature_dim, bias=False))

    if 'channel' in args.prompt_mode:
        student.t2i2 = torch.nn.Linear(text_dim, feature_dim, bias=False)
        student.se_block = torch.nn.Sequential(torch.nn.Linear(feature_dim*2, feature_dim, bias=True),
                                               torch.nn.Sigmoid(),
                                               torch.nn.Linear(feature_dim, feature_dim),
                                               torch.nn.Sigmoid(),)

    student = student.cuda(args.gpu)

    optim_params_id = [id(param) for param in student.t2i.parameters()]
    if 'channel' in args.prompt_mode:
        optim_params_id += [id(param) for param in student.t2i2.parameters()]  # se_block is not included. use smaller lr for se_block
        # optim_params_id += [id(param) for param in student.se_block.parameters()]
    optim_params = [param for param in student.parameters() if id(param) in optim_params_id]
    other_params = [param for param in student.parameters() if id(param) not in optim_params_id]
    if args.optim == 'sgd':
        optim = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == 'adamw':
        optim = torch.optim.AdamW([{'params': optim_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
                                   {'params': other_params, 'lr': args.encoder_lr}], weight_decay=5e-2)
    else:
        raise ValueError(f'unknown optim: {args.optim}')

    if args.resume:
        args.init = args.resume
    if args.init:
        checkpoint = torch.load(args.init, map_location=f'cuda:{args.gpu}')
        student.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        raise ValueError('must provide pre-trained model')

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=f'cuda:{args.gpu}')
        student.load_state_dict(checkpoint['state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f'load checkpoint at epoch {start_epoch}')

    if args.test:
        test(test_text, student, test_loader, 0, args)
        return

    best_acc = 0.
    for epoch in range(start_epoch, args.epochs):
        train(train_text, student, train_loader, optim, epoch, args)

        if (epoch + 1) % args.test_freq == 0:
            acc = test(test_text, student, test_loader, epoch, args)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': student.state_dict(),
            'optimizer': optim.state_dict(),
        }
        torch.save(checkpoint, args.checkpoint_dir + f'checkpoint_epoch_latest.pth')
        if (epoch + 1) % args.save_freq == 0:
            torch.save(checkpoint, args.checkpoint_dir + f'checkpoint_epoch_{epoch + 1:03d}.pth')
        if (epoch + 1) % args.test_freq == 0 and acc > best_acc:
            best_acc = acc
            torch.save(checkpoint, args.checkpoint_dir + f'checkpoint_epoch_best.pth')


def get_text_feature(teacher, dataset, args):
    class_idx = dataset.dataset.classes
    idx2text = dataset.idx2text
    if args.no_template:
        text = [idx2text[idx] for idx in class_idx]
    else:
        text = ['A photo of ' + idx2text[idx] for idx in class_idx]

    teacher.eval()
    if args.nlp_model == 'clip':
        text_token = clip.tokenize(text).cuda(args.gpu)
        if args.text_length != -1:
            text_token = text_token[:, :args.text_length]
        with torch.no_grad():
            text_feature = teacher.encode_text(text_token)
            text_feature = text_feature.float()
    else:
        with torch.no_grad():
            text_feature = teacher.encode(text)
            text_feature = torch.tensor(text_feature).cuda(args.gpu)

    return text_feature


def train(text, student, train_loader, optim, epoch, args):
    student.train()
    losses = 0.
    accs = 0.
    for idx, episode in enumerate(train_loader):
        image = episode[0].cuda(args.gpu)  # way * (shot+15)
        glabels = episode[1].cuda(args.gpu)
        labels = torch.arange(args.train_way).unsqueeze(-1).repeat(1, 15).view(-1).cuda(args.gpu)

        image = image.view(args.train_way, args.shot+15, *image.shape[1:])
        sup, que = image[:, :args.shot].contiguous(), image[:, args.shot:].contiguous()
        sup, que = sup.view(-1, *sup.shape[2:]), que.view(-1, *que.shape[2:])

        glabels = glabels.view(args.train_way, args.shot+15)[:, :args.shot]
        glabels = glabels.contiguous().view(-1)
        text_features = text[glabels]
        if args.prompt_mode == 'spatial':
            text_features = student.t2i(text_features)
            _, sup_im_features = student.forward_with_semantic_prompt(sup, text_features, args)
        else:
            _, sup_im_features = student.forward_with_semantic_prompt_channel(sup, text_features, args)

        sup_im_features = sup_im_features.view(args.train_way, args.shot, -1).mean(dim=1)

        _, que_im_features = student(que)

        sim = F.normalize(que_im_features, dim=-1) @ F.normalize(sup_im_features, dim=-1).t()
        loss = F.cross_entropy(sim / args.t, labels)
        losses += loss.item()
        _, pred = sim.max(-1)
        accs += labels.eq(pred).sum().float().item() / labels.shape[0]

        optim.zero_grad()
        loss.backward()
        optim.step()

        if idx % args.print_step == 0 or idx == len(train_loader) - 1:
            print_string = f'Train epoch: {epoch}, step: {idx:3d}, loss: {losses / (idx + 1):.4f}, acc: {accs * 100 / (idx + 1):.2f}'
            print(print_string)
    args.logger.add_scalar('train/loss', losses / len(train_loader), epoch)
    args.logger.add_scalar('train/acc', accs / len(train_loader), epoch)


def test(text, student, test_loader, epoch, args):
    student.eval()
    accs = []
    with torch.no_grad():
        for episode in test_loader:
            if args.aug_support == 1:
                # use prototype classifier
                image = episode[0].cuda(args.gpu)  # way * (shot+15)
                glabels = episode[1].cuda(args.gpu)
                labels = torch.arange(args.way).unsqueeze(-1).repeat(1, 15).view(-1).cuda(args.gpu)

                image = image.view(args.way, args.shot + 15, *image.shape[1:])
                sup, que = image[:, :args.shot].contiguous(), image[:, args.shot:].contiguous()
                sup, que = sup.view(-1, *sup.shape[2:]), que.view(-1, *que.shape[2:])

                glabels = glabels.view(args.way, args.shot + 15)[:, :args.shot]
                glabels = glabels.contiguous().view(-1)
                text_features = text[glabels]
                if args.prompt_mode == 'spatial':
                    text_features = student.t2i(text_features)
                    _, sup_im_features = student.forward_with_semantic_prompt(sup, text_features, args)
                else:
                    _, sup_im_features = student.forward_with_semantic_prompt_channel(sup, text_features, args)
                _, que_im_features = student(que)

                if args.test_classifier == 'prototype':
                    sup_im_features = sup_im_features.view(args.way, args.shot, -1).mean(dim=1)
                    sim = F.normalize(que_im_features, dim=-1) @ F.normalize(sup_im_features, dim=-1).t()
                    _, pred = sim.max(-1)
                elif args.test_classifier == 'fc':
                    x_train = F.normalize(sup_im_features, dim=-1).cpu().numpy()
                    y_train = torch.arange(args.way).unsqueeze(-1).repeat(1, args.shot).view(-1).numpy()
                    # x_test = F.normalize(que_im_features, dim=-1).cpu().numpy()
                    x_test = que_im_features.cpu().numpy()
                    from sklearn.linear_model import LogisticRegression
                    clf = LogisticRegression(penalty='l2',
                                             random_state=0,
                                             C=1,
                                             solver='lbfgs',
                                             max_iter=1000,
                                             multi_class='multinomial')
                    clf.fit(x_train, y_train)
                    pred = clf.predict(x_test)
                    pred = torch.tensor(pred).cuda(args.gpu)

            elif args.aug_support > 1:
                # use logistic regression classifier
                image = torch.cat(episode[0]).cuda(args.gpu)  # aug_support * way * (shot+15)
                glabels = episode[1].cuda(args.gpu)
                labels = torch.arange(args.way).unsqueeze(-1).repeat(1, 15).view(-1).cuda(args.gpu)

                image = image.view(args.aug_support, args.way, args.shot + 15, *image.shape[1:])
                sup = image[:, :, :args.shot].contiguous().view(-1, *image.shape[3:])
                que = image[0, :, args.shot:].contiguous().view(-1, *image.shape[3:])

                glabels = glabels.view(args.way, args.shot + 15)[:, :args.shot]
                glabels = glabels.unsqueeze(0).repeat(args.aug_support, 1, 1).contiguous().view(-1)
                text_features = text[glabels]
                # text_features = student.t2i(text_features)
                # _, sup_im_features = student.forward_with_semantic_prompt(sup, text_features, args)
                if args.prompt_mode == 'spatial':
                    text_features = student.t2i(text_features)
                    _, sup_im_features = student.forward_with_semantic_prompt(sup, text_features, args)
                else:
                    _, sup_im_features = student.forward_with_semantic_prompt_channel(sup, text_features, args)

                _, que_im_features = student(que)

                if args.test_classifier == 'prototype':
                    sup_im_features = sup_im_features.view(args.aug_support, args.way, args.shot, -1).mean(dim=0).mean(dim=1)
                    sim = F.normalize(que_im_features, dim=-1) @ F.normalize(sup_im_features, dim=-1).t()
                    _, pred = sim.max(-1)
                elif args.test_classifier == 'fc':
                    x_train = F.normalize(sup_im_features, dim=-1).cpu().numpy()
                    y_train = torch.arange(args.way).unsqueeze(0).unsqueeze(-1).repeat(args.aug_support, 1, args.shot).view(-1).numpy()
                    x_test = F.normalize(que_im_features, dim=-1).cpu().numpy()
                    from sklearn.linear_model import LogisticRegression
                    clf = LogisticRegression(penalty='l2',
                                             random_state=0,
                                             C=1.0,
                                             solver='lbfgs',
                                             max_iter=1000,
                                             multi_class='multinomial')
                    clf.fit(x_train, y_train)
                    pred = clf.predict(x_test)
                    pred = torch.tensor(pred).cuda(args.gpu)

            acc = labels.eq(pred).sum().float().item() / labels.shape[0]
            accs.append(acc)

    m, h = mean_confidence_interval(accs)
    print(f'Test epoch: {epoch}, test acc: {m * 100:.2f}+-{h * 100:.2f}')
    args.logger.add_scalar('test/acc', m * 100, epoch)

    return m


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='debug')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100'])
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--image_size', type=int, default=224, choices=[224, 84])
    parser.add_argument('--aug', action='store_true', default=True)
    parser.add_argument('--rand_aug', action='store_true')
    parser.add_argument('--aug_support', type=int, default=1)
    parser.add_argument('--model', type=str, default='visformer-t', choices=['visformer-t', 'visformer-t-84'])
    parser.add_argument('--nlp_model', type=str, default='clip', choices=['clip', 'glove', 'mpnet'])
    parser.add_argument('--prompt_mode', type=str, default='spatial+channel', choices=['spatial', 'channel', 'spatial+channel'])
    parser.add_argument('--no_template', action='store_true')
    parser.add_argument('--eqnorm', action='store_true', default=True)
    parser.add_argument('--stage', type=float, default=3.2, choices=[2, 2.1, 2.2, 2.3, 3, 3.1, 3.2, 3.3])
    parser.add_argument('--projector', type=str, default='linear', choices=['linear', 'mlp', 'mlp3'])
    parser.add_argument('--avg', type=str, default='all', choices=['all', 'patch', 'head'])
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--optim', type=str, default='adamw', choices=['sgd', 'adamw'])
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--encoder_lr', type=float, default=1e-6)
    parser.add_argument('--init', type=str, default='checkpoint/miniImageNet/visformer-t/pre-train/checkpoint_epoch_800.pth')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--text_length', type=int, default=20)
    parser.add_argument('--train_way', type=int, default=-1)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train_episodes', type=int, default=-1)
    parser.add_argument('--episodes', type=int, default=600)
    parser.add_argument('--test_classifier', type=str, default='prototype', choices=['prototype', 'fc'])
    parser.add_argument('--print_step', type=int, default=100)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=20)

    args = parser.parse_args()
    if args.seed >= 0:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    main(args)

