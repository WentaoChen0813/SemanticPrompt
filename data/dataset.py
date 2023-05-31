import torchvision


# train_dataset_path = {
#         'miniImageNet': 'dataset/miniImagenet/base',
#         'tieredImageNet': 'dataset/tieredImageNet/base',
#         'CIFAR-FS': 'dataset/cifar100/base',
#         'FC100': 'dataset/FC100_hd/base',
#     }
#
# val_dataset_path = {
#         'miniImageNet': 'dataset/miniImagenet/val',
#         'tieredImageNet': 'dataset/tieredImageNet/val',
#         'CIFAR-FS': 'dataset/cifar100/val',
#         'FC100': 'dataset/FC100_hd/val',
#     }
#
# test_dataset_path = {
#         'miniImageNet': 'dataset/miniImagenet/novel',
#         'tieredImageNet': 'dataset/tieredImageNet/novel',
#         'CIFAR-FS': 'dataset/cifar100/novel',
#         'FC100': 'dataset/FC100_hd/novel',
#     }
#
# dataset_path = {
#         'miniImageNet': ['dataset/miniImagenet/base', 'dataset/miniImagenet/val', 'dataset/miniImagenet/novel'],
#         'tieredImageNet': ['dataset/tieredImageNet/base', 'dataset/tieredImageNet/val', 'dataset/tieredImageNet/novel'],
#         'CIFAR-FS': ['dataset/cifar100/base', 'dataset/cifar100/val', 'dataset/cifar100/novel'],
#         'FC100': ['dataset/FC100_hd/base', 'dataset/FC100_hd/val', 'dataset/FC100_hd/novel'],
# }

train_dataset_path = {
        'miniImageNet': '../dataset/miniImagenet/base',
        'tieredImageNet': '../dataset/tieredImageNet/base',
        'CUB': '../dataset/CUB/base',
        'CIFAR-FS': '../dataset/cifar100/base',
        'FC100': '../dataset/FC100/base',
        'FC100_hd': '../dataset/FC100_hd/base',
        'CropDisease': '../dataset/CD-FSL/CropDisease/base'
    }

val_dataset_path = {
        'miniImageNet': '../dataset/miniImagenet/val',
        'tieredImageNet': '../dataset/tieredImageNet/val',
        'CUB': '../dataset/CUB/val',
        'CIFAR-FS': '../dataset/cifar100/val',
        'FC100': '../dataset/FC100/val',
        'CropDisease': '../dataset/CD-FSL/CropDisease/val'
    }

test_dataset_path = {
        'miniImageNet': '../dataset/miniImagenet/novel',
        'tieredImageNet': '../dataset/tieredImageNet/novel',
        'CUB': '../dataset/CUB/novel',
        'CIFAR-FS': '../dataset/cifar100/novel',
        'FC100': '../dataset/FC100/novel',
        'FC100_hd': '../dataset/FC100_hd/novel',
        'EuroSAT': '../dataset/CD-FSL/EuroSAT/2750',
        'CropDisease': '../dataset/CD-FSL/CropDisease/novel'
    }

dataset_path = {
        'miniImageNet': ['../dataset/miniImagenet/base', '../dataset/miniImagenet/val', '../dataset/miniImagenet/novel'],
        'tieredImageNet': ['../dataset/tieredImageNet/base', '../dataset/tieredImageNet/val', '../dataset/tieredImageNet/novel'],
        'CUB': ['../dataset/CUB/base', '../dataset/CUB/val', '../dataset/CUB/novel'],
        'CIFAR-FS': ['../dataset/cifar100/base', '../dataset/cifar100/val', '../dataset/cifar100/novel'],
        'FC100': ['../dataset/FC100/base', '../dataset/FC100/val', '../dataset/FC100/novel'],
        'CropDisease': ['../dataset/CD-FSL/CropDisease/train'],
        'EuroSAT': ['../dataset/CD-FSL/EuroSAT/2750']
}


class DatasetWithTextLabel(object):
    def __init__(self, dataset_name, aug, split='test'):
        self.dataset_name = dataset_name
        if split == 'train':
            dataset_path = train_dataset_path[dataset_name]
        elif split == 'val':
            dataset_path = val_dataset_path[dataset_name]
        elif split == 'test':
            dataset_path = test_dataset_path[dataset_name]
        self.dataset = torchvision.datasets.ImageFolder(dataset_path, aug)
        self.idx2text = {}
        if dataset_name == 'miniImageNet' or dataset_name == 'tieredImageNet':
            with open('data/ImageNet_idx2text.txt', 'r') as f:
                for line in f.readlines():
                    idx, _, text = line.strip().split()
                    text = text.replace('_', ' ')
                    self.idx2text[idx] = text
        elif dataset_name == 'FC100':
            with open('data/cifar100_idx2text.txt', 'r') as f:
                for line in f.readlines():
                    idx, text = line.strip().split()
                    idx = idx.strip(':')
                    text = text.replace('_', ' ')
                    self.idx2text[idx] = text
        elif dataset_name == 'CIFAR-FS':
            for idx in self.dataset.classes:
                text = idx.replace('_', ' ')
                self.idx2text[idx] = text

    def __getitem__(self, i):
        image, label = self.dataset[i]
        text = self.dataset.classes[label]
        text = self.idx2text[text]
        # text prompt: A photo of a {label}
        text = 'A photo of a ' + text
        return image, label, text

    def __len__(self):
        return len(self.dataset)
