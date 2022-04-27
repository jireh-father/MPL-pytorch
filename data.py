import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.utils.data as data
from augmentation import RandAugmentCIFAR
from torchvision.datasets.folder import ImageFolder

import json
import os
import random

logger = logging.getLogger(__name__)

cifar10_mean = (0.491400, 0.482158, 0.4465231)
cifar10_std = (0.247032, 0.243485, 0.2615877)
cifar100_mean = (0.507075, 0.486549, 0.440918)
cifar100_std = (0.267334, 0.256438, 0.276151)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


def get_cifar10(args):
    if args.randaug:
        n, m = args.randaug
    else:
        n, m = 2, 10  # default
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])
    transform_finetune = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        RandAugmentCIFAR(n=n, m=m),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(args.data_path, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs, finetune_idxs = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        args.data_path, train_labeled_idxs, train=True,
        transform=transform_labeled
    )
    finetune_dataset = CIFAR10SSL(
        args.data_path, finetune_idxs, train=True,
        transform=transform_finetune
    )
    train_unlabeled_dataset = CIFAR10SSL(
        args.data_path, train_unlabeled_idxs,
        train=True,
        transform=TransformMPL(args, mean=cifar10_mean, std=cifar10_std)
    )

    test_dataset = datasets.CIFAR10(args.data_path, train=False,
                                    transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, finetune_dataset


def get_cifar100(args):
    if args.randaug:
        n, m = args.randaug
    else:
        n, m = 2, 10  # default
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])
    transform_finetune = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        RandAugmentCIFAR(n=n, m=m),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(args.data_path, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs, finetune_idxs = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        args.data_path, train_labeled_idxs, train=True,
        transform=transform_labeled
    )
    finetune_dataset = CIFAR100SSL(
        args.data_path, finetune_idxs, train=True,
        transform=transform_finetune
    )
    train_unlabeled_dataset = CIFAR100SSL(
        args.data_path, train_unlabeled_idxs, train=True,
        transform=TransformMPL(args, mean=cifar100_mean, std=cifar100_std)
    )

    test_dataset = datasets.CIFAR100(args.data_path, train=False,
                                     transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, finetune_dataset


def get_fashion_transforms(input_size):
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    import albumentations as al
    from albumentations.pytorch import ToTensorV2

    return al.Compose(
        [
            al.LongestMaxSize(max_size=round(input_size * 1.1),
                              interpolation=cv2.INTER_AREA),
            al.augmentations.transforms.PadIfNeeded(min_height=round(input_size * 1.1),
                                                    min_width=round(input_size * 1.1),
                                                    border_mode=cv2.BORDER_CONSTANT,
                                                    value=0),
            al.RandomCrop(input_size, input_size),
            al.HorizontalFlip(p=0.5),
            al.Normalize(),
            ToTensorV2()
        ])


def get_fashion_aug_transforms(input_size):
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    import albumentations as al
    from albumentations.pytorch import ToTensorV2

    return al.Compose(
        [
            al.LongestMaxSize(max_size=round(input_size * 1.1),
                              interpolation=cv2.INTER_AREA),
            al.augmentations.transforms.PadIfNeeded(min_height=round(input_size * 1.1),
                                                    min_width=round(input_size * 1.1),
                                                    border_mode=cv2.BORDER_CONSTANT,
                                                    value=0),
            al.RandomCrop(input_size, input_size),
            al.HorizontalFlip(p=0.5),
            al.ShiftScaleRotate(shift_limit=0.1,
                                scale_limit=0.1,
                                rotate_limit=30,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=0,
                                p=0.5),
            al.Rotate(limit=(-10, 10), border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.25),
            al.RandomBrightnessContrast(p=0.25),
            al.RandomGamma(p=0.25),
            al.OneOf([
                al.HueSaturationValue(),
                al.RGBShift(),
                al.CLAHE(),
            ], p=0.25),
            al.OneOf([
                al.GaussNoise(),
                al.ISONoise(),
                al.MultiplicativeNoise(),
            ], p=0.25),
            al.Blur(blur_limit=3, p=0.25),
            al.Downscale(p=0.2, interpolation=cv2.INTER_LINEAR),
            al.ImageCompression(quality_lower=80, p=0.2),
            al.Normalize(),
            ToTensorV2()
        ])


def get_fashion_test_transforms(size):
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    import albumentations as al
    from albumentations.pytorch import ToTensorV2
    return al.Compose(
        [
            al.LongestMaxSize(max_size=round(size * 1.1),
                              interpolation=cv2.INTER_AREA),
            al.augmentations.transforms.PadIfNeeded(min_height=round(size * 1.1),
                                                    min_width=round(size * 1.1),
                                                    border_mode=cv2.BORDER_CONSTANT,
                                                    value=0),
            al.CenterCrop(height=size, width=size),
            al.Normalize(),
            ToTensorV2()
        ])


def get_fashion_attribute(args):
    transform_labeled = get_fashion_transforms(args.resize)
    transform_finetune = get_fashion_aug_transforms(args.resize)
    transform_val = get_fashion_test_transforms(args.resize)

    # dataset_class = FashionAttributeMultiLabelDataset if args.is_multi_label_dataset else FashionAttributeDataset
    dataset_class = FashionAttributeDataset

    train_labeled_dataset = dataset_class(args.train_label_json, args.label_type, args.data_path, transform_labeled,
                                          use_albumentations=True)

    finetune_dataset = None
    if args.finetune_data_path:
        finetune_dataset = dataset_class(args.finetune_label_json, args.label_type, args.finetune_data_path,
                                         transform_finetune,
                                         use_albumentations=True)

    if args.use_unlabeled_one_folder:
        train_unlabeled_dataset = FashionAttributeUnlabeledDatasetOneFolder(args.unlabeled_data_path,
                                                                            TransformMPLFashion(args),
                                                                            use_albumentations=True)
    else:
        train_unlabeled_dataset = FashionAttributeUnlabeledDataset(args.unlabeled_data_path,
                                                                   TransformMPLFashion(args), use_albumentations=True)

    test_dataset = dataset_class(args.test_label_json, args.label_type, args.test_data_path, transform_val,
                                 use_albumentations=True)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, finetune_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all training data
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx_ex = np.hstack([labeled_idx for _ in range(num_expand_x)])
        np.random.shuffle(labeled_idx_ex)
        np.random.shuffle(labeled_idx)
        return labeled_idx_ex, unlabeled_idx, labeled_idx
    else:
        np.random.shuffle(labeled_idx)
        return labeled_idx, unlabeled_idx, labeled_idx


def x_u_split_test(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        labeled_idx.extend(idx[:label_per_class])
        unlabeled_idx.extend(idx[label_per_class:])
    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.array(unlabeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])

    np.random.shuffle(labeled_idx)
    np.random.shuffle(unlabeled_idx)
    return labeled_idx, unlabeled_idx


class TransformMPL(object):
    def __init__(self, args, mean, std):
        if args.randaug:
            n, m = args.randaug
        else:
            n, m = 2, 10  # default

        self.ori = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize * 0.125),
                                  fill=128,
                                  padding_mode='constant')])
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize * 0.125),
                                  fill=128,
                                  padding_mode='constant'),
            RandAugmentCIFAR(n=n, m=m)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)
        return self.normalize(ori), self.normalize(aug)


class TransformMPLFashion(object):
    def __init__(self, args):
        self.ori = get_fashion_transforms(args.resize)
        self.aug = get_fashion_aug_transforms(args.resize)

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)
        return ori, aug


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class FashionAttributeDataset(ImageFolder):
    def __init__(self, multi_label_json, label_type, root, transform=None, use_albumentations=False):
        super(FashionAttributeDataset, self).__init__(root, transform=transform)
        multi_label_dict = json.load(open(multi_label_json, encoding='utf-8'))
        label_dict = multi_label_dict[label_type]

        keys = list(label_dict.keys())
        keys.sort()

        self.root = root
        self.use_albumentations = use_albumentations

        self.samples = []
        for i, k in enumerate(keys):
            self.samples += list(zip(label_dict[k], [i] * len(label_dict[k])))

    def __getitem__(self, index):
        while True:
            try:
                path, target = self.samples[index]
                path = os.path.join(self.root, path)
                sample = self.loader(path)

                if self.transform is not None:
                    if self.use_albumentations:
                        sample = self.transform(image=np.array(sample))['image']
                    else:
                        sample = self.transform(sample)
                return sample, target
            except Exception as e:
                # traceback.print_exc()
                print(str(e), path)
                index = random.randint(0, len(self) - 1)


class FashionAttributeMultiLabelDataset(ImageFolder):
    def __init__(self, multi_label_json, label_type, root, transform=None):
        super(FashionAttributeMultiLabelDataset, self).__init__(root, transform=transform)
        multi_label_dict = json.load(open(multi_label_json, encoding='utf-8'))
        label_dict = multi_label_dict[label_type]

        class_dict = {}
        for k in label_dict:
            vals = label_dict[k]
            for v in vals:
                class_dict[v] = True

        classes = list(class_dict.keys())
        classes.sort()
        self.classes = classes
        class_map = {c: i for i, c in enumerate(classes)}
        self.samples = []
        for k in label_dict:
            vals = label_dict[k]
            labels = []
            for v in vals:
                labels.append(class_map[v])
            self.samples.append((k, labels))

        self.root = root

    def __getitem__(self, index):
        while True:
            try:
                path, multi_labels = self.samples[index]
                target = [0] * len(self.classes)
                for l in multi_labels:
                    target[l] = 1
                path = os.path.join(self.root, path)
                sample = self.loader(path)
                if self.transform is not None:
                    sample = self.transform(sample)
                # albumentations transform style
                # if self.transform is not None:
                #     sample = self.transform(image=sample)['image']
                return sample, np.array(target, dtype=np.float32)
            except Exception as e:
                # traceback.print_exc()
                print(str(e), path)
                index = random.randint(0, len(self) - 1)


class FashionAttributeUnlabeledDataset(ImageFolder):
    def __init__(self, root, transform=None, use_albumentations=False):
        super(FashionAttributeUnlabeledDataset, self).__init__(root, transform=transform)
        self.use_albumentations = use_albumentations

    def __getitem__(self, index):
        while True:
            try:
                path, target = self.samples[index]
                sample = self.loader(path)
                if self.transform is not None:
                    if self.use_albumentations:
                        sample = self.transform(image=np.array(sample))['image']
                    else:
                        sample = self.transform(sample)

                return sample, target
            except Exception as e:
                # traceback.print_exc()
                print(str(e), path)
                index = random.randint(0, len(self) - 1)


class FashionAttributeUnlabeledDatasetOneFolder(data.Dataset):
    def __init__(self, root, transform=None, use_albumentations=False):
        import glob
        self.samples = glob.glob(os.path.join(root, "*"))
        self.transform = transform
        from torchvision.datasets.folder import default_loader
        self.loader = default_loader
        self.use_albumentations = use_albumentations

    def __getitem__(self, index):
        while True:
            try:
                path = self.samples[index]
                sample = self.loader(path)
                if self.transform is not None:
                    if self.use_albumentations:
                        sample = self.transform(image=np.array(sample))['image']
                    else:
                        sample = self.transform(sample)
                return sample, 1
            except Exception as e:
                # traceback.print_exc()
                print(str(e), path)
                index = random.randint(0, len(self) - 1)

    def __len__(self):
        return len(self.samples)


DATASET_GETTERS = {
    'cifar10': get_cifar10,
    'cifar100': get_cifar100,
    'fashion_category': get_fashion_attribute
}
