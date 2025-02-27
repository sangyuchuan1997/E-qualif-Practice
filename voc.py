import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import torchvision.transforms as T
from models import FasterRCNN


def load_data(type='det'):
    if type == 'det':
        detection_dataset = torchvision.datasets.VOCDetection(
            root="./data/voc/detection", year="2012", image_set="train", download=True
        )
        return detection_dataset

    if type == 'seg':
        segmentation_dataset = torchvision.datasets.VOCSegmentation(
            root="./data/voc/segmentation", year="2012", image_set="train", download=True
        )
        return segmentation_dataset


def visualize_voc_sample(dataset, idx=0):
    img, annotation = dataset[idx]

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    objects = annotation["annotation"]["object"]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(objects)))

    for obj, color in zip(objects, colors):
        name = obj["name"]
        bbox = obj["bndbox"]

        x1 = float(bbox["xmin"])
        y1 = float(bbox["ymin"])
        x2 = float(bbox["xmax"])
        y2 = float(bbox["ymax"])

        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle(
            (x1, y1), width, height, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)

        ax.text(x1, y1-5, name, bbox=dict(facecolor=color,
                alpha=0.5), fontsize=12, color="white")

    plt.axis("off")
    plt.title(f'Sample {idx}: {annotation["annotation"]["filename"]}')
    plt.show()


class VOCDatasetFasterRCNN(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.class_to_idx = {
            'background': 0,
            'aeroplane': 1,
            'bicycle': 2,
            'bird': 3,
            'boat': 4,
            'bottle': 5,
            'bus': 6,
            'car': 7,
            'cat': 8,
            'chair': 9,
            'cow': 10,
            'diningtable': 11,
            'dog': 12,
            'horse': 13,
            'motorbike': 14,
            'person': 15,
            'pottedplant': 16,
            'sheep': 17,
            'sofa': 18,
            'train': 19,
            'tvmonitor': 20
        }

    def __getitem__(self, index):
        img, target = self.dataset[index]

        # annotation解析
        boxes = []
        labels = []

        objects = target['annotation']['object']
        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            bbox = obj['bndbox']
            boxes.append([
                float(bbox['xmin']),
                float(bbox['ymin']),
                float(bbox['xmax']),
                float(bbox['ymax'])
            ])
            labels.append(self.class_to_idx[obj['name']])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'iamge_id': torch.tensor([index]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(objects),), dtype=torch.int64)
        }

        if self.transform:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.dataset)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            height, width = image.shape[-2:]
            image = F.hflip(image)
            target['boxes'][:, [0, 2]] = width - target['boxes'][:, [2, 0]]
        return image, target


def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    detection_datasets = load_data(type='det')
    # データの可視化
    # visualize_voc_sample(detection_datasets, 0)

    # Data前処理
    dataset = VOCDatasetFasterRCNN(
        detection_datasets, transform=get_transform(train=True))
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    model = FasterRCNN(num_classes=21)
    device = torch.device("mps")
    model.to(device)
    
    x = torch.randn(2, 3, 800, 800).to(device)
    cls_scores, bbox_preds, rois = model(x)
    print("Classification scores shape:", cls_scores.shape)
    print("Bounding box predictions shape:", bbox_preds.shape)
    print("RoIs shape:", rois.shape)
    # train(model, trainloader, testloader, criterion, optimizer)
