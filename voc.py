import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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

    fig, ax = plt.subplot(1)
    ax.imshow(img)

    objects = annotation["annotation"]["object"]
    if not isinstance(object, list):
        objects = [objects]

    # 各オブジェクトに対してバウンディングボックスを描画
    for obj in objects:
        name = obj["name"]
        bbox = obj["bndbox"]
        xmin = float(bbox["xmin"])
        ymin = float(bbox["ymin"])
        xmax = float(bbox["xmax"])
        ymax = float(bbox["ymax"])

        # バウンディングボックスの描画
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle(
            (xmin, ymin), width, height, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)

        # ラベルの追加
        plt.text(
            xmin, ymin - 5, name, bbox=dict(facecolor="red", alpha=0.5), color="white"
        )

    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    detection_datasets = load_data(type='det')
    visualize_voc_sample(detection_datasets, 0)
    # model, criterion, optimizer = init_model(model='ResNet')
    # train(model, trainloader, testloader, criterion, optimizer)
