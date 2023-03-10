import time
from pathlib import Path

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataLoader import *
from Metrics import *

DATA_ROOT = Path('./Data')
IMAGES = DATA_ROOT / Path('images')
MASKS = DATA_ROOT / Path('masks')
LABELS = ['background', 'car', 'wheel', 'lights', 'window']
SAVE_MODELS_PATH = Path('models')

device = 'mps'
# device = 'cpu'
# device = 'gpu'
device = torch.device(device)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_figure_size(figsize=(8, 6), dpi=120):
    plt.figure(figsize=figsize, dpi=dpi)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def save_model(model, fn):
    SAVE_MODELS_PATH.mkdir(exist_ok=True)
    torch.save(model, f'./{SAVE_MODELS_PATH}/{fn}')

def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler):
    train_losses = []
    test_losses = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1
    not_improve = 0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        # training loop

        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            # training phase
            image_tiles, mask_tiles = data['img'], data['mask']
            # print(image_tiles.size(), mask_tiles.size())

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)
            # forward
            output = model(image)
            # print(f'output size {output.size()} mask size {mask.size()}')
            loss = criterion(output[0], mask[0])
            # evaluation metrics
            iou_score += mIoU(output, mask)
            accuracy += pixel_accuracy(output, mask)
            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            # step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()

        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0

            # validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    # reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data['img'], data['mask']

                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)
                    output = model(image)
                    # evaluation metrics
                    val_iou_score += mIoU(output, mask)
                    test_accuracy += pixel_accuracy(output, mask)
                    # loss
                    loss = criterion(output[0], mask[0])
                    test_loss += loss.item()

            # calculatio mean for each batch
            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(val_loader))

            print(f'loss train: {running_loss}')
            print(f'loss train: {test_loss}')

            if min_loss > (test_loss / len(val_loader)):
                print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss / len(val_loader))))
                min_loss = (test_loss / len(val_loader))

                print('saving model...')
                save_model(model, 'Unet-Mobilenet_v2_mIoU-{:.3f}_loss-{:.3f}.pt'.
                           format(val_iou_score / len(val_loader),
                                  min_loss
                                  ))

            if (test_loss / len(val_loader)) > min_loss:
                not_improve += 1
                min_loss = (test_loss / len(val_loader))
                print(f'Loss Not Decrease for {not_improve} time')
                if not_improve == 7:
                    print('Loss not decrease for 7 times, Stop Training')
                    break

            # iou
            val_iou.append(val_iou_score / len(val_loader))
            train_iou.append(iou_score / len(train_loader))
            train_acc.append(accuracy / len(train_loader))
            val_acc.append(test_accuracy / len(val_loader))
            print("Epoch:{}/{}..".format(e + 1, epochs),
                  "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                  "Val Loss: {:.3f}..".format(test_loss / len(val_loader)),
                  "Train mIoU:{:.3f}..".format(iou_score / len(train_loader)),
                  "Val mIoU: {:.3f}..".format(val_iou_score / len(val_loader)),
                  "Train Acc:{:.3f}..".format(accuracy / len(train_loader)),
                  "Val Acc:{:.3f}..".format(test_accuracy / len(val_loader)),
                  "Time: {:.2f}m".format((time.time() - since) / 60))

    history = {'train_loss': train_losses, 'val_loss': test_losses,
               'train_miou': train_iou, 'val_miou': val_iou,
               'train_acc': train_acc, 'val_acc': val_acc,
               'lrs': lrs}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    return history


def main():
    max_lr = 1e-3
    epoch = 15
    weight_decay = 1e-4

    img_list = list(IMAGES.glob('*'))
    mask_list = list(MASKS.glob('*'))

    imgs_train, imgs_test, mask_train, mask_test = train_test_split(img_list, mask_list, 0.8)

    # размер входных изображений должен быть кратен размеру первой свертки
    trf_img = transforms.Compose([
        transforms.ColorJitter(0.7, 0.7, 0.7, 0),
        ToFloatTensor(),
        # transforms.Normalize(mean=0.5, std=0.5)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        Padding(32),
    ])

    trf_mask = transforms.Compose([
        ToFloatTensor(),
        Padding(32),
    ])

    data_train = CarBodyDataset(imgs_train, mask_train, trf_img, trf_mask)
    data_test = CarBodyDataset(imgs_test, mask_test, trf_img, trf_mask)

    train_loader = DataLoader(data_train, batch_size=1, shuffle=False)
    test_loader = DataLoader(data_test, batch_size=1, shuffle=True)

    model = smp.Unet('mobilenet_v2',
                     encoder_weights='imagenet',
                     classes=1,
                     activation=None,
                     encoder_depth=3,
                     decoder_channels=[64, 32, 16],
                     # encoder_depth=5,
                     # decoder_channels=[256, 128, 64, 32, 16]
                     )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                                steps_per_epoch=len(train_loader))

    history = fit(epoch, model, train_loader, test_loader, criterion, optimizer, sched)


if __name__ == '__main__':
    main()
