import os
import time
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm.auto import tqdm

from config import (
    BATCH_SIZE,
    DEVICE,
    NUM_CLASSES,
    NUM_EPOCH,
    OUT_DIR,
    RESUME,
    SAVE_MODEL_EPOCH,
    SAVE_PLOTS_EPOCH,
    START_EPOCH,
    CHECKPOINT,
)
from dataset import VictimModelDataset
from model import create_FasterRCNN_model
from utils import Averager, collate_fn

plt.style.use("ggplot")


def train(train_data_loader, model):
    print("Training...")
    global train_itr, train_loss_list, args

    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)

        train_loss_hist.send(loss_value)

        losses.backward()
        optimizer.step()

        train_itr += 1

        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list


# function for running validation iterations
def validate(valid_data_loader, model):
    print("Validating")
    global val_itr, val_loss_list, map

    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list


if __name__ == "__main__":
    if not os.path.exists(OUT_DIR):
        os.mkdir(os.path.abspath(OUT_DIR))
    transform_chain = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset = VictimModelDataset(
        image_folder="img",
        label_folder="labels/pascal",
        root_dir="./dataset",
        classes_file="types.txt",
        transform=transform_chain,
    )

    train_indices = torch.randperm(len(dataset.train_indexes)).tolist()
    val_indices = torch.randperm(len(dataset.val_indexes)).tolist()
    classes = dataset.classes
    dataset = Subset(dataset, train_indices)
    dataset_test = Subset(dataset, val_indices)
    dataset_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    dataset_loader_test = DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    model = create_FasterRCNN_model(num_classes=NUM_CLASSES)
    model.to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)

    if RESUME:
        model.load_state_dict(CHECKPOINT["model_state_dict"])
        optimizer.load_state_dict(CHECKPOINT["optimizer_state_dict"])
    model.train()

    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    train_loss_list = []
    val_loss_list = []

    MODEL_NAME = "model"
    for epoch in range(START_EPOCH, NUM_EPOCH):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCH}")

        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()

        # create two subplots, one for each, training and validation
        figure_1, train_ax = plt.subplots()
        figure_2, valid_ax = plt.subplots()

        # start timer and carry out training and validation
        start = time.time()
        train_loss = train(dataset_loader, model)
        val_loss = validate(dataset_loader_test, model)
        print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")
        print(f"Epoch #{epoch} validation loss: {val_loss_hist.value:.3f}")
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        if (epoch + 1) % SAVE_MODEL_EPOCH == 0:  # save model after every n epochs
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"{OUT_DIR}/model{epoch+1}.pth",
            )
            print("SAVING MODEL COMPLETE...\n")

        if (epoch + 1) % SAVE_PLOTS_EPOCH == 0:  # save loss plots after n epochs
            train_ax.plot(train_loss, color="blue")
            train_ax.set_xlabel("iterations")
            train_ax.set_ylabel("train loss")
            valid_ax.plot(val_loss, color="red")
            valid_ax.set_xlabel("iterations")
            valid_ax.set_ylabel("validation loss")
            figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch+1}.png")
            figure_2.savefig(f"{OUT_DIR}/valid_loss_{epoch+1}.png")
            print("SAVING PLOTS COMPLETE...")

        if (epoch + 1) == NUM_EPOCH:  # save loss plots and model once at the end
            train_ax.plot(train_loss, color="blue")
            train_ax.set_xlabel("iterations")
            train_ax.set_ylabel("train loss")
            valid_ax.plot(val_loss, color="red")
            valid_ax.set_xlabel("iterations")
            valid_ax.set_ylabel("validation loss")
            figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch+1}.png")
            figure_2.savefig(f"{OUT_DIR}/valid_loss_{epoch+1}.png")

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"{OUT_DIR}/model{epoch+1}.pth",
            )

        plt.close("all")
