import torch
import os
import cv2
import dask.dataframe as dd
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from config import TRAIN_DIR, VALID_DIR, ATT_DIR


class VictimModelDataset(Dataset):
    """Dataset based on the output from the victim model"""

    def __init__(
        self, label_folder, image_folder, root_dir, classes_file, transform=None
    ):
        self.indexes = sorted(
            [
                file.split(".")[0]
                for file in os.listdir(os.path.join(root_dir, label_folder))
            ],
            key=lambda x: int(x),
        )
        self.permuted_indexes = torch.randperm(len(self.indexes))
        self.root_dir = root_dir
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        classes_path = os.path.join(os.path.abspath(root_dir), classes_file)
        self.classes = np.array(dd.read_csv(classes_path))
        self.classes = self.classes.reshape((len(self.classes)))
        self.train_indexes = getIndexes(
            TRAIN_DIR, "train.txt", slice(5000), self.permuted_indexes
        )  # getSaveTrainIndexes(self.permuted_indexes)
        self.val_indexes = getIndexes(
            VALID_DIR, "val.txt", slice(-100, -1), self.permuted_indexes
        )  # getSaveValIndexes(self.permuted_indexes)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.root_dir, self.image_folder, f"{self.indexes[idx]}.jpg"
        )

        label_name = os.path.join(
            os.path.abspath(self.root_dir),
            self.label_folder,
            f"{self.indexes[idx]}.txt",
        )

        label_df = dd.read_csv(
            label_name, delimiter=" ", names=["label", "xmin", "ymin", "xmax", "ymax"]
        )

        boxes = np.array(label_df[["xmin", "ymin", "xmax", "ymax"]])

        for box in boxes:
            if box[0] > box[2]:
                box[0], box[2] = box[2], box[0]
            if box[1] > box[3]:
                box[1], box[3] = box[3], box[1]

        labels = label_df[["label"]].values.compute()
        for label in labels:
            label[0] += 1

        image = getImage(img_name)

        if self.transform:
            image = self.transform(image)

        targets = {
            "boxes": torch.from_numpy(boxes).type(torch.float32),
            "labels": torch.from_numpy(labels).reshape(len(labels)),
        }

        return image, targets


class AttackDataset(Dataset):
    def __init__(self, image_folder, label_folder, root_dir, transform=None):
        self.indexes = sorted(
            [
                file.split(".")[0]
                for file in os.listdir(os.path.join(root_dir, image_folder))
            ],
            key=lambda x: int(x),
        )
        self.permuted = torch.randperm(len(self.indexes))
        self.root_dir = root_dir
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.model_train_val_indexes = np.concatenate(
            (
                getIndexes(TRAIN_DIR, "train.txt", slice(5000)),
                getIndexes(
                    VALID_DIR, "val.txt", slice(-100, -1)
                ),
            )
        )
        self.attack_indexes = self.getSaveAttackIndexes(self.model_train_val_indexes)

    def __len__(self):
        return len(self.attack_indexes)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.root_dir, self.image_folder, f"{self.attack_indexes[idx]}.jpg"
        )

        image = getImage(img_name)

        if self.transform:
            image = self.transform(image)

        return {"img": image.type(torch.float32), "name": self.attack_indexes[idx]}

    def getSaveAttackIndexes(self, opt_out_indexes):
        indexes = []
        x = 1
        if not os.path.exists(f"{ATT_DIR}/attack.txt"):
            try:
                os.mkdir(f"{ATT_DIR}")
            except:
                pass
            with open(f"{ATT_DIR}/attack.txt", "w") as f:
                for i in self.permuted:
                    if int(i) not in opt_out_indexes and os.path.exists(f"./dataset/img/{i}.jpg"):
                        f.writelines(f"{i}\n")
                        x += 1
        indexes = np.array(dd.read_csv(f"{ATT_DIR}/attack.txt", header=None))
        indexes = np.reshape(indexes, len(indexes))
        return indexes


def getImage(path):
    with open(path, "rb") as file:
        img_bytes = file.read()
    img = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return image


def getIndexes(path_to_check_for_indexes, file_name, slicer, permuted_indexes=[]):
    if not os.path.exists(path_to_check_for_indexes):
        os.mkdir(path_to_check_for_indexes)
        indexes = []
        with open(f"{path_to_check_for_indexes}/{file_name}", "w") as f:
            for i in permuted_indexes[slicer]:
                f.writelines(f"{i}\n")
                indexes.append(f"{i}")
        return indexes
    indexes = np.array(
        dd.read_csv(f"{path_to_check_for_indexes}/{file_name}", header=None)
    )
    indexes = np.reshape(indexes, len(indexes))
    return indexes


# def getSaveTrainIndexes(permuted_indexes = []):
#     if not os.path.exists(f"{TRAIN_DIR}"):
#         os.mkdir(f"{TRAIN_DIR}")
#         with open(f"{TRAIN_DIR}/train.txt", "w") as f:
#             for i in permuted_indexes[:5000]:
#                 f.writelines(f"{i}\n")
#     train_indexes = np.array(dd.read_csv(f"{TRAIN_DIR}/train.txt", header=None))
#     train_indexes = np.reshape(train_indexes, len(train_indexes))
#     return train_indexes

# def getSaveValIndexes(permuted_indexes = []):
#     if not os.path.exists(f"{VALID_DIR}"):
#         os.mkdir(f"{VALID_DIR}")
#         with open(f"{VALID_DIR}/val.txt", "w") as f:
#             for i in permuted_indexes[-100:]:
#                 f.writelines(f"{i}\n")
#     val_indexes = np.array(dd.read_csv(f"{VALID_DIR}/val.txt", header=None))
#     val_indexes = np.reshape(val_indexes, len(val_indexes))
#     return val_indexes
