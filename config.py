import torch
import numpy as np
import dask.dataframe as dd

BATCH_SIZE = 1
RESIZE_TO = 640
START_EPOCH = 13
NUM_EPOCH = 20
RESUME = True
WEIGHTS = "./outputs/model13.pth"
CHECKPOINT = torch.load(WEIGHTS)

DEVICE = torch.device('mps') if torch.backends.mps.is_available() and torch.backends.mps.is_built() else torch.device('cpu')
# DEVICE = torch.device('cpu')

TRAIN_DIR = './dataset/train'
VALID_DIR = './dataset/val'
ATT_DIR = './dataset/attack'

CLASSES_LIST = np.array(dd.read_csv('./dataset/types.txt'))
CLASSES = CLASSES_LIST.reshape(len(CLASSES_LIST))
NUM_CLASSES = len(CLASSES)

VISUALIZE_TRANSFORMED_IMAGES = False

OUT_DIR = './outputs'
SAVE_PLOTS_EPOCH = 1
SAVE_MODEL_EPOCH = 1