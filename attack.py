import argparse
import os

import cv2
import numpy as np
import torch as t

from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms

from config import CLASSES, DEVICE, NUM_CLASSES
from dataset import AttackDataset
from infer import draw_bbox, predict
from model import create_FasterRCNN_model
from daedalus import Daedalus

transform = transforms.Compose([
    transforms.ToTensor(),
])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default=["all", "most", "least", "single"], help="Attack mode")
    parser.add_argument('--att_class', help="Class to attack in 'single' mode")
    parser.add_argument('--conf', default=0.3, type=float, help="Confidence of attack")
    parser.add_argument('--num_ex', default=10, type=int, help="Number of adversarial examples")
    parser.add_argument('--batch_size', default=1, type=int, help="Batch size")
    parser.add_argument('--steps', default=5, type=int, help="Number of times to adjust constant with binary search")
    parser.add_argument('--consts', default=2, type=int, help="Initial constant to pick as a first guess")
    parser.add_argument('--max_itr', default=10000, type=int, help="Maximum number of iterations to perfrom gradient descent")
    parser.add_argument('--early_abort', default=False, type=bool, help="Abort gradient descent early if no improvements")
    parser.add_argument('--lr', default=1e-2, type=float, help="Rate at which convergence occurs, larger values => converge faster => less accurate results")
    parser.add_argument('--out', default="./adv", help="Path to save outputs")
    parser.add_argument('--weights', help="Path to saved weights")
    parser.add_argument('--num_cls', default=NUM_CLASSES, help="Number of classes", type=int)
    return parser.parse_args()

def main(args):
    # pull necessary args out
    mode, attack_cls, conf, num_ex, steps, consts, max_itr, early_abort, lr, num_cls = args['mode'], args['att_class'], args['conf'], args['num_ex'], args['steps'], \
        args['consts'], args['max_itr'], args['early_abort'], args['lr'], args['num_cls']
    weights = args['weights']
    batch_size = args['batch_size']
    out_root = os.path.abspath(args['out'])

    # root path for all output
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    
    # root path for adversarial
    if not os.path.exists(f"{out_root}/adversarial"):
        os.mkdir(f"{out_root}/adversarial")

    if not os.path.exists(f"{out_root}/adversarial/{lr}"):
        os.mkdir(f"{out_root}/adversarial/{lr}")
    
    # path for adversarial example
    if not os.path.exists(f"{out_root}/adversarial/{lr}/example"):
        os.mkdir(f"{out_root}/adversarial/{lr}/example")

    # load model weights and initialize model
    checkpoint = t.load(weights)

    model = create_FasterRCNN_model(NUM_CLASSES, version="v2")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    # initialize the dataset to perform the attack on
    dataset = AttackDataset(image_folder='img', root_dir='./dataset', label_folder='labels/pascal', transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    attacker = Daedalus(model, attack_cls, mode, (3, 1280, 1920), batch_size, conf, lr, steps, max_itr, early_abort, consts, num_cls, DEVICE)

    COLORS = np.random.uniform(0, 255, size=(NUM_CLASSES, 3))

    for i, data in enumerate(loader):
        attacker.attack(data, out_root)

if __name__ == '__main__':
    args = parse_args().__dict__
    main(args)

