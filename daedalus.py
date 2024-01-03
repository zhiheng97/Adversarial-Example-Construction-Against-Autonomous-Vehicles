import os

import cv2
import numpy as np
import torch as t
from torchvision.transforms import transforms

from utils import Logger
from infer import draw_bbox
from config import CLASSES, NUM_CLASSES

COLORS = np.random.uniform(0, 255, size=(NUM_CLASSES, 3))
# partitions should be integer-like types
def dynamic_partition(data: t.Tensor, partitions: t.Tensor, num_partitions=None):
    assert len(partitions.shape) == 1, "Only one dimensional partitions supported"
    assert (
        data.shape[0] == partitions.shape[0]
    ), "Partitions requires the same size as data"

    if num_partitions is None:
        num_partitions = max(t.unique(partitions))

    return [data[partitions == i] for i in range(num_partitions)]


class Daedalus:
    def __init__(
        self,
        model,
        target_class,
        attack_mode,
        shape,
        batch_size,
        confidence,
        learning_rate,
        binary_search_step,
        max_iterations,
        abort_early,
        initial_const,
        num_cls,
        device,
    ):
        self.LR = learning_rate
        self.MAX_ITR = max_iterations
        self.BSS = binary_search_step
        self.AE = abort_early
        self.initial_consts = initial_const
        self.batch_size = batch_size
        self.repeat = binary_search_step >= 6
        self.model = model
        self.confidence = confidence
        self.target_class = target_class
        self.attack_mode = attack_mode
        self.num_cls = num_cls
        self.device = device

        self.logger = Logger('./logs/log.txt').logger

    def select_class(self, target_class, boxes, scores, labels, mode="all"):
        box_classes = labels
        class_counts = t.bincount(labels)
        print(class_counts)

        if mode == "all":
            selected_boxes = t.reshape(boxes, [self.batch_size, -1, 4])
            if scores == None:
                return selected_boxes, None, selected_scores
            return selected_boxes, None, None
        elif mode == "most":
            selected_cls = t.argmax(class_counts)
        elif mode == "least":
            class_counts = t.where(
                t.equal(class_counts, 0),
                int(1e6) * t.ones_like(class_counts, dtype=t.int32),
                class_counts,
            )
            selected_cls = t.argmin(class_counts)
        elif mode == "single":
            file = "./dataset/types.txt"
            with open(file) as f:
                class_names = f.readlines()
            class_names = [c.strip() for c in class_names]
            selected_cls = class_names.index(target_class.numpy())
        selected_cls = selected_cls.type(t.int32)
        index = t.equal(box_classes, selected_cls).type(t.int32)
        _, selected_boxes = dynamic_partition(boxes, index, num_partitions=2)
        _, selected_scores = dynamic_partition(scores, index, num_partitions=2)
        selected_boxes = t.reshape(selected_boxes, [self.batch_size, -1, 4])
        selected_scores = t.reshape(selected_scores, [self.batch_size, -1, self.num_cls])
        if scores == None:
            return selected_boxes, None, selected_scores
        _, selected_objectness = dynamic_partition(scores, index, num_partitions=2)
        selected_objectness = t.reshape(selected_objectness, [self.batch_size, -1, 1])
        return selected_boxes, selected_objectness, selected_scores

    def run(self, img):
        newimg = self.pertubations + img

        outs = self.model(newimg)
        boxes = outs[0]["boxes"]
        labels = outs[0]["labels"]
        scores = outs[0]["scores"]
        logits = outs[0]['logits']
        boxes, _, _ = self.select_class(
            self.target_class, boxes, scores, labels, mode=self.attack_mode
        )

        x1 = boxes[..., 0:1] / img.shape[3]
        y1 = boxes[..., 1:2] / img.shape[2]
        x2 = boxes[..., 2:3] / img.shape[3]
        y2 = boxes[..., 3:4] / img.shape[2]
        self.bw = t.abs(x2 - x1)
        self.bh = t.abs(y1 - y2)

        l2dist = t.sum(t.square(newimg - img), [1, 2, 3])

        box_scores = t.max(logits, dim=-1, keepdim=True).values

        x = t.square(box_scores - 1)
        loss1_1_x = t.mean(x)

        f3 = t.mean(t.square(t.multiply(self.bw, self.bh)))

        loss_adv = loss1_1_x + f3
        loss1 = t.mean(self.consts * loss_adv).type(t.float32)
        loss2 = t.mean(l2dist).type(t.float32)
        self.loss = t.add(loss1, loss2)

        # propagate the loss, this updates the perturbations tensor
        self.loss.backward()
        self.optimizer.step()
        
        return (
            self.loss,
            l2dist,
            loss_adv.unsqueeze(0).detach().cpu().numpy(),
            newimg,
            loss1_1_x,
            f3
        )

    def attack_batch(self, imgs, file_name):
        def check_success(loss, init_loss):
            return loss <= init_loss * (1 - self.confidence)
        
        if not os.path.exists(f"./adv/adversarial/{self.LR}/example/{file_name}"):
            os.mkdir(f"./adv/adversarial/{self.LR}/example/{file_name}")

        batch_size = self.batch_size

        lower_bound = t.tensor(np.zeros(batch_size)).type(t.float32).to(self.device)
        self.consts = (
            t.from_numpy(np.ones(batch_size) * self.initial_consts)
            .type(t.float32)
            .to(self.device)
        )
        upper_bound = t.tensor(np.ones(batch_size) * 1e10).type(t.float32).to(self.device)

        o_bestl2 = [1e10] * batch_size
        o_bestloss = [1e10] * batch_size
        o_bestattack = [np.zeros(imgs[0].shape)] * batch_size
        
        #init_loss, _, init_adv_loss, _, _, _ = self.run(imgs)
        #self.logger.info(f"Initial Loss: {init_loss}, Initial Adversarial Loss: {init_adv_loss}")

        for outer_step in range(self.BSS):
            self.pertubations = t.tensor(
            np.zeros((imgs.shape)), dtype=t.float32
            ).to(self.device)
            self.pertubations.requires_grad = True

            self.loss = t.Tensor(np.zeros((1))).to(self.device)
            self.loss.requires_grad = True

            optimizer = t.optim.Adam([self.pertubations], lr=self.LR)
            self.optimizer = optimizer
            self.optimizer.zero_grad()
        
            init_loss, _, init_adv_loss, _, _, _ = self.run(imgs)
            self.logger.info(f"Initial Loss: {init_loss}, Initial Adversarial Loss: {init_adv_loss}")

            bestl2 = [1e10] * batch_size
            bestloss = [1e10] * batch_size

            if self.repeat and outer_step == self.BSS - 1:
                self.consts = upper_bound

            self.logger.info(f"Adjusted c to: {self.consts}")
            prev = init_loss * 1.1
            for iteration in range(self.MAX_ITR):

                l, l2s, l1s, nimgs, loss1_1_x, f3 = self.run(imgs)

                self.logger.info(f"=== iteration: {iteration} ===")
                self.logger.info(
                    f"Loss values of box confidence and dimension: {loss1_1_x, f3}"
                )
                self.logger.info(f"Adversarial loss: {l1s}")
                self.logger.info(f"Distortions: {l2s}")

                if self.AE and iteration % (self.MAX_ITR // 10) == 0:
                    if iteration != 0 and l > prev * 0.9999:
                        break
                    prev = 1

                for e, (l1, l2, ii) in enumerate(zip(l1s, l2s, nimgs)):
                    if l2 < bestl2[e] and check_success(l1, init_adv_loss[e]):
                        bestl2[e] = l2
                        bestloss[e] = l1
                        self.logger.info(f"Found best l2 and loss: {l2, l1}")
                        saveImage(ii.detach().cpu().numpy(), f'./adv/adversarial/{self.LR}/example/{file_name}/{outer_step}_{iteration}.jpg')
                    if l2 < o_bestl2[e] and check_success(l1, init_adv_loss[e]):
                        o_bestl2[e] = l2
                        o_bestloss[e] = l1
                        o_bestattack[e] = ii
                        self.logger.info(f"Found overall best l2, loss and attack: {l2, l1}")
                        saveImage(ii.detach().cpu().numpy(), f'./adv/adversarial/{self.LR}/example/{file_name}/{outer_step}_overall_best.jpg')

            for e in range(batch_size):
                if check_success(l1s[e], init_adv_loss[e]):
                    upper_bound[e] = min(upper_bound[e], self.consts[e])
                    if upper_bound[e] < 1e9:
                        self.consts[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    lower_bound[e] = max(lower_bound[e], self.consts[e])
                    if upper_bound[e] < 1e9:
                        self.consts[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        self.consts[e] *= 10
        #o_bestl2 = np.array([l2.detach().cpu() for l2 in o_bestl2])
        #return o_bestattack, o_bestl2

    def attack(self, data, out_path):
        r = []
        ds = []
        print(f"Go up to {len(data['img'])}")
        for i in range(0, len(data['img']), self.batch_size):
            imgs = data['img'].to(self.device)
            print(f"Tick {i}")
            self.attack_batch(imgs, data['name'][0])
        #return np.array(r), np.array(ds)

def saveImage(image, path):
    img_shape = image.shape
    if img_shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    image *= 255.
    cv2.imwrite(path, image)
