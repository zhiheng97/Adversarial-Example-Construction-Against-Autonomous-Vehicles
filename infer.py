import os
import argparse
import cv2
import torch
import numpy as np
import dask.dataframe as dd
import gc

from torchvision import transforms
from torchmetrics.detection import MeanAveragePrecision
from PIL import Image
from model import create_FasterRCNN_model
from utils import LabelEnum

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to model to use for inference', required=True)
    parser.add_argument('-o', '--output', default='./output', help='Path to output folder')
    parser.add_argument('-cls', '--classes', default='./dataset/types.txt', help='Path to label classes', required=True)
    parser.add_argument('--use_gpu', type=bool, default=False, help='To use GPU to accelerate or not')
    parser.add_argument('--image', help="Image to run inference on", required=True)
    parser.add_argument('--metrics', default=False, type=bool, help="To calculate metrics (mAP)")
    
    return parser.parse_args().__dict__

def predict(image, model, threshold, labels, device, transform = None):

    if transform:
        image = transform(image).to(device)
        image = image.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(image.to(device))

    pred_classes = [labels[i] for i in outputs[0]['labels'].cpu().numpy()]

    pred_scores = outputs[0]['scores'].detach().cpu().numpy()

    pred_bbox = outputs[0]['boxes'].detach().cpu().numpy()

    labels = []
    scores = []
    boxes = []
    for i in range(len(pred_scores)):
        if pred_scores[i] >= threshold:
          labels.append(LabelEnum[pred_classes[i]].value)
          scores.append(pred_scores[i])  
          boxes.append(torch.from_numpy(pred_bbox[i]))

    return boxes, pred_classes, labels, scores

def draw_bbox(boxes, classes, labels, image, COLORS):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(image, (int(box[0]), int(box[1])),
                             (int(box[2]), int(box[3])),
                      color, 2)
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)
    return image

def main(args):
    if not os.path.exists(os.path.abspath(args['output'])):
        os.mkdir(os.path.abspath(args['output']))

    if args['use_gpu']:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built() and torch.backends.mps.is_macos13_or_newer():
            args['use_gpu'] = 'mps'
        elif torch.cuda.is_available():
            args['use_gpu'] = 'cuda'
        else:
            args['use_gpu'] = 'cpu'   
    else:
        args['use_gpu'] = 'cpu'        
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    classes = np.array(dd.read_csv(args['classes']))
    classes = classes.reshape(len(classes))
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    model = create_FasterRCNN_model(len(classes), version="v2")
    checkpoint = torch.load(args['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval().to(args['use_gpu'])
    with open(os.path.abspath(args['image']), "rb") as file:
        img_bytes = file.read()
    img = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if args['metrics']:
        gt_file = f"./dataset/labels/pascal/{args['image'].split('.')[1].split('/')[-1]}.txt"
        targets_df = dd.read_csv(os.path.abspath(gt_file), header=None, names=['label', 'xmin', 'ymin', 'xmax', 'ymax'], delimiter=' ')
        gt_labels = np.array(targets_df[['label']])
        gt_labels = np.reshape(gt_labels, len(gt_labels))
        gt_labels = torch.from_numpy(gt_labels)
        gt_boxes = torch.from_numpy(np.array(targets_df[['xmin', 'ymin', 'xmax', 'ymax']]))
        mAP = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    boxes, out_classes, labels, scores = predict(image, model, 0., classes, args['use_gpu'], transform=transform)
    boxes = torch.stack([box for  box in boxes])
    labels = torch.from_numpy(np.array(labels))
    scores = torch.from_numpy(np.array(scores))
    if args['metrics']:
        mAP.update(preds=[{'boxes': boxes, 'scores': scores, 'labels': labels}], target=[{'boxes': gt_boxes, 'labels': gt_labels}])
        print(mAP.compute())
    image = draw_bbox(boxes, out_classes, labels, image, COLORS)
    img_name = args['image'].split('/')[-1]
    path = f"{args['output']}/{img_name}"
    cv2.imwrite(path, image)

if __name__ == '__main__':
    args = parse_args()

    main(args)
    gc.collect()