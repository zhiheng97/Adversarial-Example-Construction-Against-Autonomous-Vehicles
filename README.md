# Adversarial Example Construction Against Autonomous Vehicles

---

**Undergraduate Final Year Project**

*Completed by Loh Zhi Heng, supervised by A/P Tan Rui, and presented to A/P Douglas Maskell and A/P Tan Rui on 8 December 2023.*

*Special thanks to Research Associate Guo Dongfang for managing any technical difficulties, and resources required.*

*Link to FYP Paper - https://hdl.handle.net/10356/171944*

---

## Abstract

>With autonomous vehicles (AVs) approaching widespread adoption, there is a need to
emphasize safety as it must not be neglected. Touted to be free from errors commonly
made by humans, they are nevertheless not immune to attacks with malicious intent.
In general, AVs utilize a variety of machine-learning models and sensors to help them
understand their environment. However, based on past research on machine learning
models, it is understood that they may be susceptible to adversarial attacks. In this
paper, Daedalus, an attack algorithm that exploits the vulnerability in Non-Maximum
Suppression (NMS) is used to generate adversarial examples using a surrogate model.
The perturbations on the images are nearly imperceptible. The generated images are
subsequently evaluated against the Single-Stage Monocular 3D Object Detection via
Key Point Estimation (SMOKE) utilized in Baidu Apollo’s Autonomous Driving
System for camera-based object detection. In addition, look into potential mitigations
that could be implemented to mitigate Daedalus.

## Setup
---

### Environments used

- Baidu Apollo 7.0
- Google Colaboratory
- MBP M1 Pro, MacOS Ventura 13.5.1 

### Running the project

*You may skip the model extraction and training the surrogate model phase if you do not intend to run it as a black-box project*

#### Model Extraction

Google Colaboratory was used as Baidu Apollo's SMOKE model requires the CUDA cores to run inference on images. If you do own a system that has CUDA cores available you may run it on your own system. 

The `SMOKE_extraction.ipynb` is used for extracting the inference values. It is based on the Apollo Object Detection file `smoke_object_detection.cc`. 

It uses the Waymo Open Dataset V2 Perception dataset as the model was trained on it by Apollo Developers. This allows the surrogate model to have a better approximation of SMOKE's model weights.

The project runs the extraction over 5 times, each time changing the camera view.

#### Training the Surrogate Model

The project uses the PyTorch V2 implementation of Faster RCNN as the base model with Stochastic Gradient Descent as the optimizer.

Do note that the labels would require `__background__` as the first item in the list.

The script to train the surrogate model is `engine.py`. The script tries to use the MPS backend found in Apple Silicon devices, else it defaults to CPU backend. However, at the time of the project, certain operations are yet to be implemented on MPS, hence the environment variable of `PYTORCH_ENABLE_MPS_FALLBACK=1`. 

In the event that, you're running out of memory, you can set the environment variable `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.1`, but it is not recomended!

The optimizer uses the following configurations:

- Learning rate: 0.0001
- Momentum: 0.9
- Weight decay: 0.0005

The model is then trained on 5100 randomly selected images from the Waymo Open Dataset V2, with 100 images used as the validation dataset for 20 epochs achieving a validation loss of around `0.2`.

#### Generating Adversarial Examples

The project references the [Daedalus implementation](https://github.com/NeuralSec/Daedalus-attack), however it is ported over to PyTorch instead of using the TensorFlow framework.

Moreover, the project only implements the l<sub>2</sub> attack, attacking all categories, and using the loss function f<sub>3</sub> as detailed in the paper.

Additionally, some modifications to the PyTorch implementation of Faster RCNN are required, as Daedalus requires the `class_probabilities` as an input. The modifications to be made are in `postprocess_detections.py` and `roi_heads.py`. This can be avoided if you can find a Faster RCNN implementation that returns this as an output.

The script that generates the adversarial examples is `attack.py`, and it takes in several arguments:

- --mode: The attack mode
- --conf: The confidence of the attack
- --num_ex: Number of examples
- --batch_size: The size of each batch
- --steps: The incremental step
- --consts: The constant to increment by
- --max_itr: The maximum iterations
- --lr: The learning rate
- --out: The directory to store the outputs
- --weights: The directory that stores the Faster RCNN weights
- --num_cls: Number of classes
- --early_abort: To abort early if the iteration doesn't get better
- --use_gpu: To indicate that you want to use GPU

A total of 600 adversarial examples were generated, 100 images per configuration:

- Conf: 0.3, 0.7
- LR: 1e-1, 1e-2, 1e-3
- Const: 2
- Max Iterations: 1000
- Step: 5
- Model: all

### Results

Performance was evaluated using the Mean Average Precision (MAP) metric, the following table shows the performance.

| Conf | LR  | mAP50 | mAP75 | mAR10 | mAR100 |
|:----:|:---:|:-----:|:-----:|:-----:|:------:|
|0.3|0.1|0.3342|0.2703|0.3087|0.3270|
|0.3|0.01|0.6461|0.6256|0.6224|0.6801|
|0.3|0.0001|0.8708|0.8675|0.8253|0.9020|
|0.7|0.1|0.3456|0.2868|0.3276|0.3362|
|0.7|0.01|0.5545|0.5357|0.5241|0.5722|
|0.7|0.0001|0.8391|0.8331|0.7940|0.8655|

From the above table, it shows that although the results are not as impressive as seen in the Daedalus Paper, it is still dangerous as it shows the the Non-Maximum Supression (NMS) algorithm can be defeated.

### Conclusion

In summary, this paper has presented the use of Daedalus in a black-box environment to generate adversarial examples in an attempt to defeat Baidu Apollo's SMOKE model, while maintaining near imperceptible perturbations to the clean image.

Potentially, adversaries could use Daedalus to generate a dataset that has been poisoned by these imperceptible perturbations causing object detections models to falsely detect objects in the scene.

A potential mitigation could be through the use of Multi-Sensor Fusion (MSF) methods to mitigate the possibility of such an attack.

### Citations
`@artical{9313033,  
author={Wang, Derui and Li, Chaoran and Wen, Sheng and Han, Qing-Long and Nepal, Surya and Zhang, Xiangyu and Xiang, Yang}, 
journal={IEEE Transactions on Cybernetics},  
title={Daedalus: Breaking Nonmaximum Suppression in Object Detection via Adversarial Examples},  
year={2021}, 
volume={}, 
number={},
pages={1-14},
doi={10.1109/TCYB.2020.3041481}}`

`@InProceedings{Sun_2020_CVPR, author = {Sun, Pei and Kretzschmar, Henrik and Dotiwalla, Xerxes and Chouard, Aurelien and Patnaik, Vijaysai and Tsui, Paul and Guo, James and Zhou, Yin and Chai, Yuning and Caine, Benjamin and Vasudevan, Vijay and Han, Wei and Ngiam, Jiquan and Zhao, Hang and Timofeev, Aleksei and Ettinger, Scott and Krivokon, Maxim and Gao, Amy and Joshi, Aditya and Zhang, Yu and Shlens, Jonathon and Chen, Zhifeng and Anguelov, Dragomir}, title = {Scalability in Perception for Autonomous Driving: Waymo Open Dataset}, booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, month = {June}, year = {2020} }`
