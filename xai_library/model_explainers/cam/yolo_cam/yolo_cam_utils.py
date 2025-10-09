import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import (EigenCAM, EigenGradCAM, GradCAM, GradCAMPlusPlus,
                              HiResCAM, LayerCAM, RandomCAM, XGradCAM)
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics import YOLO
from ultralytics.utils.ops import xywh2xyxy


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        post_result, pre_post_boxes, post_boxes = self.post_process(model_output[0])
        return [[post_result, pre_post_boxes]]

    def release(self):
        for handle in self.handles:
            handle.remove()

class YOLOTarget(torch.nn.Module):
    def __init__(self, output_type, conf, ratio, target_class=None):
        super().__init__()
        self.output_type = output_type
        self.conf = conf
        self.ratio = ratio
        self.target_class = target_class  # Target class to focus on

    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        detected_classes = set()  # Collect unique class IDs detected
        for i in range(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            class_id = post_result[i].argmax().item()
            detected_classes.add(class_id)  # Record the detected class ID

            if self.target_class is not None and class_id != self.target_class:
                # If target_class is set, do not gather detections of other classes in the result
                continue

            if self.output_type == 'class' or self.output_type == 'all':
                result.append(post_result[i].max())
            elif self.output_type == 'box' or self.output_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])

        # Suggest detected classes if no target_class is specified
        if self.target_class is None and detected_classes:
            print(f"Detected classes: {sorted(detected_classes)}")
            print("Specify a `target_class` in the parameters to focus on a specific class.")
        
        return sum(result)

class YOLOHeatmap:
    def __init__(self, params):
        self.device = params['device']
        self.target_class = params.get('target_class', None)  # Fetch target_class from params
        self.model = YOLO(params['weight']).to(self.device)
        self.model.model.eval()
        for param in self.model.parameters():
            param.requires_grad = True
        self.target = YOLOTarget(
            params['backward_type'], 
            params['conf_threshold'], 
            params['ratio'], 
            self.target_class
        )
        self.target_layers = [self.model.model.model[l] for l in params['layer']]
        self.method = eval(params['method'])(self.model.model, self.target_layers)
        self.method.activations_and_grads = ActivationsAndGradients(self.model.model, self.target_layers, None)
        self.conf_threshold = params['conf_threshold']
        self.ratio = params['ratio']
        self.show_box = params['show_box']
        self.renormalize = params['renormalize']
        self.colors = np.random.uniform(0, 255, size=(len(self.model.names), 3)).astype(int)

    def process(self, img_path, save_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image at path {img_path} could not be loaded.")
        img, _, _ = letterbox(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
        tensor.requires_grad = True

        try:
            grayscale_cam = self.method(tensor, [self.target])
            plt.imshow(grayscale_cam.squeeze(), cmap='jet')
            plt.colorbar()
            plt.show()
            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

            pred = self.model(tensor)[0]
            pred = self.post_process(pred)
            if self.renormalize:
                cam_image = self.renormalize_cam_in_bounding_boxes(
                    pred[:, :4].cpu().detach().numpy().astype(np.int32), 
                    img, 
                    grayscale_cam
                )
            if self.show_box:
                for data in pred:
                    data = data.cpu().detach().numpy()
                    cam_image = self.draw_detections(
                        data[:4], 
                        self.colors[int(data[4:].argmax())], 
                        f'{self.model.names[int(data[4:].argmax())]} {float(data[4:].max()):.2f}', 
                        cam_image
                    )

            cam_image = Image.fromarray(cam_image)
            #cam_image.save(save_path)
        except AttributeError as e:
            print("Attribute Error found ....")
            return
    def __call__(self, img_path, save_path):
        if os.path.isdir(img_path):
            for img_path_ in os.listdir(img_path):
                self.process(f'{img_path}/{img_path_}', f'{save_path}/{img_path_}')
        else:
            self.process(img_path, f'{save_path}/result.png')



def get_params():
    params = {
        'weight': 'EXPLib/dl_component/CNN/Object_Detectors/yolov8/yolov8n.pt',
        'device': 'cuda:0',
        'method': 'HiResCAM',  # Choose CAM method
        'layer': [10, 12, 14, 16, 18],
        'backward_type': 'box',  # 'class', 'box', 'all'
        'conf_threshold': 0.2,
        'ratio': 0.02,
        'show_box': False,
        'renormalize': True,
        'target_class': None  # This can be None or class ID. Set to None to find the possible classes before set to a number
    }
    return params
