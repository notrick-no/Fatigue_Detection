import os.path as osp

import cv2
import matplotlib.cm as cm
import numpy as np
import torch.hub
import os
import model
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchsummary import summary
from recognition.grad_cam import BackPropagation, GradCAM,GuidedBackPropagation

eye_shape = (48,48)
mouth_shape = (64,64)
classes = [
    'Close',
    'Open',
]



def get_gradient_image(gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    return np.uint8(gradient)


def get_gradcam_image(gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    return np.uint8(gcam)
    
def guided_backprop_image(img, raw_img, shape, net):
    img = torch.stack([img])
    bp = BackPropagation(model=net)
    probs, ids = bp.forward(img)
    
    #img = img.requires_grad_()
    gcam = GradCAM(model=net)
    _ = gcam.forward(img)

    gbp = GuidedBackPropagation(model=net)
    _ = gbp.forward(img)
    
    #img = img.requires_grad_()

    # Guided Backpropagation
    actual_status = ids[:, 0]
    gbp.backward(ids=actual_status.reshape(1, 1))
    gradients = gbp.generate()

    # Grad-CAM
    gcam.backward(ids=actual_status.reshape(1, 1))
    regions = gcam.generate(target_layer='last_conv')

    # Get Images
    prob = probs.data[:, 0]
    if actual_status == 0:
        prob = probs.data[:,1]

    prob_image = np.zeros((shape[0], 60, 3), np.uint8)
    cv2.putText(prob_image, '%.1f%%' % (prob * 100), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)

    guided_bpg_image = get_gradient_image(gradients[0])
    guided_bpg_image = cv2.merge((guided_bpg_image, guided_bpg_image, guided_bpg_image))

    grad_cam_image = get_gradcam_image(gcam=regions[0, 0], raw_image=raw_img)
    guided_gradcam_image = get_gradient_image(torch.mul(regions, gradients)[0])
    guided_gradcam_image = cv2.merge((guided_gradcam_image, guided_gradcam_image, guided_gradcam_image))
    #print(classes[actual_status.data]) #, probs.data[:,0] * 100

    return actual_status.data
    
def eye_check(raw_eye):
    raw_eye = cv2.resize(raw_eye, eye_shape, interpolation=cv2.INTER_LINEAR)
    model_name='model_79_98_0.0414.t7'
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    
    eye = transform_test(Image.fromarray(raw_eye).convert('L'))

    
    net = model.Model(num_classes=len(classes))
    checkpoint = torch.load(os.path.join('D:/吕良伟毕业论文+PPT+算法程序/吕良伟毕业论文+PPT+算法程序/Fatigue Detector/Fatigue_Detector/trained/', model_name), map_location=torch.device('cpu'))
    
    net.load_state_dict(checkpoint['net'])
    net.eval()

    #summary(net, (1, eye_shape[0], eye_shape[1]), device='cpu')
    
    #for index, image in enumerate(images):
    result = guided_backprop_image(eye, raw_eye, eye_shape, net)
    #print(result)
        
    return result

def mouth_check(raw_mouth):
    raw_mouth = cv2.resize(raw_mouth, mouth_shape, interpolation=cv2.INTER_LINEAR)
    model_name='model_188_99_0.0125.t7'
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    
    mouth = transform_test(Image.fromarray(raw_mouth).convert('L'))

    
    net = model.Model(num_classes=len(classes))
    checkpoint = torch.load(os.path.join('D:/吕良伟毕业论文+PPT+算法程序/吕良伟毕业论文+PPT+算法程序/Fatigue Detector/Fatigue_Detector/trained/', model_name), map_location=torch.device('cpu'))
    
    net.load_state_dict(checkpoint['net'])
    net.eval()

    #summary(net, (1, mouth_shape[0], mouth_shape[1]), device='cpu')
    
    #for index, image in enumerate(images):
    result = guided_backprop_image(mouth, raw_mouth, mouth_shape, net)
    #print(result)
        
    return result