from __future__ import print_function
import cv2
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time
from data import cfg_mnet, cfg_slim, cfg_rfb
from layers.functions.prior_box import PriorBox
from utils.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from models.net_slim import Slim
from models.net_rfb import RFB
from utils.box_utils import decode, decode_landm

import torch.hub
from recognition.check import eye_check, mouth_check

parser = argparse.ArgumentParser(description='Test')
#解析参数，argparse 模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息
parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or slim or RFB')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--long_side', default=640, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()



def check_keys(model, pretrained_state_dict):            #   以下全部都是pth转pt
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):                               #pytorch加载模型
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):                 #pytorch加载模型
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)       #全局不求导

    cfg = None
    net = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
        net = RetinaFace(cfg = cfg, phase = 'test')
    elif args.network == "slim":
        cfg = cfg_slim
        net = Slim(cfg = cfg, phase = 'test')
    elif args.network == "RFB":
        cfg = cfg_rfb
        net = RFB(cfg = cfg, phase = 'test')
    else:
        print("Don't support network!")
        exit(0)

    net = load_model(net, args.trained_model, args.cpu)
    net.eval()  #测试模型
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True   #增加运行效率
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)
    
    #ind = 0
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        #ind = ind + 1
        flag, img_raw = cap.read()
        # testing begin
        for i in range(1):
            #image_path = "./img/sample.jpg"

            #img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = np.float32(img_raw)

            # testing scale
            target_size = args.long_side
            max_size = args.long_side
            im_shape = img.shape
            im_size_min = np.min(im_shape[0:2])    #取得长宽的大小值
            im_size_max = np.max(im_shape[0:2])
            resize = float(target_size) / float(im_size_min)   #防止超过最大最小值
            # prevent bigger axis from being more than max_size:
            if np.round(resize * im_size_max) > max_size:
                resize = float(max_size) / float(im_size_max)
            if args.origin_size:
                resize = 1

            if resize != 1:
                img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
                #对img进行缩放，缩放比例为resize
            im_height, im_width, _ = img.shape


            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)

            tic = time.time()
            loc, conf, landms = net(img)  # forward pass
            #print('net forward time: {:.4f}'.format(time.time() - tic))

            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])
            scale1 = scale1.to(device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > args.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, args.nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:args.keep_top_k, :]
            landms = landms[:args.keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)

            # show image
            if args.save_image:
                for b in dets:
                    if b[4] < args.vis_thres:
                        continue
                    text = "{:.4f}".format(b[4])
                    b = list(map(int, b))
                    #cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                    cx = b[0]
                    cy = b[1] + 12
                    #cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255)
                    #print(b)
                    eye_dis_half = int((( (abs(b[5]-b[7]) **2) + (abs(b[6]-b[8]) **2) ) ** 0.5)/2)
                    eye_dis_half_half = int((( (abs(b[5]-b[7]) **2) + (abs(b[6]-b[8]) **2) ) ** 0.5)/3)
                    eye_dis_3half = int((( (abs(b[5]-b[7]) **2) + (abs(b[6]-b[8]) **2) ) ** 0.5)/4)
                    
                    left_eye_x = b[5]-eye_dis_half_half
                    left_eye_y = b[6]-eye_dis_3half
                    left_eye_w = eye_dis_half_half + eye_dis_half_half
                    left_eye_h = eye_dis_3half * 2
                    left_eye = img_raw[left_eye_y:left_eye_y + left_eye_h + 8, left_eye_x:left_eye_x + left_eye_w]
                    print('left_shape',left_eye.shape)
                    right_eye_x = b[7]-eye_dis_half_half
                    right_eye_y = b[8]-eye_dis_3half
                    right_eye_w = eye_dis_half_half + eye_dis_half_half
                    right_eye_h = eye_dis_3half * 2
                    right_eye = img_raw[right_eye_y:right_eye_y + right_eye_h + 8, right_eye_x:right_eye_x + right_eye_w]
                    print('right_shape',right_eye.shape)
                    
                    distance_half = int((( (abs(b[11]-b[13]) **2) + (abs(b[11]-b[13]) **2) ) ** 0.5)/2)
                    distance_half_half = int((( (abs(b[11]-b[13]) **2) + (abs(b[11]-b[13]) **2) ) ** 0.5)/4)
                    mouthcentre_x = int((b[11]+b[13])/2)
                    mouthcentre_y = int((b[12]+b[14])/2)+10
                    
                    mouth_x = mouthcentre_x - distance_half
                    mouth_y = mouthcentre_y-distance_half
                    mouth_w = distance_half * 2
                    mouth_h = distance_half * 2
                    mouth = img_raw[mouth_y:mouth_y + mouth_h , mouth_x:mouth_x + mouth_w]
                    print('mouth_shape',mouth.shape)
                    '''
                    left_name = "left_eye_close.jpg"
                    right_name = "right_eye_close.jpg"
                    cv2.imwrite(left_name, left_eye)
                    cv2.imwrite(right_name, right_eye)
                    
                    left_eye = cv2.imread(left_name)
                    right_eye = cv2.imread(right_name)'''
                    torch.set_grad_enabled(True)
                    left_result = eye_check(left_eye)
                    right_result = eye_check(right_eye)
                    mouth_result = mouth_check(mouth)
                    torch.set_grad_enabled(False)
                    
                   
                    
                    

                    #print(distance_half*2)
                    
                    #cv2.rectangle(img_raw, (b[5]-eye_dis_half_half, b[6]-eye_dis_3half), (b[5]+eye_dis_half_half, b[6]+eye_dis_3half), (0, 0, 255), 2)
                    #cv2.rectangle(img_raw, (b[7]-eye_dis_half_half, b[8]-eye_dis_3half), (b[7]+eye_dis_half_half, b[8]+eye_dis_3half), (0, 255, 255), 2)
                    #cv2.rectangle(img_raw, (mouthcentre_x-distance_half, mouthcentre_y-distance_half), (mouthcentre_x+distance_half, mouthcentre_y+distance_half), (255, 0, 0), 2)
                    # landms
                    cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                    cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                    cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                    cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                    cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
                # save image
                cv2.imshow("cap", img_raw)
                
                '''
                if ind == 1:
                    mouth_name = "mouth_close.jpg"
                    cv2.imwrite(mouth_name, mouth)
                    
                    left_name = "left_eye_close.jpg"
                    right_name = "right_eye_close.jpg"
                    cv2.imwrite(left_name, left_eye)
                    cv2.imwrite(right_name, right_eye) '''
                #cv2.imwrite(name, img_raw)
            # if the `q` key was pressed, break from the loop

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 释放摄像头
    cap.release()
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
