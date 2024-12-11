# -*- coding: utf-8 -*- 

import numpy as np              # 数据处理的库numpy
import cv2                      # 图像处理的库OpenCv
import wx                       # 构造显示界面的GUI
import wx.xrc
import wx.adv
# import the necessary packages
#from imutils.video import FileVideoStream
#from imutils.video import VideoStream
#from imutils import face_utils
import argparse
#import imutils
import datetime,time
import math
import os
from pytorch_infer import inference

import torch
import torch.backends.cudnn as cudnn
from data import cfg_mnet, cfg_slim, cfg_rfb
from layers.functions.prior_box import PriorBox
from utils.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from models.net_slim import Slim
from models.net_rfb import RFB
from utils.box_utils import decode, decode_landm

import torch.hub
#import model
from recognition.check import eye_check, mouth_check

COVER = './images/shangda8.png'

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='D:\Fatigue Detector\Fatigue_Detector\weights\mobilenet0.25_Final.pth',
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
#这个 check_keys 函数用于比较模型的 state_dict 和预训练模型的 state_dict 中的键（key），并检查它们是否匹配。它会输出缺失的键、未使用的键和
# 已经使用的键的数量，帮助您判断模型和预训练权重之间是否有不匹配的地方。
def check_keys(model, pretrained_state_dict):
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

#remove_prefix 函数的作用是从模型的 state_dict 中移除参数名的前缀（通常是 "module."），这在加载分布式训练或多GPU训练保存的模型权重时非常有用。
#使用 PyTorch 的 torch.nn.DataParallel 进行多GPU训练时，模型的 state_dict 中保存的参数（如权重、偏置等）会自动加上前缀 'module.'
#为了使这些参数在单GPU环境或没有 DataParallel 的环境中能够正确加载，通常需要去掉这些多余的 'module.' 前缀。
def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

#这个 load_model 函数用于加载预训练模型的权重，并根据是否将模型加载到 CPU 或 GPU 来调整模型的加载方式。
#返回加载了预训练权重的模型。
def load_model(model, pretrained_path, load_to_cpu):
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


class Fatigue_detecting(wx.Frame):
    #使用 wxPython 创建图形界面应用程序的代码，主要用于视频流疲劳检测，包括打哈欠、闭眼、离位等功能
    def __init__( self, parent, title ):
        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = title, pos = wx.DefaultPosition, size = wx.Size( 873,535 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
                
        self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )
        self.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_MENU ) )

        bSizer1 = wx.BoxSizer( wx.VERTICAL )
        bSizer2 = wx.BoxSizer( wx.HORIZONTAL )
        bSizer3 = wx.BoxSizer( wx.VERTICAL )

        self.m_animCtrl1 = wx.adv.AnimationCtrl( self, wx.ID_ANY, wx.adv.NullAnimation, wx.DefaultPosition, wx.DefaultSize, wx.adv.AC_DEFAULT_STYLE ) 
        bSizer3.Add( self.m_animCtrl1, 1, wx.ALL|wx.EXPAND, 5 )        
        bSizer2.Add( bSizer3, 9, wx.EXPAND, 5 )
        bSizer4 = wx.BoxSizer( wx.VERTICAL )
        sbSizer1 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"参数设置" ), wx.VERTICAL )
        sbSizer2 = wx.StaticBoxSizer( wx.StaticBox( sbSizer1.GetStaticBox(), wx.ID_ANY, u"视频源" ), wx.VERTICAL )
        gSizer1 = wx.GridSizer( 0, 2, 0, 8 )
        m_choice1Choices = [ u"摄像头ID_0", u"摄像头ID_1", u"摄像头ID_2" ]
        self.m_choice1 = wx.Choice( sbSizer2.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.Size( 90,25 ), m_choice1Choices, 0 )
        self.m_choice1.SetSelection( 0 )
        gSizer1.Add( self.m_choice1, 0, wx.ALL, 5 )
        self.camera_button1 = wx.Button( sbSizer2.GetStaticBox(), wx.ID_ANY, u"开始检测", wx.DefaultPosition, wx.Size( 90,25 ), 0 )
        gSizer1.Add( self.camera_button1, 0, wx.ALL, 5 )
        self.vedio_button2 = wx.Button( sbSizer2.GetStaticBox(), wx.ID_ANY, u"打开视频文件", wx.DefaultPosition, wx.Size( 90,25 ), 0 )
        gSizer1.Add( self.vedio_button2, 0, wx.ALL, 5 )

        self.off_button3 = wx.Button( sbSizer2.GetStaticBox(), wx.ID_ANY, u"暂停", wx.DefaultPosition, wx.Size( 90,25 ), 0 )
        gSizer1.Add( self.off_button3, 0, wx.ALL, 5 )
        sbSizer2.Add( gSizer1, 1, wx.EXPAND, 5 )
        sbSizer1.Add( sbSizer2, 2, wx.EXPAND, 5 )
        sbSizer3 = wx.StaticBoxSizer( wx.StaticBox( sbSizer1.GetStaticBox(), wx.ID_ANY, u"疲劳检测" ), wx.VERTICAL )
        bSizer5 = wx.BoxSizer( wx.HORIZONTAL )
        self.yawn_checkBox1 = wx.CheckBox( sbSizer3.GetStaticBox(), wx.ID_ANY, u"打哈欠检测", wx.Point( -1,-1 ), wx.Size( -1,15 ), 0 )
        self.yawn_checkBox1.SetValue(True) 
        bSizer5.Add( self.yawn_checkBox1, 0, wx.ALL, 5 )
        self.blink_checkBox2 = wx.CheckBox( sbSizer3.GetStaticBox(), wx.ID_ANY, u"闭眼检测", wx.Point( -1,-1 ), wx.Size( -1,15 ), 0 )
        self.blink_checkBox2.SetValue(True) 
        bSizer5.Add( self.blink_checkBox2, 0, wx.ALL, 5 )
        sbSizer3.Add( bSizer5, 1, wx.EXPAND, 5 )
        
        bSizer6 = wx.BoxSizer( wx.HORIZONTAL )
        self.face_checkBox4 = wx.CheckBox( sbSizer3.GetStaticBox(), wx.ID_ANY, u"离位检测", wx.DefaultPosition, wx.Size( -1,15 ), 0 )
        self.face_checkBox4.SetValue(True) 
        bSizer6.Add( self.face_checkBox4, 0, wx.ALL, 5 )        
        
        

        self.reset_button4 = wx.Button( sbSizer3.GetStaticBox(), wx.ID_ANY, u"重置", wx.DefaultPosition, wx.Size( -1,22 ), 0 )
        bSizer6.Add( self.reset_button4, 0, wx.ALL, 5 )  
        sbSizer3.Add( bSizer6, 1, wx.EXPAND, 5 )
        sbSizer1.Add( sbSizer3, 2, 0, 5 )

        sbSizer6 = wx.StaticBoxSizer( wx.StaticBox( sbSizer1.GetStaticBox(), wx.ID_ANY, u"状态输出" ), wx.VERTICAL )
        self.m_textCtrl3 = wx.TextCtrl( sbSizer6.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_MULTILINE|wx.TE_READONLY )
        sbSizer6.Add( self.m_textCtrl3, 1, wx.ALL|wx.EXPAND, 5 )
        sbSizer1.Add( sbSizer6, 5, wx.EXPAND, 5 )
        bSizer4.Add( sbSizer1, 1, wx.EXPAND, 5 )
        bSizer2.Add( bSizer4, 3, wx.EXPAND, 5 )
        bSizer1.Add( bSizer2, 1, wx.EXPAND, 5 )

        self.SetSizer( bSizer1 )  
        self.Layout()
        self.Centre( wx.BOTH )
        
        # Connect Events
        self.m_choice1.Bind( wx.EVT_CHOICE, self.cameraid_choice )#绑定事件
        self.camera_button1.Bind( wx.EVT_BUTTON, self.camera_on )#开
        self.vedio_button2.Bind( wx.EVT_BUTTON, self.vedio_on )
        self.off_button3.Bind( wx.EVT_BUTTON, self.off )#关
        self.reset_button4.Bind( wx.EVT_BUTTON, self.reset)

        
        # 封面图片
        self.image_cover = wx.Image(COVER, wx.BITMAP_TYPE_ANY)
        # 显示图片在m_animCtrl1上
        self.bmp = wx.StaticBitmap(self.m_animCtrl1, -1, wx.Bitmap(self.image_cover))

        # 设置窗口标题的图标
        self.icon = wx.Icon('./images/shangda2.ico', wx.BITMAP_TYPE_ICO)
        self.SetIcon(self.icon)
        # 系统事件
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        
        print("wxpython界面初始化加载完成！")
        
        """参数"""
        # 默认为摄像头0
        self.VIDEO_STREAM = 0
        self.CAMERA_STYLE = False # False未打开摄像头，True摄像头已打
        
        self.EYE_AR_CONSEC_FRAMES = 3
        self.MOUTH_AR_CONSEC_FRAMES = 5
        self.LONG_EYE_AR_CONSEC_FRAMES = 15
        
        self.long_EyeClose = 0
        self.EyeClose = 0
        self.blink_count = 0
        self.MouthOpen = 0
        self.eyetimes = 2
        self.yawm_count = 0
        self.alltimes = 23
        self.mask = False
        self.k = 0


   #在 Python 中，__del__ 是一个特殊方法，通常用于对象销毁时的清理工作。它在对象被销毁之前自动调用，类似于其他语言中的析构函数（destructor）
    def __del__( self ):
        pass

    def _learning_face(self,event):
        #关闭自动梯度计算
        torch.set_grad_enabled(False)

        cfg = None
        net = None
        #这段代码通过检查命令行传入的 args.network 参数来选择适当的神经网络模型。
        #每个网络结构对应一个配置文件（如 cfg_mnet、cfg_slim、cfg_rfb），并且根据选定的网络类型初始化相应的模型（如 RetinaFace、Slim、RFB）。
        #如果传入的网络类型不在预期范围内，程序会输出错误并退出。
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
        #载预训练模型权重（通过 load_model 函数）。
        #设置模型为评估模式（eval()），以关闭 dropout 和 batch normalization 等训练时特有的行为。
        #打印模型结构以及模型加载完成的提示信息。
        #启用 cuDNN 优化，提升 GPU 上卷积操作的性能（适用于固定输入大小的模型）。
        #根据 args.cpu 参数选择设备（CPU 或 GPU）。
        #将模型移动到选定的计算设备上。
        net = load_model(net, args.trained_model, args.cpu)
        net.eval()
        print('Finished loading model!')
        print(net)
        cudnn.benchmark = True
        device = torch.device("cpu" if args.cpu else "cuda")
        net = net.to(device)

        
        #建cv2摄像头对象，这里使用电脑自带摄像头，如果接了外部摄像头，则自动切换到外部摄像头
        self.cap = cv2.VideoCapture(self.VIDEO_STREAM)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 24)
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 32)



        if self.cap.isOpened()==True:#  返回true/false 检查初始化是否成功
            self.CAMERA_STYLE = True
            self.m_textCtrl3.AppendText(u"打开摄像头成功!!!\n")
        else:
            self.m_textCtrl3.AppendText(u"摄像头打开失败!!!\n")
            #显示封面图
            self.bmp.SetBitmap(wx.Bitmap(self.image_cover))
        # 成功打开视频，循环读取视频流
        start_time = time.time()
        while(self.cap.isOpened()):
            #ind = ind + 1
            flag, img_raw = self.cap.read()
            #img_raw.set(cv2.CAP_PROP_FRAME_HEIGHT, 24)
            #img_raw.set(cv2.CAP_PROP_FRAME_WIDTH, 32)
            # testing begin
            for i in range(1):


                # img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
                # read_frame_stamp = time.time()
                # if (flag):
                #     inference(img_raw,
                #               conf_thresh=0.8,
                #               iou_thresh=0.5,
                #               target_shape=(360, 360),
                #               draw_result=True,
                #               show_result=False)
                    # cv2.imshow('image', img_raw[:, :, ::-1])

                img = np.float32(img_raw)

                # 测试缩放比
                target_size = args.long_side
                max_size = args.long_side
                im_shape = img.shape
                im_size_min = np.min(im_shape[0:2])
                im_size_max = np.max(im_shape[0:2])
                resize = float(target_size) / float(im_size_min)
                # 防止更大的轴超过最大尺寸:
                if np.round(resize * im_size_max) > max_size:
                    resize = float(max_size) / float(im_size_max)
                if args.origin_size:
                    resize = 1

                if resize != 1:
                    img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
                im_height, im_width, _ = img.shape

                #图像归一化和转换为张量
                scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                img -= (104, 117, 123)
                img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img).unsqueeze(0)
                img = img.to(device)
                scale = scale.to(device)

                #向前传播
                tic = time.time()
                loc, conf, landms = net(img)
                #print('net forward time: {:.4f}'.format(time.time() - tic))

                #生成锚框和解码
                priorbox = PriorBox(cfg, image_size=(im_height, im_width))
                priors = priorbox.forward()
                priors = priors.to(device)
                prior_data = priors.data
                boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
                boxes = boxes * scale / resize
                boxes = boxes.cpu().numpy()
                scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
                #解码关键点
                landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
                scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                       img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                       img.shape[3], img.shape[2]])
                scale1 = scale1.to(device)
                landms = landms * scale1 / resize
                landms = landms.cpu().numpy()

                # ignore low scores筛选低置信度的框
                inds = np.where(scores > args.confidence_threshold)[0]
                boxes = boxes[inds]
                landms = landms[inds]
                scores = scores[inds]

                # keep top-K before NMS 非极大值抑制
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

                # keep top-K faster NMS 保持top-K检测
                dets = dets[:args.keep_top_k, :]
                landms = landms[:args.keep_top_k, :]

                dets = np.concatenate((dets, landms), axis=1)

                #height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                #print(height)
                #width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                #print(width)
                # show image

     #上面的代码是处理输入图像，通过模型推理生成边界框和人脸关键点，然后通过一系列后处理步骤（如缩放、解码、NMS）来优化检测结果，最后输出前 K
     # 个检测结果。这是典型的人脸检测工作流程，涉及图像的预处理、模型推理、解码、NMS 等技术。
                if args.save_image:
                    Face = False
                    Mask = False
                    for b in dets:   #遍历面部框
                        if b[4] < args.vis_thres:   # 跳过低置信度框
                            continue
                        Face = True   # 如果检测到面部并且通过 inference 函数检测到戴口罩的标志（返回值为 1），则设置 Mask = True

                        face_mask_out = inference(img_raw,
                                  conf_thresh=0.8,
                                  iou_thresh=0.5,
                                  target_shape=(360, 360),
                                  draw_result=True,
                                  show_result=False)
                        #print(face_mask_out[0][0])
                        if face_mask_out:
                            if face_mask_out[0][0]== 1:
                                    Mask = True
                            # landms
                            #cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                            #cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                            #cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                            #cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                            #cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
                            if Mask:
                                text = "{:.4f}".format(b[4]) #显示置信度
                                b = list(map(int, b))
                                #cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                                cx = b[0]
                                cy = b[1]
                                #cv2.putText(img_raw, text, (cx, cy),
                                            #cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                #print(b)
                                eye_dis_half = int((( (abs(b[5]-b[7]) **2) + (abs(b[6]-b[8]) **2) ) ** 0.5)/2)
                                eye_dis_half_half = int((( (abs(b[5]-b[7]) **2) + (abs(b[6]-b[8]) **2) ) ** 0.5)/3)
                                eye_dis_3half = int((( (abs(b[5]-b[7]) **2) + (abs(b[6]-b[8]) **2) ) ** 0.5)/4)

                                left_eye_x = b[5]-eye_dis_half
                                left_eye_y = b[6]-eye_dis_half
                                left_eye_w = eye_dis_half * 2
                                left_eye_h = eye_dis_half * 2
                                left_eye = img_raw[left_eye_y:left_eye_y + left_eye_h, left_eye_x:left_eye_x + left_eye_w]

                                right_eye_x = b[7]-eye_dis_half
                                right_eye_y = b[8]-eye_dis_half
                                right_eye_w = eye_dis_half * 2
                                right_eye_h = eye_dis_half * 2
                                right_eye = img_raw[right_eye_y:right_eye_y + right_eye_h, right_eye_x:right_eye_x + right_eye_w]


                                distance_half = int((( (abs(b[11]-b[13]) **2) + (abs(b[11]-b[13]) **2) ) ** 0.5)/2)
                                distance_half_half = int((( (abs(b[11]-b[13]) **2) + (abs(b[11]-b[13]) **2) ) ** 0.5)/4)
                                mouthcentre_x = int((b[11]+b[13])/2)
                                mouthcentre_y = int((b[12]+b[14])/2)+10

                                mouth_x = mouthcentre_x - distance_half
                                mouth_y = mouthcentre_y-distance_half - 8
                                mouth_w = distance_half * 2
                                mouth_h = distance_half * 2 + 8
                                mouth = img_raw[mouth_y:mouth_y + mouth_h , mouth_x:mouth_x + mouth_w]

                                #以上代码计算出面部的眼睛和嘴巴的坐标

                                torch.set_grad_enabled(True)
                                #眼睛检测
                                if self.blink_checkBox2.GetValue()== True:
                                    left_result = eye_check(left_eye)
                                    right_result = eye_check(right_eye)

                                    cv2.rectangle(img_raw, (b[5]-eye_dis_half, b[6]-eye_dis_3half), (b[5]+eye_dis_half, b[6]+eye_dis_3half), (0, 0, 255), 2)
                                    cv2.rectangle(img_raw, (b[7]-eye_dis_half, b[8]-eye_dis_3half), (b[7]+eye_dis_half, b[8]+eye_dis_3half), (0, 255, 255), 2)

                                    '''
                                    if left_result == 1:
                                        cv2.putText(img_raw, "Open", (left_eye_x, left_eye_y-12),
                                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                    elif left_result == 0:
                                        cv2.putText(img_raw, "Close", (left_eye_x, left_eye_y-12),
                                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                               
                                    if right_result == 1:
                                        cv2.putText(img_raw, "Open", (right_eye_x, right_eye_y-12),
                                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                    elif right_result == 0:
                                        cv2.putText(img_raw, "Close", (right_eye_x, right_eye_y-12),
                                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255)) '''

                                    if left_result != right_result:
                                        if left_result == 1 or right_result == 1:
                                            cv2.putText(img_raw, "Open", (left_eye_x, left_eye_y-12),
                                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                            cv2.putText(img_raw, "Open", (right_eye_x, right_eye_y-12),
                                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                        else:
                                            cv2.putText(img_raw, "Close", (left_eye_x, left_eye_y-12),
                                               cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                            cv2.putText(img_raw, "Close", (right_eye_x, right_eye_y-12),
                                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                                    elif left_result == 0:
                                        cv2.putText(img_raw, "Close", (left_eye_x, left_eye_y-12),
                                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                        cv2.putText(img_raw, "Close", (right_eye_x, right_eye_y-12),
                                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                        self.EyeClose += 1
                                        self.eyetimes += 1
                                        self.alltimes += 1
                                        #print(self.EyeClose)
                                        #！长时间闭眼警告
                                        if self.EyeClose >= self.LONG_EYE_AR_CONSEC_FRAMES:
                                            if self.long_EyeClose == 0:
                                                self.long_EyeClose+= 1
                                                self.m_textCtrl3.AppendText(time.strftime('%Y-%m-%d %H:%M ', time.localtime())+u"长时间闭眼\n")
                                        #cv2.putText(img_raw, "Eye Close", (250, 100),
                                        #        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 191, 255),2)
                                    elif left_result == 1:
                                        cv2.putText(img_raw, "Open", (left_eye_x, left_eye_y-12),
                                           cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                        cv2.putText(img_raw, "Open", (right_eye_x, right_eye_y-12),
                                           cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                        self.alltimes += 1
                                        #cv2.putText(img_raw, "Eye Open", (250, 100),
                                        #        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 191, 255),2)
                                else:
                                    self.EyeClose = 0
                                    self.blink_count = 0
                                    self.long_EyeClose = 0

                                #cv2.putText(img_raw, "Blinks: {}".format(self.blink_count), (450, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)


                                if self.yawn_checkBox1.GetValue()== True:
                                    mouth_result = mouth_check(mouth)
                                    cv2.rectangle(img_raw, (mouthcentre_x-distance_half, mouthcentre_y-distance_half), (mouthcentre_x+distance_half, mouthcentre_y+distance_half), (0, 255, 127), 2)

                                    if mouth_result == 1:
                                        self.MouthOpen += 1
                                        cv2.putText(img_raw, "Open", (mouth_x, mouth_y-12),
                                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                        #cv2.putText(img_raw, "Mouth Open", (200, 100),
                                        #        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 191, 255),2)
                                    elif mouth_result == 0:
                                        if self.MouthOpen >= self.MOUTH_AR_CONSEC_FRAMES:
                                            self.yawm_count += 1
                                            self.m_textCtrl3.AppendText(time.strftime('%Y-%m-%d %H:%M ', time.localtime())+u"打哈欠\n")
                                        self.MouthOpen = 0
                                        cv2.putText(img_raw, "Close", (mouth_x, mouth_y-12),
                                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                        #cv2.putText(img_raw, "Mouth Close", (200, 100),
                                        #        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 191, 255),2)
                                else:
                                    self.MouthOpen = 0
                                    self.yawm_count = 0

                                #cv2.putText(img_raw, "Yawning: {}".format(self.yawm_count), (450, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                                torch.set_grad_enabled(False)

                                #if self.blink_count >= 5 or self.yawm_count >= 5 or self.long_EyeClose == 1:
                                 #       cv2.putText(img_raw, "SLEEP!!!", (cx-70, cy-10),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                                PERCLOS = self.eyetimes/self.alltimes
                                cv2.putText(img_raw, "PERCLOS: {}".format("%.2f" % PERCLOS), (20, 456),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                                if PERCLOS >= 0.15:
                                    cv2.putText(img_raw, "PERCLOS: {}".format("%.2f" % PERCLOS), (20, 456),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                                    self.k = 1
                                if self.long_EyeClose==1:
                                    self.k = 1
                                if PERCLOS >= 0.13 and self.yawm_count!=0:
                                    cv2.putText(img_raw, "PERCLOS: {}".format("%.2f" % PERCLOS), (20, 456),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                                    self.k = 1

                                if self.k == 1:
                                    cv2.putText(img_raw, "SLEEP!!!", (cx - 80, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (0, 0, 255),3)
                                mid_time = time.time()
                                if (time.localtime(start_time).tm_sec - time.localtime(mid_time).tm_sec) % 30 == 0:
                                    start_time = time.time()
                                    self.yawm_count = 0
                                    self.eyetimes = 2
                                    self.alltimes = 23

                            else:
                                text = "{:.4f}".format(b[4])
                                b = list(map(int, b))
                                # cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                                cx = b[0]
                                cy = b[1]
                                # cv2.putText(img_raw, text, (cx, cy),
                                # cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                # print(b)
                                eye_dis_half = int((((abs(b[5] - b[7]) ** 2) + (abs(b[6] - b[8]) ** 2)) ** 0.5) / 2)
                                eye_dis_half_half = int((((abs(b[5] - b[7]) ** 2) + (abs(b[6] - b[8]) ** 2)) ** 0.5) / 3)
                                eye_dis_3half = int((((abs(b[5] - b[7]) ** 2) + (abs(b[6] - b[8]) ** 2)) ** 0.5) / 4)

                                left_eye_x = b[5] - eye_dis_half
                                left_eye_y = b[6] - eye_dis_half
                                left_eye_w = eye_dis_half * 2
                                left_eye_h = eye_dis_half * 2
                                left_eye = img_raw[left_eye_y:left_eye_y + left_eye_h, left_eye_x:left_eye_x + left_eye_w]

                                right_eye_x = b[7] - eye_dis_half
                                right_eye_y = b[8] - eye_dis_half
                                right_eye_w = eye_dis_half * 2
                                right_eye_h = eye_dis_half * 2
                                right_eye = img_raw[right_eye_y:right_eye_y + right_eye_h,
                                            right_eye_x:right_eye_x + right_eye_w]

                                # distance_half = int((((abs(b[11] - b[13]) ** 2) + (abs(b[11] - b[13]) ** 2)) ** 0.5) / 2)
                                # distance_half_half = int(
                                #     (((abs(b[11] - b[13]) ** 2) + (abs(b[11] - b[13]) ** 2)) ** 0.5) / 4)
                                # mouthcentre_x = int((b[11] + b[13]) / 2)
                                # mouthcentre_y = int((b[12] + b[14]) / 2) + 10
                                #
                                # mouth_x = mouthcentre_x - distance_half
                                # mouth_y = mouthcentre_y - distance_half - 8
                                # mouth_w = distance_half * 2
                                # mouth_h = distance_half * 2 + 8
                                # mouth = img_raw[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w]

                                torch.set_grad_enabled(True)
                                if self.blink_checkBox2.GetValue() == True:
                                    left_result = eye_check(left_eye)
                                    right_result = eye_check(right_eye)

                                    cv2.rectangle(img_raw, (b[5] - eye_dis_half, b[6] - eye_dis_3half),
                                                  (b[5] + eye_dis_half, b[6] + eye_dis_3half), (0, 0, 255), 2)
                                    cv2.rectangle(img_raw, (b[7] - eye_dis_half, b[8] - eye_dis_3half),
                                                  (b[7] + eye_dis_half, b[8] + eye_dis_3half), (0, 255, 255), 2)

                                    '''
                                    if left_result == 1:
                                        cv2.putText(img_raw, "Open", (left_eye_x, left_eye_y-12),
                                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                    elif left_result == 0:
                                        cv2.putText(img_raw, "Close", (left_eye_x, left_eye_y-12),
                                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    
                                    if right_result == 1:
                                        cv2.putText(img_raw, "Open", (right_eye_x, right_eye_y-12),
                                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                    elif right_result == 0:
                                        cv2.putText(img_raw, "Close", (right_eye_x, right_eye_y-12),
                                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255)) '''

                                    if left_result != right_result:
                                        if left_result == 1 or right_result == 1:
                                            cv2.putText(img_raw, "Open", (left_eye_x, left_eye_y - 12),
                                                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                            cv2.putText(img_raw, "Open", (right_eye_x, right_eye_y - 12),
                                                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                        else:
                                            cv2.putText(img_raw, "Close", (left_eye_x, left_eye_y - 12),
                                                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                            cv2.putText(img_raw, "Close", (right_eye_x, right_eye_y - 12),
                                                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                                    elif left_result == 0:
                                        cv2.putText(img_raw, "Close", (left_eye_x, left_eye_y - 12),
                                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                        cv2.putText(img_raw, "Close", (right_eye_x, right_eye_y - 12),
                                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                        self.EyeClose += 1
                                        self.eyetimes += 1
                                        self.alltimes += 1
                                        if self.EyeClose >= self.LONG_EYE_AR_CONSEC_FRAMES:
                                            if self.long_EyeClose == 0:
                                                self.long_EyeClose = 1
                                                self.m_textCtrl3.AppendText(
                                                    time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"长时间闭眼\n")
                                        # cv2.putText(img_raw, "Eye Close", (250, 100),
                                        #        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 191, 255),2)
                                    elif left_result == 1:
                                        cv2.putText(img_raw, "Open", (left_eye_x, left_eye_y - 12),
                                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                        cv2.putText(img_raw, "Open", (right_eye_x, right_eye_y - 12),
                                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                        self.alltimes += 1
                                        # if self.EyeClose >= self.EYE_AR_CONSEC_FRAMES:
                                        #     self.blink_count += 1
                                        #     self.m_textCtrl3.AppendText(
                                        #         time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"眨眼\n")
                                        # self.EyeClose = 0
                                        # cv2.putText(img_raw, "Eye Open", (250, 100),
                                        #        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 191, 255),2)
                                else:
                                    self.EyeClose = 0
                                    self.blink_count = 0
                                    self.long_EyeClose = 0

                                # cv2.putText(img_raw, "Blinks: {}".format(self.blink_count), (450, 30),
                                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                                # if self.yawn_checkBox1.GetValue() == True:
                                #     mouth_result = mouth_check(mouth)
                                #     cv2.rectangle(img_raw, (mouthcentre_x - distance_half, mouthcentre_y - distance_half),
                                #                   (mouthcentre_x + distance_half, mouthcentre_y + distance_half),
                                #                   (0, 255, 127), 2)
                                #
                                #     if mouth_result == 1:
                                #         self.MouthOpen += 1
                                #         cv2.putText(img_raw, "Open", (mouth_x, mouth_y - 12),
                                #                     cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                #         # cv2.putText(img_raw, "Mouth Open", (200, 100),
                                #         #        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 191, 255),2)
                                #     elif mouth_result == 0:
                                #         if self.MouthOpen >= self.MOUTH_AR_CONSEC_FRAMES:
                                #             self.yawm_count += 1
                                #             self.m_textCtrl3.AppendText(
                                #                 time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"打哈欠\n")
                                #         self.MouthOpen = 0
                                #         cv2.putText(img_raw, "Close", (mouth_x, mouth_y - 12),
                                #                     cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                #         # cv2.putText(img_raw, "Mouth Close", (200, 100),
                                #         #        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 191, 255),2)
                                # else:
                                #     self.MouthOpen = 0
                                #     self.yawm_count = 0
                                #
                                # cv2.putText(img_raw, "Yawning: {}".format(self.yawm_count), (450, 60),
                                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                                torch.set_grad_enabled(False)
                                PERCLOS = self.eyetimes / self.alltimes
                                cv2.putText(img_raw, "PERCLOS: {}".format("%.2f" % PERCLOS), (20, 456),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                                if PERCLOS >= 0.15:
                                    cv2.putText(img_raw, "PERCLOS: {}".format("%.2f" % PERCLOS), (20, 456),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                                    self.k = 1
                                if self.long_EyeClose==1:
                                    self.k = 1
                                if self.k == 1:
                                    cv2.putText(img_raw, "SLEEP!!!", (cx - 80, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (0, 0, 255),3)
                                mid_time = time.time()
                                if (time.localtime(start_time).tm_sec - time.localtime(mid_time).tm_sec) % 30 == 0:
                                    start_time = time.time()
                                    self.yawm_count = 0
                                    self.eyetimes = 2
                                    self.alltimes = 23


                    if Face == False:
                        if self.face_checkBox4.GetValue()== True:
                            cv2.putText(img_raw, "No Face", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),3, cv2.LINE_AA)
                            self.blink_count = 0
                            self.yawm_count  = 0
                            self.EyeClose    = 0
                            self.blink_count = 0
                            self.MouthOpen   = 0
                            self.yawm_count  = 0
                            self.long_EyeClose = 0

            #opencv中imread的图片内部是BGR排序，wxPython的StaticBitmap需要的图片是RGB排序，不转换会出现颜色变换
            height,width = img_raw.shape[:2]
            image1 = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            pic = wx.Bitmap.FromBuffer(width,height,image1)
            # 显示图片在panel上：
            self.bmp.SetBitmap(pic)

        # 释放摄像头
        self.cap.release()

        # do a bit of cleanup
        #cv2.destroyAllWindows()

    def camera_on(self,event):
        """使用多线程，子线程运行后台的程序，主线程更新前台的UI，这样不会互相影响"""
        import _thread
        # 创建子线程，按钮调用这个方法，
        _thread.start_new_thread(self._learning_face, (event,))
    
    def cameraid_choice( self, event ):
        # 摄像头编号
        cameraid = int(event.GetString()[-1])# 截取最后一个字符
        if cameraid == 0:
            self.m_textCtrl3.AppendText(u"准备打开本地摄像头!!!\n")
        if cameraid == 1 or cameraid == 2:
            self.m_textCtrl3.AppendText(u"准备打开外置摄像头!!!\n")
        self.VIDEO_STREAM = cameraid
        
    def vedio_on( self, event ):  
        if self.CAMERA_STYLE == True :# 释放摄像头资源
            self.blink_count = 0
            self.yawm_count  = 0
            self.EyeClose    = 0
            self.blink_count = 0   
            self.MouthOpen   = 0
            self.yawm_count  = 0  
            self.long_EyeClose = 0            
            # 弹出关闭摄像头提示窗口
            dlg = wx.MessageDialog(None, u'确定要关闭摄像头？', u'操作提示', wx.YES_NO | wx.ICON_QUESTION)
            if(dlg.ShowModal() == wx.ID_YES):
                self.cap.release()#释放摄像头
                self.bmp.SetBitmap(wx.Bitmap(self.image_cover))#封面
                dlg.Destroy()#取消弹窗
        # 选择文件夹对话框窗口
        dialog = wx.FileDialog(self,u"选择视频检测",os.getcwd(),'',wildcard="(*.mp4)|*.mp4",style=wx.FD_OPEN | wx.FD_CHANGE_DIR)
        if dialog.ShowModal() == wx.ID_OK:
            #如果确定了选择的文件夹，将文件夹路径写到m_textCtrl3控件
            self.m_textCtrl3.SetValue(u"文件路径:"+dialog.GetPath()+"\n")
            self.VIDEO_STREAM = str(dialog.GetPath())# 更新全局变量路径
            dialog.Destroy
            """使用多线程，子线程运行后台的程序，主线程更新前台的UI，这样不会互相影响"""
            import _thread
            # 创建子线程，按钮调用这个方法，
            _thread.start_new_thread(self._learning_face, (event,))
    def reset( self, event ):
        self.blink_count = 0
        self.yawm_count  = 0
        self.EyeClose    = 0
        self.blink_count = 0   
        self.MouthOpen   = 0
        self.yawm_count  = 0  
        self.long_EyeClose = 0
        self.eyetimes = 2
        self.alltimes = 23
        self.k = 0

    def off(self,event):
        """关闭摄像头，显示封面页"""
        self.cap.release()
        self.bmp.SetBitmap(wx.Bitmap(self.image_cover))
        
    def OnClose(self, evt):
        """关闭窗口事件函数"""
        dlg = wx.MessageDialog(None, u'确定要关闭本窗口？', u'操作提示', wx.YES_NO | wx.ICON_QUESTION)
        if(dlg.ShowModal() == wx.ID_YES):
            self.Destroy()
        print("检测结束，成功退出程序!!!")

            
class main_app(wx.App):
    """
     在OnInit() 里边申请Frame类，这样能保证一定是在app后调用，
     这个函数是app执行完自己的__init__函数后就会执行
    """
    # OnInit 方法在主事件循环开始前被wxPython系统调用，是wxpython独有的
    def OnInit(self):
        self.frame = Fatigue_detecting(parent=None,title="Fatigue_Detector")
        self.frame.Show(True)
        return True   

    
if __name__ == "__main__":
    app = main_app()
    app.MainLoop()

