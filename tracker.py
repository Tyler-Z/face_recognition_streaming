from __future__ import print_function

import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
#from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from utils.timer import Timer
import sys
import time

from sort import *
# sys.argv = ['test.py', '-s', '--vis_thres', '0.6', '--cpu']
sys.argv = ['test.py', '-s', '--vis_thres', '0.6']


parser = argparse.ArgumentParser(description='FaceBoxes')
parser.add_argument('-m', '--trained_model', default='weights/FaceBoxes.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
# parser.add_argument('--dataset', default='PASCAL', type=str, choices=['AFW', 'PASCAL', 'FDDB'], help='dataset')
parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--show_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()


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


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


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


def gstreamer_pipeline(
    capture_width=640,
    capture_height=360,
    display_width=640,
    display_height=360,
    framerate=12,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def draw(frame,x1,y1,x2,y2,color,obj_id):
    # color = colors[int(obj_id) % len(colors)]
    # color = tuple([i * 255 for i in color])    
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),color, 4)
    # cv2.rectangle(frame, (int(x1), int(y1-35)), (int(x1+len(cls)*19+60),int(y1)), color, -1)
    cv2.putText(frame, "face-" + str(int(obj_id)), (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 
        1, (255,255,255), 3)
    return(frame)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # net and model
    net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)


    tracker = Sort(max_age=2, min_hits=4)


#     capture = cv2.VideoCapture("./data/video/140p.mp4")
#     capture = cv2.VideoCapture("./data/video/240p.mp4")
    # capture = cv2.VideoCapture("./data/video/360p.mp4")
    # capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    capture = cv2.VideoCapture("./data/video/480p.mp4")
    #capture = cv2.VideoCapture("./data/video/720p.mp4")
#     capture = cv2.VideoCapture("./data/video/1080p.mp4")

    _t = {'forward_pass': Timer(), 'misc': Timer()}
    
    while True:
        # image_path = testset_folder + img_name + '.jpg'
        # img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        _, img_raw = capture.read()
        # img_raw = cv2.flip(img_raw, 1)

        t1 = time.time()

        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        loc, conf = net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale # / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        #keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        _t['misc'].toc()

        if len(dets) == 0:
            dets = np.empty((0,5))
        tracks = tracker.update(dets)


        for j,(x1, y1, x2, y2, obj_id) in enumerate(tracks):
            img_raw = draw(img_raw, x1, y1, x2, y2, (255,255,255), obj_id)
        cv2.imshow('frame',img_raw)

        # # show boxed image
        # if args.show_image:
        #     for b in dets:
        #         if b[4] < args.vis_thres:
        #             continue
        #         text = "{:.4f}".format(b[4])
        #         b = list(map(int, b))
        #         cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        #         cx = b[0]
        #         cy = b[1] + 12
        #         cv2.putText(img_raw, text, (cx, cy),
        #                     cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        #     cv2.imshow('res', img_raw)
        #     cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        t2 = time.time()
        print("fps: %f" % (1/(t2-t1)))
