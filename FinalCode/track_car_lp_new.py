#!/usr/bin/env python3

import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from numpy import random


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages, LoadStreams, letterbox
from yolov7.utils.general import (check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, check_file)
from yolov7.utils.torch_utils import select_device, time_synchronized
from yolov7.utils.plots import plot_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

import easyocr

import json


VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes


@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes

        save_crop_lp=False,  # save cropped LP prediction boxes

        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
        yolo_weights = Path(yolo_weights[0])
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    save_dir = Path(save_dir)
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    model = attempt_load(Path(yolo_weights), map_location=device)  # load FP32 model
    names, = model.names,
    stride = model.stride.max().cpu().numpy()  # model stride
    imgsz = check_img_size(imgsz[0], s=stride)  # check image size

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        nr_sources = 1
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                half,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
        strongsort_list[i].model.warmup()
    outputs = [None] * nr_sources
    
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    
    #********************************************************************#
    #lp_yolo_weights, trace =  opt.weights, not opt.no_trace
    lp_weights, trace =  "weights/lp_best.pt", False
    #save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    #webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    #    ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    #save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    #set_logging()
    #device = select_device(opt.device)
    #half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    lp_model = attempt_load(Path(lp_weights), map_location=device)  # load FP32 model
    stride = int(lp_model.stride.max())  # model stride
    #imgsz = check_img_size(imgsz, s=stride)  # check img_size
    lp_names, = lp_model.names
    #load easyocr model 
    easyocr_reader = easyocr.Reader(['en'])

    #if trace:
    #    lp_model = TracedModel(lp_model, device, imgsz)

    #********************************************************************#
    data = {'frame_id': 0, 'vehc_id':0, 'vehc_cls':'', 'vehc_bb':[], 'vehc_conf': 0, 'LP_bb':[], 'LP_conf':0, 'LP_txt':'', 'LP_txt_conf':0}
    
    lp_path = str(save_dir)+'/LP/'
    vehicle_path = str(save_dir)+'/Vehicle/'
    if save_crop_lp: os.mkdir(lp_path)
    if save_crop: os.mkdir(vehicle_path)

    # Run tracking
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap) in enumerate(dataset):
        s = ''
        t1 = time_synchronized()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_synchronized()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im)
        t3 = time_synchronized()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred[0], conf_thres, iou_thres, classes, agnostic_nms)
        dt[2] += time_synchronized() - t3
        
        data['frame_id']=frame_idx
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop
            imc_lp = im0.copy() if save_crop_lp else im0

            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort
                t4 = time_synchronized()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_synchronized()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
    
                        bboxes = output[0:4].astype(int)
                        id = int(output[4])
                        cls = int(output[5])

                        if(save_txt):
                            bbox_left = int(output[0])
                            bbox_top = int(output[1])
                            bbox_w = int(output[2] - output[0])
                            bbox_h = int(output[3] - output[1])
                            data['vehc_bb'] = [bbox_left, bbox_top, bbox_w, bbox_h]
                        
                        data['vehc_id'] = id
                        data['vehc_cls'] = names[cls]    
                        data['vehc_conf'] = f"{conf:.2f}"
                        #................................................................................#
                        #frame_car_h = bboxes[1]
                        #frame_car_w = bboxes[0]
                        # Padded resize
                        car_im0 = im0[bboxes[1]:bboxes[3],bboxes[0]:bboxes[2]]
                        car_im0_copy = imc_lp[bboxes[1]:bboxes[3],bboxes[0]:bboxes[2]]
                        
                        car_rs_img = letterbox(car_im0_copy, new_shape=imgsz, stride=stride)[0]
                        car_rs_img = car_rs_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                        car_rs_img = np.ascontiguousarray(car_rs_img)
                        

                        # Convert                       
                        car_rs_img = torch.from_numpy(car_rs_img).to(device)
                        car_rs_img = car_rs_img.half() if half else car_rs_img.float()  # uint8 to fp16/32
                        car_rs_img /= 255.0  # 0 - 255 to 0.0 - 1.0
                        if car_rs_img.ndimension() == 3:
                            car_rs_img = car_rs_img.unsqueeze(0)
                        #print("shape of the car_rs_img",car_rs_img.shape)
                        #print("shape of image im",im.shape)                        
                        # Inference
                        lp_t1 = time_synchronized()
                        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                            lp_pred = lp_model(car_rs_img)[0]
                        lp_t2 = time_synchronized()

                        # Apply NMS
                        lp_pred = non_max_suppression(lp_pred, conf_thres, iou_thres, agnostic=agnostic_nms)
                        lp_t3 = time_synchronized()
                        
 
                        # Process lp_detections
                      
                        for lp_i, lp_det in enumerate(lp_pred):  # detections per image
                            #lp_det is a list of lists [[bb,conf,class]] so 6 elements in the list                         
                            
                            if len(lp_det):
                                #print("lp_det:",lp_det)
                                # Rescale boxes from img_size to im0 size
                                lp_gn = torch.tensor(car_im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                                lp_det[:, 0:4] = scale_coords(car_rs_img.shape[2:], lp_det[:, :4], car_im0.shape).round() #xyxy

                                cls_lp = lp_det[0][5]
                                conf_lp = f"{lp_det[0][4]:.2f}"
                                
                                xyxy_lp = lp_det[0][0:4].detach().cpu().numpy()
                                lp_xmin,lp_ymin,lp_xmax,lp_ymax = xyxy_lp.astype(int)

                                #print("conf_lp:",conf_lp)
                                label_lp = f"LP {conf_lp}"
                                # Print results

                                
                                data['LP_conf'] = conf_lp

                                #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                                lp_im0 = car_im0_copy[lp_ymin:lp_ymax,lp_xmin:lp_xmax]                    
                                #easyocr_result = easyocr_reader.recognize(lp_im0)#,paragraph="False")   
                                easyocr_result = easyocr_reader.readtext(lp_im0)     #readtext works better thatn recognize                                     
                                ocr_label = " "    
                                ocr_conf = " "                  
                                #doing for each license plate
                                for res in easyocr_result:
                                    if res[2] > 0.1:
                                        ocr_label += res[1] + " "
                                        ocr_conf += f'{res[2]:.2f} '
                                  
                                if ocr_label == ' ':
                                    data['LP_txt']= 'NA'
                                    data['LP_txt_conf']= 'NA'
                                else:
                                    data['LP_txt']= ocr_label
                                    data['LP_txt_conf']= ocr_conf
                                
                                


                                 
                                label_lp = label_lp + ocr_label
                                print("label per detection :",label_lp)
                                plot_one_box(lp_det[0][0:4], car_im0, label=label_lp, color=colors[int(cls_lp)], line_thickness=2)
                                # Write results
                                #lp_box_label += lp_det[:,0:4][0] + " Class:LP " + "Conf: "+ conf_lp
                                #print("det value for each prediction - ",det)
                                #for *xyxy_lp, conf_lp, cls_lp in reversed(lp_det):
                                if save_txt:  # Write to file
                                    #xyxy_lp = lp_det[:, 0:4]
                                    xyxy2xywh(torch.tensor(xyxy_lp).view(1, 4))[0]
                                    xywhs_lp = xyxy2xywh(torch.tensor(xyxy_lp).view(1, 4))[0].detach().cpu().numpy().astype(int)
                                    #xywh_lp = (xyxy_lp(torch.tensor(xyxy_lp).view(1, 4)) / lp_gn).view(-1).tolist()  # normalized xywh
                                    #line_lp = (frame_idx + 1, id, cls_lp, *xywhs_lp, conf_lp, ocr_label)# if opt.save_conf else (cls_lp, *xywhs_lp, ocr_label)  # label format
                                    data['LP_bb'] = xywhs_lp.tolist()
                                    #print('type:',type(data['LP_bb']))

                                    with open(txt_path + '.txt', 'a') as f:
                                        #f.write(('%g ' * 10) % (frame_idx + 1, id, xywhs_lp[0],xywhs_lp[1],xywhs_lp[2],xywhs_lp[3], -1, -1,-1, i))
                                        #f.write(" "+ label_lp + '\n')
                                        json.dump(data, f, ensure_ascii=False)
                                        f.write('\n')

                                if save_crop_lp:
                                    cv2.imwrite(str(save_dir)+'/LP/'+ str(frame_idx) +"_"+ str(id)+ocr_label+'.jpg',lp_im0)

                                #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                                #................................................................................# 
                            else:
                                data['LP_bb']='NA'
                                data['LP_bb_conf']='NA'
                                data['LP_txt']='NA'
                                data['LP_txt_conf']='NA'
                        '''
                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                str_label =str(id)+' '+str(names[int(cls)]) + " "+ f'{conf:.2f}'
                                f.write(('%g ' * 10) % (frame_idx + 1, id, bbox_left, bbox_top, bbox_w, bbox_h,-1, -1, -1, i)) # MOT format
                                f.write(" "+str_label + '\n')
                        '''
                        if save_vid or save_crop or show_vid:  # Add bbox to image

                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[cls]} {conf:.2f}'))
                            plot_one_box(bboxes, im0, label=label, color=colors[cls], line_thickness=2)
                            if save_crop:
                                cv2.imwrite(vehicle_path + str(frame_idx)+"_"+label+'.jpg',car_im0_copy)
                    
                        

                    
                      
                print(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

            else:
                strongsort_list[i].increment_ages()
                print('No detections')
                data = {'frame_id': frame_idx, 'vehc_id':'NA', 'vehc_cls':'NA', 'vehc_bb':'NA', 'vehc_conf':'NA', 'LP_bb':'NA', 'LP_conf':'NA', 'LP_txt':'NA', 'LP_txt_conf':'NA'}
                

            # Stream results
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    #vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, imgsz, imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')

    parser.add_argument('--save-crop-lp', action='store_true', help='save cropped LP prediction boxes')

    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
