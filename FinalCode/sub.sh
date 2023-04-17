# detect.py
executable      = track_car_lp_new.py
arguments       = "--conf-thres 0.25 --source ../test/20230222_115854.mp4 --device 0 --classes 1 2 3 5 7 --save-crop-lp --save-crop --save-vid  --save-txt --strong-sort-weights weights/osnet_x0_25_msmt17.pt --yolo-weights weights/yolov7.pt"
output          = lp_yolo_ocr_output.txt
log             = lp_yolo_ocr_log.log
error 		= lp_yolo_ocr_error.log

getenv=True
+GPUJob = true
requirements = (TARGET.GPUSlot)
request_GPUs = 1

request_cpus   = 1
request_disk   = 2 GB
request_memory = 2 GB

requirements = InMastodon
+Group = "GRAD"
+ProjectDescription="training grasp learning algorithm for object lifting simulation"
+Project="AI_ROBOTICS"

queue

