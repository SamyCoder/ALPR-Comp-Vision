# detect.py
executable      = detect_lp.py
arguments       = "-python detect_lp.py --source ../../test/20230222_115854.mp4 --weights ../weights/yolov7.pt --conf 0.25 --save-txt --save-conf --device 0 --classes 1 2 3 5 7"
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

