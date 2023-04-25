# detect.py
executable      = test_lp.py
arguments       = "--source ../../test/sample/*/* --weights ../weights/yolov7.pt --conf 0.25 --save-txt-lp --nosave --save-conf --device 0 --classes 1 2 3 5 7"
output          = lp_test_yolo_ocr_output.txt
log             = lp_test_yolo_ocr_log.log
error 		= lp_test_yolo_ocr_error.log

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

#"--source ../../test/sample/*/* --weights ../weights/yolov7.pt --conf 0.25 --save-txt-lp --save-txt --nosave --save-conf --device 0 --classes 1 2 3 5 7"
