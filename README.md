# ALPR-Comp-Vision
Automatic license plate recognition (ALPR) architecture for processing low-resolution video from moving cellphone cameras.

Our code is build off from https://github.com/mikel-brostrom/Yolov7_StrongSORT_OSNet.git

Follow these steps to reproduce our code: 

Create a new environmet with the command <<conda create -n "alpr" python=3.8.16>>

Do <<pip install -r requirements.txt>>
  
Create a 'user_network' folder in ~/.EasyOCR/user_network, and add the best_accuracy.py and best_accuracy.yaml files there.

Create a 'model' folder in ~/.EasyOCR/model, and add best_accuracy.pth file there

Run the command to <<python main.py --conf-thres 0.25 --source ../test/20230222_115854.mp4 --device 0 --save-crop-lp --save-crop --save-vid  --save-txt --strong-sort-weights weights/osnet_x0_25_msmt17.pt --yolo-weights weights/yolov7.pt --classes 1 2 3 5 7>>
