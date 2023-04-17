How to use these custom ocr models?

1. Create a 'user_network' folder in ~/.EasyOCR/user_network, and add your custom_model.py and custom_model.yaml files there.

2. Create a model folder in ~/.EasyOCR/model, and add custom_model.pth file there. 

3. The way we use OCR in our code remains the same, except the change: 
reader = easyocr.Reader(['en'], recog_network='custom_model')

4. Note that we still need easyocr package and remember to name all the 3 files above (.py, .yaml, .pth) same.
