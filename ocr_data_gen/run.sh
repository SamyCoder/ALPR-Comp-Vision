#!/bin/bash

# clone https://github.com/Belval/TextRecognitionDataGenerator to use this file,
# and add/replace "fonts" to "trdg/fonts" (contains various license plate style fonts)
# creates OCR data using fonts in "trdg/fonts/latin/" with the following options:

# -c [COUNT] = 50
# -rs [RANDOM STRINGS] = True
#   -let [USE LETTERS] = True
#   -num [USE NUMBERS] = True
# -k [SKEW] = 5 degrees
#   -rk [RANDOM SKEW] = True (skews between neg -k and pos -k value)
# -bl [BLUR] = 3 pixels (I think it's pixels)
#   -rbl [RANDOM BLUR] = True (blur between neg -bl and pos -bl value)
# -ca [CASE] = upper (uppercase letters only)
# -f [FORMAT] = 128 (height of image in pixels - width is automatically set based on string length)


python trdg/run.py -c 50 -rs -let -num -k 5 -rk -bl 3 -rbl -ca upper -f 128
