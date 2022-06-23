import config
import cv2
import argparse
import numpy as np
if config.INFERENCE_ENGINE_TYPE == 'opencv':
    import text_detection_cv as text_detection
    import text_recognition_cv as text_recognition

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, type=str, help="path to input image")
args = vars(ap.parse_args())

def  main():
    try:
        image = cv2.imread(args["image"])
        td = text_detection.PixelLinkDecoder()
        tr = text_recognition.TextRecognizer()

        img1, bounding_rects = td.inference(image)
        texts,img2 = tr.inference(image, bounding_rects) 
        
        print('Result:', texts)
        op =  cv2.addWeighted(img1,0.3,img2,0.9,1)

        cv2.imshow('Detected text', op)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()

