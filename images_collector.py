import cv2 # OpenCV lib
import os
import time
import uuid # Naming files

IMAGES_PATH = 'Tensorflow/workspace/images/collectedimages'

labels = ['Hello']
number_imgs = 15

for label in labels:
    directory = f"Tensorflow/workspace/images/collectedimages/{label}"
    os.makedirs(directory, exist_ok = True)
    cap = cv2.VideoCapture(1) # Initialise video capture (webacam device number = 1)
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    for imgnum in range(number_imgs):
        ret, frame = cap.read() # Setting up capture
        imgname = os.path.join(IMAGES_PATH, label, label + '.' + '{}.jpg'.format(str(uuid.uuid1()))) # Defining img name
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release() # Stop video capture 