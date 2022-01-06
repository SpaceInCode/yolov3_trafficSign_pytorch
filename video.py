import cv2
import numpy as np
from numpy.matlib import repmat
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from detection_and_classification import *

cap=cv2.VideoCapture(
    'C:\\Grade 4\\Machine Learning\\Code\\TSD7\\yolov3_trafficSign_pytorch\\video\\road_video.MOV',
    # 'C:\\Grade 4\\Machine Learning\\Code\\TSD7\\yolov3_trafficSign_pytorch\\video\\video1.mp4',
    # 'C:\\Grade 4\\Machine Learning\\Code\\TSD7\\yolov3_trafficSign_pytorch\\video\\video2.mp4',
    # 'C:\\Grade 4\\Machine Learning\\Code\\TSD7\\yolov3_trafficSign_pytorch\\video\\video3.mp4',
    )
fps=cap.get(cv2.CAP_PROP_FPS)
size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

videoWriter=cv2.VideoWriter(
    'C:\\Grade 4\\Machine Learning\\Code\\TSD7\\yolov3_trafficSign_pytorch\\video\\output.mp4',
    # 'C:\\Grade 4\\Machine Learning\\Code\\TSD7\\yolov3_trafficSign_pytorch\\video\\output1.mp4',
    # 'C:\\Grade 4\\Machine Learning\\Code\\TSD7\\yolov3_trafficSign_pytorch\\video\\output2.mp4',
    # 'C:\\Grade 4\\Machine Learning\\Code\\TSD7\\yolov3_trafficSign_pytorch\\video\\output3.mp4',
                cv2.VideoWriter_fourcc(*'mp4v'),(fps),
                # size,
                (640,480)
                )

def segment():
    while cap.isOpened():
        success,frame=cap.read()
        if success==False:
            break
        # Load the frame and convert to RGB
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # pixels=frame.reshape((-1, 3))
        # img=np.float32(pixels)
        
        segment_image=video_process(frame=frame)

        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        # K = n_cls
        # attempts=10
        # _,labels,centers=cv2.kmeans(img,K,None,criteria,attempts,
        #                         cv2.KMEANS_PP_CENTERS)

        # centers = np.uint8(centers)
        # segment_image = centers[labels.flatten()]
        # segment_image2= segment_image.reshape(frame.shape)
        

        videoWriter.write(segment_image)

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    segment()