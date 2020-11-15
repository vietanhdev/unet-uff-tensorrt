import os
import cv2
import tensorrt
from utils import inference as inference_utils
from utils import model as model_utils


list_videos = ["/home/anhnv/DATA/video_non_polyp/Non-polyp Colon 200117.mp4"]

# Model used for inference
uff_model_path = None
trt_model_path = None

# Confidence threshold for drawing bounding box
VISUALIZATION_THRESHOLD = 0.5

# Precision command line argument -> TRT Engine datatype
TRT_PRECISION_TO_DATATYPE = {
    16: trt.DataType.HALF,
    32: trt.DataType.FLOAT
}

MAX_BATCH_SIZE = 1

# Set up all TensorRT data structures needed for inference
trt_inference_wrapper = inference_utils.TRTInference(
    trt_engine_path, uff_model_path,
    trt_engine_datatype=TRT_PRECISION_TO_DATATYPE(32),
    batch_size = MAX_BATCH_SIZE
)


for video in list_videos:
    print(video)
    cap = cv2.VideoCapture(video)
    
    while(cap.isOpened()):
        start = time.time()
        ret, frame = cap.read()
        if ret == False:
            break
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        output = trt_inference_wrapper(image)
        print(output)
        mask = frame

        h, w, c = mask.shape
        tmp = np.zeros([h, w*2, 3],dtype=np.uint8)
        tmp[:, :w, :] = frame
        tmp[:, w:, :] = mask
        cv2.imshow('Result')

        end = time.time()
        