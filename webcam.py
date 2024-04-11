# TensorFlow Lite 모델을 사용하여 실시간 웹캠 피드에서 객체 탐지를 수행합니다. 각 객체 주변에 상자와 점수를 그립니다.
# FPS를 향상시키기 위해 웹캠 객체는 메인 프로그램과 별도의 스레드에서 실행됩니다.
# 이 스크립트는 PiCamera나 일반 USB 웹캠 모두에서 작동합니다.

# 이 코드는 TensorFlow Lite 이미지 분류 예제에서 기반을 가져왔습니다.
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
# OpenCV를 사용하여 상자와 라벨을 그리는 자체 방법을 추가했습니다.

# 패키지 임포트
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

# 웹캠에서 비디오 스트리밍을 별도의 처리 스레드로 관리하는 카메라 객체를 정의합니다.
# 출처: Adrian Rosebrock, PyImageSearch, https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Picamera에서 비디오 스트리밍을 제어하는 카메라 객체입니다."""
    def __init__(self,resolution=(640,480),framerate=30):
        # PiCamera와 카메라 이미지 스트림을 초기화합니다.
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # 스트림에서 첫 번째 프레임을 읽습니다.
        (self.grabbed, self.frame) = self.stream.read()

	    # 카메라가 정지됐을 때를 제어하는 변수입니다.
        self.stopped = False

    def start(self):
	    # 비디오 스트림에서 프레임을 읽는 스레드를 시작합니다.
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # 스레드가 정지될 때까지 무한히 반복합니다.
        while True:
            # 카메라가 정지되면 스레드를 정지합니다.
            if self.stopped:
                # 카메라 자원을 닫습니다.
                self.stream.release()
                return

            # 그렇지 않으면 스트림에서 다음 프레임을 가져옵니다.
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	    # 가장 최근의 프레임을 반환합니다.
        return self.frame

    def stop(self):
	    # 카메라와 스레드를 정지해야 함을 나타냅니다.
        self.stopped = True

# 입력 인자를 정의하고 파싱합니다.
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='폴더가 .tflite 파일이 위치한 곳입니다', required=True)
parser.add_argument('--graph', help='.tflite 파일의 이름이 detect.tflite와 다를 경우', default='detect.tflite')
parser.add_argument('--labels', help='labelmap.txt 파일의 이름이 다를 경우', default='labelmap.txt')
parser.add_argument('--threshold', help='탐지된 객체를 표시하기 위한 최소 신뢰 임계값', default=0.5)
parser.add_argument('--resolution', help='원하는 웹캠 해상도 in WxH. 입력된 해상도를 웹캠이 지원하지 않으면 오류가 발생할 수 있습니다.', default='1280x720')
parser.add_argument('--edgetpu', help='탐지를 가속하기 위해 Coral Edge TPU Accelerator 사용', action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# TensorFlow 라이브러리를 임포트합니다.
# tflite_runtime이 설치되어 있으면 tflite_runtime에서 interpreter를 임포트하고, 그렇지 않으면 일반 tensorflow에서 임포트합니다.
# Coral Edge TPU를 사용하는 경우, load_delegate 라이브러리를 임포트합니다.
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# Edge TPU를 사용하는 경우, Edge TPU 모델에 대한 파일 이름을 지정합니다.
if use_TPU:
    # 사용자가 .tflite 파일의 이름을 지정한 경우 해당 이름을 사용하고, 그렇지 않으면 기본값인 'edgetpu.tflite'를 사용합니다.
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# 현재 작업 디렉토리의 경로를 가져옵니다.
CWD_PATH = os.getcwd()

# 객체 탐지에 사용되는 모델이 포함된 .tflite 파일의 경로입니다.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# 라벨 맵 파일의 경로입니다.
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# 라벨 맵을 로드합니다.
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# COCO "starter model"을 사용하는 경우 라벨 맵에 대한 이상한 수정이 필요합니다.
# 첫 번째 라벨이 '???'인데, 이것을 제거해야 합니다.
if labels[0] == '???':
    del(labels[0])

# TensorFlow Lite 모델을 로드합니다.
# Edge TPU를 사용하는 경우, 특별한 load_delegate 인자를 사용합니다.
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# 모델 세부 정보를 가져옵니다.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# 이 모델이 TF2 또는 TF1로 생성되었는지 확인하기 위해 출력 레이어 이름을 확인합니다.
# TF2와 TF1 모델의 출력이 다르게 정렬되기 때문입니다.
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # TF2 모델입니다.
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # TF1 모델입니다.
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# 프레임 속도 계산을 초기화합니다.
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# 비디오 스트림을 초기화합니다.
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

# 무한 반복하여 카메라에서 연속적으로 프레임을 캡처합니다.
while True:

    # 프레임 속도를 계산하기 위한 타이머를 시작합니다.
    t1 = cv2.getTickCount()

    # 비디오 스트림에서 프레임을 가져옵니다.
    frame1 = videostream.read()

    # 프레임을 취득하고 예상된 모양 [1xHxWx3]으로 크기를 조정합니다.
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # 모델이 비정규화된 경우(즉, 모델이 양자화되지 않은 경우) 픽셀 값을 정규화합니다.
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # 이미지를 입력으로 사용하여 실제 탐지를 수행하기 위해 모델을 실행합니다.
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # 탐지 결과를 검색합니다.
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # 탐지된 객체의 경계 상자 좌표
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # 탐지된 객체의 클래스 인덱스
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # 탐지된 객체의 신뢰도

    # 모든 탐지를 반복하고 신뢰도가 최소 임계값 이상인 경우 탐지 상자를 그립니다.
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # 경계 상자 좌표를 가져오고 상자를 그립니다.
            # 해석기는 이미지 차원을 벗어난 좌표를 반환할 수 있으므로, 이미지 내에 있도록 강제로 조정해야 합니다.
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # 라벨을 그립니다.
            object_name = labels[int(classes[i])] # 클래스 인덱스를 사용하여 "labels" 배열에서 객체 이름을 찾습니다.
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # 예: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # 폰트 크기를 가져옵니다.
            label_ymin = max(ymin, labelSize[1] + 10) # 창의 맨 위에 라벨을 너무 가까이 그리지 않도록 합니다.
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # 라벨 텍스트를 넣을 하얀 상자를 그립니다.
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # 라벨 텍스트를 그립니다.

    # 프레임의 모서리에 프레임 속도를 그립니다.
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # 모든 결과가 프레임에 그려졌으므로 이를 표시할 시간입니다.
    cv2.imshow('Object detector', frame)

    # 프레임 속도를 계산합니다.
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # 'q'를 누르면 종료합니다.
    if cv2.waitKey(1) == ord('q'):
        break

# 정리 작업
cv2.destroyAllWindows()
videostream.stop()
