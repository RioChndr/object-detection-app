import cv2 as cv
import numpy as np


class obj_detector:
    hasDetected = False
    detectedObj = []
    listObj = {}
    capVideo = None
    photo_target = None

    def __init__(self, weightModel, configModel, namesModel):

        self.net = cv.dnn.readNet(weightModel, configModel)
        self.classes = []
        with open(namesModel, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1]
                              for i in self.net.getUnconnectedOutLayers()]
        # print(output_layers)
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.tracker_obj = []

    def set_photo(self, img):
        self.photo_target = cv.imread(img)

    def detect_obj(self, frame=None):
        # init
        self.listObj = {}
        self.hasDetected = True
        if frame is not None:
            self.photo_target = frame
        img_ori = self.photo_target
        img = cv.UMat(img_ori)

        timer = cv.getTickCount()
        height, width, channels = img_ori.shape
        net = self.net
        blob = cv.dnn.blobFromImage(
            img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        # net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
        outs = net.forward(self.output_layers)

        self.detectedObj = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > .5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    self.detectedObj.append([
                        class_id,
                        str(self.classes[class_id]),
                        [x, y, w, h]
                    ])
        self.time_process = (cv.getTickCount() - timer) / cv.getTickFrequency()
        return self.detectedObj

    def label_img(self, bbox, label, color):
        img = self.photo_target

        font = cv.FONT_HERSHEY_PLAIN
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        center_x = (w / 2) + x
        center_y = (h / 2) + y
        cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv.putText(img, label, (x, y+30), font, 1.5, color, 2)
        cv.circle(img, (int(center_x), int(center_y)), 2, color, thickness=2)
        self.photo_target = img

    def label_obj(self):
        self.detect_obj()
        for obj in self.detectedObj:
            id_class_obj = obj[0]
            class_obj = obj[1]
            pos_obj = obj[2]
            color_obj = self.colors[id_class_obj]
            self.label_img(pos_obj, class_obj, color_obj)
            if class_obj in self.listObj:
                self.listObj[class_obj] += 1
            else:
                self.listObj[class_obj] = 1

    def show_image(self):
        cv.imshow('result', self.photo_target)

    def get_image(self):
        # Change from BGR (opencv) to RGB for tkinter
        return self.cvrt_img(self.photo_target)

    def cvrt_img(self, img):
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)

    def capture_video(self, path):
        self.capVideo = cv.VideoCapture(path)
        return self.capVideo.isOpened()

    def read_frame(self):
        ret, frame = self.capVideo.read()
        return frame

    def detect_frame(self, frame):
        self.photo_target = frame
        self.label_obj()
        result = self.get_image()
        return result
