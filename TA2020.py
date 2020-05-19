import layout
import tkinter as tk
import cv2 as cv
import copy
from brain import obj_detector as brain_obj_detector
from brainTracker import TrackerSystem
from tkinter import filedialog
from PIL import ImageTk, Image


class ObjectDetectorApp:

    PHOTO = "Photo"
    VIDEO = "Video"
    DELAYWINDOW = 100
    count_frame = 0

    selectedFile = {
        'path': None,
        'name': None,
        'format': None,
        'type': None
    }

    def __init__(self, weight, cfg, names):
        self.brain = brain_obj_detector(weight, cfg, names)
        self.brain_tracker = TrackerSystem(weight, cfg, names)

    def findFile(self):
        FilePath = filedialog.askopenfilename()

        # When user close filedialog
        if len(FilePath) == 0:
            return

        photoFormats = ['png', 'jpg', 'jpeg', 'gif']
        videoFormats = ['mp4', 'mkv', '3gp']

        splitName = FilePath.split("/")

        fileName = splitName[len(splitName)-1]
        splitFormat = fileName.split(".")
        # last index is format file
        fileFormat = splitFormat[len(splitFormat)-1]
        self.selectedFile['path'] = FilePath
        self.selectedFile['name'] = fileName
        self.selectedFile['format'] = fileFormat
        print(FilePath)

        if fileFormat in photoFormats:
            self.selectedFile['type'] = self.PHOTO
        elif fileFormat in videoFormats:
            self.selectedFile['type'] = self.VIDEO
        else:
            self.selectedFile['type'] = None

        self.showToPanel()
        layout.setInfo(None)

    def showToPanel(self):
        layout.txtNameFile.set(self.selectedFile['name'])
        if self.selectedFile['type'] == self.VIDEO:
            self.showVideo2Panel()
            return
        self.showImage2Panel(self.selectedFile['path'])

    def showImage2Panel(self, path=None, frame=None):
        if path is not None:
            loadImg = Image.open(path)
        elif frame is not None:
            loadImg = frame

        wImage, hImage = loadImg.size

        maxHeight = 700

        adaptWidth = wImage * (maxHeight/hImage)
        adaptHeight = maxHeight

        loadImg = loadImg.resize((int(adaptWidth), int(adaptHeight)))

        img = ImageTk.PhotoImage(loadImg)
        layout.showImage.config(image=img)
        layout.showImage.image = img

    def showVideo2Panel(self):

        cap = self.brain_tracker.capture_video(self.selectedFile['path'])
        # cap = cv.VideoCapture(self.selectedFile['path'])
        if cap == False:
            layout.rootWindow.showerror("error", "file video tidak ditemukan")
            return

        frstFrame = self.brain_tracker.read_frame()
        self.thumpnail = copy.deepcopy(frstFrame)
        frstFrame = self.brain_tracker.cvrt_img(frstFrame)
        frstFrame = Image.fromarray(frstFrame)
        self.showImage2Panel(frame=frstFrame)

    def runDetector(self):
        if self.selectedFile['type'] == self.PHOTO:
            self.detectorImage()
        elif self.selectedFile['type'] == self.VIDEO:
            self.detectorVideo()
        else:
            layout.rootWindow.showerror("error", "file video tidak ditemukan")
            return False

    def detectorImage(self):

        layout.setProgressBar(10)

        self.brain.set_photo(self.selectedFile['path'])
        self.brain.label_obj()  # process
        result_img = self.brain.get_image()

        # time process
        time_consumn = self.brain.time_process

        # generate from array to image
        result_img = Image.fromarray(result_img)
        self.showImage2Panel(frame=result_img)

        layout.setProgressBar(100)
        layout.setTxtLastSpd(time_consumn)

        detectedObj = self.brain.listObj
        layout.setInfo(detectedObj)

    def detectorVideo(self):
        self.updateDetectorVideo()

    def updateDetectorVideo(self):
        self.count_frame += 1
        frame = self.brain_tracker.read_frame()
        result = self.brain_tracker.track_object_inframe(
            frame, self.count_frame)

        # time process
        time_consumn = self.brain_tracker.time_process

        # generate from array to image
        result_img = Image.fromarray(result)
        self.showImage2Panel(frame=result_img)

        layout.setTxtLastSpd(time_consumn)

        detectedObj = self.brain_tracker.listObj
        countedObj = self.brain_tracker.objects_counted
        layout.setInfo(detectedObj, countedObj)

        layout.rootWindow.after(self.DELAYWINDOW, self.updateDetectorVideo)

    def selectROI(self):
        layout.set_messagebox_info(
            "Setting ROI", "Select area ROI and press ENTER to save it or ESC to cancel")
        new_thumpnail = self.brain_tracker.set_ROI(self.thumpnail)
        new_thumpnail = self.brain_tracker.cvrt_img(new_thumpnail)
        new_thumpnail = Image.fromarray(new_thumpnail)
        self.showImage2Panel(frame=new_thumpnail)


if __name__ == "__main__":
    objDetector = ObjectDetectorApp('yolov3/yolov3.weights',
                                    'yolov3/yolov3.cfg',
                                    'yolov3/coco.names')

    layout.runButton.config(command=objDetector.runDetector)
    layout.chooseFile.config(command=objDetector.findFile)
    layout.setROIbtn.config(command=objDetector.selectROI)

    layout.rootWindow.mainloop()  # put this at end file
