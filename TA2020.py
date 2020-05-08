import layout
import tkinter as tk
import cv2 as cv
from brain import obj_detector
from tkinter import filedialog
from PIL import ImageTk, Image


PHOTO = "Photo"
VIDEO = "Video"

selectedFile = {
    'path': None,
    'name': None,
    'format': None,
    'type': None
}
brain = obj_detector('yolov3/yolov3.weights',
                         'yolov3/yolov3.cfg',
                         'yolov3/coco.names')

def findFile():
    FilePath = filedialog.askopenfilename()
    if FilePath is None:
        return
    photoFormats = ['png', 'jpg', 'jpeg', 'gif']
    videoFormats = ['mp4', 'mkv', '3gp']

    splitName = FilePath.split("/")

    fileName = splitName[len(splitName)-1]
    splitFormat = fileName.split(".")
    fileFormat = splitFormat[len(splitFormat)-1]  # last index is format file
    selectedFile['path'] = FilePath
    selectedFile['name'] = fileName
    selectedFile['format'] = fileFormat
    print(FilePath)

    if fileFormat in photoFormats:
        selectedFile['type'] = PHOTO
    elif fileFormat in videoFormats:
        selectedFile['type'] = VIDEO
    else:
        selectedFile['type'] = None

    showToPanel()
    layout.setInfo(None)


def showToPanel():
    layout.txtNameFile.set(selectedFile['name'])
    if selectedFile['type'] == VIDEO:
        showVideo2Panel()
        return
    showImage2Panel(selectedFile['path'])


def showImage2Panel(path = None, frame= None):
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

def showVideo2Panel():
    cap = cv.VideoCapture(selectedFile['path'])
    if cap.isOpened() == False:
        layout.rootWindow.showerror("error", "file video tidak ditemukan")
        return
    
    ret, frstFrame = cap.read()
    frstFrame = brain.cvrt_img(frstFrame)
    frstFrame = Image.fromarray(frstFrame)
    showImage2Panel(frame=frstFrame)


def scanFile():
    global brain
    layout.setProgressBar(10)
    
    brain.set_photo(selectedFile['path'])
    brain.label_obj()  # process
    result_img = brain.get_image()

    # time process
    time_consumn = brain.time_process

    # generate from array to image
    result_img = Image.fromarray(result_img)
    showImage2Panel(frame=result_img)

    layout.setProgressBar(100)
    layout.setTxtLastSpd(time_consumn)

    detectedObj = brain.listObj
    layout.setInfo(detectedObj)


layout.runButton.config(command=scanFile)
layout.chooseFile.config(command=findFile)

layout.rootWindow.mainloop()  # put this at end file
