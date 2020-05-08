import layout
import tkinter as tk
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


def findFile():
    FilePath = filedialog.askopenfilename()
    photoFormats = ['png', 'jpg', 'jpeg', 'gif']
    videoFormats = ['mp4', 'mkv', '3gp']

    splitName = FilePath.split("/")

    fileName = splitName[len(splitName)-1]
    splitFormat = fileName.split(".")
    fileFormat = splitFormat[len(splitFormat)-1]  # last index is format file
    selectedFile['path'] = FilePath
    selectedFile['name'] = fileName
    selectedFile['format'] = fileFormat

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
    showImage2Panel(selectedFile['path'])


def showImage2Panel(path, isArray=False):
    if isArray:
        loadImg = path
    else:
        loadImg = Image.open(path)

    wImage, hImage = loadImg.size

    maxHeight = 700

    adaptWidth = wImage * (maxHeight/hImage)
    adaptHeight = maxHeight

    loadImg = loadImg.resize((int(adaptWidth), int(adaptHeight)))

    img = ImageTk.PhotoImage(loadImg)
    layout.showImage.config(image=img)
    layout.showImage.image = img


def scanFile():
    layout.setProgressBar(10)
    brain = obj_detector('yolov3/yolov3.weights',
                         'yolov3/yolov3.cfg',
                         'yolov3/coco.names')
    brain.set_photo(selectedFile['path'])
    brain.label_obj()  # process
    result_img = brain.get_image()

    # time process
    time_consumn = brain.time_process

    # generate from array to image
    result_img = Image.fromarray(result_img)
    showImage2Panel(result_img, True)

    layout.setProgressBar(100)
    layout.setTxtLastSpd(time_consumn)

    detectedObj = brain.listObj
    layout.setInfo(detectedObj)


layout.runButton.config(command=scanFile)
layout.chooseFile.config(command=findFile)

layout.rootWindow.mainloop()  # put this at end file
