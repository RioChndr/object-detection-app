from tkinter import *
from tkinter import ttk
from tkinter.font import Font

# configure GUI

rootWindow = Tk(className="Object Detection and Counting | TA RIO 2020")
rootWindow.resizable(0, 0)  # don't allow resizeing in x or y direction

txtNameFile = StringVar()
fileNameEntry = Entry(rootWindow, state="disabled",
                      width=40, textvariable=txtNameFile)
fileNameEntry.grid(row=0, column=0, sticky=W, ipadx=20)

chooseFile = Button(rootWindow, text="Choose Video/Photo",
                    padx=5, pady=5)
chooseFile.grid(row=0, column=1, sticky=W)

runButton = Button(rootWindow, text="RUN", padx=5,
                      pady=5, width=20, height=2)
runButton.grid(row=0, column=2)

showImage = Label(rootWindow, text="Display Video")
showImage.grid(row=1, column=0, columnspan=2, rowspan=3)

setROIbtn = Button(rootWindow, text="Setting ROI", padx=5, pady=5, width=20)
setROIbtn.grid(row=1, column=2, sticky=N)

infoROI = Label(rootWindow, text="Stat ROI : None/OK")
infoROI.grid(row=2, column=2, sticky=NW)


# """
# Informasi :

# ~ On Frame ~
# Mobil : 10
# Motor : 20

# ~ Object Counted ~
# Mobil : 20
# Motor : 10
# """
fontStyle = Font(family="Lucida Grande", size=15)

txtInfoDetail = StringVar()
detailInfo = Message(rootWindow, textvariable=txtInfoDetail, aspect=200,
                     justify="left", font=fontStyle)
detailInfo.grid(row=3, column=2, sticky="w")

progressBar = ttk.Progressbar(rootWindow, orient=HORIZONTAL,
                              length=400, mode='determinate')
progressBar.grid(row=4, column=0, columnspan=2)
progressBar['value'] = 50

txtStepFrame = StringVar()
stepFrame = Label(rootWindow, textvariable=txtStepFrame, pady=10)
stepFrame.grid(row=5, column=0, sticky=W)

txtLastSpd = StringVar()
lastSpeed = Label(rootWindow, textvariable=txtLastSpd)
lastSpeed.grid(row=6, column=0, sticky=W)


def setTxtLastSpd(spd):
    txtLastSpd.set("Last Speed Detected : {} s".format(spd))


def setTxtStepFrame(currentFrame, lenFrame):
    txtStepFrame.set("Frame {} of {}".format(currentFrame, lenFrame))


def setProgressBar(value):

    progressBar['value'] = value

# Berupa array
# e.g {"mobil" : 10, "motor" : 20}


def setInfo(dataOnFrame, dataCounted=None):
    info = "Informasi :\n \n"
    info = "{}~ On Frame~\n".format(info)
    for k, v in dataOnFrame.items():
        info = "{}{}:{}\n".format(info, k, v)

    if dataCounted is None:
        dataCounted = dataOnFrame

    info = "{}\n\n~Counted Object~\n".format(info)
    for k, v in dataCounted.items():
        info = "{}{}:{}\n".format(info, k, v)

    txtInfoDetail.set(info)
