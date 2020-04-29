from tkinter import *
from tkinter import ttk

# configure GUI

rootWindow = Tk(className="Object Detection and Counting | TA RIO 2020")
rootWindow.resizable(0, 0)  # don't allow resizeing in x or y direction

nameVideo = Entry(rootWindow, state="disabled", width=40)
nameVideo.grid(row=0, column=0)

chooseVideo = Button(rootWindow, text="Choose Video", padx=5, pady=5)
chooseVideo.grid(row=0, column=1, sticky=W)

controlVideo = Button(rootWindow, text="RUN", padx=5,
                      pady=5, width=20, height=2)
controlVideo.grid(row=0, column=2)

showImage = Label(rootWindow, text="Display Video")
showImage.grid(row=1, column=0, columnspan=2, rowspan=3)

setROIbtn = Button(rootWindow, text="Setting ROI", padx=5, pady=5, width=20)
setROIbtn.grid(row=1, column=2)

infoROI = Label(rootWindow, text="Stat ROI : None/OK")
infoROI.grid(row=2, column=2)

infoText = """
Informasi : 

~ On Frame ~
Mobil : 10
Motor : 20

~ Object Counted ~
Mobil : 20
Motor : 10
"""
detailInfo = Message(rootWindow, text=infoText, aspect=200,
                     justify="left")
detailInfo.grid(row=3, column=2, sticky="w")

progressBar = ttk.Progressbar(rootWindow, orient=HORIZONTAL,
                              length=400, mode='determinate')
progressBar.grid(row=4, column=0, columnspan=2)
progressBar['value'] = 50

stepFrame = Label(rootWindow, text="Frame 59 of 69", pady=10)
stepFrame.grid(row=5, column=0, sticky=W)

lastSpeed = Label(rootWindow, text="Last Speed Detected : 2.5123 s")
lastSpeed.grid(row=6, column=0, sticky=W)

rootWindow.mainloop()
