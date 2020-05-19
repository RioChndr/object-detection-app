from brain import obj_detector
import cv2 as cv
import copy


class TrackerSystem(obj_detector):
    objects_tracked = []
    objects_counted = {}
    frame_for_detect = 20
    number_frame = 0  # Iteration
    tracker_type = "CSRT"
    box_ROI = None

    def __init__(self, weightModel, configModel, namesModel):
        super().__init__(weightModel, configModel, namesModel)

    def set_tracker(self, id=None, name=None):
        tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD',
                         'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        if id is not None:
            tracker_type = tracker_types[id]
        if name is not None:
            tracker_type = name

        if tracker_type == 'BOOSTING':
            tracker = cv.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv.TrackerCSRT_create()
        return tracker

    def set_frame_for_detect(self, frame_for_detect):
        # Every [frame_for_detect] frame, do detect
        self.frame_for_detect = frame_for_detect

    def is_inside_box(self, child_boxes, parent_box):
        x1, y1, w1, h1 = child_boxes
        x2, y2, w2, h2 = parent_box

        center2_x = (w2 / 2) + x2
        center2_y = (h2 / 2) + y2
        # print("child_boxes = {}, parent_box = {}".format(child_boxes, child_box))
        # print("center 2 = {}, {}".format(center2_x, center2_y))
        if center2_x > x1 and center2_x < w1 + x1:
            if center2_y > y1 and center2_y < h1 + y1:
                return True
        return False

    def is_out_frame(self, bbox):
        # Init
        f_height, f_width, channels = self.photo_target.shape

        f_width = f_width - (f_width * .005)  # dikurangi 2 persen
        f_height = f_height - (f_height * .005)  # dikurangi 2 persen
        x1, y1, w1, h1 = bbox
        center_x = (w1 / 2) + x1
        center_y = (h1 / 2) + y1
        if x1 < 0 or w1 + x1 > f_width or y1 < 0 or h1 + y1 > f_height:
            return True
        return False

    def detect_obj_frame(self):
        detected_obj = self.detect_obj(frame=self.photo_target)

        if len(self.objects_tracked) == 0:
            for obj in detected_obj:
                class_id, label, boxes = obj
                t_cv = self.set_tracker(name=self.tracker_type)
                self.append_obj_tracker(
                    t_cv, self.photo_target, boxes, class_id)
        else:
            for obj in detected_obj:
                class_id, label, pos_obj = obj

                # find the nearest point/coordinate
                i = 0
                old_object = [None, None, None, None]
                old_object[3] = False  # Default value
                for obj_tracked in self.objects_tracked:
                    pos_obj_tracked = obj_tracked[2]

                    is_there = self.is_inside_box(pos_obj_tracked, pos_obj)
                    if is_there:
                        old_object = self.objects_tracked[i]
                        del self.objects_tracked[i]
                    i += 1
                t_cv = self.set_tracker(name=self.tracker_type)
                self.append_obj_tracker(
                    t_cv, self.photo_target, pos_obj, class_id, old_object[3])

        return

    def update_track_object(self):
        delete_id_object = []
        if len(self.objects_tracked) > 0:
            i = 0
            for obj in self.objects_tracked:
                tracker, class_id, bbox, is_counted = obj
                # Update tracker system
                ok, new_bbox = tracker.update(self.photo_target)
                if ok:
                    is_out = self.is_out_frame(new_bbox)
                    if is_out:
                        del self.objects_tracked[i]
                    else:
                        self.objects_tracked[i][2] = new_bbox

                        j = 0
                        pos_current_obj = self.objects_tracked[i][2]
                        for other_obj in self.objects_tracked:
                            if i == j:
                                continue
                            pos_other_obj = other_obj[2]
                            is_collapse = self.is_inside_box(
                                pos_other_obj, pos_current_obj)
                            if is_collapse:
                                print("hapus id : {} ".format(j))
                                del self.objects_tracked[j]
                            j += 1
                i += 1
        return

    def append_obj_tracker(self, tracker, img, bbox, class_id_label, is_counted=False):
        tracker.init(img, tuple(bbox))
        self.objects_tracked.append(
            [tracker, class_id_label, bbox, is_counted])

    # RUN THIS TRACKER !!
    def extract_frame(self, frame, number_frame):
        self.number_frame = number_frame
        self.photo_target = frame

        timer = cv.getTickCount()
        if self.number_frame == 1 or number_frame % self.frame_for_detect == 0:
            self.detect_obj_frame()
        else:
            self.update_track_object()
        self.check_objects_trough_ROI()
        self.label_obj_tracker()
        self.show_ROI_box()
        self.time_process = (cv.getTickCount() - timer) / cv.getTickFrequency()
        return self.get_image()

    def label_obj_tracker(self):
        self.listObj = {}
        for obj in self.objects_tracked:
            tracker, class_id, bbox, is_counted = obj
            class_label = self.classes[class_id]
            color_label = self.colors[class_id]
            if is_counted == True:
                color_label = (237, 224, 36)  # Yellow
                class_label = None
            self.label_img(bbox, class_label, color_label)
            if class_label in self.listObj:
                self.listObj[class_label] += 1
            else:
                self.listObj[class_label] = 1

    # Pos_y is position Y for ROI, area can be TOP/BOTTOM

    def set_ROI(self, frame):
        ori_frame = copy.deepcopy(frame)  # For copy frame,
        font = cv.FONT_HERSHEY_PLAIN
        color = (200, 200, 0)
        txt_info = "Beri kotak ROI dengan klik beberapa ruang, lalu tekan enter jika selesai"
        cv.putText(frame, txt_info, (10, 10), font, 1, color)
        self.box_ROI = cv.selectROI(frame)
        cv.waitKey(1)
        cv.destroyAllWindows()

        return self.show_ROI_box(ori_frame)

    def check_objects_trough_ROI(self):

        for i in range(len(self.objects_tracked)):
            if self.objects_tracked[i][3] == True:
                continue
            if self.is_trough_ROI(self.objects_tracked[i]) == True:
                self.objects_tracked[i][3] = True  # Is in ? YES !!
                self.add_object_counted(self.objects_tracked[i])

    def is_trough_ROI(self, obj):
        return self.is_inside_box(self.box_ROI, obj[2])

    def show_ROI_box(self, frame=None):
        if frame is None:
            frame = self.photo_target
        color = (234, 237, 76)
        x, y, w, h = self.box_ROI
        p1 = (int(x), int(y))
        p2 = (int(x + w), int(y + h))

        cv.rectangle(frame, p1, p2, color, 2, 1)
        self.photo_target = frame
        return frame

    def add_object_counted(self, obj):
        tracker, class_id, bbox, is_counted = obj
        class_label = self.classes[class_id]
        color_label = self.colors[class_id]
        if class_label in self.objects_counted:
            self.objects_counted[class_label] += 1
        else:
            self.objects_counted[class_label] = 1
