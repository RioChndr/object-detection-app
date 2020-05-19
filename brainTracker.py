from brain import obj_detector
import cv2 as cv
import copy


class TrackerSystem(obj_detector):
    objects_tracked = []
    objects_counted = {}
    frame_for_detect = 20
    count_frame = 0  # Iteration
    tracker_type = "CSRT"
    box_ROI = None

    def __init__(self, weightModel, configModel, namesModel):
        super().__init__(weightModel, configModel, namesModel)

    # RUN THIS TRACKER !!
    def track_object_inframe(self, frame, count_frame):
        self.count_frame = count_frame
        self.photo_target = frame

        timer = cv.getTickCount()
        if self.count_frame == 1 or count_frame % self.frame_for_detect == 0:
            self.detect_object()
        else:
            self.update_track_object()
        self.check_objects_trough_ROI()
        self.label_obj_tracker()
        self.show_ROI_box()
        self.time_process = (cv.getTickCount() - timer) / cv.getTickFrequency()
        return self.get_image()

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

    def is_inside_box(self, parent_box, child_box):
        parent = {}
        child = {}
        parent['x'], parent['y'], parent['w'], parent['h'] = parent_box
        child['x'], child['y'], child['w'], child['h'] = child_box

        centerChildX = (child['w'] / 2) + child['x']
        centerChildY = (child['h'] / 2) + child['y']
        if centerChildX > parent['x'] and centerChildX < parent['w'] + parent['x']:
            if centerChildY > parent['y'] and centerChildY < parent['h'] + parent['y']:
                return True
        return False

    def is_out_frame(self, bbox):
        # Init
        f_height, f_width, channels = self.photo_target.shape

        f_width = f_width - (f_width * .005)  # dikurangi 2 persen
        f_height = f_height - (f_height * .005)  # dikurangi 2 persen
        x1, y1, w1, h1 = bbox
        if x1 < 0 or w1 + x1 > f_width or y1 < 0 or h1 + y1 > f_height:
            return True
        return False

    def detect_object(self):
        detected_obj = self.detect_obj(frame=self.photo_target)

        if len(self.objects_tracked) == 0:
            # First Object added
            for current_obj in detected_obj:
                class_id, label, boxes = current_obj
                t_cv = self.set_tracker(name=self.tracker_type)
                self.append_obj_tracker(
                    t_cv, self.photo_target, boxes, class_id)
        else:
            i = 0
            for current_obj in detected_obj:
                class_id, label, pos_obj = current_obj

                # find the nearest point/coordinate
                i = 0
                old_object = [None, None, None, None]
                old_object[3] = False  # Default value
                obj_collapsed = self.is_collapse_other_object(
                    i)

                old_object = obj_collapsed

                t_cv = self.set_tracker(name=self.tracker_type)
                self.append_obj_tracker(
                    t_cv, self.photo_target, pos_obj, class_id, old_object[3])
                i += 1
        return

    def update_track_object(self):
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
                        self.is_collapse_other_object(i)
                i += 1
        return True

    def is_collapse_other_object(self, id_object):
        id_other_obj = 0
        object_collapsed = [None, None, None, None]  # Default format object
        pos_current_obj = self.objects_tracked[id_object][2]

        for other_obj in self.objects_tracked:
            if id_object == id_other_obj:
                continue
            pos_other_obj = other_obj[2]
            is_collapse = self.is_inside_box(
                pos_other_obj, pos_current_obj)
            if is_collapse:
                object_collapsed = self.objects_tracked[id_other_obj]
                del self.objects_tracked[id_other_obj]
                break
            id_other_obj += 1
        return object_collapsed

    def append_obj_tracker(self, tracker, img, bbox, class_id_label, is_counted=False):
        tracker.init(img, tuple(bbox))
        self.objects_tracked.append(
            [tracker, class_id_label, bbox, is_counted])

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
