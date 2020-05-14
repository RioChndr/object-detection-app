from brain import obj_detector
import cv2 as cv

class TrackerSystem(obj_detector):
    objects_tracked = []
    frame_for_detect = 1
    number_frame = 0  # Iteration
    tracker_type = "CSRT"

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

    def is_inside_box(box_1, box_2):
        x1, y1, w1, h1 = box_1
        x2, y2, w2, h2 = box_2

        center2_x = (w2 / 2) + x2
        center2_y = (h2 / 2) + y2
        # print("box_1 = {}, box_2 = {}".format(box_1, box_2))
        # print("center 2 = {}, {}".format(center2_x, center2_y))
        if center2_x > x1 and center2_x < w1 + x1:
            if center2_y > y1 and center2_y < h1 + y1:
                return True
        return False

    def is_out_frame(self, bbox):
        # Init
        f_width, f_height, channels = self.photo_target

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

                # Check, is current object is collapsed with object tracked,
                # if so, then delete it without check is object has different class id
                i = 0
                for obj_tracked in self.objects_tracked:
                    pos_obj_tracked = obj_tracked[3]

                    is_there = self.is_inside_box(pos_obj_tracked, pos_obj)
                    if is_there:
                        del self.objects_tracked[i]
                    i += 1
                t_cv = self.set_tracker(name=self.tracker_type)
                self.append_obj_tracker(
                    t_cv, self.photo_target, pos_obj, class_id)

        return

    def update_track_object(self):
        if len(self.objects_tracked) > 0:
            i = 0
            for obj in self.objects_tracked:
                tracker, class_id, bbox = obj
                # Update tracker system
                ok, new_bbox = tracker.update(self.photo_target)
                if ok:
                    is_out = self.is_out_frame(new_bbox)
                    if is_out:
                        del self.objects_tracked[i]
                    else:
                        self.objects_tracked[i][2] = new_bbox
                i += 1
        return

    # Pos_y is position Y for ROI, area can be TOP/BOTTOM

    def set_ROI(self, pos_y, area=None):
        pass

    def check_objects_trough_ROI(self):
        pass

    def is_trough_ROI(self, object):
        pass

    def append_obj_tracker(self, tracker, img, bbox, class_id_label):
        tracker.init(img, tuple(bbox))
        self.objects_tracked.append([tracker, class_id_label, bbox])

    def extract_frame(self, frame, number_frame):
        self.number_frame = number_frame
        self.photo_target = frame

        if self.number_frame == 0 or number_frame % self.frame_for_detect == 0:
            self.detect_obj_frame()
        else:
            self.update_track_object()
        self.label_obj_tracker()

    def label_obj_tracker(self):
        for obj in self.objects_tracked:
            tracker, class_id, bbox = obj
            class_label = self.classes[class_id]
            color_label = self.colors[class_id]
            self.label_img(bbox, class_label, color_label)
            if class_label in self.listObj:
                self.listObj[class_label] += 1
            else:
                self.listObj[class_label] = 1
