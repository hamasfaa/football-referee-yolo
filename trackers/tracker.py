from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
    def detect_frames(self, frames):
        batch_size=20
        detections=[]
        for i in range(0,len(frames),batch_size):
            detections+=self.model.predict(frames[i:i+batch_size], conf=0.1)
            break
        return detections
    
    def get_object_track(self, frames):
        detections= self.detect_frames(frames)
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            # print(cls_names)
            cls_names_inv = {v: i for i, v in cls_names.items()}
            # print(cls_names_inv)
            
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind]=cls_names_inv['player']
                    
            detection_with_traks= self.tracker.update_with_detections(detection_supervision)
            
            print(detection_with_traks)