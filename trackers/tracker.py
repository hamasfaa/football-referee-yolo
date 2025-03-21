from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
import pandas as pd
import numpy as np
sys.path.append('../')
from utils import get_center_of_bbox, get_width_of_bbox

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
    def interpolate_ball_position(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[])for x in ball_positions if x is not None]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions=df_ball_positions.bfill()
        
        ball_positions = [{1:{'bbox':x}}for x in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions
        
    def detect_frames(self, frames):
        batch_size=20
        detections=[]
        for i in range(0,len(frames),batch_size):
            detections+=self.model.predict(frames[i:i+batch_size], conf=0.1)
        return detections
    
    def get_object_track(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        
        detections= self.detect_frames(frames)
        
        tracks={
            "players": [], # list of dictionaries, each dictionary contains the tracks of players in a frame
            "referees": [],
            "ball": [],
        }
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            # print(cls_names)
            cls_names_inv = {v: i for i, v in cls_names.items()}
            # print(cls_names_inv)
            
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind]=cls_names_inv['player']
                    
            detection_with_tracks= self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id]={"bbox": bbox}
                    
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id]={"bbox": bbox}
                    
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1]={"bbox": bbox}
                
            # print(detection_with_tracks)
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
            
        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])
        
        x_center, y_center = get_center_of_bbox(bbox)
        width = get_width_of_bbox(bbox)
        
        cv2.ellipse(frame, center=(x_center, y2),axes=(int(width), int(0.35*width)), angle=0, startAngle=-45 , endAngle=235, color=color, thickness=2, lineType=cv2.LINE_4)
        
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2)+15
        y2_rect = (y2 + rectangle_height//2)+15
        
        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
            x1_text = x1_rect + 12
            if track_id > 100:
                x1_text -=10
            
            cv2.putText(frame, str(track_id), (int(x1_text), int(y2_rect-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)     
        
        return frame
    
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        
        triangle_points = np.array([[x, y], [x-10, y-20], [x+10, y-20]])
        
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0),2)
        
        return frame
        
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        team_ball_control_frame = team_ball_control[:frame_num+1]
        team_1_num = team_ball_control_frame[team_ball_control_frame==1].shape[0]
        team_2_num = team_ball_control_frame[team_ball_control_frame==2].shape[0]
        
        team_1 = team_1_num/(team_1_num+team_2_num)
        team_2 = team_2_num/(team_1_num+team_2_num)
        
        cv2.putText(frame, f"Team 1: {team_1*100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Team 2: {team_2*100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        return frame
    
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            
            # Draw
            for track_id, player in player_dict.items():
                # bbox = player["bbox"]
                # frame = sv.draw_bbox(frame, bbox, (0, 255, 0))
                color= player.get("team_color", (0,255,255))
                team_id = player.get("team_id", 0)
                frame = self.draw_ellipse(frame, player["bbox"], color, team_id)
                
                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (255, 0, 0))
                
            for _, referee in referee_dict.items():
                # bbox = referee["bbox"]
                # frame = sv.draw_bbox(frame, bbox, (0, 0, 255))
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 0, 255), None)
                
            for _, ball in ball_dict.items():
                # bbox = ball["bbox"]
                # frame = sv.draw_bbox(frame, bbox, (255, 0, 0))
                frame = self.draw_triangle(frame, ball["bbox"], (255, 0, 0))
            
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            
            output_video_frames.append(frame)
        
        return output_video_frames
            
