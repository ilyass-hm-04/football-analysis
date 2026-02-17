from ultralytics import YOLO
import supervision as sv
import pickle
import cv2
import os
import sys
sys.path.append("../")
from utils import get_centre_bbox , get_bbox_width,get_foot_position
import numpy as np
import pandas as pd


class Tracker :
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_centre_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position





    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        #interpolation

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
    
        return ball_positions

    def detect_frames(self,frames):
        batch_size = 20
        detections = []
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections

    def get_object_trackers(self, frames,read_from_stub = False,stub_path = None):

        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,"rb") as f:
                traks = pickle.load(f)
            return traks

        detections = self.detect_frames(frames)
        
        Tracks = {
                "players" : [],
                "referees" : [],
                "ball" : [],
            }
        
        for frame_num , detection in enumerate(detections):
            cls_name = detection.names
            cls_name_inv = {v:k for k,v in cls_name.items()}

            #Convert detections to supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #Convert goalkeepr into a player
            for object_num , classe_id in enumerate(detection_supervision.class_id):
                if cls_name[classe_id] == 'goalkeeper':
                    detection_supervision.class_id[object_num] = cls_name_inv['player']

            #Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            Tracks["players"].append({})
            Tracks["referees"].append({})
            Tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                tracker_id = frame_detection[4]

                if cls_id == cls_name_inv['player']:
                    Tracks["players"][frame_num][tracker_id] = {"bbox" : bbox}
                
                if cls_id == cls_name_inv['referee']:
                    Tracks["referees"][frame_num][tracker_id] = {"bbox" : bbox}
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_name_inv['ball']:
                    Tracks["ball"][frame_num][1] = {"bbox" : bbox}


        #Save tracks
        if stub_path is not None:
            with open(stub_path,"wb") as f:
                pickle.dump(Tracks,f)
        return Tracks
            
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_centre , _ = get_centre_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(frame,
                    center=(x_centre,y2),
                    axes=(int(width),int(0.36*width)),
                    angle=0.0,
                    startAngle=-45,
                    endAngle=235,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_4)
        
        rectange_width = 40
        rectange_height = 20
        x1_rect = x_centre - rectange_width//2
        x2_rect = x_centre + rectange_width//2
        y1_rect = y2 - rectange_height//2 +20
        y2_rect = y2 + rectange_height//2 - 20

        if track_id is not None :
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect)),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED
                          )
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(frame,
                        f"{track_id}",
                        (int(x1_text),int(y2_rect+13)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0,0,0),
                        2
            )

        return frame
    
    def draw_triangle(self,frame,bbox,color):
        y = int(bbox[1])
        x,_ = get_centre_bbox(bbox)
        triangle_point = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
            ])
        cv2.drawContours(frame,[triangle_point],0,color,cv2.FILLED)
        cv2.drawContours(frame,[triangle_point],0,(0,0,0),2)
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Ensure 'frame' is valid
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("The 'frame' is invalid or not a NumPy array.")
        
        # Ensure frame dimensions are correct (3 channels for RGB or BGR)
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError("The 'frame' should be a 3-channel image (RGB or BGR).")
        
        # Draw a semi-transparent rectangle
        overlay = frame.copy()  # Ensure this operation works
        
        # Define rectangle coordinates and transparency
        top_left = (1350, 850)
        bottom_right = (1900, 970)
        cv2.rectangle(overlay, top_left, bottom_right, (255, 255, 255), -1)
        alpha = 0.4  # Transparency level
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Calculate ball control statistics
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_num_frames = np.sum(team_ball_control_till_frame == 1)
        team_2_num_frames = np.sum(team_ball_control_till_frame == 2)

        # Avoid division by zero
        total_frames = team_1_num_frames + team_2_num_frames
        if total_frames == 0:
            team_1, team_2 = 0.0, 0.0
        else:
            team_1 = team_1_num_frames / total_frames
            team_2 = team_2_num_frames / total_frames
        
        # Add text to the frame
        cv2.putText(frame, f"Team 1 Ball Control : {team_1 * 100:.2f}%", (1400, 900),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control : {team_2 * 100:.2f}%", (1400, 950),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotaions(self,video_frames,tracks,team_ball_control):
        output_video_frames = []
        for frame_num,frame in enumerate(video_frames):

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            #Draw players
            for track_id,player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame,player["bbox"],color,track_id)

                if player.get('has_ball',False):
                    frame = self.draw_triangle(frame,player['bbox'],(0,0,255))

            #Draw referees
            for track_id,referee in referee_dict.items():
                frame = self.draw_ellipse(frame,referee["bbox"],(0,255,255))
            
            #Draw ball
            for track_id,ball in ball_dict.items():
                frame = self.draw_triangle(frame,ball["bbox"],(0,255,0))

            #Draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)


            output_video_frames.append(frame)
        
        return output_video_frames
        