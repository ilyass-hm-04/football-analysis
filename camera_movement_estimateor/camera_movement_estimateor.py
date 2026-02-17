import cv2
import numpy as np
import os
import pickle
import sys 
sys.path.append('../')
from utils import measure_distance,measure_xy_distance

class CameraMovementEstimateor():
    def __init__(self,frame):

        self.mindistance = 5


        self.lk_params =dict(
            winSize =(15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20] = 1
        mask_features[:,900:1050] = 1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 8,
            mask = mask_features
        )
    

    def add_adjust_positions_to_tracks(self,tracks,camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted
        

    def get_camera_movement(self,frames,read_from_stub=False,stub_path=None):
        #Read stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f :
                return pickle.load(f)

        camera_movement = [[0,0]]*len(frames)

        old_gray = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
        old_features  = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num],cv2.COLOR_BGR2GRAY)
            new_features, _,_ = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_features,None,**self.lk_params)
            

            max_distance = 0
            camera_movement_x , camera_movement_y = 0,0

            for i , (old,new) in enumerate(zip(old_features,new_features)):
                new_features_points = new.ravel()
                old_features_points = old.ravel()

                distance = measure_distance(old_features_points,new_features_points)
                if distance >max_distance:
                    max_distance = distance
                    camera_movement_x , camera_movement_y = measure_xy_distance(old_features_points,new_features_points)
            
            if max_distance> self.mindistance:
                camera_movement[frame_num] = [camera_movement_x,camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray,**self.features)
            
            if frame_gray is None or not isinstance(frame_gray, np.ndarray):
                raise ValueError("The 'frame' is invalid or not a NumPy array.")
            
            old_gray = frame_gray.copy()
        
        if stub_path is not None :
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)

        return camera_movement
    
    def draw_camera_movement(self,frames,camera_movement_per_frame):
        output_frames = []
        for fram_num , frame in enumerate(frames):
            if frame is None or not isinstance(frame, np.ndarray):
                raise ValueError("The 'frame' is invalid or not a NumPy array.")

            # Reduce resolution if memory is a concern
            if frame.shape[0] > 720 or frame.shape[1] > 1280:  # Example threshold
                frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))  # Resize to half

            # Ensure frame is uint8 (minimize memory usage)
            overlay = frame.astype(np.uint8)

            cv2.rectangle(overlay,(0,0),(300,60),(255,255,255),-1)
            alpha = 0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement , y_movement = camera_movement_per_frame[fram_num]
            frame =cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            frame = cv2.putText(frame, f"Camera Movement Y : {y_movement:.2f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            output_frames.append(frame)
        
        return output_frames