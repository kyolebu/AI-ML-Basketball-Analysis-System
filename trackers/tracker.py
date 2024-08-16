from ultralytics import YOLO
import supervision as sv 
import pickle
import os
import sys
import cv2
sys.path.append("../")
from utils import get_bbox_width, get_center_of_bbox

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)



    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf = 0.1)
            detections += detections_batch
        return detections
    


    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,"rb") as f:
                tracks = pickle.load(f)
            return tracks

        # get detections to get tracking
        detections = self.detect_frames(frames)

        tracks = {
            "Player":[],  # for each frame, it has the players and their bounding box
            "Ref":[],
            "Ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            tracks["Player"].append({})
            tracks["Ref"].append({})
            tracks["Ball"].append({})

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['Player']:
                    tracks["Player"][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_names_inv['Ref']:
                    tracks["Ref"][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_names_inv['Ball']:
                    tracks["Ball"][frame_num][3] = {"box":bbox}
        
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks,f)

        return tracks
    


    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        return frame


    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["Player"][frame_num]
            ref_dict = tracks["Ref"][frame_num]
            ball_dict = tracks["Ball"][frame_num]

            # draw players
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], (0,0,255), track_id)


            output_video_frames.append(frame)
        return output_video_frames

