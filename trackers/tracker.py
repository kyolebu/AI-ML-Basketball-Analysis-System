from ultralytics import YOLO
import supervision as sv 

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.track(frames[i:i+batch_size], conf = 0.1)
            detections += detections_batch
            break
        return detections
    
    def get_object_tracks(self, frames):
        # get detections to get tracking
        detections = self.detect_frames(frames)

        tracks = {
            "players":[],
            "referees":[],
            "ball":[]
        }


        for frame_num, detection in enumerate(detections):
            # convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            break