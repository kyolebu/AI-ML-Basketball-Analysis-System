from utils import read_video, save_video
from trackers import Tracker

def main():
    # read video
    video_frames = read_video('input_videos/2024_Finals_Game_5.mp4')

    # initialize Tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames)

    # save video
    save_video(video_frames, 'output_videos/output_video.avi')

    
if __name__ == '__main__':
    main()