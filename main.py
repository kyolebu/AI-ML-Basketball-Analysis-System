from utils import read_video, save_video
from trackers import Tracker

def main():
    # read video
    video_frames = read_video('input_videos/2024_Finals_Game_5.mp4')

    # initialize Tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')

    # draw output
    ## draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    
    # save video

    
if __name__ == '__main__':
    main()