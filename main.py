from utils import save_video, read_video
from trackers import Tracker

def main():
    # Read Video
    vdo_frame = read_video('vdos/08fd33_4.mp4')

    # Initializing traker
    traker = Tracker('models/best.pt')

    # Getting tracker object
    tracks = traker.get_object_tracks(vdo_frame, read_from_stub=True, stub_path= 'stubs/track_stub.pkl')

    # Drawing annotation
    output_vdo_frame = traker.draw_annotations(vdo_frame, tracks)

    # Save Video
    save_video(output_vdo_frame, 'output_videos/output.avi')

if __name__ == '__main__':
    main()