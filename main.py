from utils import save_video, read_video
from trackers import Tracker
from team_assigner import TeamAssigner
import cv2

def main():
    # Read Video
    vdo_frame = read_video('vdos/08fd33_4.mp4')

    # Initializing traker
    traker = Tracker('models/best.pt')

    # Getting tracker object
    tracks = traker.get_object_tracks(vdo_frame, read_from_stub=True, stub_path= 'stubs/track_stub.pkl')


    # Assigning Team
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(vdo_frame[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(vdo_frame[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]


    ### Saving a single player image
    for track_id, player in tracks['players'][0].items():
        bbox = player['bbox']
        frame = vdo_frame[0]

        saved_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Saving image
        cv2.imwrite(f'output_videos/player_img.jpg', saved_img)
        break # as I want only one image

    # Drawing annotation
    output_vdo_frame = traker.draw_annotations(vdo_frame, tracks)

    # Save Video
    save_video(output_vdo_frame, 'output_videos/output.avi')

if __name__ == '__main__':
    main()