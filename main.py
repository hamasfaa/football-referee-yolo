from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner

def main():
    video_frames = read_video('input_videos/08fd33_4.mp4')
    
    
    tracker = Tracker('models/best.pt')
    
    tracks = tracker.get_object_track(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl') 
    
    # save sample img
    # for track_id, player in tracks["players"][0].items():
    #     bbox = player["bbox"]   
    #     frame = video_frames[0]
        
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
    #     cv2.imwrite(f"output_videos/player.jpg", cropped_image)
    #     break
    
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])
    
    for frame_number, player_track in enumerate(tracks["players"]):
        for player_id, player in player_track.items():
            team_id = team_assigner.get_player_team(video_frames[frame_number], player["bbox"], player_id)
            
            tracks["players"][frame_number][player_id]["team_id"] = team_id
            tracks["players"][frame_number][player_id]["team_color"] = team_assigner.team_colors[team_id]
            
    # Draw circles around the players and referees
    output_frames = tracker.draw_annotations(video_frames, tracks)
    
    save_video(output_frames, 'output_videos/output.avi')
    
if __name__ == "__main__":
    main()