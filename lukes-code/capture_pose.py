import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import utils

import os
import json

mp_drawing = mp.solutions.drawing_utils # type: ignore
mp_drawing_styles = mp.solutions.drawing_styles # type: ignore
mp_holistic = mp.solutions.holistic # type: ignore

def get_pose(frames, writeable=False):
  '''get pose info from frames (np.uint8, T H W C, BGR)'''
  with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    if writeable:
      return [holistic.process(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    else:
      # faster?
      results = []
      for frame in frames:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        result = holistic.process(image)
        results.append(result)
        image.flags.writeable = True
      return results    
    
## IO
def draw_pose_mediapipe(frames, pose_info):
  '''draw pose info on frames (np.uint8, T H W C, BGR)'''
  # Custom drawing specifications for smaller, cleaner face landmarks
  face_landmark_style = mp_drawing.DrawingSpec(
    color=(0, 255, 0),  # Green color (BGR format)
    thickness=1,
    circle_radius=1     # Much smaller circles
  )
  
  face_connection_style = mp_drawing.DrawingSpec(
    color=(255, 255, 255),  # White connections
    thickness=1
  )

  for frame, results in zip(frames, pose_info):
    #Face
    if results.face_landmarks:  
      mp_drawing.draw_landmarks(
        frame,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=face_landmark_style,
        connection_drawing_spec=face_connection_style)
    #Pose
    if results.pose_landmarks:
      mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    # Left Hand
    if results.left_hand_landmarks:
      mp_drawing.draw_landmarks(
        frame,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS)
    # Right Hand
    if results.right_hand_landmarks:
      mp_drawing.draw_landmarks(
        frame,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS)
  return frames
    
def draw_pose_openpose(frame, pose_info):
  person = pose_info['people'][0]
  pose_keypoints2d = person['pose_keypoints_2d']
  face_keypoints2d = person['face_keypoints_2d']
  left_hand_keypoints2d = person['hand_left_keypoints_2d']
  right_hand_keypoints2d = person['hand_right_keypoints_2d']
  # print(f'Pose keypoints: {len(pose_keypoints2d)//3}, Face keypoints: {len(face_keypoints2d)//3}, '
  #       f'Left hand keypoints: {len(left_hand_keypoints2d)//3}, Right hand keypoints: {len(right_hand_keypoints2d)//3}')
  # # Draw pose keypoints
  for i in range(0, len(pose_keypoints2d), 3):
    x, y, confidence = pose_keypoints2d[i:i+3]
    print(f'Pose keypoint {i//3}: ({x}, {y}), confidence: {confidence}')
    if confidence > 0.1:  # Threshold for visibility
      cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)  # Green circle
  # Draw face keypoints
  for i in range(0, len(face_keypoints2d), 3):
    x, y, confidence = face_keypoints2d[i:i+3]
    if confidence > 0.1:  # Threshold for visibility
      cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)  # Blue circle
  # Draw left hand keypoints
  for i in range(0, len(left_hand_keypoints2d), 3):
    x, y, confidence = left_hand_keypoints2d[i:i+3]
    if confidence > 0.1:  # Threshold for visibility
      cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 0), -1)  # Yellow circle
  # Draw right hand keypoints
  for i in range(0, len(right_hand_keypoints2d), 3):
    x, y, confidence = right_hand_keypoints2d[i:i+3]
    if confidence > 0.1:  # Threshold for visibility
      cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 255), -1)  # Magenta circle
  return frame
    
#########Testing functions
def test_get_draw():
  frames = utils.cv_load('./media/69241.mp4', 0, 10, all=True)
  pose_info = get_pose(frames, writeable=True)
  frames = draw_pose_mediapipe(frames, pose_info)
  path = os.path.expanduser('~/Code/SSH_files')
  if os.path.exists(path):
    path = os.path.join(path, 'test.mp4')
    utils.save_video(frames, path)
  else:
    print('Path does not exist:', path)

def checkpose():

  '''evaluate existing pose data'''
  for _ in range(3): print('') # just for spacing
  pose_data_path = '../data/pose_per_individual_videos/00295/image_00001_keypoints.json'
  with open(pose_data_path, 'r') as f:
    pose_data = json.load(f)
  
  frames = utils.cv_load('../data/WLASL2000/00295.mp4', 0, 0, True)
  frame0001 = frames[0]
  pose_frame = draw_pose_openpose(frame0001, pose_data)
  out_path = './output/pose_frame_0001.jpg'
  cv2.imwrite(out_path, pose_frame)
  print(f'Saved pose frame to {out_path}')
    
############ Finished tests
def main():
  # test_get_draw()
  checkpose()
  print('Test completed')

if __name__ == '__main__':
  main()
