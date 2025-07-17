import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import utils

import os

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
def draw_pose(frames, pose_info):
  '''draw pose info on frames (np.uint8, T H W C, BGR)'''
  for frame, results in frames, pose_info:
    #Face
    if results.face_landmarks:  
      mp_drawing.draw_landmarks(
        frame,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
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
    
#########Testing functions
def test_get_draw():
  frames = utils.cv_load('./69241.mp4', 0, 10, all=True)
  pose_info = get_pose(frames, writeable=True)
  frames = draw_pose(frames, pose_info)
  path = '~/Code/SSH_files'
  if os.path.exists(path):
    path = os.path.join(path, 'test.mp4')
    utils.save_video(frames, '~/Code/SSH_files/test.mp4')
  else:
    print('Path does not exist:', path)
############ Finished tests
def main():
  test_get_draw()
  print('Test completed')

if __name__ == '__main__':
  main()