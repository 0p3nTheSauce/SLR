import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import utils

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def get_pose(frames):
  '''get pose info from frames (np.uint8, T H W C, BGR)'''
  with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    return [holistic.process(
      cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    
## IO
def draw_pose(frames, pose_info):
  '''draw pose info on frames (np.uint8, T H W C, BGR)'''
  for frame, results in frames, pose_info:
    #Face
    mp_drawing.draw_landmarks(
      frame,
      results.face_landmarks,
      mp_holistic.FACEMESH_CONTOURS,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp_drawing_styles
      .get_default_face_mesh_contours_style())
    #Pose
    mp_drawing.draw_landmarks(
      frame,
      results.pose_landmarks,
      mp_holistic.POSE_CONNECTIONS,
      landmark_drawing_spec=mp_drawing_styles
      .get_default_pose_landmarks_style())
    
    

def main():
  pass

if __name__ == '__main__':
  main()