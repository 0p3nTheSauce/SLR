import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

class MediaPipeVideoAnalyzer:
    def __init__(self):
        """Initialize MediaPipe solutions"""
        self.mp_drawing = mp.solutions.drawing_utils # type: ignore
        self.mp_drawing_styles = mp.solutions.drawing_styles # type: ignore
        
        # Initialize different MediaPipe solutions
        self.mp_pose = mp.solutions.pose # type: ignore
        self.mp_hands = mp.solutions.hands # type: ignore
        self.mp_face_mesh = mp.solutions.face_mesh # type: ignore
        self.mp_holistic = mp.solutions.holistic # type: ignore
        
        # Initialize pose estimation
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        # Initialize hand tracking
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Initialize face mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize holistic (combined solution)
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Data storage for visualization
        self.pose_data = []
        self.hand_data = []
        self.face_data = []
        
    def process_video_pose(self, video_path, output_path=None, show_live=True):
        """Process video with pose estimation"""
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        self.pose_data = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            # Extract pose landmarks
            if results.pose_landmarks:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
                self.pose_data.append(landmarks)
                
                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            # Add frame info
            cv2.putText(frame, f'Frame: {frame_count}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if output_path:
                out.write(frame) # type: ignore
            
            if show_live:
                cv2.imshow('MediaPipe Pose', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
        
        cap.release()
        if output_path:
            out.release() # type: ignore
        cv2.destroyAllWindows()
        
        print(f"Processed {frame_count} frames")
        print(f"Pose data collected for {len(self.pose_data)} frames")
        
        return self.pose_data
    
    def process_video_hands(self, video_path, output_path=None, show_live=True):
        """Process video with hand tracking"""
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        self.hand_data = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Extract hand landmarks
            if results.multi_hand_landmarks:
                frame_hands = []
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_points = []
                    for lm in hand_landmarks.landmark:
                        hand_points.append([lm.x, lm.y, lm.z])
                    frame_hands.append(hand_points)
                    
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                self.hand_data.append(frame_hands)
            else:
                self.hand_data.append([])
            
            cv2.putText(frame, f'Frame: {frame_count}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if output_path:
                out.write(frame) # type: ignore
            
            if show_live:
                cv2.imshow('MediaPipe Hands', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
        
        cap.release()
        if output_path:
            out.release() # type: ignore
        cv2.destroyAllWindows()
        
        return self.hand_data
    
    def process_video_holistic(self, video_path, output_path=None, show_live=True):
        """Process video with holistic solution (pose + hands + face)"""
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        holistic_data = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb_frame)
            
            # Store frame data
            frame_data = {
                'pose': None,
                'left_hand': None,
                'right_hand': None,
                'face': None
            }
            
            # Draw pose
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
                pose_points = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
                frame_data['pose'] = pose_points # type: ignore
            
            # Draw hands
            if results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                left_hand_points = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
                frame_data['left_hand'] = left_hand_points # type: ignore
            
            if results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                right_hand_points = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
                frame_data['right_hand'] = right_hand_points # type: ignore
            
            # Draw face
            if results.face_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS)
                face_points = [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]
                frame_data['face'] = face_points # type: ignore
            
            holistic_data.append(frame_data)
            
            cv2.putText(frame, f'Frame: {frame_count}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if output_path:
                out.write(frame) # type: ignore
            
            if show_live:
                cv2.imshow('MediaPipe Holistic', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
        
        cap.release()
        if output_path:
            out.release() # type: ignore
        cv2.destroyAllWindows()
        
        return holistic_data
    
    def visualize_pose_data(self, pose_data, save_plot=False):
        """Visualize pose data with matplotlib"""
        if not pose_data:
            print("No pose data to visualize")
            return
        
        # Extract key joint positions over time
        frames = len(pose_data)
        
        # MediaPipe pose landmark indices
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        
        # Extract coordinates
        left_wrist_x = [frame[LEFT_WRIST][0] for frame in pose_data]
        left_wrist_y = [frame[LEFT_WRIST][1] for frame in pose_data]
        right_wrist_x = [frame[RIGHT_WRIST][0] for frame in pose_data]
        right_wrist_y = [frame[RIGHT_WRIST][1] for frame in pose_data]
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Wrist trajectories
        axes[0, 0].plot(left_wrist_x, left_wrist_y, 'b-', label='Left Wrist', alpha=0.7)
        axes[0, 0].plot(right_wrist_x, right_wrist_y, 'r-', label='Right Wrist', alpha=0.7)
        axes[0, 0].set_title('Wrist Trajectories')
        axes[0, 0].set_xlabel('X Position')
        axes[0, 0].set_ylabel('Y Position')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].invert_yaxis()  # Invert Y to match image coordinates
        
        # Plot 2: X positions over time
        axes[0, 1].plot(range(frames), left_wrist_x, 'b-', label='Left Wrist X')
        axes[0, 1].plot(range(frames), right_wrist_x, 'r-', label='Right Wrist X')
        axes[0, 1].set_title('X Positions Over Time')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('X Position')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Y positions over time
        axes[1, 0].plot(range(frames), left_wrist_y, 'b-', label='Left Wrist Y')
        axes[1, 0].plot(range(frames), right_wrist_y, 'r-', label='Right Wrist Y')
        axes[1, 0].set_title('Y Positions Over Time')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Y Position')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Distance between wrists
        wrist_distance = [np.sqrt((lx - rx)**2 + (ly - ry)**2) 
                         for lx, ly, rx, ry in zip(left_wrist_x, left_wrist_y, 
                                                  right_wrist_x, right_wrist_y)]
        axes[1, 1].plot(range(frames), wrist_distance, 'g-', linewidth=2)
        axes[1, 1].set_title('Distance Between Wrists')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Distance')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('pose_analysis.png', dpi=300, bbox_inches='tight')
            print("Plot saved as 'pose_analysis.png'")
        
        plt.show()
    
    def create_pose_heatmap(self, pose_data, save_plot=False):
        """Create a heatmap showing pose landmark positions"""
        if not pose_data:
            print("No pose data to visualize")
            return
        
        # Create heatmap data
        heatmap_data = np.zeros((33, len(pose_data)))  # 33 pose landmarks
        
        for frame_idx, frame_data in enumerate(pose_data):
            for landmark_idx, landmark in enumerate(frame_data):
                # Use visibility as intensity
                heatmap_data[landmark_idx, frame_idx] = landmark[3]  # visibility
        
        # Create the heatmap
        plt.figure(figsize=(15, 8))
        plt.imshow(heatmap_data, cmap='hot', interpolation='nearest', aspect='auto')
        plt.colorbar(label='Visibility')
        plt.title('Pose Landmarks Visibility Heatmap')
        plt.xlabel('Frame')
        plt.ylabel('Landmark Index')
        
        # Add landmark labels
        landmark_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]
        
        plt.yticks(range(33), landmark_names[:33], fontsize=8)
        
        if save_plot:
            plt.savefig('pose_heatmap.png', dpi=300, bbox_inches='tight')
            print("Heatmap saved as 'pose_heatmap.png'")
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = MediaPipeVideoAnalyzer()
    
    # Example 1: Process video with pose estimation
    print("Processing video with pose estimation...")
    video_path = "00333.mp4"  # Replace with your video path
    
    # Process video (set show_live=False if you don't want live preview)
    pose_data = analyzer.process_video_pose(
        video_path, 
        output_path="pose_output.mp4", 
        show_live=False
    )
    
    # Visualize the results
    if pose_data:
        analyzer.visualize_pose_data(pose_data, save_plot=True)
        analyzer.create_pose_heatmap(pose_data, save_plot=True)
    
    # Example 2: Process video with hand tracking
    print("Processing video with hand tracking...")
    hand_data = analyzer.process_video_hands(
        video_path, 
        output_path="hands_output.mp4", 
        show_live=False
    )
    
    # Example 3: Process video with holistic solution
    print("Processing video with holistic solution...")
    holistic_data = analyzer.process_video_holistic(
        video_path, 
        output_path="holistic_output.mp4", 
        show_live=False
    )
    
    print("Video processing complete!")
    print("Check the output files and generated plots.")