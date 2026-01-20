from typing import List, Union, Tuple
import torch
import cv2
from logging import Logger
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from pathlib import Path
import shutil
from argparse import ArgumentParser
import wandb
import time
import subprocess
from typing import Callable, Optional, Dict, Any
from multiprocessing.synchronize import Event as EventClass


################ GPU ###################

class gpu_manager:
	@classmethod
	def get_gpu_memory_usage(cls, gpu_id: int=0) -> Tuple[float, float]:
		"""Get GPU usage across all processes

		Args:
			gpu_id (int, optional): ID of GPU. Defaults to 0.

		Raises:
			RuntimeError: nvidia-smi failed with subprocess
			ValueError: Unexpected nvidia-smi output
			ValueError: Failed to parse GPU memory usage:

		Returns:
			Tuple[float, float]: used, total in GiB
		"""

		result = subprocess.run(
			[
				"nvidia-smi",
				f"--id={gpu_id}",
				"--query-gpu=memory.used,memory.total",
				"--format=csv,noheader,nounits",
			],
			capture_output=True,
			text=True,
		)
		if result.returncode != 0:
			raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
		output = result.stdout.strip()
		parts = output.split(",")
		if len(parts) != 2:
			raise ValueError(f"Unexpected nvidia-smi output: '{output}'")
		try:
			used, total = map(float, parts)
		except Exception as e:
			raise ValueError(f"Failed to parse GPU memory usage: '{output}'") from e

		return used / 1024, total / 1024  # In GB


	@classmethod
	def output(cls, logger: Optional[Logger], message: str) -> None:
		if logger is None:
			print(message)
		else:
			logger.info(message)

	@classmethod
	def wait_for_completion(
		cls,
		check_interval: int = 3600,  # 1 hour
		confirm_interval: int = 60,  # 1 minute
		num_checks: int = 5,  # confirm consistency over 5 minutes
		gpu_id: int = 0,
		max_util_gb: float = 1.0,  # Maximum memory usage in GB
		logger: Optional[Logger] = None,
		event: Optional[EventClass] = None
	) -> bool: 
		"""Wait for GPU memory to be free before proceeding

		Args:
				check_interval (int, optional): Wait period before checking again. Defaults to 3600 (1 hour).
				confirm_interval (int, optional): Wait period before checking again when confirming. Defaults to 60 (1 minute).
				num_checks: (int, optional): Confirm consistency over how many checks. Defaults to 5 (5 minutes).
				gpu_id (int, optional): CUDA GPU. Defaults to 0.
				max_util_gb (float, optional): Threshold GPU usage to trigger waiting. Defaults to 1.0 (GB).
				logger: (Logger, optional): Optionally supply a logger
				event: (Optional[EventClass], optional): Optionally supply a multiprocessing stopping event.
		Returns:
				bool: Whether monitoring was killed by the user, either through CTRL+C or stop event.
		"""
		assert torch.cuda.is_available(), "CUDA is not available"

		used, total = cls.get_gpu_memory_usage(gpu_id)

		proceed = used > max_util_gb

		while proceed:
			cls.output(logger, "")
			cls.output(logger,
				f"Monitoring GPU: {gpu_id}, current memory usage: {used:.2f}/{total:.2f} GB ({used / total * 100:.1f}%)"
			)
			cls.output(logger,f"Last checked at {time.strftime('%Y-%m-%d %H:%M:%S')}")
			try:
				time.sleep(check_interval)
				used, _ = cls.get_gpu_memory_usage(gpu_id)
			except KeyboardInterrupt:
				cls.output(logger,"")
				cls.output(logger,"Monitoring interrupted by user")
				return False

			if used <= max_util_gb and cls._confirm_usage(
				confirm_interval, num_checks, gpu_id, max_util_gb
			):
				break
			elif event is not None and event.is_set():
				cls.output(logger, "")
				cls.output(logger, "Stop event detected")
				return False

		cls.output(logger,"")
		cls.output(logger,
			f"GPU: {gpu_id} is available, current memory usage: {used:.2f}/{total:.2f} GB ({used / total * 100:.1f}%)"
		)

		return True

	@classmethod
	def _confirm_usage(
		cls,
		confirm_interval: int = 60,  # 1 minute
		num_checks: int = 5,  # confirm consistency over 5 minutes
		gpu_id: int = 0,
		max_util_gb: float = 1.0,
	) -> bool:
		for _ in range(num_checks):
			used, _ = cls.get_gpu_memory_usage(gpu_id)
			if used > max_util_gb:
				return False
			time.sleep(confirm_interval)
		return True

############## wandb ##################
class wandb_manager:
	@classmethod
	def get_run_id(
		cls, run_name, entity: str, project: str, idx: Optional[int] = None
	) -> Optional[str]:
		api = wandb.Api()

		runs = api.runs(f"{entity}/{project}")
		ids = []
		for run in runs:
			if run.name == run_name:
				ids.append(run.id)

		if len(ids) == 0:
			print(f"No runs found with name: {run_name}")
			return None
		elif len(ids) > 1:
			print(f"Multiple runs found with name: {run_name}")
			if isinstance(idx, int) and abs(idx) < len(runs):
				print(f"Returning id for idx: {idx}")
				return ids[idx]
			else:
				print("No idx supplied, returning None")
				return None
		else:
			return ids[0]

	@classmethod
	def list_runs(
		cls,
		entity: str,
		project: str,
		disp: bool = False,
	) -> list:
		"""
		List all runs for a given entity and project.

		Args:
			entity (str): The wandb entity.
			project (str): The wandb project.
			disp (bool, optional): If True, prints details of each run.

		Returns:
			list: A list of wandb run objects, not strings.
		"""
		api = wandb.Api()
		runs = api.runs(f"{entity}/{project}")

		if disp:
			for run in runs:
				print(f"Run ID: {run.id}")
				print(f"Run name: {run.name}")
				print(f"State: {run.state}")
				print(f"Created: {run.created_at}")
				print("---")

		return runs

	@classmethod
	def run_present(cls, run_id: str, runs: List) -> bool:
		return any([run.id == run_id for run in runs])

	@classmethod
	def validate_runId(cls, run_id: str, entity: str, project: str) -> bool:
		return cls.run_present(run_id, cls.list_runs(entity, project))

############### Input ##################

def ask_nicely(message: str,
							 requirment: Callable[[str], bool],
							 error: Optional[str] = None) -> str:
	passed=False
	ans = 'none'

	while not passed:
		passed=True
		ans = input(message)
		try:
			if not requirment(ans):
				print(error)
				passed=False
		except Exception as e:
			print(e)
			passed=False
	return ans

############# pretty printing ##############

def print_dicto(diction):
	print(string_nested_dict(diction))
	
def string_nested_dict(diction):
	ans = ""
	if type(diction) is dict:
		ans += "{\n"
		for key, value in diction.items():
			ans += f'{key} : {string_nested_dict(value)}\n'
		ans += "}\n"
	else:
		ans += str(diction)
	return ans


def str_dict(dic: Dict[str, Any], disp: bool = False) -> str:
	"""Print dictionary in a more readable format.

	Args:
		dic (Dict[str, Any]): Dictionary to print
	"""
	maxl_k = max([len(key) for key in dic.keys()])
	st = "{\n"
	for key in dic.keys():
		st += f"\t{key:<{maxl_k}} : {dic[key]}\n"
	st += "}"
	if disp:
		print(st)
	return st

def print_dict(diction):
	print(json.dumps(diction, indent=4))

################# Loading #####################
	
def load_rgb_frames_from_video(video_path : str, start : int, end : int
															, all : bool =False) -> torch.Tensor:
	return cv_to_torch(cv_load(video_path, start, end, all))

def cv_load(video_path:str|Path, start:int, end:int, all:bool=False) -> np.ndarray:
	video_path = Path(video_path)  
	if not video_path.exists():
		raise FileNotFoundError(f'File {video_path} does not exist')
	cap = cv2.VideoCapture(str(video_path))
	frames = []
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		frames.append(frame)
	if len(frames) > 0:
		if all:
			return np.asarray(frames) 
		else:
			return np.asarray(frames[start:end])
	else:
		raise ValueError(f"No frames were loaded for file {video_path}")

################## Saving #####################

def save_video(frames, path, fps=30):
	'''Arguments:
	frames : numpy tensor (T H W C) BGR (cv format)
	path : output path (string) 
	fps : int'''
	
	if len(frames.shape) != 4:
		raise ValueError(f"Expected 4D tensor (T,H,W,C), got shape {frames.shape}")
	
	# Ensure frames are uint8
	if frames.dtype != np.uint8:
		if frames.max() <= 1.0:
			frames = (frames * 255).astype(np.uint8)
		else:
			frames = frames.astype(np.uint8)
	
	fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
	width = frames.shape[2]
	height = frames.shape[1]
	out = cv2.VideoWriter(path, fourcc, fps, (width, height))
	if not out.isOpened():
		raise RuntimeError(f"Could not open video writer for {path}")
	
	for frame in frames:
		out.write(frame)
	out.release()
	
################## Displaying  #####################
	
	################  CV based ####################
	
def watch_video(frames=None, path='',wait=33, title='Video'):
	if frames is None and not path:
		raise ValueError('pass either a tensor or path')
	elif frames is not None:
		if not (isinstance(frames, torch.Tensor) or isinstance(frames, np.ndarray)):
			raise ValueError('frames must be torch.Tensor or np.ndarray')
		if frames.dtype == torch.uint8:
			frames = torch_to_cv(frames) #type: ignore
		elif frames.dtype != np.uint8:
			raise ValueError('frames must be either torch.Tensor with dtype torch.uint8 and shape (T, C, H, W) in RGB format, or np.ndarray with dtype np.uint8 and shape (T, H, W, C) in BGR format')
		for img in frames:
			cv2.imshow(f'{title} from tensor', img) #type: ignore
			key = cv2.waitKey(wait) & 0xFF
			if key == ord('q') or key == 27:
				break
			cv2.imshow(f'{title} from tensor', img)
			key = cv2.waitKey(wait) & 0xFF
			if key == ord('q') or key == 27:
				break
	else:
		cap = cv2.VideoCapture(path)
		beg = True
		while cap.isOpened():
			ret, img = cap.read()
			if not ret:
				print('no frames')
				if beg:
					continue
				else:
					print('finished') 
					break #works for files, not well for webcam
			cv2.imshow(f'{title} from file', img)
			beg = False
			key = cv2.waitKey(wait) & 0xFF
			if key == ord('q') or key == 27:
				break
	cv2.destroyAllWindows()
			
def show_bbox(frames, bbox):
	#frames has shape (num_frames, channels, height, width)
	#bbox is a list of [x1, y1, x2, y2]
	x1, y1, x2, y2 = bbox
	print("Bounding box coordinates:", bbox)
	for i in range(frames.shape[0]):
		frame = frames[i].permute(1, 2, 0).cpu().numpy()  # Change to (height, width, channels)
		cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw the bounding box
		cv2.imshow('Frame with Bounding Box', frame)
		cv2.waitKey(0)  # Display each frame for 100 ms
	cv2.destroyAllWindows()

def cv_display(frames: np.ndarray,  output: Optional[Union[str, Path]] = None):
	for i, frame in enumerate(frames):
		cv2.imshow('Cropped Frames', frame)  # Display the first frame
		cv2.waitKey(100)  # Wait for a key press to close the window
		if output is not None:
			cv2.imwrite(f"{output}/frame_{i:04d}.jpg", frame)
	cv2.destroyAllWindows()

def cv_save(frames: np.ndarray, output: Union[str, Path]):
	Path(output).mkdir(parents=True, exist_ok=True)
	for i, img in enumerate(frames):
		cv2.imwrite(f"{output}/frame_{i:04d}.jpg", img)  # Save each frame as an image
	print(f"Cropped frames saved to {output}.")

	################# Plot based #########################

def plt_display(frames: torch.Tensor, num: int, size: Tuple[float, float]=(5.0,5.0), adapt: bool=False, output: Optional[Union[str, Path]]=None):
	"""
	Visualise a subset of frames using matplotlib.

	Args:
		frames (torch.Tensor): 4D tensor of shape (T, C, H, W). Channels expected in RGB order.
		num (int): Number of frames to display (evenly sampled). Must be >= 1.
		size (tuple, optional): Figure size (width, height) in inches. Defaults to (5, 5).
		adapt (bool, optional): If True, scale `size` based on frame resolution. Defaults to False.
		output (str | Path, optional): If provided, directory to save each plotted frame as 'frame{i}.png'.

	Returns:
		None

	Raises:
		ValueError: If num < 1 or frames has incompatible shape.
	"""
	if output:
		output = Path(output)
		output.mkdir(parents=True, exist_ok=True)
	
	if adapt:
		factor = 5 / 256
		w, h = frames.shape[2], frames.shape[3]
		size = (w * factor, h * factor)
		
	if num < 1:
		raise ValueError("num must be >= 1")
	
	num_frames = len(frames)
	if num_frames <= num:
		step = 1
	else:
		step = num_frames // num
	
	for i, frame in enumerate(frames[::step]):
		# Permute from (C, H, W) to (H, W, C)
		np_frame = frame.permute(1, 2, 0).cpu().numpy()
		
		# Normalize to [0, 1] range
		np_frame = (np_frame - np_frame.min()) / (np_frame.max() - np_frame.min())
		
		plt.figure(figsize=size)
		plt.imshow(np_frame)
		plt.axis('off')
		if output:
			plt.savefig(output / f'frame{i}.png', bbox_inches='tight', pad_inches=0)
		plt.show()
	
################### Conversions #####################
		
def torch_to_cv(frames: torch.Tensor) -> np.ndarray:
	'''convert 4D torch tensor (T C H W) uint8 to opencv format'''
	frames = frames.permute(0, 2, 3, 1) # change to T H W C
	np_frames = frames.numpy()
	np_frames_col = np.asarray([
		cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
		for np_frame in np_frames
		])
	return np_frames_col
	
def torch_to_mediapipe(frames : torch.Tensor) -> np.ndarray:
	'''convert 4D torch.uint8 (T C H W) to mediapipe format'''
	frames = frames.permute(0, 2, 3, 1) # convert to (T H W C)
	np_array = frames.numpy()
	return np_array

def cv_to_torch(frames):
	'''convert 4D opencvformat to torch Tensor (T C H W)'''
	frames_rgb = np.asarray([
		cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
		for cv_frame in frames])
	torch_frames = torch.from_numpy(frames_rgb)
	torch_frames = torch_frames.permute(0,3,1,2)
	return torch_frames
	
####################     Plotting utilities  ####################

def plot_from_lists(train_loss, val_loss=None, 
													title='Training Loss Curve',
													xlabel='Epochs',
													ylabel='Loss',
													save_path=None,
													show=True):
	# Extract the three components from training loss
	# wall_times = [point[0] for point in train_loss]
	# steps = [point[1] for point in train_loss]
	# values = [point[2] for point in train_loss]
	steps = list(range(len(train_loss)))
	plt.figure(figsize=(10, 6))
	
	# Plot training loss
	plt.plot(steps, train_loss, color='orange', linewidth=2, label='Training Loss')
	
	# Plot validation loss if provided
	if val_loss:
		# val_steps = [point[1] for point in val_loss]
		# val_values = [point[2] for point in val_loss]
		
		plt.plot(steps, val_loss, color='blue', linewidth=2, label='Validation Loss')
	
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.grid(True, alpha=0.3)
	plt.legend()
	
	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"Plot saved to: {save_path}")
	
	if show:
		plt.show()

##################### Misc ###################################

def extract_num(fname):
	num_substrs = re.findall(r'\d+', fname)
	if len(num_substrs) > 1:
		num_str_concat = ' '.join(sub for sub in num_substrs)
		print('fname has multuple number substrings: ')
		print(num_str_concat)
		idx = -1
		while True:
			ans = input('zfill which substring? [-1]')
			if ans == '':
				break
			elif ans.isdigit():
				idx = int(ans)
				if -len(num_substrs) <= idx < len(num_substrs):
					break
				else:
					print(f'invalid index: {idx} for {len(num_substrs)} substrings')
			else:
				print(f'{ans} is not a digit')
		return num_substrs[idx]
	elif len(num_substrs) == 0:
		print(f'No valid number substrings found in {fname}')
		return
	else:
		return num_substrs[0]
		 
def is_removable(f:Path,rem_files:list[str]) -> bool:
	if not f.is_file():
		return False
	for r in rem_files:
		if f.name.endswith(r):
			return True
	return False 

def clean_checkpoints(paths, ask=False, add_zfill=True, decimals=3, rem_empty=False, rem_files= []):
	for path in paths:
		remove = rem_empty
		path_obj = Path(path)
		# print(path_obj.name)
		# Find checkpoint directories
		check_point_dirs = [item.name for item in path_obj.iterdir() 
												if item.is_dir() and 'checkpoint' in item.name]
		
		if rem_files:
			to_rem = [p for p in path_obj.iterdir() if is_removable(p, rem_files)]
			for f in to_rem:
				remove = True
				if ask:
					ans = 'none'
					print(f'{f} is set to be removed')
					while ans not in ('y', '', 'n'):
						ans = input('Delete [y]/n: ')
					remove = ans != 'n'
				if remove:
					print(f'Deleting {f}')
					f.unlink()
				else:
					print(f'Skipping {f}')
						
		if len(check_point_dirs) == 0 or all([is_empty(path_obj / d) for d in check_point_dirs]):
			
			if ask and rem_empty:
				print(f'No checkpoints found in {path}') 
				ans = 'none'
				while ans not in ('y', '', 'n'):
					ans = input('Delete [y]/n: ')
				
				if ans == 'n':
					remove = False
			else:
				print(f"Warning, no checkpoints found in {path}")
							
			if rem_empty:
			
				try:
					shutil.rmtree(path)
					print(f"Removed directory and all contents: {path}")
				except OSError as e:
					print(f"Error removing {path}: {e}")
			else:
				print(f'Warning, no checkpoints found in {path}') 
			continue
		# else:
			# print("not emtpy")
			# print(path_obj.name)
		
		
		for check in check_point_dirs:
			to_empty = path_obj / check
			
			dirty = sorted([item.name for item in to_empty.iterdir() if item.is_file()])
			files = [file for file in dirty 
							 if file.endswith('.pth')
							 and 'best' not in file]
			
			if add_zfill:
				for i, f in enumerate(files):
					num = extract_num(f)
					if num is None:
						continue
					files[i] = f.replace(num, num.zfill(decimals))
					
			if len(files) <= 2:
				continue 
			
			#leave best.pth and the last checkpoint
			if files[-1] == 'psuedo_checkpoint.pth':
				print("Ignoring psuedo_checkpoint")
				files = files[:-1]
			to_remove = files[:-1]  # not great safety wise, assumes files sort correctly
			if ask:
				ans = 'none'
				while ans != 'y' and ans != '' and ans != 'n':
					
					print(f'Only keep checkpoint {files[-1]} in {to_empty} (excluding best.pth and non pth files)?')
					ans = input(f'Delete up to {files[-2]}[y]/n: ')
				
				if ans == 'n':
					continue

			for file in to_remove:
				file_path = to_empty / file
				file_path.unlink()
				print(f'removed {file} in {to_empty}')

def is_empty(path):
	return not any(Path(path).iterdir())
		
def clean_experiments(path, ask=False, rem_empty=False, rem_files=[]):
	path_obj = Path(path)
	
	if not path_obj.exists():
		raise FileNotFoundError(f'Experiments path: {path} was not found')
	
	sub_paths = [item for item in path_obj.iterdir() if item.is_dir()]
	
	return clean_checkpoints(sub_paths, ask=ask,rem_empty=rem_empty, rem_files=rem_files)
		
def clean_runs(path, ask=False, rem_empty=False, rem_files=[]):
	path_obj = Path(path)
	
	if not path_obj.exists():
		raise FileNotFoundError(f'Runs directory: {path} was not found')
	sub_paths = [item for item in path_obj.iterdir() if item.is_dir()]
	
	for p in sub_paths:
		clean_experiments(p, ask, rem_empty, rem_files)
	

def enum_dir(path:str|Path, make:bool=False, decimals:int=3):
	'''Enumerate filenames'''
	path = Path(path)
	path_str = str(path)
	if path.exists():
		if not path_str[-1].isdigit():
			path_str += '0'.zfill(decimals)
		path = Path(path_str)
		while path.exists():
			num = int(path_str[-decimals:]) 
			path_str = path_str[:-decimals] + str(num + 1).zfill(decimals)
			path = Path(path_str)
	if make:
		path.mkdir(parents=True, exist_ok=True)
	return path

def main():
	#TODO: the output for empty runs has an issue
	parser = ArgumentParser(description='utils.py')
	parser.add_argument('-da', '--dont_ask', action='store_true', help='Clean files and directories without confirming first')
	parser.add_argument('-re', '--remove_empty', action='store_true', help='Remove empty directories')
	parser.add_argument('-rf', '--remove_files', nargs='+', type=str, help='Remove files with provided suffixes (e.g. .png .txt) ' )
	args = parser.parse_args()
	
	clean_runs(path='runs',
						 ask=not args.dont_ask,
						 rem_empty=args.remove_empty,
						 rem_files=args.remove_files)
 

if __name__ == '__main__':
	main()
	