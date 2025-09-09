import torch
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import v2
import os
from video_dataset import VideoDataset
from torch.utils.data import DataLoader
# from models.pytorch_r3d import Resnet3D18_basic
# from configs import Config
import tqdm
import torch.nn.functional as F

from train import set_seed, get_model
from pathlib import Path
import random
import configs
import utils
import gc

from typing import Optional
#################################### Testing #################################

def cleanup_memory():
	"""Cleanup GPU and CPU memory"""
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.cuda.synchronize()
	gc.collect()

def set_seed(seed=42):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def sep_arch_exp(path:str|Path):
	path_obj = Path(path)
	exp_no = path_obj.name[-3:]
	arch = path_obj.name[:-7]
	return arch, exp_no 
	
def create_runs_dict(imp_path:str|Path='wlasl_implemented_info.json', runs_path:str|Path='runs') -> dict:
	with open(imp_path, 'r') as f:
		imp_info = json.load(f)
	
	runs_path = Path(runs_path)
	available_models = imp_info['models'].keys()
 
	runs_dict = {}
	for split in imp_info['splits']:
		folder = runs_path / split
		split_dict = {}
	
		for arch in available_models:
			split_dict[arch] = []
	 
		for path in folder.iterdir():
			if not path.is_dir():
				continue
			arch, exp_no = sep_arch_exp(path)
			if arch not in available_models:
				continue
			split_dict[arch].append(exp_no)
	 
		runs_dict[split] = split_dict
	
	return runs_dict

def summarise_results(runs_dict):
  for split in runs_dict.keys():
    for arch in runs_dict[split].keys():
      for i, exp_no in enumerate([split][arch]):
        output = Path(f'runs/{split}/{arch}_exp{exp_no}')
    

			
def test_all(runs_dict:dict, imp_path:str|Path='wlasl_implemented_info.json',
						 classes_path:str|Path='wlasl_class_list.json',
							root_path:str|Path='../data/WLASL/WLASL2000',
						 test_last:bool=False, top_k:bool=True, plot:bool=False,disp:bool=False,
			 			output:Optional[str|Path] = 'wlasl_runs_results.json') -> dict:
	
	with open(imp_path, 'r') as f:
		imp_info = json.load(f)
	
	if not imp_info:
		raise ValueError(f'Implemented info empty')
	
	root = Path(root_path)
	
	for split in runs_dict.keys(): #e.g. asl100
		
		labels = Path(f'./preprocessed/labels/{split}')
		print(f'Processing split: {split}')
		
		for arch in runs_dict[split].keys(): #e.g. S3D
			
			print(f'With architecture: {arch}')
			
			for i, exp_no in enumerate(runs_dict[split][arch]): #e.g. 001
				print(f'Experiment no: {exp_no}')

				cleanup_memory()
				
				config_path = Path(f'configfiles/{split}/{arch}_{exp_no}.ini')
				output = Path(f'runs/{split}/{arch}_exp{exp_no}')
				save_path = output / 'checkpoints'
				arg_dict = {
					'architecture' : arch,
					'exp_no': exp_no,
					'split' : split,
					'root' : root,
					'labels' : labels,
					'save_path' : save_path,
					'config_path' : config_path 
				}
				config = configs.load_config({'config_path':config_path})
				configs.print_config(config)
				
				#setup data
				
				model_info = imp_info['models'][arch]
				utils.print_dict(model_info)
				
				final_t = v2.Compose([
					v2.Lambda(lambda x: x.float() / 255.0),
					v2.Normalize(mean=model_info['mean'], std=model_info['std']),
					v2.Lambda(lambda x: x.permute(1,0,2,3)) 
				])
				
				test_transforms = v2.Compose([v2.CenterCrop(config['data']['frame_size']),
																final_t])
				
				test_instances = labels / 'test_instances_fixed_frange_bboxes_len.json'
				val_instances = labels / 'val_instances_fixed_frange_bboxes_len.json'
				test_classes = labels / 'test_classes_fixed_frange_bboxes_len.json'
				val_classes = labels / 'val_classes_fixed_frange_bboxes_len.json'
				
				test_set = VideoDataset(root, test_instances, test_classes,
						transforms=test_transforms, num_frames=config['data']['num_frames'])
				val_set = VideoDataset(root, val_instances, val_classes,
						transforms=test_transforms, num_frames=config['data']['num_frames'])

				test_loader = DataLoader(test_set,
						batch_size=1, shuffle=True, num_workers=2,pin_memory=False, drop_last=False)
				val_loader = DataLoader(val_set,
						batch_size=1, shuffle=True, num_workers=2,pin_memory=False, drop_last=False)

				num_classes = len(set(test_set.classes))
				print(f'Number of samples: {len(test_set)}')
				print(f'Number of classes: {num_classes}')
				
				#setup model
				
				model = get_model(model_info['idx'], num_classes)
				
				if test_last: #some of these may have valid best.pth, others not
					checkpoint_paths = [x for x in save_path.iterdir() if x.name.endswith('.pth')]
					if len(checkpoint_paths) > 2:
						#contains more than just best and last
						s = sorted(checkpoint_paths)
						checkpoint_paths = [checkpoint_paths[0]] + [checkpoint_paths[-1]]
				else:
					checkpoint_paths = [save_path / 'best.pth']
				
				if len(checkpoint_paths) == 0:
					print(f'Warning: no weights found for {save_path}')
					continue

				for check_path in checkpoint_paths:
					
					print(f'Checkpoint: {check_path}')
					checkpoint = torch.load(check_path)
					if check_path.name == 'best.pth':
						try:
							model.load_state_dict(checkpoint)
						except Exception as e:
							print(f"Failed to load checkpoint: {check_path}")
							continue
					else:
						try:
							model.load_state_dict(checkpoint['model_state_dict'])
						except Exception as e:
							print(f"Failed to load checkpoint: {check_path}")
							continue
					
					#test it
					
					if top_k:
						print('Val')
						val_res = test_top_k(
							model=model,
							test_loader=val_loader,
							save_path=output / check_path.name.replace('.pth', '_val-top-k.txt')
						)
						print('Test')
						test_res = test_top_k(
							model=model,
							test_loader=test_loader,
							save_path=output / check_path.name.replace('.pth', '_test-top-k.txt')
						)
						experiment = {
							"checkpoint" 	: check_path.name.replace('.pth', ''),
							"val set" 		: val_res,
							"test set"		: test_res 
						}
						runs_dict[split][arch][i] = experiment #update runs_dict as we go
			
					if plot:
						accuracy, class_report, all_preds, all_targets = test_model(model, test_loader)
						# print(f'Test accuracy: {accuracy}')
						plot_heatmap(
							report=class_report,
							classes_path=classes_path,
							title='Test set Classification Report',
							save_path=output / check_path.name.replace('.pth', '_test-heatmap.png'),
							disp=disp
						)
						plot_bar_graph(
							report=class_report,
							classes_path=classes_path,
							title='Test set Classification Report',
							save_path=output / check_path.name.replace('.pth', '_test-bargraph.png'),
							disp=disp
						)
						plot_confusion_matrix(
							y_true=all_targets,
							y_pred=all_preds,
							classes_path=classes_path,
							size=(15,15),
							title='Test set Classification Report',
							save_path=output / check_path.name.replace('.pth', '_test-confmat.png'),
							disp=disp
						)
	
	#save modified runs_dict
	if output is not None:
		with open(output, 'r') as f:
			json.dump(runs_dict, f, indent=2)
	return runs_dict
	
	
def test_model(model, test_loader):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()
	all_preds = []
	all_targets = []
	
	with torch.no_grad():
		for item in tqdm.tqdm(test_loader, desc='Testing'):
			data, target = item['frames'], item['label_num']
			data, target = data.to(device), target.to(device)
			output = model(data)
			_, preds = torch.max(output, 1)
			all_preds.extend(preds.cpu().numpy())
			all_targets.extend(target.cpu().numpy())
	
	accuracy = accuracy_score(all_targets, all_preds)
	report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
	
	return accuracy, report, all_preds, all_targets

def test_top_k(model, test_loader, seed=None, verbose=False, save_path=None):
	
	if seed is not None:
		set_seed(0)
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()
	
	correct = 0
	correct_5 = 0
	correct_10 = 0
	
	num_classes = len(set(test_loader.dataset.classes))
	
	top1_fp = np.zeros(num_classes, dtype=np.int64)
	top1_tp = np.zeros(num_classes, dtype=np.int64)
	
	top5_fp = np.zeros(num_classes, dtype=np.int64)
	top5_tp = np.zeros(num_classes, dtype=np.int64)
	
	top10_fp = np.zeros(num_classes, dtype=np.int64)
	top10_tp = np.zeros(num_classes, dtype=np.int64)
	
	for item in tqdm.tqdm(test_loader, desc="Testing"):
		data, target = item['frames'], item['label_num'] 
		data, target = data.to(device), target.to(device)
		
		predictions = model(data)

		out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
		
		if target[0].item() in out_labels[-5:]:
			correct_5 += 1
			top5_tp[target[0].item()] += 1
		else:
			top5_fp[target[0].item()] += 1
		if target[0].item() in out_labels[-10:]:
			correct_10 += 1
			top10_tp[target[0].item()] += 1
		else:
			top10_fp[target[0].item()] += 1
		if torch.argmax(predictions[0]).item() == target[0].item():
			correct += 1
			top1_tp[target[0].item()] += 1
		else:
			top1_fp[target[0].item()] += 1
		
		if verbose:
			print(f"Video ID: {item['video_id']}\n\
							Correct 1: {float(correct) / len(test_loader)}\n\
							Correct 5: {float(correct_5) / len(test_loader)}\n\
							Correct 10: {float(correct_10) / len(test_loader)}")

	#per class accuracy
	top1_per_class = np.mean(top1_tp / (top1_tp + top1_fp))
	top5_per_class = np.mean(top5_tp / (top5_tp + top5_fp))
	top10_per_class = np.mean(top10_tp / (top10_tp + top10_fp))
	top1_per_instance = correct / len(test_loader)
	top5_per_instance = correct_5 / len(test_loader)
	top10_per_instance = correct_10 / len(test_loader)
	fstr = 'top-k average per class acc: {}, {}, {}'.format(top1_per_class, top5_per_class, top10_per_class)
	fstr2 = 'top-k per instance acc: {}, {}, {}'.format(top1_per_instance, top5_per_instance, top10_per_instance)
	print(fstr)
	print(fstr2)
	result = {
		'per_class': [top1_per_class,top5_per_class,top10_per_class],
		'per_instance': [top1_per_instance, top5_per_instance, top10_per_instance]
	}
	if save_path is not None:
		with open(save_path, 'w') as f:
			json.dump(result, f, indent=2)
	 
	return result

			
###############################  Plottting #############################################################

def plot_heatmap(report, classes_path, title='Classification Report Heatmap', save_path=None, disp=True):
	with open(classes_path, 'r') as f:
		test_classes = json.load(f)
	
	df = pd.DataFrame(report).iloc[:-1,:].T
	num_classes_to_plot = min(len(df)-2, len(test_classes))
	
	plt.figure(figsize=(10,10))
	sns.heatmap(df.iloc[:num_classes_to_plot, :3], annot=True, cmap='Blues', fmt='.2f', 
							xticklabels=['Precision', 'Recall', 'F1-Score'],
							yticklabels=[test_classes[i] for i in range(num_classes_to_plot)])
	plt.title(title)
	plt.tight_layout()
	if save_path:
		plt.savefig(save_path, )
	if disp:
		plt.show()

def plot_heatmap_reports_metric(reports, classes_path, metric, names):
	with open(classes_path, 'r') as f:
		test_classes = json.load(f)
		
	assert len(reports) == len(names)
	
	classes = list(reports[0].keys())[:-3]
	metric_scores = [[report[cls][metric] 
										for cls in classes] for report in reports]
	
	df = pd.DataFrame(metric_scores, 
										index=names,
										columns=classes)
	df = df.T
	
	num_to_plot = min(len(classes), len(test_classes))
	
	plt.figure(figsize=(10, 10))
	sns.heatmap(df.iloc[:num_to_plot, :], 
							annot=True, 
							cmap='Blues', 
							fmt='.2f',
							xticklabels=names,
							yticklabels=[test_classes[i] for i in range(num_to_plot)])
	plt.title(f'Classification Report Heatmap - {metric.title()}')
	plt.tight_layout()
	plt.show()
	
		
def plot_bar_graph(report, classes_path,title='Classification Report - Per Class Metrics',
									 save_path=None, disp=True):
	with open(classes_path, 'r') as f:
		test_classes = json.load(f)
	
	classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
	metrics = ['precision', 'recall', 'f1-score']

	# Prepare data for plotting
	precision = [report[cls]['precision'] for cls in classes]
	recall = [report[cls]['recall'] for cls in classes]
	f1_score = [report[cls]['f1-score'] for cls in classes]

	# Create bar plot
	x = np.arange(len(classes))
	width = 0.25

	fig, ax = plt.subplots(figsize=(10, 18))
	bars1 = ax.barh(x - width, precision, height=width, label='Precision', alpha=0.8)
	bars2 = ax.barh(x, recall, height=width, label='Recall', alpha=0.8)
	bars3 = ax.barh(x + width, f1_score, height=width, label='F1-Score', alpha=0.8)

	ax.set_ylabel('Classes')
	ax.set_xlabel('Scores')
	ax.set_title(title)
	ax.set_yticks(x)
	
	# Fix: Only use as many class names as we have classes in the report
	num_classes = len(classes)
	class_labels = [test_classes[int(cls)] if int(cls) < len(test_classes) else f"Class_{cls}" 
									for cls in classes]
	ax.set_yticklabels(class_labels)
	
	ax.legend()
	ax.set_xlim(0, 1.1)

	plt.tight_layout()
	if save_path:
		plt.savefig(save_path)
	if disp:
		plt.show()
	
def plot_bar_graph_reports_metric(reports, classes_path, metric, names):
	with open(classes_path, 'r') as f:
		test_classes = json.load(f)
	classes = list(reports[0].keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
	
	num_reports = len(reports)
	assert num_reports == len(names) #it may be better to extract names from reports
	
	#Create bar plot
	x = np.arange(len(classes))
	width = 0.8 / num_reports  # 0.8 gives good spacing, adjust as needed

	fig, ax = plt.subplots(figsize=(10, 18))
	for i, report in enumerate(reports):
		metric_list = [report[cls][metric] for cls in classes]
		offset = (i - (num_reports - 1) / 2) * width  # Center the bars
		ax.barh(x + offset, metric_list, height=width, 
						label=f'{names[i]}', alpha=0.8)
	
	ax.set_ylabel('Classes')
	ax.set_xlabel(f'{metric} scores')
	ax.set_title(f'{metric}')

	class_labels = [test_classes[int(cls)] if int(cls) < len(test_classes) else f"Class_{cls}" 
									for cls in classes]
	ax.set_yticks(x)  # This is the key missing line!
	ax.set_yticklabels(class_labels)
	
	ax.legend()
	ax.set_xlim(0, 1.1)

	plt.tight_layout()
	plt.show() 
	
def plot_confusion_matrix(y_true, y_pred, classes_path=None, num_classes=100,
													title="Confusion Matrix", size=(10, 8), row_perc=True,
													save_path=None, disp=True):
	"""
	Plot confusion matrix from true and predicted labels
	
	Parameters:
	y_true: array-like, true labels
	y_pred: array-like, predicted labels  
	classes_path: str, path to JSON file with class names (optional)
	title: str, plot title
	"""
	
	# Create confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	
	# Load class names if provided
	class_names = None
	if classes_path is not None and num_classes:
		with open(classes_path, 'r') as f:
			test_classes = json.load(f)
	
		class_names = test_classes[:num_classes]
	
	if row_perc:
		cm_row_percent = cm / cm.sum(axis=1, keepdims=True) * 100  # Normalize each row
		cm_row_percent = np.nan_to_num(cm_row_percent).round(2)     # Handle division by zero
		cm = cm_row_percent
		title += ' rowise normalised'
	
	plt.figure(figsize=size)
	sns.heatmap(
		cm, 
		annot=False, 
		fmt='d', 
		cmap='Blues',
		linewidths=0.5,      # Add gridlines between cells
		linecolor='gray'     # Gridline color (e.g., gray, white, black)
		)
	plt.title(title)
	plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=90, fontsize=8) #type: ignore
	plt.yticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=0, fontsize=8) #type: ignore
	plt.xlabel("Predicted", fontsize=12)
	plt.ylabel("True", fontsize=12)
	plt.tight_layout()
	if save_path:
		plt.savefig(save_path)
	if disp:
		plt.show()

########################### Other testing functions #############



if __name__ == '__main__':
	# config_path = './configfiles/asl100.ini'
	# configs = Config(config_path)
	# run_test_r3d18_1( output='runs/asl100/r3d18_exp5')

	
	runs_dict = create_runs_dict()
	result_dict = test_all(runs_dict, test_last=True, top_k=True, plot=True)
	utils.print_dict(result_dict)
	
