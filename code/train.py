import torch # type: ignore
# import os
import json
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

#local imports
import numpy as np
import random
import wandb
from video_dataset import VideoDataset
from configs import load_config, print_config, take_args
from models.pytorch_mvit import MViTv2S_basic, MViTv1B_basic
from models.pytorch_swin3d import Swin3DBig_basic, Swin3DSmall_basic, Swin3DTiny_basic
from models.pytorch_r3d import Resnet2_plus1D_18_basic, Resnet3D_18_basic
from models.pytorch_s3d import S3D_basic
from stopping import EarlyStopper
# from code.experiments.scripts.quewing import get_run_id

from typing import Optional, List

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
	) -> list[str]:
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


ENTITY= 'ljgoodall2001-rhodes-university'
# PROJECT = 'WLASL - SLR'
PROJECT = 'WLASL-100'
PROJECT_BASE = 'WLASL'


def set_seed(seed=42):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def setup_data(mean, std, config):
	
	final_transform =  v2.Compose([
		v2.Lambda(lambda x: x.float() / 255.0),
		v2.Normalize(mean=mean, std=std),
		v2.Lambda(lambda x: x.permute(1,0,2,3)) 
	])
	
	#setup dataset
	train_transforms = v2.Compose([v2.RandomCrop(config.data['frame_size']),
																 v2.RandomHorizontalFlip(),
																 final_transform])
	test_transforms = v2.Compose([v2.CenterCrop(config.data['frame_size']),
																final_transform])
	
 
	label_path = Path(config.admin['labels'])
 
	# train_instances = os.path.join(config.admin['labels'], 'train_instances_fixed_frange_bboxes_len.json')
	# val_instances = os.path.join(config.admin['labels'],'val_instances_fixed_frange_bboxes_len.json' )
	# train_classes = os.path.join(config.admin['labels'], 'train_classes_fixed_frange_bboxes_len.json')
	# val_classes = os.path.join(config.admin['labels'],'val_classes_fixed_frange_bboxes_len.json' )
	
	train_instances = label_path / 'train_instances_fixed_frange_bboxes_len.json'
	val_instances = label_path / 'val_instances_fixed_frange_bboxes_len.json'
	train_classes = label_path / 'train_classes_fixed_frange_bboxes_len.json'
	val_classes = label_path / 'val_classes_fixed_frange_bboxes_len.json'

	dataset = VideoDataset(config.admin['root'],train_instances, train_classes,
		transforms=train_transforms, num_frames=config.data['num_frames'])
	dataloader = DataLoader(dataset, batch_size=config.training['batch_size'],
		shuffle=True, num_workers=2,pin_memory=True)
	num_classes = len(set(dataset.classes))
	
	val_dataset = VideoDataset(config.admin['root'], val_instances, val_classes,
		transforms=test_transforms, num_frames=config.data['num_frames'])
	val_dataloader = DataLoader(val_dataset,
		batch_size=config.training['batch_size'], shuffle=True, num_workers=2,pin_memory=False)
	val_classes = len(set(val_dataset.classes))
	assert num_classes == val_classes
	
	dataloaders = {'train': dataloader, 'val': val_dataloader}
	
	return dataloaders, num_classes
	

def get_model(model_idx:int, num_classes:int, drop_p:Optional[float]=None):
	model_constructors = {
				0: MViTv1B_basic,
				1: MViTv2S_basic,
				2: Swin3DBig_basic,
				3: Swin3DSmall_basic,
				4: Swin3DTiny_basic,
				5: Resnet2_plus1D_18_basic,
				6: Resnet3D_18_basic,
				7: S3D_basic,
		}
	#TODO: the is definitely a cleaner way to do this
	model_names = {
				0: "MViTv1B_basic",
				1: "MViTv2S_basic",
				2: "Swin3DBig_basic",
				3: "Swin3DSmall_basic",
				4: "Swin3DTiny_basic",
				5: "Resnet2_plus1D_18_basic",
				6: "Resnet3D_18_basic",
				7: "S3D_basic",
		}
	if model_idx not in model_constructors:
		raise ValueError(f"Unknown model_idx: {model_idx}")
	if drop_p is not None:
		model = model_constructors[model_idx](num_classes=num_classes, drop_p=drop_p)
	else:
		model = model_constructors[model_idx](num_classes)
	print(f"Successfully using model {model_names[model_idx]}")
 
	return model	
 
 
def train_loop(model_info, wandb_run, load=None, save_every=5,
								 recover=False, seed=None):
	
	if seed is not None:
		set_seed(seed)

	config = wandb_run.config
	
	#setup transforms
	final_transform = v2.Compose([
		v2.Lambda(lambda x: x.float() / 255.0),
		v2.Normalize(mean=model_info['mean'], std=model_info['mean']),
		v2.Lambda(lambda x: x.permute(1,0,2,3)) 
	])
	
	#should actually dirst resize before cropping
	
 
	train_transforms = v2.Compose([v2.RandomCrop(config.data['frame_size']),
																 v2.RandomHorizontalFlip(),
																 final_transform])
	test_transforms = v2.Compose([v2.CenterCrop(config.data['frame_size']),
																final_transform])
	
	#setup data
	label_path = Path(config.admin['labels'])
 
	train_instances = label_path / 'train_instances_fixed_frange_bboxes_len.json'
	val_instances = label_path / 'val_instances_fixed_frange_bboxes_len.json'
	train_classes = label_path / 'train_classes_fixed_frange_bboxes_len.json'
	val_classes = label_path / 'val_classes_fixed_frange_bboxes_len.json'
	
	dataset = VideoDataset(config.admin['root'],train_instances, train_classes,
		transforms=train_transforms, num_frames=config.data['num_frames'])
	dataloader = DataLoader(dataset, batch_size=config.training['batch_size'],
		shuffle=True, num_workers=2,pin_memory=True)
	num_classes = len(set(dataset.classes))
	
	val_dataset = VideoDataset(config.admin['root'], val_instances, val_classes,
		transforms=test_transforms, num_frames=config.data['num_frames'])
	val_dataloader = DataLoader(val_dataset,
		batch_size=config.training['batch_size'], shuffle=True, num_workers=2,pin_memory=False)
	val_classes = len(set(val_dataset.classes))
	assert num_classes == val_classes
	
	dataloaders = {'train': dataloader, 'val': val_dataloader}
	
	#model, metrics, optimizer, schedular, loss

	try:
		drop_p = config.model_params['drop_p']
	except Exception as _:
		drop_p = None
 
	model = get_model(model_info['idx'], num_classes, drop_p)
	
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	
	steps = 0
	epoch = 0 
	# best_val_score=0.0
	best_val_score=float('inf')
	
	param_groups = [
		{
			'params': model.backbone.parameters(),
			'lr': config.optimizer['backbone_init_lr'],  # Low LR for pretrained backbone
			'weight_decay': config.optimizer['backbone_weight_decay'] #also higher weight decay
		},
		{
			'params': model.classifier.parameters(), 
			'lr': config.optimizer['classifier_init_lr'],  # Higher LR for new classifier
			'weight_decay': config.optimizer['classifier_weight_decay'] #lower weight decay
		}
	]
	
	optimizer = optim.AdamW(param_groups, 
													eps = config.optimizer['eps'])
	
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
																									 T_max=config.scheduler['tmax'],
																									 eta_min=config.scheduler['eta_min'])
	
	loss_func = nn.CrossEntropyLoss()

	#usi  
	save_path = Path(config.admin['save_path'])

	#if we are continuing from last checkpoint, set 'load'
	if recover: 
		fname = ''
		# if os.path.exists(config.admin['save_path']):
		if save_path.exists():
			# files = sorted(os.listdir(config.admin['save_path']))
			files = sorted([f.name for f in save_path.iterdir() if f.is_file()])
			if len(files) > 0:
				fname = files[-1]
			else:
				raise ValueError(f"Directory: {save_path} is empty")
		else:
			raise ValueError(f"Could not find directory: {save_path}")
				#setup recovery
		
		load = save_path / fname
	else:
		#make sure save path exists
		save_path.mkdir(parents=True, exist_ok=True)
		
	
	#early stopping setup
	es_info = config.training['early_stopping']
	stopping_metrics = {
		'val' : {
			'loss': 0.0,
			'acc': 0.0
		},
		'train' : {
			'loss': 0.0,
			'acc': 0.0
		}
	}
	stopper = EarlyStopper(
		arg_dict=es_info,
		wandb_run=wandb_run
	)
	
	if load:
		load_path = Path(load)
		if load_path.exists():
			checkpoint = torch.load(load, map_location=device)
			model.load_state_dict(checkpoint['model_state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
			stopper.load_state_dict(checkpoint['stopper_state_dict'])
			epoch = checkpoint['epoch'] + 1
			steps = checkpoint['steps']
			best_val_score = checkpoint['best_val_score']
			
			print(f"Resuming from epoch {epoch}, steps {steps}")
			print(f"Loaded model from {load}")
		else:
			cont = input(f"Checkpoint {load} does not exist, starting from scratch? [y]")
			if cont.lower() != 'y':
				return
			epoch = 0
			steps = 0
	
	#train it
	while epoch < config.training['max_epoch'] and not stopper.stop:
		# print(f"Step {steps}/{config.training['max_steps']}")
		print(f"Epoch {epoch}/{config.training['max_epoch']}")
		print('-'*10)
		
		epoch += 1
		#training and validation stage
		for phase in ['train', 'val']:
			
			if phase == 'train':
				model.train()
			else:
				model.eval()

			#Reset matrics for this phase
			running_loss = 0.0
			running_corrects = 0
			total_samples = 0

			#for gradient accumulation  
			accumulated_loss = 0.0
			accumulated_steps = 0
			optimizer.zero_grad()
			
			for batch_idx, item in enumerate(dataloaders[phase]):
				data, target = item['frames'], item['label_num']
				data, target = data.to(device), target.to(device)
				batch_size = data.size(0)
				total_samples += batch_size
				
				if phase == 'train':
					model_output = model(data)
				else:
					with torch.no_grad():
						model_output = model(data)
				
				#Accumulate metrics
				loss = loss_func(model_output, target)        
				running_loss += loss.item() * batch_size  
				_, predicted = model_output.max(1)
				running_corrects += predicted.eq(target).sum().item()
				
				if phase == 'train':
					scaled_loss = loss / config.training['update_per_step']
					scaled_loss.backward()
					
					accumulated_loss += loss.item()
					accumulated_steps += 1
					
					if accumulated_steps == config.training['update_per_step']:
						optimizer.step()
						optimizer.zero_grad()
						steps += 1
						
						# Print progress every few steps
						if steps % 10 == 0:
							avg_acc_loss = accumulated_loss / accumulated_steps
							current_acc = 100.0 * running_corrects / total_samples
							print(f'Step {steps}: Accumulated Loss: {avg_acc_loss:.4f}, '
										f'Current Accuracy: {current_acc:.2f}%')
							
							wandb_run.log({
								'Loss/Train_Step': avg_acc_loss,
								'Accuracy/Train_Step': current_acc,
								'Step': steps
							})
						
						# Reset accumulation
						accumulated_loss = 0.0
						accumulated_steps = 0
			
			#calculate  epoch metrics
			epoch_loss = running_loss / total_samples # Average loss per sample
			epoch_acc = 100.0 * running_corrects / total_samples
			
			#early stopping logic
			stopping_metrics[phase]['loss'] = epoch_loss
			stopping_metrics[phase]['acc'] = epoch_acc
			if phase == stopper.phase:
				stopper.step(stopping_metrics[phase][stopper.metric])
			
			print(f'{phase.upper()} - Epoch {epoch}:')
			print(f'  Loss: {epoch_loss:.4f}')
			print(f'  Accuracy: {epoch_acc:.2f}% ({running_corrects}/{total_samples})')
			
			wandb_run.log({
				f'Loss/{phase.capitalize()}': epoch_loss,
				f'Accuracy/{phase.capitalize()}': epoch_acc,
				'Epoch': epoch
			})
			
			# Validation specific logic
			if phase == 'val':
				# Save best model
				if epoch_loss < best_val_score:
					best_val_score = epoch_loss
					model_name = save_path / 'best.pth'
					torch.save(model.state_dict(), model_name)
					print(f'New best model saved: {model_name} (Loss: {epoch_loss:.2f}%)')
			
				# Step scheduler with validation loss
				scheduler.step() 
				
				print(f'Best validation loss so far: {best_val_score:.2f}%')
			
		# Save checkpoint
		if epoch % save_every == 0 or not epoch < config.training['max_epoch'] or stopper.stop:
				checkpoint_data = {
					'epoch': epoch,
					'steps': steps,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'scheduler_state_dict': scheduler.state_dict(),
					'best_val_score': best_val_score,
					'stopper_state_dict': stopper.state_dict() 
				}
				checkpoint_path = save_path / f'checkpoint_{str(epoch).zfill(3)}.pth'
				

				torch.save(checkpoint_data, checkpoint_path)
					
				print(f'Checkpoint saved: {checkpoint_path}')
				
	print('Finished training successfully')
	# wandb_run.finish()


def main():
	with open('./wlasl_implemented_info.json') as f:
		info = json.load(f)
	available_splits = info['splits']
	model_info = info['models']
	
	maybe_args = take_args(available_splits, model_info.keys())
	if isinstance(maybe_args, tuple):
		arg_dict, tags, project, entity = maybe_args
	else:
		print(f'Need tuple not: {type(maybe_args)}')
		return
	config = load_config(arg_dict, verbose=True)
	
	print_config(config)

	proceed = input("Confirm: y/n: ")
	if proceed.lower() == 'y':
		admin = config['admin']
		model_specifcs = model_info[admin['model']]
		run_id = config['admin']['run_id'] if 'run_id' in config['admin'] else None
			
		#setup wandb run
		run_name = f"{admin['model']}_{admin['split']}_exp{admin['exp_no']}"
		if admin['recover']:
			if 'run_id' in config['admin']:
				run_id = config['admin']['run_id']
			else: 
				# run_id = get_run_id(run_name, ENTITY, project)
				run_id = wandb_manager.get_run_id(
					run_name,
					entity,
					project,
					idx=-1 #last if same name
				)
			if run_id is None:
				print("Run id nto found automatically, pass as arg instead")
				return
				
			print(f"Resuming run with ID: {run_id}")
			run = wandb.init(
				entity=entity,
				project=project,
				id=run_id,
				resume='must',
				name=run_name,
				tags=tags,
				config=config      
			)
		else:
			print(f"Starting new run with name: {run_name}")
			run = wandb.init(
				entity=ENTITY,
				project=project,
				name=run_name,
				tags=tags,
				config=config      
			)
		print(f"Run ID: {run.id}")
		print(f"Run name: {run.name}")  # Human-readable name
		print(f"Run path: {run.path}")  # entity/project/run_id format
		

		train_loop(model_specifcs, run, recover=admin['recover'])
	else:
		print("Training cancelled")
	
if __name__ == '__main__':
	main()
	# list_runs()