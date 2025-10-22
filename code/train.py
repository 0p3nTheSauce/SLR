from typing import Optional, Dict, Union
import torch  # type: ignore
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from pathlib import Path
import wandb
from wandb.sdk.wandb_run import Run
# local imports

from video_dataset import VideoDataset
from configs import load_config, print_config, take_args, LABEL_SUFFIX
from stopping import EarlyStopper
from models import get_model, norm_vals
from utils import wandb_manager, set_seed


def setup_data(mean, std, config):
	final_transform = v2.Compose(
		[
			v2.Lambda(lambda x: x.float() / 255.0),
			v2.Normalize(mean=mean, std=std),
			v2.Lambda(lambda x: x.permute(1, 0, 2, 3)),
		]
	)

	# setup dataset
	train_transforms = v2.Compose(
		[
			v2.RandomCrop(config.data["frame_size"]),
			v2.RandomHorizontalFlip(),
			final_transform,
		]
	)
	test_transforms = v2.Compose(
		[v2.CenterCrop(config.data["frame_size"]), final_transform]
	)

	label_path = Path(config.admin["labels"])

	train_instances = label_path / f"train_instances_{LABEL_SUFFIX}"
	val_instances = label_path / f"val_instances_{LABEL_SUFFIX}"
	train_classes = label_path / f"train_classes_{LABEL_SUFFIX}"
	val_classes = label_path / f"val_classes_{LABEL_SUFFIX}"

	dataset = VideoDataset(
		config.admin["root"],
		train_instances,
		train_classes,
		transforms=train_transforms,
		num_frames=config.data["num_frames"],
	)
	dataloader = DataLoader(
		dataset,
		batch_size=config.training["batch_size"],
		shuffle=True,
		num_workers=2,
		pin_memory=True,
	)
	num_classes = len(set(dataset.classes))

	val_dataset = VideoDataset(
		config.admin["root"],
		val_instances,
		val_classes,
		transforms=test_transforms,
		num_frames=config.data["num_frames"],
	)
	val_dataloader = DataLoader(
		val_dataset,
		batch_size=config.training["batch_size"],
		shuffle=True,
		num_workers=2,
		pin_memory=False,
	)
	val_classes = len(set(val_dataset.classes))
	assert num_classes == val_classes

	dataloaders = {"train": dataloader, "val": val_dataloader}

	return dataloaders, num_classes


def get_scheduler(
	optimizer: optim.Optimizer, sched_conf: Optional[dict] = None
) -> LRScheduler:
	"""Get learning rate scheduler based on config."""
	if sched_conf is None:
		# Identity scheduler - multiplies LR by 1.0 (no change)
		return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

	return optim.lr_scheduler.CosineAnnealingLR(
		optimizer, T_max=sched_conf["tmax"], eta_min=sched_conf.get("eta_min", 0)
	)

def get_scheduler2(
	optimizer: optim.Optimizer, sched_conf: Optional[dict] = None
):
	"""Get learning rate scheduler based on config."""
	if sched_conf is None:
		# Identity scheduler - multiplies LR by 1.0 (no change)
		return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

	# if 'warmup_epochs' in sched_conf:
	# 	warmup_scheduler = optim.lr_scheduler.LinearLR(
	# 		optimizer,
	# 		start_factor=sched_conf['start_factor'], 
	# 		end_factor=sched_conf['end_factor'],
	# 		total_iters=sched_conf['warmup_epochs']
	# 	)
	# else:
	# 	warmup_scheduler


	if sched_conf["type"] == "CosineAnnealingLR":
		if 'warmup_epochs' in sched_conf:
			warmup_scheduler = optim.lr_scheduler.LinearLR(
				optimizer,
    			start_factor=sched_conf['start_factor'], 
				end_factor=sched_conf['end_factor'],
				total_iters=sched_conf['warmup_epochs']
			)
			cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
				optimizer, T_max=sched_conf["tmax"], eta_min=sched_conf["eta_min"]
			)
			return optim.lr_scheduler.SequentialLR(
				optimizer,
				schedulers=[warmup_scheduler, cosine_scheduler],
				milestones=[sched_conf['warmup_epochs']]
			)
		else:
			return optim.lr_scheduler.CosineAnnealingLR(
				optimizer, T_max=sched_conf["tmax"], eta_min=sched_conf["eta_min"]
			)
	else:
		raise ValueError(f"Scheduler type {sched_conf['type']} not recognized.")

def train_loop(
	model_name: str, 
	wandb_run: Run, 
	load: Optional[Union[Path, str]]=None,
	save_every: int = 5,
	recover: bool = False,
	seed: Optional[int] =None
) -> None:
	"""Train loop for video classification model.

	Args:
		model_name (str): Name of the model to train.
		wandb_run (Run): Wandb run instance for logging, and config.
		load (Optional[Union[Path, str]], optional): Path to checkpoint to load, otherwise don't load checkpoint. Defaults to None.
		save_every (int, optional): Period of saving (epochs). Defaults to 5.
		recover (bool, optional): Continue from a failed run. Defaults to False.
		seed (Optional[int], optional): Random seed value, otherwise no random seed. Defaults to None.

	"""
	
	
	if seed is not None:
		set_seed(seed)

	config = wandb_run.config

	model_info = norm_vals(model_name)

	dataloaders, num_classes = setup_data(model_info["mean"], model_info["std"], config)

	try:
		drop_p = config.model_params["drop_p"]
	except Exception as _:
		drop_p = 0.0

	model = get_model(model_name, num_classes, drop_p)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	steps = 0
	epoch = 0
	# best_val_score=0.0
	best_val_score = float("inf")

	param_groups = [
		{
			"params": model.backbone.parameters(),
			"lr": config.optimizer[
				"backbone_init_lr"
			],  # Low LR for pretrained backbone
			"weight_decay": config.optimizer[
				"backbone_weight_decay"
			],  # also higher weight decay
		},
		{
			"params": model.classifier.parameters(),
			"lr": config.optimizer[
				"classifier_init_lr"
			],  # Higher LR for new classifier
			"weight_decay": config.optimizer[
				"classifier_weight_decay"
			],  # lower weight decay
		},
	]

	optimizer = optim.AdamW(param_groups, eps=config.optimizer["eps"])

	scheduler = get_scheduler(optimizer, config.get("scheduler", None))

	loss_func = nn.CrossEntropyLoss()

	# usi
	save_path = Path(config.admin["save_path"])

	# if we are continuing from last checkpoint, set 'load'
	if recover:
		fname = ""

		files = sorted([f.name for f in save_path.iterdir() if f.is_file()])
		if len(files) > 0:
			fname = files[-1]

		load = save_path / fname
	else:
		# make sure save path exists
		save_path.mkdir(parents=True, exist_ok=True)

	# early stopping setup
	es_info = config.training["early_stopping"]
	stopping_metrics = {
		"val": {"loss": 0.0, "acc": 0.0},
		"train": {"loss": 0.0, "acc": 0.0},
	}
	stopper = EarlyStopper(arg_dict=es_info, wandb_run=wandb_run)

	if load:
		load_path = Path(load)
		if load_path.exists():
			checkpoint = torch.load(load, map_location=device)
			model.load_state_dict(checkpoint["model_state_dict"])
			optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
			scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
			stopper.load_state_dict(checkpoint["stopper_state_dict"])
			epoch = checkpoint["epoch"] + 1
			steps = checkpoint["steps"]
			best_val_score = checkpoint["best_val_score"]

			print(f"Resuming from epoch {epoch}, steps {steps}")
			print(f"Loaded model from {load}")
		else:
			cont = input(
				f"Checkpoint {load} does not exist, starting from scratch? [y]"
			)
			if cont.lower() != "y":
				return
			epoch = 0
			steps = 0

	# train it
	while epoch < config.training["max_epoch"] and not stopper.stop:
		# print(f"Step {steps}/{config.training['max_steps']}")
		print(f"Epoch {epoch}/{config.training['max_epoch']}")
		print("-" * 10)

		epoch += 1
		# training and validation stage
		for phase in ["train", "val"]:
			if phase == "train":
				model.train()
			else:
				model.eval()

			# Reset matrics for this phase
			running_loss = 0.0
			running_corrects = 0
			total_samples = 0

			# for gradient accumulation
			accumulated_loss = 0.0
			accumulated_steps = 0
			optimizer.zero_grad()

			for batch_idx, item in enumerate(dataloaders[phase]):
				data, target = item["frames"], item["label_num"]
				data, target = data.to(device), target.to(device)
				batch_size = data.size(0)
				total_samples += batch_size

				if phase == "train":
					model_output = model(data)
				else:
					with torch.no_grad():
						model_output = model(data)

				# Accumulate metrics
				loss = loss_func(model_output, target)
				running_loss += loss.item() * batch_size
				_, predicted = model_output.max(1)
				running_corrects += predicted.eq(target).sum().item()

				if phase == "train":
					scaled_loss = loss / config.training["update_per_step"]
					scaled_loss.backward()

					accumulated_loss += loss.item()
					accumulated_steps += 1

					if accumulated_steps == config.training["update_per_step"]:
						optimizer.step()
						optimizer.zero_grad()
						steps += 1

						# Print progress every few steps
						if steps % 10 == 0:
							avg_acc_loss = accumulated_loss / accumulated_steps
							current_acc = 100.0 * running_corrects / total_samples
							print(
								f"Step {steps}: Accumulated Loss: {avg_acc_loss:.4f}, "
								f"Current Accuracy: {current_acc:.2f}%"
							)

							wandb_run.log(
								{
									"Loss/Train_Step": avg_acc_loss,
									"Accuracy/Train_Step": current_acc,
									"Step": steps,
								}
							)

						# Reset accumulation
						accumulated_loss = 0.0
						accumulated_steps = 0

			# calculate  epoch metrics
			epoch_loss = running_loss / total_samples  # Average loss per sample
			epoch_acc = 100.0 * running_corrects / total_samples

			# early stopping logic
			stopping_metrics[phase]["loss"] = epoch_loss
			stopping_metrics[phase]["acc"] = epoch_acc
			if phase == stopper.phase:
				stopper.step(stopping_metrics[phase][stopper.metric])

			print(f"{phase.upper()} - Epoch {epoch}:")
			print(f"  Loss: {epoch_loss:.4f}")
			print(f"  Accuracy: {epoch_acc:.2f}% ({running_corrects}/{total_samples})")

			wandb_run.log(
				{
					f"Loss/{phase.capitalize()}": epoch_loss,
					f"Accuracy/{phase.capitalize()}": epoch_acc,
					"Epoch": epoch,
				}
			)

			# Validation specific logic
			if phase == "val":
				# Save best model
				if epoch_loss < best_val_score:
					best_val_score = epoch_loss
					check_name = save_path / "best.pth"
					torch.save(model.state_dict(), check_name)
					print(
						f"New best model saved: {check_name} (Loss: {epoch_loss:.2f}%)"
					)

				# Step scheduler with validation loss
				scheduler.step()

				print(f"Best validation loss so far: {best_val_score:.2f}%")

		# Save checkpoint
		if (
			epoch % save_every == 0
			or not epoch < config.training["max_epoch"]
			or stopper.stop
		):
			checkpoint_data = {
				"epoch": epoch,
				"steps": steps,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"scheduler_state_dict": scheduler.state_dict(),
				"best_val_score": best_val_score,
				"stopper_state_dict": stopper.state_dict(),
			}
			checkpoint_path = save_path / f"checkpoint_{str(epoch).zfill(3)}.pth"

			torch.save(checkpoint_data, checkpoint_path)

			print(f"Checkpoint saved: {checkpoint_path}")

	print("Finished training successfully")
	# wandb_run.finish()


def main():
	maybe_args = take_args()
	if isinstance(maybe_args, tuple):
		arg_dict, tags, project, entity = maybe_args
	else:
		print(f"Need tuple not: {type(maybe_args)}")
		return
	config = load_config(arg_dict)

	print_config(config)

	proceed = input("Confirm: y/n: ")
	if proceed.lower() == "y":
		admin = config["admin"]
		model_name = admin["model_name"]
		run_id = admin["run_id"] if "run_id" in admin else None

		# setup wandb run
		run_name = f"{model_name}_{admin['split']}_exp{admin['exp_no']}"
		if admin["recover"]:
			if "run_id" in config["admin"]:
				run_id = config["admin"]["run_id"]
			else:
				run_id = wandb_manager.get_run_id(
					run_name,
					entity,
					project,
					idx=-1,  # last if same name
				)
			if run_id is None:
				print("Run id not found automatically, pass as arg instead")
				return

			print(f"Resuming run with ID: {run_id}")
			run = wandb.init(
				entity=entity,
				project=project,
				name=run_name,
				tags=tags,
				config=config,
				id=run_id,
				resume="must",
			)
		else:
			print(f"Starting new run with name: {run_name}")
			run = wandb.init(
				entity=entity,
				project=project,
				name=run_name,
				tags=tags,
				config=config
			)
		print(f"Run ID: {run.id}")
		print(f"Run name: {run.name}")  # Human-readable name
		print(f"Run path: {run.path}")  # entity/project/run_id format

		train_loop(model_name, run, recover=admin["recover"])
		run.finish()
	else:
		print("Training cancelled")


if __name__ == "__main__":
	main()
	# list_runs()
