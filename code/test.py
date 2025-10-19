from typing import Optional, Union, Tuple, Dict, Literal
import torch
import json
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from torchvision.transforms import v2
from video_transforms import Shuffle
from video_dataset import VideoDataset
from torch.utils.data import DataLoader

# from models.pytorch_r3d import Resnet3D18_basic
# from configs import Config
import tqdm

from train import get_model
from pathlib import Path
import configs
import utils
import gc

from argparse import ArgumentParser
from utils import ask_nicely

#locals
from visualise import plot_confusion_matrix, plot_bar_graph, plot_heatmap

#################################### Utilities #################################


def cleanup_memory():
	"""Cleanup GPU and CPU memory"""
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.cuda.synchronize()
	gc.collect()

##############################   Individual-run testing   ######################################

def test_model(model, test_loader):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()
	all_preds = []
	all_targets = []

	with torch.no_grad():
		for item in tqdm.tqdm(test_loader, desc="Testing"):
			data, target = item["frames"], item["label_num"]
			data, target = data.to(device), target.to(device)
			output = model(data)
			_, preds = torch.max(output, 1)
			all_preds.extend(preds.cpu().numpy())
			all_targets.extend(target.cpu().numpy())

	accuracy = accuracy_score(all_targets, all_preds)
	report = classification_report(
		all_targets, all_preds, output_dict=True, zero_division=0
	)

	return accuracy, report, all_preds, all_targets

def test_top_k(model, test_loader, seed=None, verbose=False, save_path=None):
	if seed is not None:
		utils.set_seed(0)

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
		data, target = item["frames"], item["label_num"]
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
			print(
				f"Video ID: {item['video_id']}\n\
							Correct 1: {float(correct) / len(test_loader)}\n\
							Correct 5: {float(correct_5) / len(test_loader)}\n\
							Correct 10: {float(correct_10) / len(test_loader)}"
			)

	# per class accuracy
	top1_per_class = np.mean(top1_tp / (top1_tp + top1_fp))
	top5_per_class = np.mean(top5_tp / (top5_tp + top5_fp))
	top10_per_class = np.mean(top10_tp / (top10_tp + top10_fp))
	top1_per_instance = correct / len(test_loader)
	top5_per_instance = correct_5 / len(test_loader)
	top10_per_instance = correct_10 / len(test_loader)
	fstr = "top-k average per class acc: {}, {}, {}".format(
		top1_per_class, top5_per_class, top10_per_class
	)
	fstr2 = "top-k per instance acc: {}, {}, {}".format(
		top1_per_instance, top5_per_instance, top10_per_instance
	)
	print(fstr)
	print(fstr2)

	result = {
		"top_k_average_per_class_acc": {
			"top1": top1_per_class,
			"top5": top5_per_class,
			"top10": top10_per_class,
		},
		"top_k_per_instance_acc": {
			"top1": top1_per_instance,
			"top5": top5_per_instance,
			"top10": top10_per_instance,
		},
	}

	if save_path is not None:
		with open(save_path, "w") as f:
			json.dump(result, f, indent=2)

	return result

def test_topk_clsrep(model: torch.nn.Module,
					 test_loader: DataLoader[VideoDataset], 
					 seed: Optional[int]=None,
					 verbose: bool =False,
					 save_path: Optional[Union[str, Path]]=None) -> Tuple[Dict, Dict]:
	if seed is not None:
		utils.set_seed(0)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()
 
	all_preds = []
	all_targets = []
 
	correct = 0
	correct_5 = 0
	correct_10 = 0
 
	assert isinstance(test_loader.dataset, VideoDataset), "This function uses a custom dataset"
	num_classes = len(set(test_loader.dataset.classes))

	top1_fp = np.zeros(num_classes, dtype=np.int64)
	top1_tp = np.zeros(num_classes, dtype=np.int64)

	top5_fp = np.zeros(num_classes, dtype=np.int64)
	top5_tp = np.zeros(num_classes, dtype=np.int64)

	top10_fp = np.zeros(num_classes, dtype=np.int64)
	top10_tp = np.zeros(num_classes, dtype=np.int64)
	
	with torch.no_grad():
		for item in tqdm.tqdm(test_loader, desc="Testing"):
			data, target = item["frames"], item["label_num"]
			data, target = data.to(device), target.to(device)

			predictions = model(data)
	
			#for classification report:
			_, preds = torch.max(predictions, 1)
			all_preds.extend(preds.cpu().numpy())
			all_targets.extend(target.cpu().numpy())

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
				print(
					f"Video ID: {item['video_id']}\n\
								Correct 1: {float(correct) / len(test_loader)}\n\
								Correct 5: {float(correct_5) / len(test_loader)}\n\
								Correct 10: {float(correct_10) / len(test_loader)}"
				)

	cls_report = classification_report(
		all_targets, all_preds, output_dict=True, zero_division=0
	)
	assert isinstance(cls_report, Dict), "Sklearn machine broke"

	# per class accuracy
	top1_per_class = np.mean(top1_tp / (top1_tp + top1_fp))
	top5_per_class = np.mean(top5_tp / (top5_tp + top5_fp))
	top10_per_class = np.mean(top10_tp / (top10_tp + top10_fp))
	top1_per_instance = correct / len(test_loader)
	top5_per_instance = correct_5 / len(test_loader)
	top10_per_instance = correct_10 / len(test_loader)
	fstr = "top-k average per class acc: {}, {}, {}".format(
		top1_per_class, top5_per_class, top10_per_class
	)
	fstr2 = "top-k per instance acc: {}, {}, {}".format(
		top1_per_instance, top5_per_instance, top10_per_instance
	)
	print(fstr)
	print(fstr2)

	topk_res = {
		"top_k_average_per_class_acc": {
			"top1": top1_per_class,
			"top5": top5_per_class,
			"top10": top10_per_class,
		},
		"top_k_per_instance_acc": {
			"top1": top1_per_instance,
			"top5": top5_per_instance,
			"top10": top10_per_instance,
		},
	}

	if save_path is not None:
		with open(save_path, "w") as f:
			json.dump(topk_res, f, indent=2)

	return topk_res, cls_report
	
def _get_test_loader(perm: Optional[torch.Tensor],
					  model_info: Dict,
					frame_size: int,
					num_frames: int,
					root: Path,
					labels: Path,
					set: Literal['test', 'val']) -> DataLoader[VideoDataset]:
	if perm:
		maybe_shuffle_t = Shuffle(perm)
	else:
		maybe_shuffle_t = v2.Lambda(lambda x: x)

	final_t = v2.Compose(
		[
			maybe_shuffle_t,
			v2.Lambda(lambda x: x.float() / 255.0),
			v2.Normalize(mean=model_info["mean"], std=model_info["std"]),
			v2.Lambda(lambda x: x.permute(1, 0, 2, 3)),
		]
	)

	test_transforms = v2.Compose(
		[v2.CenterCrop(frame_size), final_t]
	)

	instances = labels / f'{set}_instances_fixed_frange_bboxes_len.json'
	classes = labels / f'{set}_classes_fixed_frange_bboxes_len.json'

	tset = VideoDataset(
		root, 
		instances,
		classes,
		num_frames=num_frames,
		transforms=test_transforms,
	)
	return DataLoader(
		tset,
		batch_size=1,
		shuffle=False,
		num_workers=2,
		pin_memory=False,
		drop_last=False
	)



def test_run(
	config: Dict,
	perm: Optional[torch.Tensor] = None,
	model: Optional[torch.nn.Module] = None,
	test_loader: Optional[DataLoader[VideoDataset]] = None,
	val_loader: Optional[DataLoader[VideoDataset]] = None,
	test_val: bool = False,
	plot:bool = False,
	disp:bool = False,
	seed: int = 42,
	):
	utils.set_seed(seed)	

	main_conf = config['config']
	admin = main_conf['admin']
	model_info = config['model_info']
	data = main_conf['data']

	results = {}

	save_path = Path(admin['save_path'])
	output = save_path.parent / 'results'
	output.mkdir(exist_ok=True)

	tloaders = {
		"test": _get_test_loader(perm,
							model_info, 
							data['frame_size'],
							data['num_frames'],
							Path(admin['root']),
							Path(admin['labels']),
							'test' 
								)	 
	}

	if test_val:
		tloaders['val'] = _get_test_loader(perm,
							model_info, 
							data['frame_size'],
							data['num_frames'],
							Path(admin['root']),
							Path(admin['labels']),
							'val' 
								)
  
	test_loader = tloaders['test']
	assert isinstance(test_loader.dataset, VideoDataset), "This function uses a custom dataset"
	num_classes = len(set(test_loader.dataset.classes))



	
	print(output)
	


##############################   Multi-run Testing  ######################################

def create_runs_dict(
	imp_path: str | Path = "wlasl_implemented_info.json",
	runs_path: str | Path = "runs/",
	output: Optional[str | Path] = None,
) -> dict[str, dict[str, list[str]]]:
	"""Create a dictionary of runs that have been done.
	Args:
		imp_path: path to implemented info json
		runs_path: path to runs directory
		output: path to write runs dict to (default None, does not save)
	Returns:
		dict: experiments done dictionary"""

	def sep_arch_exp(path: str | Path):
		path_obj = Path(path)
		exp_no = path_obj.name[-3:]
		arch = path_obj.name[:-7]
		return arch, exp_no

	with open(imp_path, "r") as f:
		imp_info = json.load(f)

	runs_path = Path(runs_path)
	available_models = imp_info["models"].keys()

	runs_dict = {}
	for split in imp_info["splits"]:
		folder = runs_path / split
		split_dict = {}

		for arch in available_models:
			split_dict[arch] = []

		for path in folder.iterdir():
			if not path.is_dir():
				continue
			arch, exp_no = sep_arch_exp(path)
			if arch not in available_models:
				raise ValueError(f"{path} name does not fit convention")
			split_dict[arch].append(exp_no)

		runs_dict[split] = split_dict

	if output:
		with open(output, "w") as f:
			json.dump(runs_dict, f, indent=2)

	return runs_dict

def create_test_dict(
	all: bool = False,
	imp_path: str | Path = "wlasl_implemented_info.json",
	runs_path: str | Path = "runs",
) -> dict[str, dict[str, list[str]]]:
	"""Create a dictionary of runs to test.
	Args:
																																	all: if True, test all experiments in runs directory
																																	imp_path: path to implemented info json
																																	runs_path: path to runs directory
	Returns:
																																	dict: experiments to test dictionary"""

	runs_done_dict = create_runs_dict(imp_path, runs_path)
	if all:
		return runs_done_dict

	to_test = {}

	with open(imp_path, "r") as f:
		imp_info = json.load(f)

	cont = "y"
	while cont == "y":
		split = ask_nicely(
			message="Please enter split: ",
			requirment=lambda x: x in imp_info["splits"],
			error=f"pick one of: {imp_info['splits']}",
		)
		to_test[split] = {}
		cont2 = cont
		while cont2 == "y":
			arch = ask_nicely(
				message="Please enter architecture name: ",
				requirment=lambda x: x in imp_info["models"],
				error=f"pick one of: {imp_info['models'].keys()}",
			)
			to_test[split][arch] = []
			cont3 = cont2
			while cont3 == "y":
				av_exps = list(map(lambda z: int(z), runs_done_dict[split][arch]))
				exp_no = ask_nicely(
					message="Please enter experiment number: ",
					requirment=lambda x: int(x) in av_exps,
					error=f"pick one of {av_exps}",
				)
				to_test[split][arch].append(exp_no.zfill(3))
				cont3 = ask_nicely(
					message="Enter another experiment? [y/n]: ",
					requirment=lambda x: x in ["y", "n"],
					error="enter y or n",
				)
			cont2 = ask_nicely(
				message="Enter another architecture? [y/n]: ",
				requirment=lambda x: x in ["y", "n"],
				error="enter y or n",
			)
		cont = ask_nicely(
			message="Enter another split? [y/n]: ",
			requirment=lambda x: x in ["y", "n"],
			error="enter y or n",
		)

	return to_test


# TODO: maybe add more fine grained control for saving outputs
# TODO: basically has to double test if plotting


def test_all(
	runs_dict: dict[str, dict[str, list[str]]],
	test_last: bool = False,
	top_k: bool = True,
	plot: bool = False,
	disp: bool = False,
	res_output: Optional[str | Path] = None,
	skip_done: bool = True,
	test_val: bool = False,
	shuffle: bool = False,
	imp_path: str | Path = "wlasl_implemented_info.json",
	classes_path: str | Path = "wlasl_class_list.json",
	runs_dir: str | Path = "./runs",
	labels_dir: str | Path = "./preprocessed/labels",
	configs_dir: str | Path = "./configfiles",
	root_dir: str | Path = "../data/WLASL/WLASL2000",
	err_output: Optional[str | Path] = "result/test_errors.json",
) -> tuple[dict[str, dict[str, list[str]]], list]:
	"""Test multiple sets experiments (saves results to experiment directory).

	Args:
		runs_dict: 		experiments to test
		test_last: 		test the weights of last epoch (best.pth always tested)
		top_k: 			report top-k accuracies
		plot: 			create plots (heatmap, bargraph, confusion matrix)
		disp: 			display plots (if plot) (using utils.visualise_frames)
		res_output: 	path to write results to (default does not save result dict)
		skip_done: 		skips directories that have been tested (if they have result files)
		test_val: 		test on the validation set as well (default only test set)
		shuffle: 		shuffle the frames before inference (using Shuffle from video_transforms)
		imp_path: 		path to implemented info
		classes_path: 	path to class list file (if plot)
		runs_dir: 		path to experiment runs directory
		labels_dir: 	path to preprocessed labels
		configs_dir: 	path to configfiles directory
		root_dir: 		path to raw videos directory
		err_output: 	file to write run errors (don't write errors if not provided)

	Returns:
		result_dict: 	all test results in one dictionary
		problem_runs: 	experiments where the model did not load (because it uses old format)"""

	problem_runs = []
	result_dict = {}  # better to make a copy

	with open(imp_path, "r") as f:
		imp_info = json.load(f)

	if not imp_info:
		raise ValueError("Implemented info empty")

	root = Path(root_dir)
	all_labels = Path(labels_dir)
	runs = Path(runs_dir)
	configfiles = Path(configs_dir)

	for split in runs_dict.keys():  # e.g. asl100
		labels = all_labels / f"{split}"
		print(f"Processing split: {split}")
		result_dict[split] = {}

		for arch in runs_dict[split].keys():  # e.g. S3D
			print(f"With architecture: {arch}")
			# result_dict[split][arch] = [{} for _ in runs_dict[split][arch]]
			result_dict[split][arch] = {}

			for i, exp_no in enumerate(runs_dict[split][arch]):  # e.g. 001
				print(f"Experiment no: {exp_no}")
				result_dict[split][arch][exp_no] = {}

				cleanup_memory()

				config_path = configfiles / f"{split}/{arch}_{exp_no}.ini"
				exp_dir = runs / f"{split}/{arch}_exp{exp_no}"
				output = exp_dir / "results"

				# ensure results subdir exists
				output.mkdir(exist_ok=True)

				if skip_done and _is_done(exp_dir):
					continue

				save_path = exp_dir / "checkpoints"

				config = configs.load_config({"config_path": config_path})
				configs.print_config(config)

				# setup data

				model_info = imp_info["models"][arch]
				utils.print_dict(model_info)

				if shuffle:
					permutation = Shuffle.create_permutation(
						config["data"]["num_frames"]
					)
					maybe_shuffle_t = Shuffle(permutation)
				else:
					maybe_shuffle_t = v2.Lambda(lambda x: x)  # nothing

				final_t = v2.Compose(
					[
						maybe_shuffle_t,
						v2.Lambda(lambda x: x.float() / 255.0),
						v2.Normalize(mean=model_info["mean"], std=model_info["std"]),
						v2.Lambda(lambda x: x.permute(1, 0, 2, 3)),
					]
				)

				test_transforms = v2.Compose(
					[v2.CenterCrop(config["data"]["frame_size"]), final_t]
				)

				test_instances = labels / "test_instances_fixed_frange_bboxes_len.json"
				val_instances = labels / "val_instances_fixed_frange_bboxes_len.json"
				test_classes = labels / "test_classes_fixed_frange_bboxes_len.json"
				val_classes = labels / "val_classes_fixed_frange_bboxes_len.json"

				test_set = VideoDataset(
					root,
					test_instances,
					test_classes,
					transforms=test_transforms,
					num_frames=config["data"]["num_frames"],
				)
				val_set = VideoDataset(
					root,
					val_instances,
					val_classes,
					transforms=test_transforms,
					num_frames=config["data"]["num_frames"],
				)

				test_loader = DataLoader(
					test_set,
					batch_size=1,
					shuffle=True,
					num_workers=2,
					pin_memory=False,
					drop_last=False,
				)
				val_loader = DataLoader(
					val_set,
					batch_size=1,
					shuffle=True,
					num_workers=2,
					pin_memory=False,
					drop_last=False,
				)

				num_classes = len(set(test_set.classes))
				print(f"Number of samples: {len(test_set)}")
				print(f"Number of classes: {num_classes}")

				# setup model

				model = get_model(model_info["idx"], num_classes)

				if test_last:  # some of these may have valid best.pth, others not
					checkpoint_paths = [
						x for x in save_path.iterdir() if x.name.endswith(".pth")
					]
					if len(checkpoint_paths) > 2:
						# contains more than just best and last
						checkpoint_paths = sorted(checkpoint_paths)
						checkpoint_paths = [checkpoint_paths[0]] + [
							checkpoint_paths[-1]
						]
				else:
					checkpoint_paths = [save_path / "best.pth"]

				if len(checkpoint_paths) == 0:
					print(f"Warning: no weights found for {save_path}")
					continue

				for check_path in checkpoint_paths:
					print(f"Checkpoint: {check_path}")
					checkpoint = torch.load(check_path)
					if check_path.name == "best.pth":
						try:
							model.load_state_dict(checkpoint)
						except Exception:
							print(f"Failed to load checkpoint: {check_path}")
							problem_runs.append(str(check_path))
							continue
					else:
						try:
							model.load_state_dict(checkpoint["model_state_dict"])
						except Exception:
							print(f"Failed to load checkpoint: {check_path}")
							problem_runs.append(str(check_path))
							continue

					# test it

					if top_k:
						if shuffle:
							suffix = "-top-k_shuffled.json"
						else:
							suffix = "-top-k.json"

						if test_val:
							print("Val")
							fname = check_path.name.replace(".pth", "_val" + suffix)
							val_res = test_top_k(
								model=model,
								test_loader=val_loader,
								save_path=output / fname,
							)
						else:
							val_res = {}

						print("Test")
						fname = check_path.name.replace(".pth", "_test" + suffix)
						test_res = test_top_k(
							model=model,
							test_loader=test_loader,
							save_path=output / fname,
						)
						experiment = {"test set": test_res}

						if test_val:
							experiment["val set"] = val_res
						result_dict[split][arch][exp_no][
							check_path.name.replace(".pth", "")
						] = experiment  # update result_dict as we go

					if plot:
						accuracy, class_report, all_preds, all_targets = test_model(
							model, test_loader
						)
						# print(f'Test accuracy: {accuracy}')
						fname = check_path.name.replace(".pth", "_test-heatmap.png")
						plot_heatmap(
							report=class_report,
							classes_path=classes_path,
							title="Test set Classification Report",
							save_path=output / fname,
							disp=disp,
						)

						fname = check_path.name.replace(".pth", "_test-bargraph.png")
						plot_bar_graph(
							report=class_report,
							classes_path=classes_path,
							title="Test set Classification Report",
							save_path=output / fname,
							disp=disp,
						)
						fname = check_path.name.replace(".pth", "_test-confmat.png")
						plot_confusion_matrix(
							y_true=all_targets,
							y_pred=all_preds,
							classes_path=classes_path,
							size=(15, 15),
							title="Test set Classification Report",
							save_path=output / fname,
							disp=disp,
						)

	# save result_dict
	if res_output:
		with open(res_output, "w") as f:
			json.dump(result_dict, f, indent=2)

	if err_output and problem_runs:
		with open(err_output, "w") as f:
			json.dump(problem_runs, f, indent=2)

	return result_dict, problem_runs


def _is_done(dir_path: str | Path) -> bool:
	"""Check if a directory has been tested (if it has a json result file)"""
	folder = Path(dir_path)
	for p in folder.iterdir():
		if p.name.endswith(".json") and ("checkpoint" in p.name or "best" in p.name):
			return True
	return False

def summarise(
	results_dict: dict[
		str, dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]]]
	],
	splits: Optional[list[str]] = None,
	model_exps: Optional[list[list[tuple[str, str]]]] = None,
	to_summarise: Optional[dict[str, dict[str, list[str]]]] = None,
	metric: str = "top_k_average_per_class_acc",
) -> dict[str, dict[str, dict[str, str] | dict[str, float]]]:
	# starting very oppinionated
	"""Summarise results over multiple splits, architectures and experiments.

	Takes the best.pth weights from the test set, only one metric and experiment per architecture per split.

	Args:
									results_dict: 	results from test_all
									splits: 		splits to summarise (e.g. ['asl100', 'asl2000'])
									model_exps: 	model experiment pairs
	"""
	if to_summarise is not None:
		splits, model_exps = _unpack_to_summarise(to_summarise)
	print(splits)
	print(model_exps)
	if splits is None or model_exps is None:
		raise ValueError(
			"Either to_summarise or splits and model_exps must be provided"
		)
	summary = {}
	for i, split in enumerate(splits):
		if split not in results_dict:
			raise ValueError(f"{split} not in results_dict")
		split_dict = results_dict[split]
		try:
			res = _sum_split(split_dict, model_exps[i], metric)
		except KeyError as e:
			raise KeyError(f"{split} does not have all models and experiments") from e
		summary[split] = res
	return summary


def _unpack_to_summarise(
	to_summarise: dict[str, dict[str, list[str]]],
) -> tuple[list[str], list[list[tuple[str, str]]]]:
	"""Unpack to_summarise dictionary into splits and model_exps lists.
	Args:
									to_summarise: 	to_test dictionary
	Returns:
									splits: 		list of splits
									model_exps: 	corresponding architecture and experiment pairs
	"""
	splits = []
	model_exps = []
	for split in to_summarise.keys():
		splits.append(split)
		model_exps.append(
			[
				(arch, exp)
				for arch in to_summarise[split].keys()
				for exp in to_summarise[split][arch]
			]
		)
	return splits, model_exps


def _sum_split(
	split_dict: dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]]],
	model_exps: list[tuple[str, str]],
	metric: str,
) -> dict[str, dict[str, str] | dict[str, float]]:
	"""Summarise results for a given split over multiple architectures and experiments.
	Args:
									split_dict: 	dictionary of results for a given split
									model_exps: 	list of tuples of architecture and experiment number to summarise (e.g. [('S3D', '001'), ('R3D', '002')])
									metric: 	metric to summarise (e.g. 'top_k_average_per_class_acc')
	Returns:
									dict: 		summarised results for the given split"""

	results = {}
	for arch, exp in model_exps:
		results[arch] = {
			"exp": exp,
		}
		if arch not in split_dict:
			raise ValueError(f"{arch} not in split_dict")
		arch_dict = split_dict[arch]
		if exp not in arch_dict:
			raise ValueError(f"{exp} not in split_dict[{arch}]")
		try:
			res = _sum_model(arch_dict, exp, metric)
		except KeyError as e:
			raise KeyError(f"{arch} {exp} does not have metric {metric}") from e
		results[arch][metric] = res
	return results


def _sum_model(
	arch_dict: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]],
	exp: str,
	metric: str,
) -> dict[str, float]:
	"""Summarise results for a given architecture over multiple experiments.
	Args:
																																	arch_dict: 	dictionary of results for a given architecture
																																	exp: 		experiment number to summarise (e.g. '001')
																																	metric: 	metric to summarise (e.g. 'top_k_average_per_class_acc')
	Returns:
																																	dict: 		summarised results for the given architecture and experiment"""

	return arch_dict[exp]["best"]["test set"][metric]



if __name__ == "__main__":
	parser = ArgumentParser(description="test.py")
	subparsers = parser.add_subparsers(
		dest="command", help="Available commands", required=True
	)
	# create test dictionary
	create_parser = subparsers.add_parser("ctd", help="Create test dictionary")
	create_parser.add_argument(
		"-a",
		"--all",
		action="store_true",
		help="Test all experiments in runs directory",
	)
	create_parser.add_argument(
		"-o",
		"--output",
		type=str,
		help="Output path for the test dictionary (default: test_dict.json)",
		required=True,
	)
	# summarise
	summarise_parser = subparsers.add_parser(
		"summarise", help="Summarise results from multiple experiments"
	)
	summarise_parser.add_argument(
		"-r",
		"--results",
		type=str,
		required=True,
		help="Path to results JSON file from test_all",
	)
	summarise_parser.add_argument(
		"-t",
		"--to_summarise",
		type=str,
		help="Path to test dictionary JSON file (if not provided, splits and model_exps must be provided)",
		required=True,
	)
	summarise_parser.add_argument(
		"-o",
		"--output",
		type=str,
		help="Output path for the summary JSON file (default: summary.json)",
		default="summary.json",
	)
	summarise_parser.add_argument(
		"-m",
		"--metric",
		type=str,
		help="Metric to summarise (default: top_k_average_per_class_acc)",
		default="top_k_average_per_class_acc",
	)

	# test from test dictionary
	test_parser = subparsers.add_parser(
		"test", help="Test experiments from a test dictionary"
	)
	test_parser.add_argument(
		"-tt",
		"--to_test",
		type=str,
		help="Path to test dictionary JSON file",
		required=True,
	)
	test_parser.add_argument(
		"-tl",
		"--test_last",
		action="store_true",
		help="Test the weights of last epoch (best.pth always tested)",
	)
	test_parser.add_argument(
		"-pt",
		"--plot",
		action="store_true",
		help="Plot results (heatmap, bargraph, confusion matrix)",
	)
	test_parser.add_argument(
		"-sd",
		"--skip_done",
		action="store_true",
		help="Skip directories that have been tested (if they have result files)",
	)
	test_parser.add_argument(
		"-o",
		"--output",
		type=str,
		help="Output path for the results JSON file (if not provided, does not save)",
	)
	test_parser.add_argument(
		"-eo",
		"--err_output",
		type=str,
		help="Output path for the errors JSON file (if not provided, does not save)",
	)
	test_parser.add_argument(
		"-sl",
		"--shuffle",
		action="store_true",
		help="Shuffle the frames before inference",
	)

	args = parser.parse_args()
	if args.command == "ctd":
		to_test = create_test_dict(all=args.all)
		with open(args.output, "w") as f:
			json.dump(to_test, f, indent=2)
		print(f"Test dictionary saved to {args.output}")
	elif args.command == "summarise":
		with open(args.results, "r") as f:
			results_dict = json.load(f)
		with open(args.to_summarise, "r") as f:
			to_summarise = json.load(f)
		summary = summarise(
			results_dict=results_dict, to_summarise=to_summarise, metric=args.metric
		)
		with open(args.output, "w") as f:
			json.dump(summary, f, indent=2)
		print(f"Summary saved to {args.output}")
	elif args.command == "test":
		with open(args.to_test, "r") as f:
			to_test = json.load(f)
		test_all(
			runs_dict=to_test,
			test_last=args.test_last,
			plot=args.plot,
			skip_done=args.skip_done,
			shuffle=args.shuffle,
			res_output=args.output,
			err_output=args.err_output,
		)
		if args.output:
			print(f"Results saved to {args.output}")
		if args.err_output:
			print(f"Errors saved to {args.err_output}")
