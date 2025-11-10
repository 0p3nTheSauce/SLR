from typing import Optional, Union, Tuple, Dict, List, Any, TypedDict, Literal, cast
from argparse import ArgumentParser
import torch
import json
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from torch.utils.data import DataLoader
import tqdm
from pathlib import Path
import gc

# locals
from visualise import plot_confusion_matrix, plot_bar_graph, plot_heatmap
from models import norm_vals, get_model
from configs import set_seed, CompRes
from video_dataset import VideoDataset, get_data_loader, get_wlasl_info
from models import avail_models
from configs import get_avail_splits, load_config, AdminInfo, RunInfo, RUNS_PATH, BaseRes, ShuffRes, TopKRes
from utils import print_dict
#################################### Utilities #################################


def cleanup_memory():
	"""Cleanup GPU and CPU memory"""
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.cuda.synchronize()
	gc.collect()

#################################### Helper classes #############################



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

	assert isinstance(report, Dict), "Sklearn machine broke"

	return accuracy, report, all_preds, all_targets


def test_top_k(model, test_loader, verbose=False, save_path=None):

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


def test_topk_clsrep(
	model: torch.nn.Module,
	test_loader: DataLoader[VideoDataset],
	verbose: bool = False,
	save_path: Optional[Union[str, Path]] = None,
) -> Tuple[
	BaseRes, Dict[str, Dict[str, float]], List[int], List[int]
]:
	"""Get the top-k accuracies (both per class and per instance) and classification report for a model on a test set.

	Args:
					model (torch.nn.Module): Initialised model to test.
					test_loader (DataLoader[VideoDataset]): Initialised dataloader for the test set.
					seed (Optional[int], optional): Random seed, if not set no seed. Defaults to None.
					verbose (bool, optional): Verbose output. Defaults to False.
					save_path (Optional[Union[str, Path]], optional): Optionally save results to json file. Defaults to None.

	Returns:
					Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], List[int], List[int]]: Dictionary of top-k accuracies (per instance and per class), classification report dictionary (sklearn style), all_targets, all_preds.
	"""

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()

	all_preds = []
	all_targets = []

	correct = 0
	correct_5 = 0
	correct_10 = 0

	assert isinstance(test_loader.dataset, VideoDataset), (
		"This function uses a custom dataset"
	)
	num_classes = len(set(test_loader.dataset.classes))

	top1_fp = np.zeros(num_classes, dtype=np.int64)
	top1_tp = np.zeros(num_classes, dtype=np.int64)

	top5_fp = np.zeros(num_classes, dtype=np.int64)
	top5_tp = np.zeros(num_classes, dtype=np.int64)

	top10_fp = np.zeros(num_classes, dtype=np.int64)
	top10_tp = np.zeros(num_classes, dtype=np.int64)


	loss_func = torch.nn.CrossEntropyLoss()
	running_loss = 0.0
	total_samples = 0

	with torch.no_grad():
		for item in tqdm.tqdm(test_loader, desc="Testing"):
			data, target = item["frames"], item["label_num"]
			data, target = data.to(device), target.to(device)
			batch_size = data.size(0)
			total_samples += batch_size

			predictions = model(data)

			#for loss
			loss = loss_func(predictions, target)
			running_loss += loss.item() * batch_size

			# for classification report:
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

	#loss
	epoch_loss = running_loss / total_samples
 
	print(f"Averag Loss: {epoch_loss:.2f}")


	topk_res = BaseRes(
		top_k_average_per_class_acc=TopKRes(
			top1=float(top1_per_class),
			top5=float(top5_per_class),
			top10=float(top10_per_class)
		),
		top_k_per_instance_acc=TopKRes(
			top1=top1_per_instance,
			top5=top5_per_instance,
			top10=top10_per_instance
		),
		average_loss=epoch_loss
  
	)
	if save_path is not None:
		with open(save_path, "w") as f:
			json.dump(topk_res, f, indent=2)

	return topk_res, cls_report, all_targets, all_preds


def collect_results(res_p: Path):
	with open(res_p, "r") as f:
		res = json.load(f)
	return res


def load_info(dirp: Path, checkname: str):
	resd = {}
	fnames = list(dirp.glob(f"{checkname}_*.json"))
	for fn in fnames:
		with open(fn, "r") as f:
			resd[fn.name.replace(".json", "")] = json.load(f)
	return resd



def test_run(
	config: RunInfo,
	set_name: Literal['test', 'val', 'train'],
	shuffle: bool = False,
	check: str = "best.pth",
	br_graph: bool = False,
	cf_matrix: bool = False,
	heatmap: bool = False,
	disp: bool = False,
	save: bool = True,
	save_img: Optional[bool] = None
) -> Union[BaseRes, ShuffRes]:
	"""Perform testing of a model according to the provided configuration.

	Args:
			config (Dict[str, Any]): Run config file.
			perm (Optional[torch.Tensor], optional): Permutation, if shuffeling frames, otherwise no shuffle. Defaults to None.
			test_val (bool, optional): Test on the val set. Defaults to False.
			test_test (bool, optional): Test on the test set. Defaults to True.
			check (str, optional): Checkpoint name. Defaults to "best.pth".
			br_graph (bool, optional): Create bar graph. Defaults to False.
			cf_matrix (bool, optional): Create confusion matrix. Defaults to False.
			heatmap (bool, optional): Create heatmap. Defaults to False.
			disp (bool, optional): Display plots. Defaults to False.
			save (bool, optional): Save results and plots. Defaults to True.
			re_test (bool, optional): Test even if results already saved. Defaults to False.

	Returns:
			Optional[Dict[str, Any]]: Results if correct parameters.
	"""
 
	if save_img is None:
		save_img = save

	set_seed()

	admin = config["admin"]
	model_name = admin["model"]
	data = config["data"]

	model_norms = norm_vals(model_name)
	results = {}

	save_path = Path(admin["save_path"])

	output = save_path.parent / "results"

	if save:
		output.mkdir(exist_ok=True)
	
	dloader, num_classes, m_permt, m_sh_et = get_data_loader(
	 	model_norms['mean'],
		model_norms['std'],
		data["frame_size"],
		data["num_frames"],
		set_info=get_wlasl_info(admin["split"], set_name=set_name),
		shuffle=shuffle,
		batch_size=1
	)

	model = get_model(model_name, num_classes, drop_p=0.0)

	check_path = save_path / check

	print(f"Loading weights from: {check_path}")

	checkpoint = torch.load(check_path)

	if check_path.name == "best.pth":
		model.load_state_dict(checkpoint)
	else:
		model.load_state_dict(checkpoint["model_state_dict"])

	if shuffle:
		suffix = "-top-k_shuffled.json"
	else:
		suffix = "-top-k.json"

	
	print(f"Testing on {set_name} set")
	fname = check_path.name.replace(".pth", f"_{set_name}{suffix}")
	save2 = output / fname
	
	topk_res, cls_report, all_targets, all_preds = test_topk_clsrep(
		model=model,
		test_loader=dloader,
		verbose=False,
	)
	
	if m_permt is not None and m_sh_et is not None: #shuffled
		results = ShuffRes(
			top_k_average_per_class_acc=topk_res['top_k_average_per_class_acc'],
			top_k_per_instance_acc=topk_res['top_k_per_instance_acc'],
			average_loss=topk_res['average_loss'],
			perm=m_permt,
			shannon_entropy=m_sh_et
		)
	else:
		results = topk_res
  
	if save:
		with open(save2, "w") as f:
			json.dump(results, f, indent=4)
		
	if heatmap:
		fname = check_path.name.replace(".pth", f"_{set_name}-heatmap.png")
		save2 = output / fname if save_img else None
		plot_heatmap(
			report=cls_report,
			title=f"{set_name.capitalize()} set Classification Report",
			save_path=save2,
			disp=disp,
		)

	if br_graph:
		fname = check_path.name.replace(".pth", f"_{set_name}-bargraph.png")
		save2 = output / fname if save_img else None
		plot_bar_graph(
			report=cls_report,
			title=f"{set_name.capitalize()} set Classification Report",
			save_path=save2,
			disp=disp,
		)

	if cf_matrix:
		fname = check_path.name.replace(".pth", f"_{set_name}-confmat.png")
		save2 = output / fname if save_img else None
		plot_confusion_matrix(
			y_true=all_targets,
			y_pred=all_preds,
			title=f"{set_name.capitalize()} set Confusion Matrix",
			save_path=save2,
			disp=disp,
		)

	return results

#TODO: can be simplified to take only admin info if each folder keeps a file on what frame rate and image size to test with
def full_test(
	config: RunInfo,
	save: bool = True
) -> CompRes: 
	"""Complete test, which includes:
		- The best validation loss, and accuracy for the whole training run 
		- The test, val and 'shuffled test' results.
	The test, val and shuffled results all contain the average loss, topk per instance, and per class accuracy.
	The shuffled results additionally contain the permutation used, and it's shannon entropy 
  
	Args:
		config (RunInfo): The run config used for training.
		save (bool, optional): Whether to save results. Defaults to True.

	Returns:
		CompRes: A results dictionary (as described above).
	"""
	save_path = Path(config['admin']['save_path'])
	files = sorted(list(save_path.iterdir()))
	last_check = torch.load(files[-1])
	best_val_acc = last_check["best_val_acc"]
	best_val_loss = last_check["best_val_loss"]
	out_dir = save_path.parent / "results"
	res_path = out_dir / "best_val_loss.json" 
 
	#test set
	test = test_run(
		config,
		'test',
		br_graph=True,
		cf_matrix=True,
		heatmap=True,
		save_img=True,
		save=False
	)
	#validation set
	val = test_run(
		config,
		'val',
		save=False
	)
	#shuffled frames test set
	test_shuff = test_run(
		config,
		'test',
		shuffle=True,
		save=False
	)
	
	results = CompRes(
		check_name='best_val',
		best_val_acc=best_val_acc,
		best_val_loss=best_val_loss,
		test=test,
		val=val,
		test_shuff=cast(ShuffRes,test_shuff)
	)
 
	if save:
		with open(res_path, 'w') as f:
			json.dump(results, f, indent=4)
   
	return results

	
	


	
def get_test_parser(prog: Optional[str] = None, desc: str = "Test a model") -> ArgumentParser:
	"""Get parser for testing configuration with subparsers for full/partial test modes

	Args:
		prog (Optional[str], optional): Script name, (e.g. testing.py). Defaults to None.
		desc (str, optional): Program desctiption. Defaults to "Test a model".

	Returns:
		ArgumentParser: Parser which takes testing arguments
	"""
	parser = ArgumentParser(description=desc, prog=prog)
	models_available = avail_models()
	splits_available = get_avail_splits()
	
	# Create subparsers for 'full' and 'partial' commands
	subparsers = parser.add_subparsers(dest='command', help='Test mode', required=True)
	
	# ============ FULL TEST SUBPARSER ============
	full_parser = subparsers.add_parser(
		'full',
		help='Run full test suite (test, val, and shuffled test with all visualizations)'
	)
	
	full_parser.add_argument(
		"model",
		type=str,
		choices=models_available,
		help=f"Model name from one of the implemented models: {models_available}",
	)
	full_parser.add_argument(
		"split",
		type=str,
		choices=splits_available,
		help=f"The class split, one of: {', '.join(splits_available)}",
	)
	full_parser.add_argument("exp_no", type=int, help="Experiment number (e.g. 10)")
	full_parser.add_argument(
		'-ds',
		'--dataset',
		type=str,
		choices=['WLASL'],
		help="Not implemented yet",
		default='WLASL'
	)
	full_parser.add_argument("-c", "--config_path", help="Path to config .ini file")
	full_parser.add_argument(
		'-se',
		'--save',
		action='store_true',
		help='Save the outputs of the test'
	)
	
	# ============ PARTIAL TEST SUBPARSER ============
	partial_parser = subparsers.add_parser(
		'partial',
		help='Run partial test on a specific set with custom options'
	)
	
	partial_parser.add_argument(
		"model",
		type=str,
		choices=models_available,
		help=f"Model name from one of the implemented models: {models_available}",
	)
	partial_parser.add_argument(
		"split",
		type=str,
		choices=splits_available,
		help=f"The class split, one of: {', '.join(splits_available)}",
	)
	partial_parser.add_argument("exp_no", type=int, help="Experiment number (e.g. 10)")
	partial_parser.add_argument(
		'-ds',
		'--dataset',
		type=str,
		choices=['WLASL'],
		help="Not implemented yet",
		default='WLASL'
	)
	partial_parser.add_argument("-c", "--config_path", help="Path to config .ini file")
	
	# Set name selection
	partial_parser.add_argument(
		'-tt',
		'--test',
		action='store_const',
		const='test',
		dest='set_name',
		help='Test on the test set'
	)
	partial_parser.add_argument(
		'-tv',
		'--val',
		action='store_const',
		const='val',
		dest='set_name',
		help='Test on the validation set'
	)
	partial_parser.add_argument(
		'-tr',
		'--train',
		action='store_const',
		const='train',
		dest='set_name',
		help='Test on the training set'
	)
	
	partial_parser.add_argument(
		'-sf',
		'--shuffle_frames',
		action='store_true',
		help='Shuffle the frames when testing'
	)
	partial_parser.add_argument(
		'-cn',
		'--checkpoint_name',
		type=str,
		help='Checkpoint name, if not best.pth',
		default='best.pth'
	)
	partial_parser.add_argument(
		'-bg',
		'--bar_graph',
		action='store_true',
		help='Plot the bar graph'
	)
	partial_parser.add_argument(
		'-cm',
		'--confusion_matrix',
		action='store_true',
		help='Plot the confusion matrix'
	)
	partial_parser.add_argument(
		'-hm',
		'--heatmap',
		action='store_true',
		help='Plot the heatmap'
	)
	partial_parser.add_argument(
		'-dy',
		'--display',
		action='store_true',
		help='Display the graphs, if they have been selected'
	)
	partial_parser.add_argument(
		'-se',
		'--save',
		action='store_true',
		help='Save the outputs of the test'
	)
	
	return parser

def main():
	parser = get_test_parser()
	args = parser.parse_args()
	
	# Validate partial test mode requires a set selection
	if args.command == 'partial' and args.set_name is None:
		parser.error("Partial test requires one of: -tt/--test, -tv/--val, or -tr/--train")
	
	exp_no = str(int(args.exp_no)).zfill(3)
	args.exp_no = exp_no
	output = Path(f"{RUNS_PATH}/{args.split}/{args.model}_exp{exp_no}")
	save_path = output / "checkpoints"
 
	if not save_path.exists() or not save_path.is_dir() or len(list(save_path.iterdir())) == 0:
		raise ValueError(f"Invalid output: {output}, must exist and be a directory that is not empty")

	args.save_path = str(save_path)
 
	# Set config path
	if args.config_path is None:
		args.config_path = f"./configfiles/{args.split}/{args.model}_{exp_no}.ini"

	admin = AdminInfo(
		model=args.model,
		dataset=args.dataset,
		split=args.split,
		exp_no=args.exp_no,
		recover=False,
		config_path=args.config_path,
		save_path=args.save_path
	)
 
	conf = load_config(admin)
 
	if args.command == 'full':
		# Run complete test suite
		print("Running full test suite...")
		results = full_test(conf, save=args.save)
		print_dict(results)
	elif args.command == 'partial':
		# Run partial test with specified parameters
		print(f"Running partial test on {args.set_name} set...")
		results = test_run(
			config=conf,
			set_name=args.set_name,
			shuffle=args.shuffle_frames,
			check=args.checkpoint_name,
			br_graph=args.bar_graph,
			cf_matrix=args.confusion_matrix,
			heatmap=args.heatmap,
			disp=args.display,
			save=args.save,
		)
		print_dict(results)

if __name__ == "__main__":
	main()
	
	