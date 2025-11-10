from typing import Optional, Union, Tuple, Dict, List, Any, TypedDict, Literal
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
from configs import set_seed
from video_dataset import VideoDataset, get_data_loader, get_wlasl_info
from models import avail_models
from configs import get_avail_splits, load_config, AdminInfo, RunInfo, RUNS_PATH
from utils import print_dict
#################################### Utilities #################################


def cleanup_memory():
	"""Cleanup GPU and CPU memory"""
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.cuda.synchronize()
	gc.collect()

#################################### Helper classes #############################

class TopKRes(TypedDict):
	top1: float
	top5: float
	top10: float

class BaseRes(TypedDict):
	top_k_average_per_class_acc: TopKRes
	top_k_per_instance_acc: TopKRes
	
class ShuffRes(BaseRes):
	perm: List[int]
	shannon_entropy: float

class CompRes(TypedDict):
	check_name: str
	best_val_loss: Optional[float]
	test: BaseRes
	val: BaseRes
	test_shuff: ShuffRes

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

	with torch.no_grad():
		for item in tqdm.tqdm(test_loader, desc="Testing"):
			data, target = item["frames"], item["label_num"]
			data, target = data.to(device), target.to(device)

			predictions = model(data)

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
		)
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
		save2 = output / fname if save else None
		plot_heatmap(
			report=cls_report,
			title=f"{set_name.capitalize()} set Classification Report",
			save_path=save2,
			disp=disp,
		)

	if br_graph:
		fname = check_path.name.replace(".pth", f"_{set_name}-bargraph.png")
		save2 = output / fname if save else None
		plot_bar_graph(
			report=cls_report,
			title=f"{set_name.capitalize()} set Classification Report",
			save_path=save2,
			disp=disp,
		)

	if cf_matrix:
		fname = check_path.name.replace(".pth", f"_{set_name}-confmat.png")
		save2 = output / fname if save else None
		plot_confusion_matrix(
			y_true=all_targets,
			y_pred=all_preds,
			title=f"{set_name.capitalize()} set Confusion Matrix",
			save_path=save2,
			disp=disp,
		)

	return results


##################### Multiple-run testing utility #########################


	
def get_test_parser(prog: Optional[str] = None,desc: str = "Test a model") -> ArgumentParser:
	"""Get parser for testing configuration

	Args:
		prog (Optional[str], optional): Script name, (e.g. testing.py). Defaults to None.
		desc (str, optional): Program desctiption. Defaults to "Test a model".

	Returns:
		ArgumentParser: Parser which takes testing arguments
	"""
	parser = ArgumentParser(description=desc, prog=prog)
	models_available = avail_models()
	splits_available = get_avail_splits()
	parser.add_argument(
		"model",
		type=str,
		choices=models_available,
		help=f"Model name from one of the implemented models: {models_available}",
	)
	parser.add_argument(
		"split",
		type=str,
		choices=splits_available,
		help=f"The class split, one of:  {', '.join(splits_available)}",
	)
	parser.add_argument("exp_no", type=int, help="Experiment number (e.g. 10)")
	parser.add_argument(
		'-ds',
		'--dataset',
		type=str,
		choices=['WLASL'],
		help="Not implemented yet",
		default='WLASL'
	)
	parser.add_argument("-c", "--config_path", help="path to config .ini file")
	parser.add_argument(
		'-sf',
		'--shuffle_frames',
		action='store_true',
		help='Shuffle the frames when testing'
	)
	parser.add_argument(
		'-sn',
		'--set_name',
		type=str,
		choices=['train', 'val', 'test'],
		help='Which set to test on'
	)
	parser.add_argument(
		'-cn',
		'--checkpoint_name',
		type=str,
		help='Checkpoint name, if not best.pth',
		default='best.pth'
	)
	parser.add_argument(
		'-bg',
		'--bar_graph',
		action='store_true',
		help='Plot the bar graph'
	)
	parser.add_argument(
		'-cm',
		'--confusion_matrix',
		action='store_true',
		help='Plot the confusion matrix'
	)
	parser.add_argument(
		'-hm',
		'--heatmap',
		action='store_true',
		help='Plot the heatmap'
	)
	parser.add_argument(
		'-dy',
		'--display',
		action='store_true',
		help='Display the graphs, if they have been selected'
	)
	parser.add_argument(
		'-se',
		'--save',
		action='store_true',
		help='Save the outputs of the test'
	)
	parser.add_argument(
		'-rt',
		'--re_test',
		action='store_true',
		help="Re-run test, even if results files already exist"
	)
	return parser

if __name__ == "__main__":
	# find_best_checkpnt(0)
	
	parser = get_test_parser()
	args = parser.parse_args()
	
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
 
	arg_dict = vars(args)
 
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
 
	results = test_run(
		config=conf,
		shuffle=args.shuffle_frames,
		set_name=args.set_name,
		check=args.checkpoint_name,
		br_graph=args.bar_graph,
		cf_matrix=args.confusion_matrix,
		heatmap=args.heatmap,
		disp=args.display,
		save=args.save,
	)
	print_dict(results)