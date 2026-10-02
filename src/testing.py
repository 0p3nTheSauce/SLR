import gc
import json
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch
import tqdm
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

from src.configs import (
    get_avail_splits,
    get_model_checkpoint_dir,
    get_model_exp_dir,
    get_model_results_dir,
    set_seed,
)
from src.models import avail_models, get_model
from src.run_types import (
    BaseRes,
    CompRes,
    DataInfo,
    MinInfo,
    ShuffleT,
    ShuffRes,
    TemporalAugs,
    TopKRes,
    is_sampler_config,
)
from src.video_dataset import (
    AVAIL_SETS,
    AVAIL_SPLITS,
    VideoDataset,
    get_data_set,
    get_wlasl_info,
)

# locals
from src.visualise import plot_bar_graph, plot_confusion_matrix, plot_heatmap

# constants

DATA_FNAME = "data_info.json"


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

    assert isinstance(report, dict), "Sklearn machine broke"

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
    fstr = f"top-k average per class acc: {top1_per_class}, {top5_per_class}, {top10_per_class}"
    fstr2 = f"top-k per instance acc: {top1_per_instance}, {top5_per_instance}, {top10_per_instance}"
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
    save_path: str | Path | None = None,
) -> tuple[BaseRes, dict[str, dict[str, float]], list[int], list[int]]:
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

            # for loss
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
    assert isinstance(cls_report, dict), "Sklearn machine broke"

    # per class accuracy
    top1_per_class = np.mean(top1_tp / (top1_tp + top1_fp))
    top5_per_class = np.mean(top5_tp / (top5_tp + top5_fp))
    top10_per_class = np.mean(top10_tp / (top10_tp + top10_fp))
    top1_per_instance = correct / len(test_loader)
    top5_per_instance = correct_5 / len(test_loader)
    top10_per_instance = correct_10 / len(test_loader)
    fstr = f"top-k average per class acc: {top1_per_class}, {top5_per_class}, {top10_per_class}"
    fstr2 = f"top-k per instance acc: {top1_per_instance}, {top5_per_instance}, {top10_per_instance}"
    print(fstr)
    print(fstr2)

    # loss
    epoch_loss = running_loss / total_samples

    print(f"Averag Loss: {epoch_loss:.2f}")

    topk_res = BaseRes(
        top_k_average_per_class_acc=TopKRes(
            top1=float(top1_per_class),
            top5=float(top5_per_class),
            top10=float(top10_per_class),
        ),
        top_k_per_instance_acc=TopKRes(
            top1=top1_per_instance, top5=top5_per_instance, top10=top10_per_instance
        ),
        average_loss=epoch_loss,
    )
    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(topk_res.model_dump(), f, indent=2)

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


def get_last_sampler(conf_list: list[TemporalAugs]):
    samplers = [(i, c) for i, c in enumerate(conf_list) if is_sampler_config(c)]
    return samplers[-1]


def setup_data(
    set_name: AVAIL_SETS,
    split: AVAIL_SPLITS,
    data_info: DataInfo,
    shuffle: bool = False,  # override for shuffle test
    pin_memory: bool = True,
    # video_length: Optional[int] = None
) -> tuple[DataLoader[VideoDataset], int, list[int] | None, float | None]:
    test_info = get_wlasl_info(split, set_name=set_name)

    # make copy to hand off to get_data_set to avoid shuffle injection in final config
    data_info_cp = data_info.model_copy(deep=True)

    aug_info = (
        data_info_cp.train_augs if set_name == "train" else data_info_cp.test_augs
    )
    if aug_info is None:
        raise ValueError(
            "Augmentation info must be provided in data_info for both train and test sets."
        )

    if shuffle:
        try:
            i, s = get_last_sampler(aug_info.temporal_aug)
            video_length = s.target_length
        except IndexError:
            raise ValueError(
                "At least one frame sampler has to be present to extract video length"
            )

        aug_info.temporal_aug.insert(i + 1, ShuffleT(num_frames=video_length))

    test_dataset, perm, shanon_entropy = get_data_set(
        set_info=test_info, data_info=data_info_cp
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return test_loader, test_dataset.num_classes, perm, shanon_entropy


def checkpoint_dir_to_result_dir(checkpoint_dir: Path) -> Path:
    """Find the corresponding results path for the provided checkpoint path

    Args:
                    checkpoint_dir (Path): Model checkpoints directory

    Returns:
                    Path: Results save path
    """
    chck_str = str(checkpoint_dir)
    reg_digits = re.search(r"\d+$", chck_str)
    if reg_digits is not None:
        reg_digits = int(reg_digits.group())
    return get_model_results_dir(checkpoint_dir.parent, reg_digits)


def test_run(
    admin: MinInfo,
    data: DataInfo,
    set_name: AVAIL_SETS,
    shuffle: bool = False,
    check: str = "best.pth",
    br_graph: bool = False,
    cf_matrix: bool = False,
    heatmap: bool = False,
    disp: bool = False,
    save: bool = True,
    save_img: bool = False,
    out_dir: Path | None = None,
) -> tuple[BaseRes | ShuffRes, dict[str, dict[str, float]], list[int], list[int]]:
    """Test a model on one test set, on one split.

    Args:
        admin (MinInfo): Information needed to load model weights.
        data (DataInfo): Information needed to locate data and apply transforms.
        set_name (AVAIL_SETS): Which set to test on.
        shuffle (bool, optional): Whether to shuffle the frames. Defaults to False.
        check (str, optional): Name of checkpoint. Defaults to "best.pth".
        br_graph (bool, optional): Plot bar graph. Defaults to False.
        cf_matrix (bool, optional): Plot confusion matrix. Defaults to False.
        heatmap (bool, optional): Plot heatmap. Defaults to False.
        disp (bool, optional): Display plots. Defaults to False.
        save (bool, optional): Save results. Defaults to True.
        save_img (bool, optional): Save images. Defaults to False.
        out_dir (Path | None, optional): Write outputs here instead of the
            directory derived from admin.save_path. Defaults to None.

    Returns:
        tuple[BaseRes | ShuffRes, dict[str, dict[str, float]], list[int], list[int]]: results (top-k + loss), cls_report, all_targets, all_preds
    """

    set_seed(admin.seed)

    model_name = admin.model

    results = {}

    save_path = Path(admin.save_path)

    output = out_dir if out_dir is not None else checkpoint_dir_to_result_dir(save_path)

    if save or save_img:
        output.mkdir(parents=True, exist_ok=True)

    dloader, num_classes, m_permt, m_sh_et = setup_data(
        set_name=set_name,
        split=admin.split,
        data_info=data,
        shuffle=shuffle,
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

    if m_permt is not None and m_sh_et is not None:  # shuffled
        results = ShuffRes(
            top_k_average_per_class_acc=topk_res.top_k_average_per_class_acc,
            top_k_per_instance_acc=topk_res.top_k_per_instance_acc,
            average_loss=topk_res.average_loss,
            perm=m_permt,
            shannon_entropy=m_sh_et,
        )
    else:
        results = topk_res

    if save:
        with open(save2, "w") as f:
            json.dump(results.model_dump(), f, indent=4)

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

    return results, cls_report, all_targets, all_preds


def save_test_sizes(data_specs: DataInfo, save_dir: Path):
    """Save the frame size and number of frames for convenient testing.

    Args:
                                    data_specs (DataInfo): Dictionary containing frame_size and num_frames
                                    save_dir (Path): Experiment directory (can retrieve with save_path.parent from AdminInfo)
    """
    fname = save_dir / DATA_FNAME
    with open(fname, "w") as f:
        json.dump(data_specs.model_dump(), f, indent=4)


def load_data_info(path: Path) -> DataInfo:
    """Load a `DataInfo` directly from a `data_info.json`-shaped file.

    Args:
        path (Path): Path to the JSON file itself (not its containing directory).

    Returns:
        DataInfo: The frame size/num_frames/augs this file describes.
    """
    with open(path, "r") as f:
        info = json.load(f)

    return DataInfo.model_validate(info)


def load_test_sizes(save_dir: Path) -> DataInfo:
    """Load the frame size and number of frames for convenient testing. This function needs
    Filename symmetry with save_test_sizes

    Args:
        save_dir (Path): Experiment directory containing data_info.json

    Returns:
        DataInfo: The frame size/num_frames/augs this run trained with.
    """
    return load_data_info(save_dir / DATA_FNAME)


def load_comp_res(save_path: Path) -> CompRes:
    """Load results from file"""
    with open(save_path, "r") as f:
        data = json.load(f)
    return CompRes.model_validate(data)


def get_res_path(save_path: Path) -> Path:
    out_dir = checkpoint_dir_to_result_dir(save_path)
    res_path = out_dir / "best_val_loss.json"  # TODO: add other types of saves?
    return res_path


def get_cls_rep_path(save_path: Path) -> Path:
    out_dir = checkpoint_dir_to_result_dir(save_path)
    res_path = out_dir / "cls_rep_all_targets_preds.json"
    return res_path


# TODO: can be simplified to take only admin info if each folder keeps a file on what frame rate and image size to test with
def full_test(
    admin: MinInfo,
    data: DataInfo | None = None,
    save: bool = True,
    re_test: bool = False,
    out_dir: Path | None = None,
) -> CompRes:
    # - The shuffled results additionally contain the permutation used, and it's shannon entropy
    """Complete test, which includes:
    - The best validation loss, and accuracy for the whole training run
    - The test, val and 'shuffled test' results.
    - The test, val and shuffled results all contain the average loss, topk per instance, and per class accuracy.

    Args:
        admin (MinInfo): Dictionary containing information on where to load weights and which dataset to use
        data (Optional[DataInfo], optional): Dictionary containing frame_size and num_frames, can be loaded automatically if data_info.json file exists. Defaults to None.
        save (bool, optional): Whether to save. Defaults to True.
        re_test (bool, optional): Re-test even if files exist. Defaults to False.
        out_dir (Path | None, optional): Write outputs here instead of the
            directory derived from admin.save_path. Defaults to None.

    Raises:
        Exception: If there is an error loading data from data_info.json

    Returns:
        CompRes: A results object (as described above).
    """
    save_path = Path(admin.save_path)

    # output
    if out_dir is not None:
        res_path = out_dir / "best_val_loss.json"
        cls_rep_path = out_dir / "cls_rep_all_targets_preds.json"
    else:
        res_path = get_res_path(save_path)
        cls_rep_path = get_cls_rep_path(save_path)

    # dont retest if exists
    if res_path.exists() and not re_test:
        return load_comp_res(res_path)
    
   
    # optionall load data
    if data is None:
        try:
            data = load_test_sizes(save_path.parent)
        except Exception:
            print(
                f"Full test failed to automatically load data info from: {save_path.parent / DATA_FNAME}"
            )
            print("Create the file, or pass as parameter instead")
            raise

    # load checkpoint
    files = sorted(save_path.iterdir())
    last_check = torch.load(files[-1])
    # extract metrics
    best_val_acc = last_check["best_val_acc"]
    best_val_loss = last_check["best_val_loss"]

    # test set
    test, cls_report, all_targets, all_preds = test_run(
        admin,
        data,
        "test",
        save=False,
    )
    # validation set
    val, _, _, _ = test_run(admin, data, "val", save=False)

    results = CompRes(
        check_name="best_val",
        best_val_acc=best_val_acc,
        best_val_loss=best_val_loss,
        test=test,
        val=val,
        # test_shuff=cast(ShuffRes, test_shuff),
    )

    class_report = {
        "cls_report": cls_report,
        "all_targets": [int(i) for i in all_targets],
        "all_preds": [int(i) for i in all_preds],
    }

    if save:
        #results folder must exist
        res_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(res_path, "w") as f:
            json.dump(results.model_dump(), f, indent=4)
        with open(cls_rep_path, "w") as f:
            # json.dump(class_report.model_dump(), f, indent=4)
            json.dump(class_report, f, indent=4)

    return results


def get_test_parser(
    prog: str | None = None, desc: str = "Test a model"
) -> ArgumentParser:
    """Get parser for testing configuration with subparsers for full/partial test modes

    Args:
                                    prog (Optional[str], optional): Script name, (e.g. testing.py). Defaults to None.
                                    desc (str, optional): Program desctiption. Defaults to "Test a model".

    Returns:
                                    ArgumentParser: Parser which takes testing arguments
    """
    models_available = avail_models()
    splits_available = get_avail_splits()

    # Shared between 'full' and 'partial': identify which run to test, either
    # a regular experiment (exp_no) or a sweep trial (sweep + run_id).
    common = ArgumentParser(add_help=False)
    common.add_argument(
        "model",
        type=str,
        choices=models_available,
        help=f"Model name from one of the implemented models: {models_available}",
    )
    common.add_argument(
        "split",
        type=str,
        choices=splits_available,
        help=f"The class split, one of: {', '.join(splits_available)}",
    )
    common.add_argument(
        "exp_no",
        type=int,
        nargs="?",
        default=None,
        help="Experiment number (e.g. 10). Omit when identifying a sweep trial with --sweep/--run_id instead.",
    )
    common.add_argument(
        "-sw",
        "--sweep",
        type=str,
        default=None,
        help="Sweep id, to test a sweep trial instead of a regular experiment (requires --run_id)",
    )
    common.add_argument(
        "-ri",
        "--run_id",
        type=str,
        default=None,
        help="Sweep trial's wandb run id (requires --sweep)",
    )
    common.add_argument(
        "-cp_d_n",
        "--checkpoint_dir_no",
        type=int,
        help="Checkpoint directory number (e.g. 10). Useful if multiple checkpoint directories",
        default=None,
    )
    common.add_argument(
        "-ds",
        "--dataset",
        type=str,
        choices=["WLASL"],
        help="Dataset name",
        default="WLASL",
    )
    common.add_argument(
        "-wp",
        "--weight_path",
        type=str,
        default=None,
        help="Checkpoint directory, overriding exp_no/--sweep+--run_id entirely (manual override, e.g. for a checkpoint outside the runs/ convention)",
    )
    common.add_argument(
        "-dp",
        "--data_path",
        type=str,
        default=None,
        help="Path to a data_info.json-shaped file, overriding the one normally read from the experiment directory",
    )
    common.add_argument(
        "-op",
        "--out_path",
        type=str,
        default=None,
        help="Directory to write results to, overriding the one normally derived from the checkpoint directory",
    )

    parser = ArgumentParser(description=desc, prog=prog)

    # Create subparsers for 'full' and 'partial' commands
    subparsers = parser.add_subparsers(dest="command", help="Test mode", required=True)

    # ============ FULL TEST SUBPARSER ============
    full_parser = subparsers.add_parser(
        "full",
        parents=[common],
        help="Run full test suite (test, val, and shuffled test with all visualizations)",
    )
    full_parser.add_argument(
        "-se", "--save", action="store_true", help="Save the outputs of the test"
    )
    full_parser.add_argument(
        "-rt", "--re_test", action="store_true", help="Retest if results exist"
    )

    # ============ PARTIAL TEST SUBPARSER ============
    partial_parser = subparsers.add_parser(
        "partial",
        parents=[common],
        help="Run partial test on a specific set with custom options",
    )

    partial_parser.add_argument(
        "set_name",
        type=str,
        choices=["test", "val", "train"],
        help="Which set to test on",
    )
    partial_parser.add_argument(
        "-sf",
        "--shuffle_frames",
        action="store_true",
        help="Shuffle the frames when testing",
    )
    partial_parser.add_argument(
        "-cp_n",
        "--checkpoint_name",
        type=str,
        help="Checkpoint name, if not best.pth",
        default="best.pth",
    )
    partial_parser.add_argument(
        "-bg", "--bar_graph", action="store_true", help="Plot the bar graph"
    )
    partial_parser.add_argument(
        "-cm",
        "--confusion_matrix",
        action="store_true",
        help="Plot the confusion matrix",
    )
    partial_parser.add_argument(
        "-hm", "--heatmap", action="store_true", help="Plot the heatmap"
    )
    partial_parser.add_argument(
        "-dy",
        "--display",
        action="store_true",
        help="Display the graphs, if they have been selected",
    )
    partial_parser.add_argument(
        "-se", "--save", action="store_true", help="Save the outputs of the test"
    )

    return parser


def _resolve_save_path(args: Namespace) -> Path:
    """Resolve the checkpoint directory to test, from exactly one of: a
    regular experiment (`exp_no`), a sweep trial (`sweep` + `run_id`), or a
    direct manual override (`weight_path`).

    Sweep trials don't live under the sequential `exp{NNN}` numbering
    (see `get_model_exp_dir`) -- they get their own directory, namespaced by
    sweep id and wandb run id, via `get_sweep_exp_dir`. `weight_path` bypasses
    both schemes entirely, for a checkpoint that doesn't live under the
    `runs/` convention at all.
    """
    if args.weight_path is not None:
        if args.exp_no is not None or args.sweep is not None or args.run_id is not None:
            raise ValueError("--weight_path cannot be combined with exp_no/--sweep/--run_id")
        if args.checkpoint_dir_no is not None:
            raise ValueError("--checkpoint_dir_no has no effect when --weight_path is given directly")
        return Path(args.weight_path)

    if args.sweep is not None or args.run_id is not None:
        from src.sweeping import get_sweep_exp_dir

        if args.sweep is None or args.run_id is None:
            raise ValueError("--sweep and --run_id must be provided together")
        if args.exp_no is not None:
            raise ValueError("exp_no cannot be combined with --sweep/--run_id")
        output = get_sweep_exp_dir(args.split, args.model, args.sweep, args.run_id)
        return get_model_checkpoint_dir(output, args.checkpoint_dir_no)

    if args.exp_no is None:
        raise ValueError("Either exp_no, --sweep/--run_id, or --weight_path must be provided")
    output = get_model_exp_dir(split=args.split, model=args.model, exp_no=args.exp_no)
    return get_model_checkpoint_dir(output, args.checkpoint_dir_no)


def main():
    parser = get_test_parser()
    args = parser.parse_args()

    save_path = _resolve_save_path(args)

    if (
        not save_path.exists()
        or not save_path.is_dir()
        or len(list(save_path.iterdir())) == 0
    ):
        raise ValueError(
            f"Invalid save path: {save_path}, must exist and be a directory that is not empty"
        )

    args.save_path = str(save_path)

    # Create minimal admin info
    admin = MinInfo(
        model=args.model,
        dataset=args.dataset,
        split=args.split,
        save_path=args.save_path,
    )

    if args.data_path is not None:
        data_info_path = Path(args.data_path)
        data = load_data_info(data_info_path)
    else:
        data_info_path = save_path.parent / DATA_FNAME
        try:
            data = load_test_sizes(save_path.parent)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {data_info_path}.")
    print(f"Loaded data info from {data_info_path}")

    out_dir = Path(args.out_path) if args.out_path is not None else None

    if args.command == "full":
        # Run complete test suite
        print("Running full test suite...")
        results = full_test(
            admin, data=data, save=args.save, re_test=args.re_test, out_dir=out_dir
        )
        print(json.dumps(results.model_dump(), indent=4))
    elif args.command == "partial":
        # Run partial test with specified parameters
        print(f"Running partial test on {args.set_name} set...")
        results, _, _, _ = test_run(
            admin=admin,
            data=data,
            set_name=args.set_name,
            shuffle=args.shuffle_frames,
            check=args.checkpoint_name,
            br_graph=args.bar_graph,
            cf_matrix=args.confusion_matrix,
            heatmap=args.heatmap,
            disp=args.display,
            save=args.save,
            out_dir=out_dir,
        )
        print(json.dumps(results.model_dump(), indent=4))


if __name__ == "__main__":
    main()
