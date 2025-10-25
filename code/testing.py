from typing import Optional, Union, Tuple, Dict, Literal, List, Any

import torch
import json
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from torchvision.transforms import v2
from video_transforms import Shuffle
from video_dataset import VideoDataset
from torch.utils.data import DataLoader
import tqdm
from pathlib import Path
import utils
import gc
from argparse import ArgumentParser

# locals
from visualise import plot_confusion_matrix, plot_bar_graph, plot_heatmap
from models import norm_vals, get_model
from configs import LABEL_SUFFIX
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

    assert isinstance(report, Dict), "Sklearn machine broke"

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


def test_topk_clsrep(
    model: torch.nn.Module,
    test_loader: DataLoader[VideoDataset],
    seed: Optional[int] = None,
    verbose: bool = False,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[
    Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], List[int], List[int]
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
    if seed is not None:
        utils.set_seed(seed)

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

    return topk_res, cls_report, all_targets, all_preds


def _get_test_loader(
    perm: Optional[torch.Tensor],
    model_norms: Dict,
    frame_size: int,
    num_frames: int,
    root: Path,
    labels: Path,
    set: Literal["test", "val"],
) -> DataLoader[VideoDataset]:
    if perm:
        maybe_shuffle_t = Shuffle(perm)
    else:
        maybe_shuffle_t = v2.Lambda(lambda x: x)

    final_t = v2.Compose(
        [
            maybe_shuffle_t,
            v2.Lambda(lambda x: x.float() / 255.0),
            v2.Normalize(mean=model_norms["mean"], std=model_norms["std"]),
            v2.Lambda(lambda x: x.permute(1, 0, 2, 3)),
        ]
    )

    test_transforms = v2.Compose([v2.CenterCrop(frame_size), final_t])

    instances = labels / f"{set}_instances_{LABEL_SUFFIX}"
    classes = labels / f"{set}_classes_{LABEL_SUFFIX}"

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
        drop_last=False,
    )


def collect_results(res_p: Path):
    with open(res_p, "r") as f:
        res = json.load(f)
    return res


def load_info(dirp: Path, checkname: str):
    resd = {}
    fnames = list(dirp.glob(f"{checkname}_*.json"))
    print('here')
    for fn in fnames:
        with open(fn, "r") as f:
            resd[fn.name.replace(".json", "")] = json.load(f)
    return resd


def test_run(
    config: Dict[str, Any],
    perm: Optional[torch.Tensor] = None,
    test_val: bool = False,
    test_test: bool = True,
    check: str = "best.pth",
    br_graph: bool = False,
    cf_matrix: bool = False,
    heatmap: bool = False,
    disp: bool = False,
    save: bool = True,
    seed: int = 42,
    re_test: bool = False,
) -> Dict[str, Any]:
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
            seed (int, optional): Testing random seed. Defaults to 42.
            re_test (bool, optional): Test even if results already saved. Defaults to False.

    Returns:
            Optional[Dict[str, Any]]: Results if correct parameters.
    """

    utils.set_seed(seed)

    admin = config["admin"]
    model_name = admin["model"]
    data = config["data"]

    model_norms = norm_vals(model_name)
    results = {}

    save_path = Path(admin["save_path"])

    output = save_path.parent / "results"

    if output.exists() and not re_test:
        return load_info(output, check.replace('.pth', ''))

    if save:
        output.mkdir(exist_ok=True)

    tloaders = {}

    if not test_test and not test_val:
        return results

    if test_test:
        tloaders["test"] = _get_test_loader(
            perm,
            model_norms,
            data["frame_size"],
            data["num_frames"],
            Path(admin["root"]),
            Path(admin["labels"]),
            "test",
        )

    if test_val:
        tloaders["val"] = _get_test_loader(
            perm,
            model_norms,
            data["frame_size"],
            data["num_frames"],
            Path(admin["root"]),
            Path(admin["labels"]),
            "val",
        )

    keys = list(tloaders.keys())
    gen_loader = tloaders[keys[0]]
    assert isinstance(gen_loader.dataset, VideoDataset), (
        "This function uses a custom dataset"
    )
    num_classes = len(set(gen_loader.dataset.classes))

    model = get_model(model_name, num_classes, drop_p=0.0)

    check_path = save_path / check

    print(f"Loading weights from: {check_path}")

    checkpoint = torch.load(check_path)

    if check_path.name == "best.pth":
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    if perm is not None:
        suffix = "-top-k_shuffled.json"
    else:
        suffix = "-top-k.json"

    for set_name, tloader in tloaders.items():
        print(f"Testing on {set_name} set")
        fname = check_path.name.replace(".pth", f"_{set_name}{suffix}")
        save2 = output / fname
        save2 = None if not save else save2

        topk_res, cls_report, all_targets, all_preds = test_topk_clsrep(
            model=model,
            test_loader=tloader,
            seed=seed,
            verbose=False,
            save_path=save2,
        )
        results[fname.replace(".json", "")] = topk_res
        heatmap, br_graph, cf_matrix = False, False, False  # skip plots if loading

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
            assert isinstance(gen_loader.dataset, VideoDataset), (
                "This function uses a custom dataset"
            )
            plot_confusion_matrix(
                y_true=all_targets,
                y_pred=all_preds,
                title=f"{set_name.capitalize()} set Confusion Matrix",
                save_path=save2,
                disp=disp,
            )

    return results


##################### Multiple-run testing utility #########################


def find_best_checkpnt(run_idx: int):
    with open("queRuns.json", "r") as f:
        all_runs = json.load(f)

    old_runs = all_runs["old_runs"]
    run = old_runs[run_idx]
    save_path = Path(run["admin"]["save_path"])
    print(f"Looking in: {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="test.py")

    parser.add_argument()
