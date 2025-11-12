from typing import Optional, Union, cast, Dict, Any, Tuple
import torch  # type: ignore
import json
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from pathlib import Path
import wandb
from wandb.sdk.wandb_run import Run
# local imports

from video_dataset import get_data_loader, get_wlasl_info
from configs import (
    load_config,
    print_config,
    get_train_parser,
    take_args,
    set_seed,
    DataInfo,
    SEED,
    RunInfo,
    WandbInfo,
    SchedInfo,
    OptimizerInfo,
)
from stopping import EarlyStopper, StopperOn
from models import get_model, norm_vals
from utils import wandb_manager
from testing import save_test_sizes


def setup_data(
    mean: Tuple[float, float, float], std: Tuple[float, float, float], config: RunInfo
):
    # NOTE: update for other datasets
    train_info = get_wlasl_info(config["admin"]["split"], set_name="train")
    val_info = get_wlasl_info(config["admin"]["split"], set_name="val")

    train_loader, num_t_classes, _, _ = get_data_loader(
        mean,
        std,
        config["data"]["frame_size"],
        config["data"]["num_frames"],
        set_info=train_info,
        batch_size=config["training"]["batch_size"],
    )
    val_loader, num_v_classes, _, _ = get_data_loader(
        mean,
        std,
        config["data"]["frame_size"],
        config["data"]["num_frames"],
        set_info=val_info,
        batch_size=1,
    )
    assert num_t_classes == num_v_classes, (
        f"Number of training classes: {num_t_classes} does not match number of validation classes: {num_v_classes}"
    )
    dataloaders = {"train": train_loader, "val": val_loader}
    return dataloaders, num_t_classes


def get_scheduler(
    optimizer: optim.Optimizer, sched_conf: Optional[SchedInfo] = None
) -> LRScheduler:
    """Get learning rate scheduler based on config."""
    no_sched = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    if sched_conf is None:
        # Identity scheduler - multiplies LR by 1.0 (no change)
        return no_sched

    warmup_sched = sched_conf["warm_up"]
    warmup_epochs = 0
    if warmup_sched is not None:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_sched["start_factor"],
            end_factor=warmup_sched["end_factor"],
            total_iters=warmup_sched["warmup_epochs"],
        )
        if sched_conf["type"] == "WarmOnly":
            return warmup_scheduler

        warmup_epochs = warmup_sched["warmup_epochs"]
    else:
        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1.0
        )

    if sched_conf["type"] == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=sched_conf["tmax"], eta_min=sched_conf["eta_min"]
        )
    elif sched_conf["type"] == "CosineAnnealingWarmRestarts":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sched_conf["t0"],
            T_mult=sched_conf["tmult"],
            eta_min=sched_conf["eta_min"],
        )
    else:
        # should not be possible to get here
        raise ValueError(f"Scheduler type {sched_conf['type']} not recognized.")

    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[warmup_epochs]
    )


def get_stopper(
    arg_dict: Optional[StopperOn] = None, wandb_run: Optional[Run] = None
) -> EarlyStopper:
    if arg_dict is None:
        return EarlyStopper(on=False)
    else:
        return EarlyStopper(arg_dict=arg_dict, wandb_run=wandb_run)


def _setup_wandb(config: RunInfo, wandb_info: WandbInfo) -> Optional[Run]:
    # wandb_info = config["wandb"]
    admin = config["admin"]

    run_name = f"{admin['model']}_{admin['split']}_exp{admin['exp_no']}"
    if admin["recover"]:
        if wandb_info["run_id"] is not None:
            run_id = wandb_info["run_id"]
        else:
            run_id = wandb_manager.get_run_id(
                run_name,
                wandb_info["entity"],
                wandb_info["project"],
                idx=-1,  # last if same name
            )
        if run_id is None:
            print("Run id not found automatically, pass as arg instead")
            return

        print(f"Resuming run with ID: {run_id}")

        run = wandb.init(
            entity=wandb_info["entity"],
            project=wandb_info["project"],
            name=run_name,
            tags=wandb_info["tags"],
            config=cast(Dict[str, Any], config),  # cast to reguler dict for wandb init
            id=run_id,
            resume="must",
        )
    else:
        print(f"Starting new run with name: {run_name}")

        run = wandb.init(
            entity=wandb_info["entity"],
            project=wandb_info["project"],
            name=run_name,
            tags=wandb_info["tags"],
            config=cast(Dict[str, Any], config),  # cast to reguler dict for wandb init
        )
    print(f"Run ID: {run.id}")
    print(f"Run name: {run.name}")  # Human-readable name
    print(f"Run path: {run.path}")  # entity/project/run_id format

    return run


def get_optimizer(model: torch.nn.Module, conf: OptimizerInfo) -> optim.AdamW:
    param_groups = [
        {
            "params": model.backbone.parameters(),
            "lr": conf["backbone_init_lr"],  # Low LR for pretrained backbone
            "weight_decay": conf["backbone_weight_decay"],  # also higher weight decay
        },
        {
            "params": model.classifier.parameters(),
            "lr": conf["classifier_init_lr"],  # Higher LR for new classifier
            "weight_decay": conf["classifier_weight_decay"],  # lower weight decay
        },
    ]

    return optim.AdamW(param_groups, eps=conf["eps"])


def train_loop(
    model_name: str,
    config: RunInfo,
    wandb_run: Run,
    load: Optional[Union[Path, str]] = None,
    save_every: int = 5,
    recover: bool = False,
    seed: Optional[int] = SEED,
) -> Optional[Dict[str, float]]:
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

    model_info = norm_vals(model_name)

    dataloaders, num_classes = setup_data(model_info["mean"], model_info["std"], config)

    drop_p = config["model_params"]["drop_p"]
    model = get_model(model_name, num_classes, drop_p)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    steps = 0
    epoch = 0
    best_val_loss = float("inf")
    best_val_acc = float("-inf")

    optimizer = get_optimizer(model, config["optimizer"])
    scheduler = get_scheduler(optimizer, config.get("scheduler", None))

    loss_func = nn.CrossEntropyLoss()

    save_path = Path(config["admin"]["save_path"])

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

    # save frame size and num frames for convenient testing
    save_test_sizes(config["data"], save_path.parent)

    # early stopping setup
    stopping_metrics = {
        "val": {"loss": 0.0, "acc": 0.0},
        "train": {"loss": 0.0, "acc": 0.0},
    }
    stopper = get_stopper(
        arg_dict=config.get("early_stopping", None), wandb_run=wandb_run
    )

    if load:
        load_path = Path(load)
        if load_path.exists():
            checkpoint = torch.load(load, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "stopper_state_dict" in checkpoint:
                stopper.load_state_dict(checkpoint["stopper_state_dict"])
            epoch = checkpoint["epoch"] + 1
            steps = checkpoint["steps"]
            # best_val_loss = checkpoint["best_val_score"]
            if "best_val_loss" in checkpoint:
                best_val_loss = checkpoint["best_val_loss"]
            if "best_val_acc" in checkpoint:
                best_val_acc = checkpoint["best_val_acc"]

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
    while epoch < config["training"]["max_epoch"] and not stopper.stop:
        print(f"Epoch {epoch}/{config['training']['max_epoch']}")
        print("-" * 10)

        epoch += 1
        # training and validation stage
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            # Reset metrics for this phase
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            # For step-level logging (only train phase)
            step_loss = 0.0
            step_corrects = 0
            step_samples = 0
            accumulated_steps = 0
            # optimizer.zero_grad()

            for item in dataloaders[phase]:
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
                    # Accumulate for step logging
                    step_loss += loss.item() * batch_size
                    step_corrects += predicted.eq(target).sum().item()
                    step_samples += batch_size

                    scaled_loss = loss / config["training"]["update_per_step"]
                    scaled_loss.backward()

                    accumulated_steps += 1

                    if accumulated_steps == config["training"]["update_per_step"]:
                        optimizer.step()
                        optimizer.zero_grad()
                        steps += 1

                        # Print step level output
                        if steps % 10 == 0:
                            avg_step_loss = step_loss / step_samples
                            step_acc = 100.0 * step_corrects / step_samples

                            print(
                                f"Step {steps}: Loss: {avg_step_loss:.4f}, "
                                f"Accuracy: {step_acc:.2f}%"
                            )

                            wandb_run.log(
                                {
                                    "Loss/Train_Step": avg_step_loss,
                                    "Accuracy/Train_Step": step_acc,
                                    "Step": steps,
                                }
                            )

                            # Reset step metrics
                            step_loss = 0.0
                            step_corrects = 0
                            step_samples = 0

                        # Reset accumulation
                        accumulated_steps = 0

            # calculate  epoch metrics
            epoch_loss = running_loss / total_samples
            epoch_acc = 100.0 * running_corrects / total_samples

            # early stopping logic
            stopping_metrics[phase]["loss"] = epoch_loss
            stopping_metrics[phase]["acc"] = epoch_acc
            if phase == stopper.phase:
                stopper.step(stopping_metrics[phase][stopper.metric])

            # print epoch level output
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
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    check_name = save_path / "best.pth"
                    torch.save(model.state_dict(), check_name)
                    print(f"Best validation loss so far: {best_val_loss:.2f}")
                    print(
                        f"New best model saved: {check_name} (Loss: {epoch_loss:.2f}%)"
                    )

                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    print(f"Best validation acc so far: {best_val_acc:.2f}")

                wandb_run.log(
                    {
                        f"Best/{phase.capitalize()}_loss": best_val_loss,
                        f"Best/{phase.capitalize()}_acc": best_val_acc,
                        "Epoch": epoch,
                    }
                )

                scheduler.step()

        # Save checkpoint
        if (
            epoch % save_every == 0
            or not epoch < config["training"]["max_epoch"]
            or stopper.stop
        ):
            checkpoint_data = {
                "epoch": epoch,
                "steps": steps,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "stopper_state_dict": stopper.state_dict(),
            }
            checkpoint_path = save_path / f"checkpoint_{str(epoch).zfill(3)}.pth"

            torch.save(checkpoint_data, checkpoint_path)

            print(f"Checkpoint saved: {checkpoint_path}")

    print("Finished training successfully")
    return {"best_val_acc": best_val_acc, "best_val_loss": best_val_loss}


def train(
    model_name: str,
    config: RunInfo,
    wandb_info: WandbInfo,
    load: Optional[Union[Path, str]] = None,
    save_every: int = 5,
    recover: bool = False,
    seed: Optional[int] = SEED,
) -> None:
    if seed is not None:
        set_seed(seed)

    wandb_run = _setup_wandb(config, wandb_info)
    if wandb_run is None:  # TODO: remove reliance on wandb
        return

    model_info = norm_vals(model_name)

    dataloaders, num_classes = setup_data(model_info["mean"], model_info["std"], config)

    drop_p = config["model_params"]["drop_p"]
    model = get_model(model_name, num_classes, drop_p)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    steps = 0
    epoch = 0
    best_val_loss = float("inf")
    best_val_acc = float("-inf")

    optimizer = get_optimizer(model, config["optimizer"])
    scheduler = get_scheduler(optimizer, config.get("scheduler", None))

    loss_func = nn.CrossEntropyLoss()

    save_path = Path(config["admin"]["save_path"])

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
    stopping_metrics = {
        "val": {"loss": 0.0, "acc": 0.0},
        "train": {"loss": 0.0, "acc": 0.0},
    }
    stopper = get_stopper(
        arg_dict=config.get("early_stopping", None), wandb_run=wandb_run
    )

    if load:
        load_path = Path(load)
        if load_path.exists():
            checkpoint = torch.load(load, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "stopper_state_dict" in checkpoint:
                stopper.load_state_dict(checkpoint["stopper_state_dict"])
            epoch = checkpoint["epoch"] + 1
            steps = checkpoint["steps"]
            # best_val_loss = checkpoint["best_val_score"]
            if "best_val_loss" in checkpoint:
                best_val_loss = checkpoint["best_val_loss"]
            if "best_val_acc" in checkpoint:
                best_val_acc = checkpoint["best_val_acc"]

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
    while epoch < config["training"]["max_epoch"] and not stopper.stop:
        print(f"Epoch {epoch}/{config['training']['max_epoch']}")
        print("-" * 10)

        epoch += 1
        # training and validation stage
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            # Reset metrics for this phase
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            # For step-level logging (only train phase)
            step_loss = 0.0
            step_corrects = 0
            step_samples = 0
            accumulated_steps = 0
            # optimizer.zero_grad()

            for item in dataloaders[phase]:
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
                    # Accumulate for step logging
                    step_loss += loss.item() * batch_size
                    step_corrects += predicted.eq(target).sum().item()
                    step_samples += batch_size

                    scaled_loss = loss / config["training"]["update_per_step"]
                    scaled_loss.backward()

                    accumulated_steps += 1

                    if accumulated_steps == config["training"]["update_per_step"]:
                        optimizer.step()
                        optimizer.zero_grad()
                        steps += 1

                        # Print step level output
                        if steps % 10 == 0:
                            avg_step_loss = step_loss / step_samples
                            step_acc = 100.0 * step_corrects / step_samples

                            print(
                                f"Step {steps}: Loss: {avg_step_loss:.4f}, "
                                f"Accuracy: {step_acc:.2f}%"
                            )

                            wandb_run.log(
                                {
                                    "Loss/Train_Step": avg_step_loss,
                                    "Accuracy/Train_Step": step_acc,
                                    "Step": steps,
                                }
                            )

                            # Reset step metrics
                            step_loss = 0.0
                            step_corrects = 0
                            step_samples = 0

                        # Reset accumulation
                        accumulated_steps = 0

            # calculate  epoch metrics
            epoch_loss = running_loss / total_samples
            epoch_acc = 100.0 * running_corrects / total_samples

            # early stopping logic
            stopping_metrics[phase]["loss"] = epoch_loss
            stopping_metrics[phase]["acc"] = epoch_acc
            if phase == stopper.phase:
                stopper.step(stopping_metrics[phase][stopper.metric])

            # print epoch level output
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
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    check_name = save_path / "best.pth"
                    torch.save(model.state_dict(), check_name)
                    print(f"Best validation loss so far: {best_val_loss:.2f}")
                    print(
                        f"New best model saved: {check_name} (Loss: {epoch_loss:.2f}%)"
                    )

                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    print(f"Best validation acc so far: {best_val_acc:.2f}")

                wandb_run.log(
                    {
                        f"Best/{phase.capitalize()}_loss": best_val_loss,
                        f"Best/{phase.capitalize()}_acc": best_val_acc,
                        "Epoch": epoch,
                    }
                )

                scheduler.step()

        # Save checkpoint
        if (
            epoch % save_every == 0
            or not epoch < config["training"]["max_epoch"]
            or stopper.stop
        ):
            checkpoint_data = {
                "epoch": epoch,
                "steps": steps,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "stopper_state_dict": stopper.state_dict(),
            }
            checkpoint_path = save_path / f"checkpoint_{str(epoch).zfill(3)}.pth"

            torch.save(checkpoint_data, checkpoint_path)

            print(f"Checkpoint saved: {checkpoint_path}")

    print("Finished training successfully")


def main():
    parser = get_train_parser()

    args = parser.parse_args()

    maybe_args = take_args(parsed_args=args, ask_bf_ovrite=not args.no_ask)
    if isinstance(maybe_args, tuple):
        admin, wandb_info = maybe_args
    else:
        print(f"Need tuple not: {type(maybe_args)}")
        return
    config = load_config(admin)

    # confirm config
    print_config(config)
    if not args.no_ask:
        proceed = input("Confirm: y/n: ")
        if proceed.lower() != "y":
            print("Training cancelled")
            return

    # setup wandb run
    run = _setup_wandb(config, wandb_info)
    if run is None:
        return

    print(f"Run ID: {run.id}")
    print(f"Run name: {run.name}")  # Human-readable name
    print(f"Run path: {run.path}")  # entity/project/run_id format

    train_loop(admin["model"], config, run, recover=admin["recover"])
    run.finish()


if __name__ == "__main__":
    main()
    # list_runs()
