from typing import Optional, Dict, Any
import shlex
import configs
import json
import torch
import torch.nn as nn
import gc
import getpass
import subprocess
from pathlib import Path

# from .server import connect_manager
import multiprocessing as mp
import time
from .shell import QueShell

# from que.shell import QueShell
from .core import (
	Que,
	connect_manager,
	_get_basic_logger,
	TO_RUN,
	CUR_RUN,
	OLD_RUNS,
	FAIL_RUNS,
	QueLocation,
)
from .tmux import tmux_manager
from run_types import (
	CentreCropConfig,
	RandomCropConfig,
	ScaleAndPadConfig,
	RunInfo,
	FailedExp,
	CompExpInfo,
	ExpInfo,
	HorizontalFlipConfig,
	AugInfo,

)

KEYS = [TO_RUN, CUR_RUN, OLD_RUNS, FAIL_RUNS]


def tmuxer():
	tman = tmux_manager()
	tman.join_session()


def timestamp():
	q = Que(_get_basic_logger())
	q.save_state(timestamp=True)


def check_gpu_memory():
	"""Helper to check GPU memory usage"""
	if torch.cuda.is_available():
		allocated = torch.cuda.memory_allocated() / 1024**3
		reserved = torch.cuda.memory_reserved() / 1024**3
		print(
			f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
		)


def gpu_worker_fixed():
	"""Simulates PyTorch GPU work with proper cleanup"""
	print("Worker: Starting GPU work...")

	try:
		model = nn.Sequential(
			nn.Linear(1000, 2000),
			nn.ReLU(),
			nn.Linear(2000, 2000),
			nn.ReLU(),
			nn.Linear(2000, 1000),
		).cuda()

		data = torch.randn(100, 1000).cuda()

		for i in range(10):
			output = model(data)
			loss = output.sum()
			loss.backward()

		print("Worker: Work complete")
		check_gpu_memory()

	finally:
		# Proper cleanup
		print("Worker: Cleaning up...")
		# del model, data
		gc.collect()
		torch.cuda.empty_cache()
		torch.cuda.synchronize()
		torch.cuda.ipc_collect()


def sim_leak():
	# CRITICAL: Use 'spawn' method
	# mp.set_start_method('spawn', force=True)

	print("=== Initial State ===")
	# check_gpu_memory()

	print("\n=== Starting Process (with fix) ===")
	p = mp.Process(target=gpu_worker_fixed)
	p.start()
	p.join()

	print("\n=== Starting Process (with fix2) ===")
	p = mp.Process(target=gpu_worker_fixed)
	p.start()
	p.join()

	print("\n=== Memory should be cleared! ===")
	print("\n=== After Process Completes ===")
	time.sleep(1)
	check_gpu_memory()


def constant():
	from .core import SERVER_MODULE_PATH

	print(SERVER_MODULE_PATH)


def activate_conda_env(env_name: str):
	tman = tmux_manager()
	tman.activate_conda_env(env_name)


def show_help():

	server = connect_manager()
	shell = QueShell(server)
	daemon_parser = shell._get_daemon_parser()
	print(daemon_parser.description)


def reconnect():
	server = connect_manager()
	server_controller = server.get_server_context()
	print(server_controller.get_state())
	ready = input("Retry? (y/n): ")
	if ready.lower() == "y":
		_cleanup(server_controller)
		print("Reconnecting...")
		reconnect()
	else:
		print("Exiting.")


def _cleanup(old_server_controller=None):
	"""Properly disconnect old proxies and reconnect"""
	# Step 1: Try to clean up old proxy connections
	if old_server_controller is not None:
		try:
			# Close the underlying connection
			old_server_controller._close()
		except Exception as _:
			pass


def daemon_start_supervisor():
	manager = connect_manager()

	daemon = manager.get_daemon()
	daemon.start_supervisor()


def daemon_stop_supervisor(t: float, hard: bool = False):
	manager = connect_manager()

	daemon = manager.get_daemon()
	daemon.stop_supervisor(stop_worker=True, timeout=t, hard=hard)


def test_create():
	que = Que(_get_basic_logger())
	arg = "R(2+1)D_18 asl100 6"
	args = shlex.split(arg)
	maybe_args = configs.take_args(sup_args=args)
	if isinstance(maybe_args, tuple):
		admin_info, wandb_info = maybe_args
		que.create_run(admin_info, wandb_info)
	else:
		print("oops")


def connect_manager_ssh(
	host: Optional[str] = None,
	ssh_user: Optional[str] = None,
	ssh_key: Optional[Path] = None,
	port: int = 50000,
	authkey: Optional[bytes] = None,
	max_retries=5,
	retry_delay=2,
):

	if authkey is None:
		password = getpass.getpass("Queue server password: ")
		authkey = password.encode()

	if ssh_user is None:
		ssh_user = Path.home().name

	ssh_cmd = [
		"ssh",
		"-N",  # don't execute a command, just tunnel
		"-L",
		f"50000:localhost:{port}",  # local port -> remote port
		"-o",
		"ExitOnForwardFailure=yes",
	]
	if ssh_key:
		ssh_cmd += ["-i", ssh_key.expanduser().as_posix()]
	ssh_cmd.append(f"{ssh_user}@{host}")

	tunnel = subprocess.Popen(ssh_cmd)
	time.sleep(2)  # give the tunnel a moment to establish

	try:
		manager = connect_manager(
			host="127.0.0.1",
			port=50000,
			authkey=authkey,
			max_retries=max_retries,
			retry_delay=retry_delay,
		)
		return manager, tunnel
	except Exception:
		tunnel.terminate()
		raise


def ssh_connect_and_test(
	host: Optional[str] = None,
	ssh_user: Optional[str] = None,
	ssh_key: Optional[Path] = None,
	port: int = 50000,
	authkey: Optional[bytes] = None,
	max_retries=5,
	retry_delay=2,
):
	manager, tunnel = connect_manager_ssh(
		host=host,
		ssh_user=ssh_user,
		ssh_key=ssh_key,
		port=port,
		authkey=authkey,
		max_retries=max_retries,
		retry_delay=retry_delay,
	)
	try:
		print("Connected to manager via SSH tunnel!")
		# You can add more tests here to interact with the manager
		que_shell = QueShell(manager)
		try:
			que_shell.cmdloop()
		except KeyboardInterrupt:
			print("\n[INFO] Exiting queShell due to keyboard interrupt.")
	finally:
		tunnel.terminate()
		print("SSH tunnel closed.")


def test_new_recover():
	q = Que(_get_basic_logger())
	q.disp_run("fail_runs", 0)
	q.recover_run(from_loc="fail_runs", clean_slate=True)
	q.disp_run("to_run", 0)


def test_copy():
	q = Que(_get_basic_logger())
	# q.disp_run("fail_runs", 0)
	# q.copy_runs("fail_runs", [0], "to_run")
	q.disp_run("to_run", 0)


def update_runs():
	q = Que(_get_basic_logger())
	# q.update_runs("to_run", [0], {"status": "updated"})
	q.disp_run("old_runs", 0)

	# q.disp_run("to_run", 0)


def update_runs2():

	data_default_dict = {
		"train_augs": {
			"normalise": True,
			"norm_dict": None,
			"frame_size_strategy": "Random_crop",
			"frame_sampler": {"max_wobble": 0, "method": "og", "randomise": False},
			"temporal_aug": [],
			"spatial_aug": ["Horizontal_flip"],
		},
		"test_augs": {
			"normalise": True,
			"norm_dict": None,
			"frame_size_strategy": "Centre_crop",
			"frame_sampler": {"max_wobble": 0, "method": "og", "randomise": False},
			"temporal_aug": [],
			"spatial_aug": [],
		},
	}

	with open("/home/luke/Code/SLR/code/que/Runs.json", "r") as f:
		all_runs = json.load(f)

	# for key in KEYS:
	que_list = all_runs[TO_RUN] + all_runs[CUR_RUN]
	new_quelist = []
	for run in que_list:
		for key in data_default_dict.keys():
			if key not in run["data"]:
				run["data"][key] = data_default_dict[key]
		ExpInfo.model_validate(run)
		new_quelist.append(run)

	all_runs[TO_RUN] = new_quelist
	all_runs[CUR_RUN] = []

	que_list = all_runs[FAIL_RUNS]
	new_quelist = []
	for run in que_list:
		for key in data_default_dict.keys():
			if key not in run["data"]:
				run["data"][key] = data_default_dict[key]
		try:
			FailedExp.model_validate(run)
		except Exception as e:
			print(e)
			continue
		new_quelist.append(run)
	all_runs[FAIL_RUNS] = new_quelist

	que_list = all_runs[OLD_RUNS]
	new_quelist = []
	for run in que_list:
		for key in data_default_dict.keys():
			if key not in run["data"]:
				run["data"][key] = data_default_dict[key]
		if isinstance(run["admin"]["exp_no"], int):
			run["admin"]["exp_no"] = str(run["admin"]["exp_no"]).zfill(3)
		CompExpInfo.model_validate(run)
		new_quelist.append(run)
	all_runs[OLD_RUNS] = new_quelist

	with open("/home/luke/Code/SLR/code/que/Runs_fixed.json", "w") as f:
		json.dump(all_runs, f, indent=4)


def fix_spatial(set_augs: Dict[str, Any]) -> Dict[str, Any]:

	replace_dict = {"Horizontal_flip": HorizontalFlipConfig().model_dump()}

	new_spatial = []
	for spatial in set_augs["spatial_aug"]:
		if isinstance(spatial, str):
			if spatial in replace_dict:
				new_spatial.append(replace_dict[spatial])
			else:
				print(f"{spatial} not in replace_dict")
		else:
			print(f"Type: {spatial} not string")
	set_augs["spatial_aug"] = new_spatial

	return set_augs


def fix_temporal(set_augs: Dict[str, Any]) -> Dict[str, Any]:

	not_allowed_keys = [
		"Shuffle"  # not sure how this snuck into some configs
	]
	aug_key = "temporal_aug"
	new_temporal = []
	for temporal in set_augs[aug_key]:
		if isinstance(temporal, str):
			if temporal in not_allowed_keys:
				print(f"Skipping: {temporal}")
			else:
				print(f"keeping: {temporal}")
				new_temporal.append(temporal)
		else:
			print(f"Type: {temporal} not string")
	set_augs[aug_key] = new_temporal

	return set_augs


def fix_all(set_augs: Dict[str, Any]) -> Dict[str, Any]:
	return fix_temporal(fix_spatial(set_augs))


def update_runs3():

	replace_dict = {"Horizontal_flip": HorizontalFlipConfig().model_dump()}

	with open("/home/luke/Code/SLR/code/que/Runs.json", "r") as f:
		all_runs = json.load(f)

	# for key in KEYS:
	que_list = all_runs[TO_RUN] + all_runs[CUR_RUN]
	new_quelist = []
	for run in que_list:
		ExpInfo.model_validate(run)
		new_quelist.append(run)

	que_list = all_runs[FAIL_RUNS]
	new_quelist = []
	for run in que_list:
		train_augs = run["data"]["train_augs"]
		test_augs = run["data"]["test_augs"]

		train_augs = fix_all(train_augs)
		test_augs = fix_all(test_augs)

		try:
			FailedExp.model_validate(run)
		except Exception as e:
			print(e)
			continue
		new_quelist.append(run)
	all_runs[FAIL_RUNS] = new_quelist

	que_list = all_runs[OLD_RUNS]
	new_quelist = []
	for run in que_list:
		train_augs = run["data"]["train_augs"]
		test_augs = run["data"]["test_augs"]

		train_augs = fix_all(train_augs)
		test_augs = fix_all(test_augs)
		CompExpInfo.model_validate(run)
		new_quelist.append(run)
	all_runs[OLD_RUNS] = new_quelist

	with open("/home/luke/Code/SLR/code/que/Runs_fixed.json", "w") as f:
		json.dump(all_runs, f, indent=4)


def validate_runs(runs_path="/home/luke/Code/SLR/code/que/Runs.json"):
	q = Que(_get_basic_logger(), runs_path=runs_path)
	# q.update_runs("to_run", [0], {"status": "updated"})
	keys: list[QueLocation] = ["to_run", "cur_run", "fail_runs", "old_runs"]
	for key in keys:
		print(str(key).capitalize())
		q.disp_runs(key)
		print("\n", "-" * 20, "\n")

	# q.disp_run("to_run", 0)


def update_runs4():

	with open("/home/luke/Code/SLR/code/que/Runs.json", "r") as f:
		all_runs = json.load(f)

	# for key in KEYS:

	que_list = all_runs[OLD_RUNS]
	new_quelist = []
	for run in que_list:
		train_augs = run["data"]["train_augs"]
		test_augs = run["data"]["test_augs"]
		AugInfo.model_validate(train_augs)
		AugInfo.model_validate(test_augs)
		CompExpInfo.model_validate(run)
		new_quelist.append(run)
	all_runs[OLD_RUNS] = new_quelist

	with open("/home/luke/Code/SLR/code/que/Runs_fixed.json", "w") as f:
		json.dump(all_runs, f, indent=4)


def update_runs5():

	def map_frame_size_strat_to_crop_config(fss, sz):
		return {
			"Centre_crop": CentreCropConfig(size=sz).model_dump(), #type: ignore 
			"Random_crop": RandomCropConfig(size=sz).model_dump(), #type: ignore
			"Scale_and_pad": ScaleAndPadConfig(size=sz).model_dump(), #type: ignore
		}[fss]

	with open("/home/luke/Code/SLR/code/que/Runs.json", "r") as f:
		all_runs = json.load(f)

	# for key in KEYS:

	# loc = KEYS[0]
	for loc in KEYS:
		que_list = all_runs[loc]
		new_quelist = []
		for run in que_list:
			data = run["data"]
			nf = data.pop("num_frames")
			fs = data.pop("frame_size")
			keys = ["test_augs", "train_augs"]
			for key in keys:
				fss = data[key].pop("frame_size_strategy")
				k = "spatial_aug"
				data[key][k] = [map_frame_size_strat_to_crop_config(fss, fs)] + data[key][k]
				k = "temporal_aug"
				if len(data[key][k]) > 0 and data[key][k][0] == 'Shuffle':
					data[key][k].pop(0)
				s = data[key].pop("frame_sampler")
				s['type'] = s.pop('method')
				s['target_length'] = nf
				data[key][k] = [s] + data[key][k]

			if loc in KEYS[:2]:
				ExpInfo.model_validate(run)
			elif loc == KEYS[2]:
				CompExpInfo.model_validate(run)
			else:
				FailedExp.model_validate(run)
			new_quelist.append(run)

		all_runs[loc] = new_quelist
	

	with open('/home/luke/Code/SLR/code/que/Runs_fixed.json', 'w') as f:
		json.dump(all_runs, f, indent=4)



def update_runs6():

	with open("/home/luke/Code/SLR/code/que/Runs.json", "r") as f:
		all_runs = json.load(f)

	# for key in KEYS:

	# loc = KEYS[0]
	for loc in KEYS:
		que_list = all_runs[loc]
		new_quelist = []
		for run in que_list:
			data = run["data"]
			# nf = data.pop("num_frames")
			# fs = data.pop("frame_size")
			keys = ["test_augs", "train_augs"]
			for key in keys:
				keys2 = [ 'spatial_aug']
				for k2 in keys2:
					for i, aug in enumerate(data[key][k2]):
						if 'size' in aug:
							frame_size = aug.pop('size')
							data[key][k2][i]['frame_size'] = frame_size
						
						

			if loc in KEYS[:2]:
				run = ExpInfo.model_validate(run).model_dump()
			elif loc == KEYS[2]:
				run = CompExpInfo.model_validate(run).model_dump()
			else:
				run = FailedExp.model_validate(run).model_dump()
			new_quelist.append(run)

		all_runs[loc] = new_quelist
	

	with open('/home/luke/Code/SLR/code/que/Runs_fixed.json', 'w') as f:
		json.dump(all_runs, f, indent=4)


def update_runs7():
	from run_types import SupervisedInfo
	
	with open("/home/luke/Code/SLR/code/que/Runs.json", "r") as f:
		all_runs = json.load(f)

	# for key in KEYS:

	for loc in KEYS:
		que_list = all_runs[loc]
		new_quelist = []
		for run in que_list:
			
   
			model_params = run["model_params"]	
			modp = SupervisedInfo.model_validate(model_params)
			run['model_params'] = modp.model_dump()
   
			if loc in KEYS[:2]:
				run = ExpInfo.model_validate(run).model_dump()
			elif loc == KEYS[2]:
				run = CompExpInfo.model_validate(run).model_dump()
			else:
				run = FailedExp.model_validate(run).model_dump()
			
	 
	 
			new_quelist.append(run)

		all_runs[loc] = new_quelist
	

	with open('/home/luke/Code/SLR/code/que/Runs_fixed.json', 'w') as f:
		json.dump(all_runs, f, indent=4)

def update_runs8():
	with open("/home/luke/Code/SLR/code/que/Runs.json", "r") as f:
		all_runs = json.load(f)

	for loc in KEYS:
		que_list = all_runs[loc]
		new_quelist = []
		for run in que_list:
			
			data = run["data"]
			keys = ["test_augs", "train_augs"]
			for key in keys:
				# keys2 = [ 'spatial_aug']
				# for k2 in keys2:
				k2 = 'temporal_aug'
				for i, aug in enumerate(data[key][k2]):
					
					if aug['type'] == 'shuffle':
						data[key][k2].pop(i)

			run['data'] = data

			if loc in KEYS[:2]:
				run = ExpInfo.model_validate(run).model_dump()
			elif loc == KEYS[2]:
				run = CompExpInfo.model_validate(run).model_dump()
			else:
				run = FailedExp.model_validate(run).model_dump()
			
	 
	 
			new_quelist.append(run)

		all_runs[loc] = new_quelist

	with open('/home/luke/Code/SLR/code/que/Runs_fixed.json', 'w') as f:
		json.dump(all_runs, f, indent=4)

def update_runs9():
	with open("/home/luke/Code/SLR/code/que/Runs.json", "r") as f:
		all_runs = json.load(f)

	for loc in KEYS:
		que_list = all_runs[loc]
		new_quelist = []
		for run in que_list:

			if loc in KEYS[:2]:
				run = ExpInfo.model_validate(run).model_dump()
			elif loc == KEYS[2]:
				run = CompExpInfo.model_validate(run).model_dump()
			else:
				run = FailedExp.model_validate(run).model_dump()
			
			new_quelist.append(run)

		all_runs[loc] = new_quelist

	with open('/home/luke/Code/SLR/code/que/Runs.json', 'w') as f:
		json.dump(all_runs, f, indent=4)

def test_set_inplace():
	from .core import Que

	firstl = 'a'
	scndl = 'g'

	dicty = {
		k : v for k, v in zip(
			[chr(i) for i in range(ord(firstl), ord(scndl))],
			range(ord(firstl), ord(scndl))
		)
	}
	rot = {chr(ord(k) + 13) : v + 13 for k, v in dicty.items()}
	opdicty = {str(v) : k for k, v in dicty.items()}
 
 
	dicty2 = {
		'd1': dicty,
		'd2': rot,
		'd3': opdicty
	}
	
	dicty3 = Que.set_nested(dicty2, ['d1', 'd3', str(ord(scndl))], scndl * 2)
 
	print(json.dumps(dicty3, indent=4))
 
def test_edit_run():
    from run_types import strict_validate
    
    demo = {
		"admin": {
                "model": "MViTv2_S",
                "dataset": "WLASL",
                "split": "asl100_bottom",
                "save_path": "runs/asl100_bottom/MViTv2_S/exp000/checkpoints",
                "seed": 42,
                "exp_no": "000",
                "recover": True,
                "config_path": "configfiles/asl100/MViTv2_S/exp016.toml",
                "weight_path": None
            },
            "training": {
                "batch_size": 4,
                "update_per_step": 2,
                "max_epoch": 200,
                "batch_size_equivalent": 8
            },
            "optimizer": {
                "eps": 1e-05,
                "backbone_init_lr": 0.0001,
                "backbone_weight_decay": 0.001,
                "classifier_init_lr": 0.001,
                "classifier_weight_decay": 0.001
            },
            "model_params": {
                "drop_p": 0.5,
                "type": "supervised"
            },
            "data": {
                "train_augs": {
                    "normalise": True,
                    "norm_dict": {
                        "mean": [
                            0.45,
                            0.45,
                            0.45
                        ],
                        "std": [
                            0.225,
                            0.225,
                            0.225
                        ]
                    },
                    "temporal_aug": [
                        {
                            "target_length": 16,
                            "max_wobble": 4,
                            "type": "chunked"
                        }
                    ],
                    "spatial_aug": [
                        {
                            "type": "HORIZONTAL_FLIP",
                            "p": 0.5
                        },
                        {
                            "frame_size": 224,
                            "type": "Centre_crop"
                        },
                        {
                            "type": "RANDAUGMENT",
                            "num_ops": 3,
                            "magnitude": 7,
                            "num_magnitude_bins": 31,
                            "interpolation": "bilinear"
                        }
                    ],
                    "strict_size": True,
                    "target_length": 16,
                    "frame_size": 224
                },
                "test_augs": {
                    "normalise": True,
                    "norm_dict": {
                        "mean": [
                            0.45,
                            0.45,
                            0.45
                        ],
                        "std": [
                            0.225,
                            0.225,
                            0.225
                        ]
                    },
                    "temporal_aug": [
                        {
                            "target_length": 16,
                            "max_wobble": 0,
                            "type": "uniform"
                        }
                    ],
                    "spatial_aug": [
                        {
                            "frame_size": 224,
                            "type": "Centre_crop"
                        }
                    ],
                    "strict_size": True,
                    "target_length": 16,
                    "frame_size": 224
                },
                "strict_size": True,
                "target_length": 16,
                "frame_size": 224
            },
            "scheduler": {
                "warm_up": None,
                "type": "CosineAnnealingWarmRestarts",
                "t0": 20,
                "tmult": 1,
                "eta_min": 0.0
            },
            "early_stopping": {
                "metric": [
                    "val",
                    "loss"
                ],
                "mode": "min",
                "patience": 15,
                "min_delta": 0.01
            },
            "wandb": {
                "entity": "ljgoodall2001-rhodes-university",
                "project": "WLASL-100_bottom",
                "tags": [
                    "Recovered"
                ],
                "run_id": None
            }
	}
    
    
    # demo = Que.set_nested(demo, ['admin', 'wandb', 'run_id'], 'abcd')
    valid = strict_validate(ExpInfo, demo)
    # valid = ExpInfo.model_validate(demo)

if __name__ == "__main__":
	# test_copy()
	# update_runs3()
	# validate_runs()
	# update_runs9()
	# test_set_inplace()
	test_edit_run()
