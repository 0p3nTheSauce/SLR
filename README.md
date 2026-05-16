# SLR

Work in progress repo focusing on training video classification architectures on the WLASL Dataset

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Installation

```bash
git clone https://github.com/0p3nTheSauce/SLR.git
cd SLR

#recommend using mini-conda 
conda env create -f wlasl_gpu.yml #GPU recommended for training / testing
conda activate wlasl

#alternatively if no GPU available 
# conda env create -f wlasl_cpu.yml 
# conda activate wlasl_cpu

#download data
mkdir data/WLASL && cd data/WLASL
wget -L "https://github.com/0p3nTheSauce/SLR/releases/download/v1.0/splits.zip"
wget -L "https://github.com/0p3nTheSauce/SLR/releases/download/v1.0/WLASL2000.zip"
unzip splits.zip
unzip WLASL2000.zip
cd ../..

#preprocess data
cd code
python preprocess.py all -ve
```

## Usage
<details>

<summary>Common arguments</summary>

Training and testing have the following **arguments** in common:

- `MODEL_NAME`: 
    One of: S3D, R3D_18, R(2+1)D_18, Swin3D_T, Swin3D_S, Swin3D_B, MViTv2_S, MViTv2_S_e, MViTv1_B, MViTv2_S_16x4, MViTv2_B_32x3, MVirTed_t, MVirTed_t_MAE 
- `SPLIT`: 
    The ASL split, one of: asl100, asl300, asl1000, asl2000
- `EXP_NO`: 
    is the experiment number (e.g. 4)

Additionally, both of them by default set a random [seed](./code/configs.py) for reproducability. (Note the [Que](./code/que/README.md) had a bug which broke RNG state during some experiments).
</details>

<details>
<summary>
Training
</summary>

### Positional arguments:

- training.py {`MODEL_NAME`} {`SPLIT`} (-en `EXP_NO` | -c `CONFIG_PATH`) [`OPTIONS`]

It is assumed that every run begins with a **config file**. This is specified in either of two ways:
- The `CONFIG_PATH` can be specified with `-c` flag. Then `EXP_NO` can be determined automatically based on existing run directories. 
- The `EXP_NO` can be specified with the `-en` flag. Then the `CONFIG_PATH` can be determined automatically if using the following naming convention (exp_no to 3 units of precision e.g. 004): 
 ./configfiles/`SPLIT`/`MODEL_NAME`/exp`EXP_NO`.toml. 

```bash
#for help python train.py -h
python training.py S3D asl100 -en 4
```

#### [Options]

`-en` and `-c` are mutually exclusive.

- -h, --help            show this help message and exit
- -en EXP_NO, --exp_no EXP_NO
                       Experiment number (e.g. 10)
- -c CONFIG_PATH, --config_path CONFIG_PATH
                        Path to config file
- -ds {WLASL}, --dataset {WLASL}
                        Not implemented yet
- -r, --recover         Recover from last checkpoint
- -ri RUN_ID, --run_id RUN_ID
                        The run id to use (especially when also using recover)
- -p PROJECT, --project PROJECT
                        wandb project name, if not WLASL-num_classes (e.g. WLASL-100)
- -et ENTITY, --entity ENTITY
                        Entity if not ljgoodall2001-rhodes-university
- -t TAGS [TAGS ...], --tags TAGS [TAGS ...]
                        Additional wandb tags
- -w WEIGHTS_PATH, --weights_path WEIGHTS_PATH
                        Path to model pretrained weights
- -na, --no_ask         Don't ask for confirmation
- -nec, --no_enum_chck  Do not enumerate the checkpoint dir num (for output)
- -f CONFIG_FILETYPE, --config_filetype CONFIG_FILETYPE
                        Config file type, defaults to: .toml

#### Output
*training.py* will construct directory structure:
```bash
runs
├── asl100
│   ├── S3D
│   │   ├──exp000
│   │   │   ├── checkpoints
│   │   │   │   ├── ...
│   │   │   │   ├── best.pth
│   │   │   │   └── checkpoint_097.pth
│   │   │   └── data_info.json
...
```
*data_info.json* contains the frame size and number of frames used in training, for easy access when testing. *best.pth* is the model checkpoint with the best validation loss. 

</details>
<details>
<summary>
Testing
</summary>

### Positional arguments

- testing.py {`MODE`} ...

where `MODE` is:
- `partial`: Run a test on a single set (e.g. test), with a number of options 
- `full`: Ticks a comprehensive set of options from `partial`, and bundles the results together. 

<details>
<summary>
Partial test
</summary>

#### Partial

- testing.py `partial` {`MODEL_NAME`} {`SPLIT`} {`EXP_NO`} {`SET`} [`OPTIONS`]

where `SET` is one of:
- `test`
- `val`
- `train`

*testing.py* `partial` has all the options available to `full` but has more fine grained control of options, and only does a single test. 

```bash
python testing.py partial S3D asl100 4 test
```
</details>

<details>
<summary>
Full test
</summary>

#### Full 

- testing.py `full` {`MODEL_NAME`} {`SPLIT`} {`EXP_NO`} [`OPTIONS`]

*testing.py* `full` will automatically choose the *best.pth* checkpoint saved by *training.py*. It evaluates on the test set, the validation set, and the test set with the frames shuffled. Additionally, it creates all the available graphs. 

```bash
python testing.py full S3D asl100 4
```
</details>

#### [OPTIONS]

<details>
<summary>
Partial test
</summary>

#### Partial 

`partial` has the options listed below, as well as all the options available to `full` test.

- -sf, --shuffle_frames
                        Shuffle the frames when testing
- -cn CHECKPOINT_NAME, --checkpoint_name CHECKPOINT_NAME
                    Checkpoint name, if not best.pth
- -bg, --bar_graph      Plot the bar graph
- -cm, --confusion_matrix
                    Plot the confusion matrix
- -hm, --heatmap        Plot the heatmap
- -dy, --display        Display the graphs, if they have been selected
</details>

<details>
<summary>
Both Full and Partial 
</summary>

#### Full & Partial

- -h, --help            show this help message and exit
- -nf NUM_FRAMES, --num_frames NUM_FRAMES
                    Number of frames (overrides data_info.json if provided)
- -fs FRAME_SIZE, --frame_size FRAME_SIZE
                    Frame size (overrides data_info.json if provided)
- -se, --save           Save the outputs of the test
</details>



#### Output

The output of *testing.py* will be a printed Dict format. The most basic unit of which is the top1, top5 and top10 accuracy scores (as decimals e.g. 0.76):

```python
class TopKRes(BaseModel):
	top1: float
	top5: float
	top10: float
```

<details>
<summary>
Partial test
</summary>

#### Partial 

*testing.py* `partial` will output either a *BaseRes*:

```python
class BaseRes(BaseModel):
	top_k_average_per_class_acc: TopKRes
	top_k_per_instance_acc: TopKRes
	average_loss: float
```

or a *ShuffRes*:

```python
class ShuffRes(BaseRes):
	perm: List[int]
	shannon_entropy: float
```

depending on whether the `shuffle_frames` argument is set. The *average_loss* is the loss calculated for the forward pass of the specified set. 

When the frames are shuffled, the permutation used, and the shannon entropy of that permutation are included in the results. The Shannon Entropy algorithm used was derived from the first answer of this stack exchange [post](https://stats.stackexchange.com/questions/78591/correlation-between-two-decks-of-cards),
which was referenced by this [webpage](https://mikesmathpage.wordpress.com/2017/04/23/card-shuffling-and-shannon-entropy/).   

Additionally, the script may output/display 0 or more graphs depending on which flags are set.

If the save flag is set, *testing.py* creates a results directory, inside runs/`SPLIT`/`MODEL_NAME`/`EXP_NO`. The results for `partial` will have the naming convention: `CHECKPOINT_NAME`_`SET`-top-k.json. For example:

```bash
runs
├── asl100
│   ├── S3D
│   │   ├──exp000
│   │   │   ├── checkpoints
│   │   │   │   ├── ...
│   │   │   │   ├── best.pth
│   │   │   │   └── checkpoint_097.pth
│   │   │   ├── data_info.json
│   │   │   └── results
│   │   │       ├── ...
│   │   │       └── best_test-top-k.json     
...
```
</details>

<details>
<summary>
Full test
</summary>

#### Full

*full* test prints a *CompRes* Dictionary:

```python
class CompRes(BaseModel):
	check_name: str
	best_val_acc: float
	best_val_loss: float
	test: BaseRes
	val: BaseRes
	test_shuff: ShuffRes
```
the complete results, contain:
- the checkpoint name (best_val for 'best validation loss')
- the best validation accuracy (as a percentage e.g. 76.0) and the best validation loss reported by *training.py*
- the results for the test set
- the results for the validation set
- the results for the shuffled test set

Additionally, all the graphs will be made (heatmap, confusion matrix, bar graph) for the test set results.

The astute observer may notice there is a duplication of validation accuracy and validation loss. This is intentional to confirm that the *training.py* and *testing.py* scripts are consistent with each other. 

If the save flag is set, *testing.py* creates a results directory, inside runs/`SPLIT`/`MODEL_NAME`/exp`EXP_NO`. `full` will save all the graphs as *.pngs* and the results in *best_val_loss.json* (the checkpoint made from the best val loss). An example is shown below:

```bash
runs
├── asl100
│   ├── S3D
│   │   ├──exp000
│   │   │   ├── checkpoints
│   │   │   │   ├── ...
│   │   │   │   ├── best.pth
│   │   │   │   └── checkpoint_097.pth
│   │   │   ├── data_info.json
│   │   │   └── results
│   │   │       ├── best_test-bargraph.png
│   │   │       ├── best_test-confmat.png
│   │   │       ├── best_test-heatmap.png
│   │   │       └── best_val_loss.json  
...
```
</details>

</details>

## Features

- Run [utils.py](./code/utils.py) to automatically clean up checkpoints.
- Use the [Que](./code/que/README.md) feature to schedule and automatically train + test runs.


## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

The data provided in this project is subject to the C-UDA license of the forked WLASL dataset. By downloading or using this data, you agree to the terms of the C-UDA and any downstream redistribution must also comply with these terms. See [C-UDA-1.0.pdf](C-UDA-1.0.pdf)

Some models used are implemented from other repositories. By using any of these models you are implicitly subject to their licenses. See the [models README.md](./code/models/README.md) for more details.  