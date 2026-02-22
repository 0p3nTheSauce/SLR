# SLR

Work in progress repo focussing on training video classification architectures on the WLASL Dataset

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
conda env create -f wlasl_gpu.yml
conda activate wlasl

#download data
mkdir data && cd data
curl -L "https://drive.google.com/uc?export=download&id=1kq0Vzt13226mODCEhhC3mpwhsSawaojS" -o splits.zip
curl -L "https://drive.google.com/uc?export=download&id=1Rv_8ZE6VDQDaY8tbJUucx1E8mcdUp-xX" -o WLASL2000.zip
unzip splits.zip
unzip WLASL2000.zip
cd ..

#preprocess data
cd code
python preprocess.py -as asl100 -ve
```

## Usage

Training and testing have the following **positional arguments** in common:

- **MODEL_NAME**: 
    One of: S3D, R3D_18, R(2+1)D_18, Swin3D_T, Swin3D_S,Swin3D_B, MViTv2_S, MViTv1_B 
- **SPLIT**: 
    The ASL split, one of: asl100, asl300, asl1000, asl2000
- **EXP_NO**: 
    is the experiment number (e.g. 4)

Additionally, both of them by default set a random seed for reproducability. 

### Training

#### Postional arguments:

- **trianing.py** **{MODEL_NAME}** **{SPLIT}** **{EXP_NO}** **[OPTIONS]**

It is assumed that every run begins with a config file. The config file path can be determined automatically if using the following naming convention (exp_no to 3 units of precision e.g. 004): 
 ./configfiles/**SPLIT**/**MODELNAME**_**EXPNO**.ini

```bash
#for help python train.py -h
python training.py S3D asl100 4
```

#### [Options]

- -h, --help            show this help message and exit
- -ds {WLASL}, --dataset {WLASL}
                    Not implemented yet
- -r, --recover         Recover from last checkpoint
- -ri RUN_ID, --run_id RUN_ID
                    The run id to use (especially when also usign recover)
- -p PROJECT, --project PROJECT
                    wandb project name, if not WLASL-num_classes (e.g. WLASL-100)
- -et ENTITY, --entity ENTITY
                    Entity if not ljgoodall2001-rhodes-university
- -ee, --enum_exp       enumerate the experiment dir num (for output)
- -ec, --enum_chck      enumerate the checkpoint dir num (for output)
- -t TAGS [TAGS ...], --tags TAGS [TAGS ...]
                    Additional wandb tags
- -c CONFIG_PATH, --config_path CONFIG_PATH
                    path to config .ini file
- -na, --no_ask         Don't ask for confirmation

#### Output
*training.py* will construct directory structure:
```bash
runs
├── asl100
│   ├── S3D
│   │   ├── checkpoints
│   │   │   ├── ...
│   │   │   ├── best.pth
│   │   │   └── checkpoint_097.pth
│   │   └── data_info.json
...
```
*data_info.json* contains the frame size and number of  frames used in training, for easy access when testing. *best.pth* is the model checkpoint with the best validation loss. 


### Testing

#### Postional arguments

- **testing.py** **{MODE}** ...

where **MODE** is:
- **partial:** Run a test on a single set (e.g. test), with a number of options 
- **full:** Ticks a comprehensive set of options from *partial*, and bundles the results together. 

##### Partial test
- **testing.py** **partial** **{MODEL_NAME}** **{SPLIT}** **{EXP_NO}** **{SET}** **[OPTIONS]**

where **SET** is on of:
- **test**
- **val**
- **train**

*testing.py* *partial* has all the options available to *full* but adds more fine grained control of options, and only does a single test. 

```bash
python testing.py partial S3D asl100 4 test
```

##### Full test

- **testing.py** **full** **{MODEL_NAME}** **{SPLIT}** **{EXP_NO}** **[OPTIONS]**

*testing.py* *full* will automatically choose the *best.pth* checkpoint saved by *training.py*. It evaluates on the test set, the validation set, and the test set with the frames shuffled. Additionally, it creates all the available graphs. 

```bash
python testing.py full S3D asl100 4
```

#### [OPTIONS]

##### Partial test

*partial* has the options listed below, as well as all the options available to *full* test.

- -sf, --shuffle_frames
                        Shuffle the frames when testing
- -cn CHECKPOINT_NAME, --checkpoint_name CHECKPOINT_NAME
                    Checkpoint name, if not best.pth
- -bg, --bar_graph      Plot the bar graph
- -cm, --confusion_matrix
                    Plot the confusion matrix
- -hm, --heatmap        Plot the heatmap
- -dy, --display        Display the graphs, if they have been selected

##### Full & Partial

- -h, --help            show this help message and exit
- -nf NUM_FRAMES, --num_frames NUM_FRAMES
                    Number of frames (overrides data_info.json if provided)
- -fs FRAME_SIZE, --frame_size FRAME_SIZE
                    Frame size (overrides data_info.json if provided)
- -se, --save           Save the outputs of the test

#### Output

The output of *testing.py* will be a printed TypedDict. The most basic unit of which is the top1, top5 and top10 accuracy scores (as decimals e.g. 0.76):

```python
class TopKRes(TypedDict):
	top1: float
	top5: float
	top10: float
```

##### Partial test

*testing.py* *partial* will output either a *BaseRes*:

```python
class BaseRes(TypedDict):
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

depending one whether the *shuffle_frames* argument is set. The *average_loss* is the loss calculated for the forward pass of the specified set. 

When the frames are shuffled, the permutation used, and the shannon entropy of that permutation are included in the results. The Shannon Entropy algorythm used was derived from the first answer of:
    https://stats.stackexchange.com/questions/78591/correlation-between-two-decks-of-cards
which was referenced by:
    https://mikesmathpage.wordpress.com/2017/04/23/card-shuffling-and-shannon-entropy/.   

Additionaly, the script may output/display 0 or more graphs depending on which flags are set.

If the save flag is set, *testing.py* creates a results directory, inside runs/**SPLIT**/**MODELNAME** _ **EXPNO**. The results for *partial* will have the naming convention: **CHECKPOINTNAME**_**SET**-top-k.json. For example:

```bash
runs
├── asl100
│   ├── S3D
│   │   ├── checkpoints
│   │   │   ├── ...
│   │   │   ├── best.pth
│   │   │   └── checkpoint_097.pth
│   │   ├── data_info.json
│   │   └── results
│   │       ├── ...
│   │       └── best_test-top-k.json     
...
```

##### Full test

*full* test prints a *CompRes* TypedDict:

```python
class CompRes(TypedDict):
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

If the save flag is set, *testing.py* creates a results directory, inside runs/**SPLIT**/**MODELNAME** _ **EXPNO**. *full* will save all the graphs as .pngs and the results in best_val_loss.json (the checkpoint made from the best val loss)

```bash
runs
├── asl100
│   ├── S3D
│   │   ├── checkpoints
│   │   │   ├── ...
│   │   │   ├── best.pth
│   │   │   └── checkpoint_097.pth
│   │   ├── data_info.json
│   │   └── results
│   │       ├── best_test-bargraph.png
│   │       ├── best_test-confmat.png
│   │       ├── best_test-heatmap.png
│   │       └── best_val_loss.json  
...
```









			


## Features

- Run [utils.py](./code/utils.py) to automatically clean up checkpoints.
- Use the [Que](./code/que/README.md) feature to schedule and automatically train + test runs.


## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.