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

Training and testing have the following positional arguments in common:

- **MODELNAME**: 
    One of: S3D, R3D_18, R(2+1)D_18, Swin3D_T, Swin3D_S,Swin3D_B, MViTv2_S, MViTv1_B as shown in ./info/implemented_info.json.
- **SPLIT**: 
    The ASL split, one of: asl100, asl300, asl1000, asl2000
- **EXPNO**: 
    is the experiment number (e.g. 4)

### Training

- *trianing.py* **MODEL_NAME** **SPLIT** **EXPERIMENT_NUMBER** **[OPTIONS]**

It is assumed that every run begins with a config file. The config file path can be determined automatically if using the following naming convention (exp_no to 3 units of precision e.g. 004):
- ./configfile/**SPLIT**/**MODELNAME**_**EXPNO**.ini

```bash
#for help python train.py -h
python training.py S3D asl100 4
```

Otherwise, specifiy the config path.

```bash
python train.py S3D asl100 4 -c './configfiles/generic/hframe_hwd.ini'
```

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
where *data_info.json* contains the frame size and number of  frames used in training
### Testing
- *testing.py* **MODE** **MODEL_NAME** **SPLIT** **EXPERIMENT_NUMBER** **[OPTIONS]**

where **MODE** is:
- **Full** Testing ticks all options of partial. It evaluates on the test set, the validation set, and the test set with the frames shuffled. 
- **partial:** Run partial test on a specific set with custom options

When the frames are shuffled, the permutation used, and the shannon entropy of that permutation 
(first answer of:
    https://stats.stackexchange.com/questions/78591/correlation-between-two-decks-of-cards
which was referenced by:
    https://mikesmathpage.wordpress.com/2017/04/23/card-shuffling-and-shannon-entropy/)



			


## Features

- Feature 1
- Feature 2
- Feature 3

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.