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


### Training






```bash
#for help python train.py -h

python train.py S3D asl100 -c './configfiles/generic/hframe_hwd.ini'
```

If config file paths follow the nameing convention...

- ./configfile/**SPLIT**/**MODELNAME**_**EXPNO**.ini

where:
- **SPLIT**: The ASL split (e.g. asl100)
- **MODELNAME**: as shown in ./info/implemented_info.json (e.g. S3D)
- **EXPNO**: is the experiment number, with 3 digit precision (e.g. 004)

example:
- ./configfiles/asl100/S3D_004.ini

... then the config path can be determined automatically:

```bash
python training.py S3D asl100 4
```

### Testing

## Features

- Feature 1
- Feature 2
- Feature 3

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.