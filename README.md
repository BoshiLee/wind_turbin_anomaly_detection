# wind_turbin_anomaly_detection

## How to install tensorflow on wsl2

### Install miniconda
## Windows
Download the  [Miniconda Installer](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe). Double-click the downloaded file and follow the instructions on the screen.
Or use the following script to install miniconda
```bash
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
start /wait "" miniconda.exe /S
del miniconda.exe
```
## MacOS
```
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

## Linux
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```
### Mac and Linux only
After installing, initialize your newly-installed Miniconda. The following commands initialize for bash and zsh shells:
```
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

## Create a new conda environment
```bash
conda create --name wind_turbine python=3.9
conda activate wind_turbine
```

### GPU Setup (NVIDIA CUDA and cuDNN are required)
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

### Install tensorflow
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify the installation
GPU
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

CPU
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('CPU'))"
```

## Pytorch

### Windows+GPU
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
### MacOS
```bash
pip3 install torch torchvision torchaudio
```