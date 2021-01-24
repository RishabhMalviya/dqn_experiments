# Overview
This repository contains an implementation of DQNs, with a number of additional tricks that have been propsed since then (see Biobliography). I have applied these DQNs to various OpenAI Gym environments. Each in it's own environment.

# Results
These are the results (before and after GIFs) for the experiments that I have run. Corresponding entries to the OpenAI leaderboards are also linked to.

# Local Setup
This setup was done on a system with these specifications:
1. **OS**: Windows 10
2. **CUDA Toolkit Version**: 11.2 (Download it from [here](https://developer.nvidia.com/Cuda-downloads))
3. **Python Version**: Python 3.6.8 (You can download the executable installer for Windows 10 from [here](https://www.python.org/ftp/python/3.6.8/python-3.6.8-amd64.exe))
But I've kept things very simple and put in lots of links, so as to make it easier for you to figure out the corresponding steps for Linux.

Here are the exact steps:
1. Clone this repository with (use [Git Bash for Windows](https://gitforwindows.org/)) `git clone https://github.com/RishabhMalviya/dqn-experiments.git`.
2. In the cloned repository, create a venv by running the following command from Powershell (venv should be installed along with the Python 3.6.8 installation form the link given above): `python -m venv ./venv`. Also, in case you're using Anaconda, you should launch Powershell by searching for "*Anaconda Powershell Prompt*" from Start.
3. We can't `pip install` the `requirements.txt` just yet because one of the dependencies, `box2d-py`, requires a build tool called Swig, which we'll have to install first: 
   1. As the [Windows documentation for Swig](http://www.swig.org/Doc1.3/Windows.html) says, *Download the [swigwin zip package](http://prdownloads.sourceforge.net/swig/swigwin-4.0.2.zip) from the [SWIG website](http://www.swig.org/download.html) and unzip into a directory. This is all that needs downloading for the Windows platform.* Note that the installation directory needs to be in your `PATH` environment variable.
   2. To get Swig to build `box2d-py` correctly, you will also have to set the following two environment variables. Change `</path/to/python>` to correspond to the python with which you created the venv:
      1. `PYTHON_INCLUDE`: `</path/to/python>/include`
      2. `PYTHON_LIB`: `</path/to/python>/libs/python36.lib`
4. Now, activate the venv by running `./venv/Scripts/activate` in Powershell.
5. Upgrade pip with `pip install -U pip`.
6. And install the requirements with `pip install -r requirements.txt`. You should adapt the first three lines of the `requirements.txt` file based on the installation command that the [PyTorch download page](https://pytorch.org/get-started/locally/) recommends for your system.
7. Finally, start a Jupyter Notebook (run `jupyter notebook` from Powershell) from the root of the repo and hack away!