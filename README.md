# Decentralized-EO
In this project we explore decentralized learning for earth observation. 
     
The project is currently built with  
- [Paseos](https://github.com/aidotse/PASEOS): A Python module that simulates the environment to operate multiple spacecraft.
- [LICOS](https://github.com/gomezzz/LICOS): A Python module for Learning Image Compression On board a Satellite constellation.
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM): A pre-trained segmentation model that can be run on mobile devices.

<!-- GETTING STARTED -->

## Getting Started

This is a brief guide how to set up the Decentralized-EO repository.

### Installation

### Building from source

To use the latest code from this repository make sure you have all the requirements installed and then clone the [GitHub](https://github.com/aidotse/Decentralized-EO.git) repository as follows ([Git](https://git-scm.com/) required):.

```
git clone https://github.com/aidotse/Decentralized-EO
```

To install the required dependencies, we recommend that you use [conda](https://docs.conda.io/en/latest/) as follows:

```
cd Decentralized-EO
conda env create -f environment.yml
```

This will create a new conda environment called `licos` and install the required software packages.
To activate the new environment, you can use:

```
conda activate licos
```

To update the environment after changes, you can use 

```
conda env update --file environment.yml --prune
```

Furthermore, if you are using a non-ARM cpu architecture and want to use MPI, remember to remove the comment before `cudatoolkits` in the environment.yml file to include the right toolkit. 


<!-- USAGE EXAMPLES -->

## Usage

### A Minimal Example -
To run a minimal example of `Decentralized-EO` simulating three satellites in orbit (see [LICOS](https://github.com/gomezzz/LICOS)), we recommend that you use

```
cd licos
mpiexec -n 3 python main.py
```

### Training a mobileSAM for EO using WorldFloods data - COMING SOON
...