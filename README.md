# Decentralized-EO
In this project we explore decentralized learning for earth observation.     
     
The project is currently built with  
- [Paseos](https://github.com/aidotse/PASEOS): A Python module that simulates the environment to operate multiple spacecraft.
    
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

This will create a new conda environment called `eo` and install the required software packages.
To activate the new environment, you can use:

```
conda activate eo
```

To update the environment after changes, you can use 

```
conda env update --file environment.yml --prune
```

### Setup src code
When installing dependencies, a local clone of the paseos repository will be installed from github and added to `modules`. 
To clone the required repository and ensure a correct implementation, we recommend that you use

```
python setup.py
```



<!-- USAGE EXAMPLES -->

## Usage

### A Minimal Example - (COMING SOON)
To run a minimal example of `Decentralized-EO`, we recommend that you use
```
python ....
```