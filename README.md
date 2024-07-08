# LabCode

**A Python package for the physical-organic chemist - designed for easy, chemical-intuition based molecular features extraction - This package is modeled after the R package MoleculaR.**
**See moleculaR's [web page](https://barkais.github.io/) for detailed guides and  documenntation**

## Installation
Currently the package is under development so the version may chagne , this package is available with pip running the command - pip install -i https://test.pypi.org/simple/ MolFeatures==0.4007.
When the package is activated for the first time all necessary supporting packages will be installed on the same enviroment as the  packge itself. 

## Usage

NOTE:

With the current version - it is only possible to analyze correctly indexed Gaussian .log files.
Make sure the common substructure is numbered correctly.

### Featurization

**See `Getting Started with Examples` in articles for a detailed guide**

#### Step 1 - Pull information from Gaussain log files

Make sure all log files you want to analyze are in one location. 
This can be done either from the command line or the GUI.
```
# Run python -m MolFeatures convert
 add picture
```
Expect the following message, indicating everything worked fine. 

`Done!`

**This action results in a set of .feather files, which are light weight files that hold all the information the package needs. As it is reasonable to assume that most users will prefer working on local machines, it is still recommended to avoid transferring heavy log files to a local machine, and it is best practice to install `MolFeatures` on both the remote and the local, and to transfer only the resulting .feather files to the local.**
 

#### Step 2 - Get molecular features

# This can be done in two ways, either from the GUI which is the prefered option when trying to extract a complete csv with many features.
# The second option is to extract features separately from the command line.

 ```
python -m MolFeatures interactive

```


