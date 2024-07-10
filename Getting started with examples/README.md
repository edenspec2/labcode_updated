**It is highly recommended to begin by running the following example hand-in-hand with this guide**

## Download example log files

Please find the example log files on the package's [Github]('(https://github.com/edenspec2/LabCode/blob/main/Getting%20started%20with%20examples/Example_log_files.zip)')

Once downloaded and unzipped, you are ready to go!

For user convenience, we demonstrate usage with a small number of molecules, such that downloading the log files directly to a local machine will stay within memory-usage reason. As stated on the home page, it is generally not the case. 

## Run extRactoR

<center><img src="figures/logs_to_feather.jpg" class="center"></center>

**A directory named 'feather_files' will be created inside the logs directory with the feather files inside it.**
**Now that we extracted the files we can run the GUI for easy feature extraction.**




```{r mol, eval=FALSE}
# Run MolFeatures 
python -m MolFeatures gui
```

<center><img src="figures/gui_dashboard.jpg" class="center"></center>

**Users are then presented with the option menu.**
**By clicking Browse to feather files directory users can choose a directory with feather files to load them**
**once loaded a list with file names will appear to let you know which files were successful**

```{r mol, eval=FALSE}
Molecules initialized : ['basic', 'm_Br', 'm_Cl', 'm_F', 'm_I', 'm_nitro', 'o_Br', 'o_Cl', 'o_F', 'o_I', 'o_nitro', 'penta_F', 'p_amine', 'p_azide', 'p_boc', 'p_Br', 'p_Cl', 'p_F', 'p_I', 'p_Me', 'p_nitro', 'p_OEt', 'p_OH', 'p_OMe', 'p_Ph', 'p_tfm']

Failed to load Molecules: []
```

Now we can start preforming other actions, lets start with visualizing one of the structures we loaded.
This image will help us select atoms for the process of feature extraction.

<center><img src="figures/visualize.jpg" width="250" height="300"></center>

<center><img src="figures/vis_example.jpg" width="500" height="450"></center>

## Features

Once we open the Feature Extraction window, we'll be presented with many questions allowing us to extract the different features.

Each of the features and their options are described with an example of what the input should look like.

<center><img src="figures/extract_features.jpg" width="800" height="500"></center>

Visualize Basic Structure - Visalizes the smallest structure in the set.
Choose Parameters - lets you choose 1) radii type that will be used for sterimol.
There are two radii systems implemented in this version, the first being Pyykko's covalent radii and the second being CPK (VDW radii). The default is set to covalent radii as it holds a definitive value for all elements of the periodic table, while CPK is only defined for a small subset of them.
2) Dipole calculation type - either directly from gaussian input or explicitly from the NBO values.

**Once all parameters are entered users can click submit for instant results which will be presented on the GUI dashboard.**
**It is recommended to use Save input/output, once clicked users will be asked to save a text file with the parameters chosen for quick results replication,
in addition a csv file will be saved with the features extracted**

#### Inputs file 

The inputs file is saved in a .RData format, which is only readable using R. To see its contents (and for advanced use - edit it for different uses) use `readRDS()`. 

```
$steRimol
$steRimol$input.vector
[1] "1 2" "1 3"

$steRimol$CPK
[1] FALSE

$steRimol$only.sub
[1] TRUE


$NBO
$NBO$atom_indices
[1] "1 2 3 15"

$NBO$difference_indices
[1] "1 2 1 3 1 15"


$Dipole
$Dipole$coor_atoms
[1] "1 2 3"

$Dipole$center_of_mass
[1] FALSE

$Dipole$center_of_mass_substructure
[1] FALSE

$Dipole$subunits_input_vector
NULL


$`Bond Vibs`
$`Bond Vibs`$atom_pairs
[1] "1 27"


$`Ring Vibs`
$`Ring Vibs`$inputs_vector
[1] "6 3 4 8 5 7"


$`Bend Vibs`
[1] NA

$Angles
$Angles$inputs_vector
[1] "4 3 1 2"


$Distances
[1] NA

$Polarizability
$Polarizability$polariz.answer
[1] "yes"

```
