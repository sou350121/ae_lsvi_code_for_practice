# Offline Contextual Bayesian Optimization

This code was built on top of code done for Offline Contextual Bayesian Optimization,
see https://github.com/fusion-ml/OCBO. The files that are added for this work,
specifically, is

```
src/strategies/joint_aelsvi.py
src/strategies/aelsvi.py
```

## Getting Set Up

The code is compatible with python 2.7. First, clone this repo and run
```
pip install -r requirements
```
By default the code leverages the [Dragonfly](https://github.com/dragonfly/dragonfly)
library. 

## Reproducing Synthetic Experiments

To run the four experiments that appear in the paper...

```
cd src
mkdir data
python ocbo.py --options <path_to_option_file>
```
where the options can be
```
options/jointbran.txt
options/jointh22.txt
options/jointh31.txt
options/jointh42.txt
```

After the simulation has finished, the plots can be reproduced by
```
cd scripts
python discrete_plotter.py --write_dir ../data --run_id <options_name>
```
