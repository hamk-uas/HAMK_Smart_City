# Smart City project - Forecasting and optimizing HVAC system parameters in a large public building.

This is repository includes Python3 scripts for smart Building Automation system development during Smart City project. The Smart City project of Häme University of Applied Sciences focuses on promoting low-carbon developments through implementation of Artificial Intelligence. To learn more about the project, read [blog post](https://blog.hamk.fi/hamk-smart/koneoppiminen-alykkaissa-rakennuksissa/) (in Finnish). The content of this repository is focused on forecasting district heating energy consumption and optimization of HVAC system controls.

The branch 'master' includes key functions from data preprocessing to Particle Swarm Optimization of HVAC controls. Package dependencies are listed in requirements.txt, and suitable .gitignore file is included in the repository.

### Overview of files
* __data_example.csv__: Exemplary data set. Includes data for multiple HVAC parameters in addition to weather variables. The whole data set has been downsampled to hourly temporal granularity.
* __preprocess.py__: Function for prosessing the data frame obtained with request_script.py to sequential data form. Formulates temporal variables used in modeling from datetime column of input data frame. Splits the data to sets of training and testing with 80% of total length used for training. Scales the data to [0,1] based on training data. Inputs features used in modeling are defined in here.
* __hyperparameter_opt.py__: Script which uses tensorflow keras tuner to find the best model composition from a search space defined as hyperparameters. This hyperparameter space varies by number of units in GRU cells, number of GRU layers in the sequential model, and the learning rate of Adam optimizer. This version uses Bayesian Optimization for selecting the model configurations from the search space and 5-fold cross-validation to assess predictive capability of a model. By default, 15 differing models are tested. The greatest performing model is saved to disk in Hierarchical Data Format (.h5). Forecasts for district heating energy consumption have been more accurate than inside air temperature ones. To enable GPU boosted performance in training of tanh-activated models, see [instructions](https://www.tensorflow.org/install/gpu) in TensorFlow documentation. 
* __optimization.py__: Particle Swarm Optimization of HVAC controls. Algorithm finds the HVAC network controls for which the energy consumption is minimal with inside temperature still being inside acceptable range. Saves control values of each iteration to disk. Single objective implementation with non-linear penalties for temperature deviations is based on Pareto optimality based multi-objective optimization presented in Wei et al. (2015) article in Energy.
* __inv_target.py__: Helper function which transforms the feature range scaling done in preprossesing back to original scale. Useful for interpretation of model outputs.

### Authors
2021 Iivo Metsä-Eerola and Genrikh Ekkerman

### Licence
Permissive Apache License 2.0
