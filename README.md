# Smart City project - Forecasting and optimizing HVAC system parameters in a large public building.

This is repository includes Python3 scripts for smart Building Automation system development during Smart City project. The Smart City project of Häme University of Applied Sciences focuses on promoting low-carbon developments through implementation of Artificial Intelligence. To learn more about the project, read [popular](https://blog.hamk.fi/hamk-smart/koneoppiminen-alykkaissa-rakennuksissa/) and [technical](https://blog.hamk.fi/hamk-smart/alykaupunki-hanke-edistaa-tekoalyn-tuotteistamista-rakennuksissa/) blog posts from HAMK Smart blog (in Finnish). The content of this repository is focused on forecasting district heating energy consumption and optimization of HVAC system controls.

The branch  'Talotohtori' focuses on the analysis of the Talotohtori buildings as well as s-building, like temperature and energy consumption prediction, two scenarios of energy optimization and control curve optimization.

### Overview of files
* __data__: Talotohtori building and s-building dataset. Both include multiple HVAC parameters with hourly temporal granularity. 

  Specifically, in Talotohtori building data, 'control on' represents the setpoint for energy consumption when the smart control system is open and 'control off' represents it when the smart control system is closed. In addition, 

  'Radiator_network_1_temperature_setpoint' is the combination of these two parameters.

* **model**: Trained model for  Talotohtori building and s-building.

* __rnn.py__:  Class structure used to train RNN models. Can be used for modeling district heating energy consumption and average inside temperature. Includes methods for data preprocessing, training the model with hyperparameter optimization, calculating prediction intervals to quantify uncertainty and visualizing the predictions. Saving and loading methods use the current directory to save training checkpoints and class attributes, including scaling and models objects.

* __main.ipynb__: Main script for training

* __tampere_building.ipynb__: Script for training Talotohtori  building data.

* __sbuilding.ipynb__: Script for training s-building data.

* **Offsets.ipynb**: A notebook for the research of the impact of the radiator network temperature offsets onto the energy consumption.

* __environment.yml__ and **requirements.txt**: Files for quick installation. The first one can be run as the from-environment installation using conda, the latter one can be used as the installation using pip.



### Article

To be continued



### Authors
2021 Iivo Metsä-Eerola and Genrikh Ekkerman

2022 Roman Tsypin and Wangkang Jin



### Licence
Permissive Apache License 2.0
