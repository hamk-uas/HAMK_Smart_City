# Smart City project - Forecasting and optimizing HVAC system parameters in a large public building.

This is repository includes Python3 scripts for smart Building Automation system development during Smart City project. The Smart City project of Häme University of Applied Sciences focuses on promoting low-carbon developments through implementation of Artificial Intelligence. To learn more about the project, read [popular](https://blog.hamk.fi/hamk-smart/koneoppiminen-alykkaissa-rakennuksissa/) and [technical](https://blog.hamk.fi/hamk-smart/alykaupunki-hanke-edistaa-tekoalyn-tuotteistamista-rakennuksissa/) blog posts from HAMK Smart blog (in Finnish). The content of this repository is focused on forecasting district heating energy consumption and optimization of HVAC system controls.

The branch 'main' includes scripts which include everything from data preprocessing to Particle Swarm Optimization of HVAC controls. Package dependencies are listed in requirements.txt, and suitable .gitignore file is included in the repository. The branch '2022' includes the analysis of the newer buildings as well as side analysis, like the analysis of radiator network offsets, developing of the potential district heat scenarios, anomaly detection and combination of the model training and inference from different buildings.

### Overview of files
* __Features.csv__: S-building raw data excluding the energy measurements.
* __Energy.csv__: S-building raw data for the energy measurements.
* __Humidity.csv__ and __Irradiance.csv__: Humidity and Irradiance observations from FMI.
* __S-building_preprocessing.ipynb__: A notebook for processing the S-building data and the related Humidity and Irradiance data.
* __data_example.csv__: Takahuhti building example data.
* __rnn.py__: Class structure used to train RNN models. Can be used for modeling district heating energy consumption and average inside temperature. Includes methods for data preprocessing, training the model with hyperparameter optimization, calculating prediction intervals to quantify uncertainty and visualizing the predictions. Saving and loading methods use the current directory to save training checkpoints and class attributes, including scaling and models objects. 
* __main.ipynb__: Main notebook for running the analysis. Methods of the class rnn are demonstrated for the user. The anomaly detection using the residual standard deviation approach is available in this script as well.  The script also has a demo for the comparison of the S-building and Tampere 2744 building.
* __Developing_Scenarios.ipynb__: An unfinished notebook for developing scenarios to research the impact of the unmeasured district heat temperature network values onto the energy consumption and energy values.
* __Offsets.ipynb__: A notebook for the research of the impact of the radiator network temperature offsets onto the energy consumption.

### Authors
2021 Iivo Metsä-Eerola and Genrikh Ekkerman <br>
2022 Roman Tsypin

### Licence
Permissive Apache License 2.0
