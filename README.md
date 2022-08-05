# Smart City project - Forecasting and optimizing HVAC system parameters in a large public building.

This is repository includes Python3 scripts for smart Building Automation system development during Smart City project. The Smart City project of Häme University of Applied Sciences focuses on promoting low-carbon developments through implementation of Artificial Intelligence. To learn more about the project, read [popular](https://blog.hamk.fi/hamk-smart/koneoppiminen-alykkaissa-rakennuksissa/) and [technical](https://blog.hamk.fi/hamk-smart/alykaupunki-hanke-edistaa-tekoalyn-tuotteistamista-rakennuksissa/) blog posts from HAMK Smart blog (in Finnish). The content of this repository is focused on forecasting district heating energy consumption and optimization of HVAC system controls.

The branch  'Talotohtori' focuses on the analysis of the Talotohtori buildings, like temperature and energy consumption prediction, two scenarios of energy optimization and control curve optimization.

### Overview of files
* __data__: Talotohtori building example data.
* __best_delay.py__:  A script to compute best energy consumption delay with least RMSE among all different delays combination.
* __temperature_prediction.py__: A script to predict inside temperature for different prediction intervals.
* __energy_prediction.py__: A script to predict energy consumption.
* __occupation.py__: A script for energy optimization to reduce the inside temperature during the time the building is occupied.
* __non_occupation.py__: A script for energy optimization to reduce the inside temperature when the building is not occupied.
* __control_curve_opti.py__: A script for control curve optimization.

### Authors
Wangkang Jin

### Licence
Permissive Apache License 2.0
