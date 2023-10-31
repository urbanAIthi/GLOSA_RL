# GLOSA_RL

## User-Centric Green Light Optimized Speed Advisory with Reinforcement Learning

This is the GitHub repository for our conference paper "User-Centric Green Light Optimized Speed Advisory with Reinforcement Learning" presented at the ITSC 2023. 

Our research addresses Green Light Optimized Speed Advisory (GLOSA), an application in the field of Intelligent Transportation Systems (ITS) for improving traffic flow and reducing emissions in urban areas. We aim at improving GLOSA, both by including traffic condition information, more specifically queue length, into the calculation of an optimal speed as well as by applying Reinforcement Learning (RL). Doing so, we incorporate rule-based classic GLOSA and RL-based GLOSA in a common comparable SUMO simulation environment.

 
## Usage:

The code is executable for different SUMO versions, but the results slightly differ depending on the version. In our study, we used SUMO 1.11.0. Also, we recommend using Python 3.10.

The main component for executing the code is the `run.py` file. The desired approach, rule-based classic or RL-based GLOSA, as well as other relevant parameters such as step size can be defined within the `config.ini` file.


### Rule-based Classic GLOSA:
The classic approach can be run directly without any training. To do so, the variable `glosa_agent` within the `config.ini` must be set to `classic`.

The core component is the `GLOSA_agent` class within `glosa.py`. Here, the optimal speed is determined based on the observed state, more precisely the upcoming traffic light phases, and other parameters from the simulation, such as information on preceding vehicles or red phase duration.


### RL-based GLOSA:

## Prerequisites:

This repository requires that you already have installed the SUMO traffic simulator. For more information on how to install SUMO, please refer to https://sumo.dlr.de/docs/Installing.html.

## Installation:
```bash
git clone https://github.com/urbanAIthi/GLOSA_RL.git

cd GLOSA_RL

conda env create -f environment.yml

conda activate glosa_rl env

# get the stable baselines3 library
git clone https://github.com/DLR-RM/stable-baselines3.git

```

## Citation:
If you use this code in your research, please cite our paper:
```bibtex
@inproceedings{schlamp2023GLOSA-RL,
  title={User-Centric Green Light Optimized Speed Advisory with Reinforcement Learning},
  author={Schlamp, Anna-Lena and Gerner, Jeremias and Bogenberger, Klaus and Schmidtner, Stefanie},
  booktitle={26th IEEE International Conference on Intelligent Transportation Systems (ITSC)},
  year={2023},
  address={Bilbao, Bizkaia, Spain},
  month={24-28 September},
  publisher={IEEE}
}
```

This work uses a cutout of the SUMO simulation network provided in: https://github.com/TUM-VT/sumo_ingolstadt.git <br>
The RL-based GLOSA approach incorporates RL-algorithms from Stable Baselines3: https://github.com/DLR-RM/stable-baselines3 <br>

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.



