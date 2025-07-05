Source code for paper "[GreenLight: Green Traffic Signal Control using Attention-based Reinforcement Learning on Fog Computing Network](https://doi.org/10.1109/IGSC64514.2024.00032)".

# Prerequisites
- [SUMO](https://eclipse.dev/sumo/) >= 1.19.
- Python >= 3.11 (older versions are not tested).

# Install
1. Clone or download this repository.
2. Install dependencies.
   ```sh
   pip install -r requirements.txt
   ```

# Usage
The model can be trained using `train.py`. The user will need to specify the path to a `.sumocfg` file using the `--sumocfg` argument. The program will use the corresponding `.net.xml` file to build the environment that the RL agent will interact with.

Use `python train.py --help` to see all accepted arguments that can be used to control the environment, RL model, and training.

After a model is trained, the `test.py` script can be used to evaluate the model. The testing simulation scenario should use the same network used for training.

For more technical details, please refer to the paper.

# Citing GreenLight
```
@INPROCEEDINGS{tang2024greenlight,
  author={Tang, Chengyu and Baskiyar, Sanjeev},
  booktitle={2024 IEEE 15th International Green and Sustainable Computing Conference (IGSC)}, 
  title={GreenLight: Green Traffic Signal Control using Attention-based Reinforcement Learning on Fog Computing Network}, 
  year={2024},
  pages={129-134},
  doi={10.1109/IGSC64514.2024.00032}
}
```
