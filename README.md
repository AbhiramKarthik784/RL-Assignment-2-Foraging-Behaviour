# Foraging Behaviour Model using Q-learning

## Requirements
- Python 3
- NumPy
- Matplotlib

## Run
```bash
pip install numpy matplotlib
python main.py
```

The program trains a Q-learning agent in a two-food-patch environment and compares it with random and heuristic strategies. It saves three graphs and `metrics.txt` in the `results/` folder.

## Actions
- 0: forage at the current patch
- 1: move to the other patch

## Note
The environment is a simplified educational simulation, not a biological model of a real animal.
