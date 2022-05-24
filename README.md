# Factorization Q-learning Initialization

## How to use
* **Set the number of antennas in the base station**. In `environment.py` change the line `self.M_ULA` to the values of your choice. The code expects M = 4, 8, 16, 32, and 64.
* **Run Q-learning and its variants algorithms**. Run the scripts `main_QL.py`, `main_DynaQ.py`, `main_DQL.py`,`main_QlL.py`,  and `main_SQL.py`. The result is the same as that in folder `Results`.  
* **Show the results**. Run the script `Comparison.ipynb` in folder `Results` to show `Figure 2`, `Figure 3`, and `Table III` in the paper.
