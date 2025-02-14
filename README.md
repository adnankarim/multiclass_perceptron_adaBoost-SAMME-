
# Perceptron Project

This project demonstrates the implementation and use of a Perceptron model as weak learner for SAAME Adaboost. Follow the instructions below to set up the environment and run the project.

---

## Requirements

- **Python Version**: Python 3.10 (higher versions are also compatible)
- **Dependencies**:
  - `scikit-learn`
  - `numpy`
  - `pandas`
  - `matplotlib`
- **Tools**: JupyterLab
- **Files**:
  - `perceptron.py`: Contains the Perceptron implementation.
  - `SAAME_adaboost.py`: Contains the SAAME Adaboost implementation.
  - `experiments.ipynb`: Notebook file for running experiments and visualizations.

---

## Installation Guide

### Step 1: Verify Python Version
Ensure Python 3.10 or a compatible version is installed by running:
```bash
python --version
```

Upgrade `pip` to the latest version:
```bash
pip install --upgrade pip
```

---

### Step 2: Install Dependencies
Run the following commands to install the required Python packages:
```bash
pip install scikit-learn
pip install numpy
pip install pandas
pip install matplotlib
```

---

### Step 3: Install JupyterLab
Install JupyterLab for running notebooks:
```bash
pip install jupyterlab
```

---

### Step 4: Ensure Required Files are Present
Make sure the following files are in the current working directory:
- `perceptron.py`: Contains the Perceptron implementation.
- `experiments.ipynb`: Jupyter notebook for running experiments.
- `SAAME_adaboost.py`: SAAME adaboost file.
---

## Running the Project

1. **Start JupyterLab**:
   Launch JupyterLab from the terminal:
   ```bash
   jupyter-lab
   ```

2. **Open `experiments.ipynb`**:
   Navigate to the `experiments.ipynb` notebook in the JupyterLab interface.

3. **Run the Notebook**:
   Execute the cells in the notebook sequentially. Ensure `perceptron.py`,`SAAME_adaboost.py` are correctly imported into the notebook.

---

## Project Structure

Your project directory should be structured as follows:
```
project/
|-- perceptron.py        # Contains the Perceptron implementation
|-- experiments.ipynb    # Notebook file for running experiments
|-- SAAME_adaboost.py experiments
|-- README.md            # Project documentation
```

---

## Notes

- Ensure all dependencies are installed to avoid runtime issues.
- If you encounter any errors, verify that `perceptron.py`,`SAAME_adaboost.py` and `experiments.ipynb` are in the same directory.


Happy coding!