sudo apt install python3-venv

python3 -m venv mlflow-client
source mlflow-client/bin/activate

python -m pip install --upgrade pip
python -m pip install mlflow scikit-learn skops
python -m pip install optuna

deactivate
