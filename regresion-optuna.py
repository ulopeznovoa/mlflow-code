import mlflow
import mlflow.sklearn
import optuna

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Configurar MLflow
mlflow.set_tracking_uri("http:// ... :5000")  #COMPLETAR
mlflow.set_experiment(" ... ") #COMPLETAR

# Preparar los datos
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.4, random_state=42, stratify=y
)


def objective(trial):

    # Optuna selecciona un valor de C entre 0.001 y 100
    C = trial.suggest_float(
        "C", 0.001, 100.0, log=True
    )

    params = {
        "C": C,
        "max_iter": 1000,
        "solver": "lbfgs",
        "random_state": 42
    }

    # Cada trial se registra como un child run
    with mlflow.start_run(
        nested=True,
        run_name=f"trial-{trial.number}"
    ):
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

        print(
            f"Trial {trial.number}: "
            f"C={C:.5f}, accuracy={accuracy:.4f}"
        )

        # Optuna intenta maximizar este valor
        return accuracy


# Parent run que agrupa todos los trials
with mlflow.start_run(run_name="optuna-study"):

    number_of_trials = 10
    mlflow.log_param("number_of_trials", number_of_trials)

    study = optuna.create_study(direction="maximize")

    study.optimize(
        objective,
        n_trials=number_of_trials
    )

    # Registrar el mejor resultado en el parent run
    mlflow.log_param("best_C", study.best_params["C"])
    mlflow.log_metric("best_accuracy", study.best_value)

    print("\nMejor resultado")
    print("C:", study.best_params["C"])
    print("Accuracy:", study.best_value)
