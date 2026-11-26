import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Configurar MLflow
mlflow.set_tracking_uri("http:// ... :5000")  #COMPLETAR
mlflow.set_experiment(" ... ") #COMPLETAR

# Preparar los datos
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Parametros del modelo
params = {"max_iter": 100, "C": 0.1}

# Crear explícitamente una ejecución
#mlflow.start_run()
mlflow.start_run(run_name=f"regression-C-{params['C']}")

model = LogisticRegression(**params)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

# Registro manual
mlflow.log_params(params)
mlflow.log_metric("accuracy", ...)  #COMPLETAR

mlflow.sklearn.log_model(
    sk_model=model, name="model", serialization_format="skops"
)

print("Accuracy:", accuracy)

mlflow.end_run()
