"""
Experimentación con MLFlow, creando funciones a partir de modelo baseline RandomForest.
El cual fue determinado a partir de validación cruzada durante la exploración del dataset.
"""
#IMPORTS
#Librerías para calcular ruta de dataset
from pathlib import Path
#Librería para recibir argumentos en la terminal
import argparse
#Librerías para carga y análisis de datos
import pandas as pd
import matplotlib.pyplot as plt
#Librerías de Scikit-learn para preprocesamiento, modelado y evaluación
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import classification_report, average_precision_score, precision_score, recall_score, f1_score, precision_recall_curve
#Librerías para experimentación en MLFlow
import mlflow
from mlflow.models import infer_signature

#VARIABLES GLOBALES
# Raíz del proyecto, calculada a partir de la ubicación de este archivo para correrse en cualquier computadora
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = str(PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv")

#Variables para modelo
TEST_SIZE = 0.3 #Definido como el valor para test split 70%/30%
TARGET = "Machine failure"
NUM_COLS = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
CAT_COLS = ['Type']
CLASS_WEIGHT = 'balanced'

#Variables para experimento MLFlow
MLFLOW_URI = "http://127.0.0.1:8080"
EXPERIMENT_NAME = "Predictive_maintenance_random_forest_model"

#DEFINICIÓN FUNCIONES
def load_data(filepath):
    """Carga el CSV y devuelve un dataFrame."""
    data = pd.read_csv(filepath)
    return data

#Separación de datos
def split_data(data, test_size, seed):
    """Separa el dataset en train/test."""
    # Separar las características (X) de la variable objetivo (y)
    X = data[NUM_COLS + CAT_COLS]
    y = data[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

#Pipeline de preprocesamiento y modelo
def build_pipeline(n_estimators, max_depth, min_samples_leaf, seed):
    """Construye el pipeline: preprocesamiento y el modelo Random Forest."""
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        class_weight=CLASS_WEIGHT, random_state=seed)
    return Pipeline([("prep", preprocessor), ("clf", model)])

#Evaluación modelo
def evaluate_model(pipe, X_test):
    """Función que recibe el pipeline y los datos de prueba."""
    y_pred_proba = pipe.predict_proba(X_test)[:,1]
    y_pred = pipe.predict(X_test)
    return y_pred_proba, y_pred

#Cálculo métricas
def calc_metrics(y_test, y_pred_proba, y_pred):
    """Función que calcula las métricas."""
    return {
        "test_pr_auc": average_precision_score(y_test, y_pred_proba),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
    }
#Argumentos recibidos en la terminal
def parse_arguments():
    """Lee los hiperparámetros desde la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelo Random Forest para mantenimiento predictivo")
    #Hiperparámetro 1: n_estimators
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Número de árboles en el Random Forest (por defecto: 100)")
    #Hiperparámetro 2: max_depth
    parser.add_argument("--max_depth", type=lambda x: int(x) if x != 'None' else None, default=None,
                        help="Profundidad máxima de los árboles (por defecto: None)")
    #Hiperparámetro 3: min_samples_leaf
    parser.add_argument("--min_samples_leaf", type=int, default=1,
                        help="Mínimo de muestras por hoja, 1 si se omite")
    #Hiperparámetro 4: seed
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para la reproducibilidad (por defecto: 42)")
    #Datos para corrida de experimento
    parser.add_argument ("--run_name", type=str, default=None,
                        help="Nombre de la corrida en MLFlow")
    parser.add_argument("--description", type=str, default="",
                        help="Descripción breve de la corrida")
    return parser.parse_args()

#Funciones para experimento MLFlow
def reg_dataset_mlflow(maintenance_df):
        """Registrar el dataset en MLFlow."""
        dataset = mlflow.data.from_pandas(
        maintenance_df,
        source=CSV_PATH,
        name="ai4i2020_predictive_maintenance",
        targets=TARGET
        )
        mlflow.log_input(dataset, context="training")

#Artefactos
def plot_confusion_matrix(y_test, y_pred):
    """Carga la matriz de confusión a MLFlow"""
    cm = confusion_matrix(
        y_test, y_pred
    )  
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No falla", "Falla"]
    ).plot(cmap=plt.cm.Blues)

    mlflow.log_figure(disp.figure_, "confusion_matrix.png")
    plt.close(disp.figure_)

def plot_prc_curve(y_test, y_pred_proba, y_pred):
    """Carga la curva de PRC-AUC a MLFlow"""
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision,
    label=f"PR-AUC = {average_precision_score(y_test, y_pred_proba):.3f}")
    ax.scatter(recall_score(y_test, y_pred), precision_score(y_test, y_pred),
        zorder=3, label="Umbral 0.5")          # umbral por defecto
    ax.axhline(y_test.mean(), color="gray", linestyle="--",
        label=f"Azar ({y_test.mean():.3f})")   # referencia de azar
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisión")
    ax.set_title("Curva Precisión-Recall:")
    ax.legend()
    mlflow.log_figure(fig, "pr_curve.png")
    plt.close(fig)

def main():
    args = parse_arguments()
    maintenance_df = load_data(CSV_PATH)
    #División dataset
    X_train, X_test, y_train, y_test = split_data(maintenance_df, 
                                                  TEST_SIZE, 
                                                  args.seed)
    #Crear pipeline
    pipe = build_pipeline(
        args.n_estimators,
        args.max_depth,
        args.min_samples_leaf,
        args.seed,
    )
    #Entrenar modelo
    pipe.fit(X_train, y_train)
    #Evaluar modelo
    y_pred_proba, y_pred = evaluate_model(pipe, X_test)
    metrics = calc_metrics(y_test, y_pred_proba, y_pred)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=args.run_name):
        reg_dataset_mlflow(maintenance_df)
        mlflow.set_tag("descripcion", args.description)
        #Parametros del experimento, excluyendo run_name y description     
        excluir = {'run_name', 'description'}
        params = {
        k: v for k, v in vars(args).items() if k not in excluir
        }
        params.update({"test_size": TEST_SIZE, "class_weight": CLASS_WEIGHT})
        #Registro de métricas y parámetros en MLFlow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        plot_confusion_matrix(y_test, y_pred)
        plot_prc_curve(y_test, y_pred_proba, y_pred)
        mlflow.sklearn.log_model(
            pipe,
            name="modelo",
            signature=infer_signature(X_train, y_pred),
            input_example=X_test.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")

    print(metrics)


if __name__ == "__main__":
    main()

