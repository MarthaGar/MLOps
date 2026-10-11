# Experimentación con MLflow: mantenimiento predictivo

Entrena un Random Forest para predecir de manera preventiva fallas de máquina (dataset AI4I 2020)
y registra cada corrida en MLflow. Utiliza la plantilla de CookieCutter-Data-Science 
simplificada para esta actividad. 

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Estructura del proyecto
```
├── LICENSE
├── Makefile
├── README.md                           <- Guía para ejecutar el proyecto.
├── data
│   ├── external
│   ├── interim
│   ├── processed
│   └── raw                             <- Dataset original (ai4i2020.csv).
├── docs                                <- Estructura de la plantilla (sin uso).
│   ├── README.md
│   ├── docs
│   └── mkdocs.yml
├── experimentacion_machine_failure     <- Estructura de la plantilla (sin uso).
│   └── __init__.py
├── models                              <- Estructura de la plantilla (sin uso).
├── notebooks                           <- Jupyter notebooks (EDA y Baseline).
│   ├── 1.0-mga-initial-baseline_model.ipynb
│   └── 2.0-mga-eda.ipynb
├── pyproject.toml                      <- Estructura de la plantilla (sin uso).
├── references                          <- Estructura de la plantilla (sin uso).
├── reports                             <- Estructura de la plantilla (sin uso).
│   └── figures
├── requirements.txt                    <- Librerías necesarias para reproducir el proyecto.
├── setup.cfg
└── src
    └── train.py                        <- Script de entrenamiento y MLflow tracking. 
```


- `src/train.py`: script de entrenamiento y evaluación de modelo.
- `data/raw/`: carpeta donde se guarda el dataset ([descarga en UCI](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)).
- `requirements.txt`: dependencias con versiones.
- `notebooks/`: Notebooks versionados.

## Cómo ejecutar

### 1. Clonar el repositorio
Abre tu terminal, clona el proyecto en tu equipo local e ingresa a la carpeta del proyecto:

```bash
git clone https://github.com/MarthaGar/MLOps.git
cd MLOps/act3_experimentacion/pred_maintenance_model
```

### 2. Crear entorno virtual
Se recomienda crear un entorno virtual para evitar conflictos con otras librerías:

```bash
# Crear entorno virtual
python -m venv .venv

# Activar en Linux / macOS / Git Bash:
source .venv/bin/activate

# Activar en Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 3. Requisitos e instalación
* Desarrollado y probado en Python 3.14.7.

Instalar las dependencias en el entorno activo:
```bash
pip install -r requirements.txt
```
### 4. Guardar dataset

El archivo `ai4i2020.csv` no está incluido en el repositorio por temas de tamaño y versionado. Para obtenerlo:

1. Descárgalo desde el [repositorio de UCI (AI4I 2020)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).
2. Asegúrate de estar parado en la raíz de este proyecto (`pred_maintenance_model`) y crea la carpeta de destino:

```bash
mkdir -p data/raw
```
3. Mueve o guarda el archivo descargado directamente en esa ruta: `data/raw/ai4i2020.csv`.

(Nota: El archivo también se adjunta en la entrega de Canvas).

### 5. Iniciar el servidor de MLFlow

En una terminal (con el entorno activo y ubicada en pred_maintenance_model):
```bash
mlflow server --host 127.0.0.1 --port 8080
```

### 6. Ejecutar el script de entrenamiento
En una segunda terminal (con el entorno activo y ubicada en pred_maintenance_model):
```bash
python src/train.py
```

### Argumentos que puede recibir train.py
Para permitir la ejecución dinámica de experimentos sin modificar el código fuente, el script expone los siguientes argumentos por línea de comandos:

| Argumento | Qué controla | Valor por defecto |
|---|---|---|
| `--n_estimators` | Número de árboles en el modelo | 100 |
| `--max_depth` | Profundidad de los árboles | None (sin límite de profundidad)|
| `--min_samples_leaf` | Muestras mínimas por hoja | 1 |
| `--seed` | Semilla para la aleatoriedad | 42 |
| `--run_name` | Nombre de la corrida en MLflow | None (MLFlow asigna un nombre) |
| `--description` | Breve descripción de la corrida | '' |

Si durante la ejecución existe alguna duda sobre los parámetros y su valor por default, se puede ejecutar el comando:
```bash
python src/train.py --help
```

### 7. Verificar registros en la interfaz de MLflow
Abrir <http://127.0.0.1:8080/> en el navegador para ver las corridas.

## Corridas realizadas
1. Modelo base
```bash
python src/train.py --run_name modelbase --description "Modelo de referencia"
```
2. Experimentación de impacto n_estimators
```bash
python src/train.py --run_name n_estimators_250 --description "Aumentar el número de árboles con respecto a modelbase" --n_estimators 250      
```
```bash            
python src/train.py --run_name n_estimators_25 --description "Disminuir el número de árboles con respecto a modelbase" --n_estimators 25          
```
3. Experimentación de impacto max_depth
```bash       
python src/train.py --run_name max_depth_1 --description "Limitar la complejidad" --max_depth 1   
```
```bash                                                       
python src/train.py --run_name max_depth_9 --description "Limitar la complejidad" --max_depth 9    
``` 
4. Experimentación de impacto min_samples_leaf
```bash                                                      
python src/train.py --run_name min_samples_leaf_4 --description "Limitar la complejidad" --min_samples_leaf 4   
```
```bash                                                       
python src/train.py --run_name min_samples_leaf_2 --description "Limitar la complejidad" --min_samples_leaf 2   
```
5. Experimentación aleatoriedad 
```  bash 
python src/train.py --run_name seed_18 --description "Verificar el impacto de la semilla" --seed 18
```
```bash
python src/train.py --run_name seed_50 --description "Verificar el impacto de la semilla" --seed 50
```
## Reproducibilidad
Prueba de reproducibilidad
```bash
python src/train.py --run_name reprod_verification --description "Verificar la reproducibilidad"   
```
### Controles de reproducibilidad
- **Semilla:** `--seed` (por defecto 42) controla las dos fuentes de aleatoriedad:
  la partición estratificada (`train_test_split`) y el `RandomForestClassifier`.
- **Dependencias:** `requirements.txt` fija las versiones exactas (`==`) y el
  proyecto se probó con Python 3.14.7.
- **Datos:** cada corrida registra el *digest* del dataset en MLflow, lo que
  permite verificar que se usó el mismo archivo `ai4i2020.csv`.
- **Verificación:** se comprobó clonando el repositorio en un entorno limpio,
  instalando `requirements.txt` y ejecutando el script; `modelbase` y
  `reprod_verification` producen el mismo PR-AUC (0.6958970295656021).
- **Alcance:** la igualdad exacta se garantiza con las mismas versiones de
  Python y librerías; con versiones distintas de scikit-learn o numpy podrían
  aparecer diferencias numéricas pequeñas.

La semilla predeterminada es 42 y se utiliza tanto para la partición estratificada de los datos como para el Random Forest. Puede modificarse mediante --seed para analizar la sensibilidad de los resultados a la aleatoriedad. Para reproducir una corrida, deben mantenerse iguales los argumentos, los datos y las versiones de las dependencias.

## Estructura del notebook
Se modificó el notebook de la actividad pasada ahora tiene el nombre 2.0-mga-eda.ipynb. Aquí se mantiene la exploración de datos y la validación cruzada que permitió elegir el modelo baseline. Todo el flujo de entrenamiento, evaluación de modelo y experimentación con MLFlow se migró al script train.py.

## Estructura train.py

La lógica se encapsuló en funciones independientes con responsabilidades únicas:
   * `load_data()`: Carga del dataset crudo.
   * `split_data()`: Partición estratificada del dataset (70% entrenamiento, 30% prueba).
   * `build_pipeline()`: Creación de un `Pipeline` de Scikit-Learn que integra el preprocesamiento (`StandardScaler` para variables numéricas, `OneHotEncoder` para categóricas) y el clasificador (`RandomForestClassifier` con `class_weight='balanced'`).
   * `evaluate_model()`: Recibe el pipeline y los datos de prueba `y_pred_proba` y `y_pred`   
   * `calc_metrics()`: Cálculo de métricas (`test_pr_auc`, `test_precision`, `test_recall`, `test_f1`).
   * `reg_dataset_mlflow()`: Registro del dataset como entrada de MLFlow.
   * `plot_confusion_matrix()`: Genera la matriz de confusión de cada corrida.
   * `plot_prc_curve()`: Genera la curva de PR-AUC de cada corrida.
