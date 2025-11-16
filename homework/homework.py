# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import gzip
import json
import os
import pickle
import zipfile

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def load_data():
    """
    Paso 1: Cargar y limpiar los datos.
    """
    # Cargar datos de entrenamiento
    with zipfile.ZipFile('files/input/train_data.csv.zip', 'r') as z:
        with z.open('train_default_of_credit_card_clients.csv') as f:
            train_df = pd.read_csv(f)
    
    # Cargar datos de prueba
    with zipfile.ZipFile('files/input/test_data.csv.zip', 'r') as z:
        with z.open('test_default_of_credit_card_clients.csv') as f:
            test_df = pd.read_csv(f)
    
    # Limpiar datos de entrenamiento
    train_df = clean_data(train_df)
    
    # Limpiar datos de prueba
    test_df = clean_data(test_df)
    
    return train_df, test_df


def clean_data(df):
    """
    Limpia un dataframe según las especificaciones del problema.
    """
    # Renombrar la columna "default payment next month" a "default"
    df = df.rename(columns={'default payment next month': 'default'})
    
    # Remover la columna "ID"
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    
    # Eliminar registros con información no disponible (valores nulos)
    df = df.dropna()
    
    # Para la columna EDUCATION, agrupar valores > 4 en la categoría "others" (4)
    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
    
    return df


def split_data(train_df, test_df):
    """
    Paso 2: Dividir los datasets en x_train, y_train, x_test, y_test.
    """
    x_train = train_df.drop(columns=['default'])
    y_train = train_df['default']
    
    x_test = test_df.drop(columns=['default'])
    y_test = test_df['default']
    
    return x_train, y_train, x_test, y_test


def create_pipeline():
    """
    Paso 3: Crear pipeline con las transformaciones y el modelo.
    """
    # Identificar columnas categóricas
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    
    # Identificar columnas numéricas
    numerical_features = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 
                         'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
                         'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                         'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    # Crear el preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
            ('num', 'passthrough', numerical_features)
        ])
    
    # Crear el pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('pca', PCA()),
        ('scaler', StandardScaler()),
        ('selectkbest', SelectKBest(score_func=f_classif)),
        ('svc', SVC())
    ])
    
    return pipeline


def optimize_model(pipeline, x_train, y_train):
    """
    Paso 4: Optimizar hiperparámetros usando validación cruzada.
    """
    # Definir el grid de hiperparámetros
    # Grid optimizado para encontrar mejores hiperparámetros
    param_grid = {
        'pca__n_components': [10, 15, 20],
        'selectkbest__k': [10, 15, 20],
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['rbf'],
        'svc__gamma': ['scale', 'auto']
    }
    
    # Crear GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Ajustar el modelo
    grid_search.fit(x_train, y_train)
    
    return grid_search


def save_model(model):
    """
    Paso 5: Guardar el modelo comprimido.
    """
    # Crear directorio si no existe
    os.makedirs('files/models', exist_ok=True)
    
    # Guardar modelo comprimido
    with gzip.open('files/models/model.pkl.gz', 'wb') as f:
        pickle.dump(model, f)


def calculate_metrics(model, x_train, y_train, x_test, y_test):
    """
    Paso 6 y 7: Calcular métricas y matrices de confusión.
    """
    # Crear directorio si no existe
    os.makedirs('files/output', exist_ok=True)
    
    # Predecir
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    # Calcular métricas para entrenamiento
    train_metrics = {
        'type': 'metrics',
        'dataset': 'train',
        'precision': precision_score(y_train, y_train_pred),
        'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1_score': f1_score(y_train, y_train_pred)
    }
    
    # Calcular métricas para prueba
    test_metrics = {
        'type': 'metrics',
        'dataset': 'test',
        'precision': precision_score(y_test, y_test_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred)
    }
    
    # Calcular matrices de confusión
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    # Formatear matrices de confusión
    train_cm = {
        'type': 'cm_matrix',
        'dataset': 'train',
        'true_0': {
            'predicted_0': int(cm_train[0, 0]),
            'predicted_1': int(cm_train[0, 1])
        },
        'true_1': {
            'predicted_0': int(cm_train[1, 0]),
            'predicted_1': int(cm_train[1, 1])
        }
    }
    
    test_cm = {
        'type': 'cm_matrix',
        'dataset': 'test',
        'true_0': {
            'predicted_0': int(cm_test[0, 0]),
            'predicted_1': int(cm_test[0, 1])
        },
        'true_1': {
            'predicted_0': int(cm_test[1, 0]),
            'predicted_1': int(cm_test[1, 1])
        }
    }
    
    # Guardar métricas
    with open('files/output/metrics.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(train_metrics) + '\n')
        f.write(json.dumps(test_metrics) + '\n')
        f.write(json.dumps(train_cm) + '\n')
        f.write(json.dumps(test_cm) + '\n')


def main():
    """
    Función principal para ejecutar todo el pipeline.
    """
    # Paso 1: Cargar y limpiar datos
    print("Paso 1: Cargando y limpiando datos...")
    train_df, test_df = load_data()
    
    # Paso 2: Dividir datos
    print("Paso 2: Dividiendo datos...")
    x_train, y_train, x_test, y_test = split_data(train_df, test_df)
    
    # Paso 3: Crear pipeline
    print("Paso 3: Creando pipeline...")
    pipeline = create_pipeline()
    
    # Paso 4: Optimizar modelo
    print("Paso 4: Optimizando hiperparámetros...")
    model = optimize_model(pipeline, x_train, y_train)
    
    print(f"Mejores parámetros: {model.best_params_}")
    print(f"Mejor score: {model.best_score_:.4f}")
    
    # Paso 5: Guardar modelo
    print("Paso 5: Guardando modelo...")
    save_model(model)
    
    # Paso 6 y 7: Calcular y guardar métricas
    print("Pasos 6 y 7: Calculando métricas y matrices de confusión...")
    calculate_metrics(model, x_train, y_train, x_test, y_test)
    
    print("¡Proceso completado!")


if __name__ == '__main__':
    main()
