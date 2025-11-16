# Instrucciones de Ejecución - LAB-03

## 📋 Resumen

Este proyecto implementa un modelo de clasificación SVC (Support Vector Classifier) para predecir el default de pago de clientes usando un pipeline completo de machine learning.

## ✅ Implementación Completa

Todos los 7 pasos del homework están implementados en `homework/homework.py`:

1. ✅ **Limpieza de datos** - Renombrar columnas, eliminar ID, limpiar NaN, agrupar EDUCATION
2. ✅ **División de datos** - x_train, y_train, x_test, y_test
3. ✅ **Pipeline ML** - OneHotEncoder → PCA → StandardScaler → SelectKBest → SVC
4. ✅ **Optimización** - GridSearchCV con 10-fold CV y balanced_accuracy
5. ✅ **Guardar modelo** - Comprimido en `files/models/model.pkl.gz`
6. ✅ **Métricas** - precision, balanced_accuracy, recall, f1_score
7. ✅ **Matrices de confusión** - Para train y test en `files/output/metrics.json`

## 🚀 Cómo ejecutar en tu PC

### 1. Configurar el entorno

**En macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**En Windows:**
```bash
python3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ejecutar el entrenamiento

```bash
# Opción 1: Ejecutar directamente
python homework/homework.py

# Opción 2: Ejecutar como módulo
python -c "from homework.homework import main; main()"
```

### 3. Verificar resultados

```bash
# Ejecutar tests
pytest tests/test_homework.py -v

# Ver archivos generados
ls -lh files/models/
ls -lh files/output/
cat files/output/metrics.json
```

## ⏱️ Tiempo de Ejecución

El proceso completo incluye:
- **GridSearchCV**: 54 combinaciones de hiperparámetros
- **Cross-validation**: 10 folds
- **Total de ajustes**: 540 entrenamientos del modelo

**Tiempo estimado**: 10-30 minutos dependiendo de tu hardware.

## 🎯 Hiperparámetros que se optimizan

```python
param_grid = {
    'pca__n_components': [10, 15, 20],      # Componentes principales
    'selectkbest__k': [10, 15, 20],         # Features más relevantes
    'svc__C': [0.1, 1, 10],                 # Regularización
    'svc__kernel': ['rbf'],                 # Kernel RBF
    'svc__gamma': ['scale', 'auto']         # Coeficiente del kernel
}
```

## 📊 Salida Esperada

### Archivos generados:

1. **`files/models/model.pkl.gz`** - Modelo entrenado comprimido (~1.2 MB)
2. **`files/output/metrics.json`** - 4 líneas JSON con:
   - Métricas de entrenamiento
   - Métricas de prueba
   - Matriz de confusión de entrenamiento
   - Matriz de confusión de prueba

### Consola durante ejecución:

```
Paso 1: Cargando y limpiando datos...
Paso 2: Dividiendo datos...
Paso 3: Creando pipeline...
Paso 4: Optimizando hiperparámetros...
Fitting 10 folds for each of 54 candidates, totalling 540 fits
Mejores parámetros: {'pca__n_components': X, 'selectkbest__k': Y, ...}
Mejor score: 0.XXXX
Paso 5: Guardando modelo...
Pasos 6 y 7: Calculando métricas y matrices de confusión...
¡Proceso completado!
```

## 📝 Estructura del Código

El archivo `homework/homework.py` contiene funciones modulares:

- `load_data()` - Carga y limpia los datos
- `clean_data(df)` - Limpieza de un dataframe
- `split_data()` - Divide en train/test
- `create_pipeline()` - Crea el pipeline de ML
- `optimize_model()` - GridSearchCV
- `save_model()` - Guarda modelo comprimido
- `calculate_metrics()` - Calcula y guarda métricas
- `main()` - Ejecuta todo el proceso

## 🔧 Ajustes Opcionales

Si el entrenamiento toma demasiado tiempo, puedes reducir el grid de búsqueda editando la función `optimize_model()`:

```python
# Grid reducido (más rápido)
param_grid = {
    'pca__n_components': [15],
    'selectkbest__k': [15],
    'svc__C': [1],
    'svc__kernel': ['rbf'],
    'svc__gamma': ['scale']
}
```

Esto ejecutará solo 10 ajustes (1 combinación × 10 folds) en lugar de 540.

## ❓ Troubleshooting

### Error: "No module named 'sklearn'"
```bash
pip install scikit-learn pandas
```

### Error: "No such file or directory: 'files/input/train_data.csv.zip'"
Verifica que estás ejecutando desde la raíz del proyecto:
```bash
cd LAB-03-prediccion-del-default-usando-svc-Pau-dna
python homework/homework.py
```

### Tests fallan con score bajo
Esto es normal si usaste un grid reducido. Usa el grid completo para mejores resultados:
```python
param_grid = {
    'pca__n_components': [10, 15, 20],
    'selectkbest__k': [10, 15, 20],
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['rbf'],
    'svc__gamma': ['scale', 'auto']
}
```

## 📚 Documentación Adicional

- Scikit-learn Pipeline: https://scikit-learn.org/stable/modules/compose.html
- GridSearchCV: https://scikit-learn.org/stable/modules/grid_search.html
- SVC: https://scikit-learn.org/stable/modules/svm.html

---

**¡El código está listo para ejecutar! Solo necesitas correr `python homework/homework.py` en tu PC.** 🎉
