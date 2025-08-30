# 🧬 Predicción de Diabetes Tipo 2

Una aplicación web interactiva desarrollada con **Streamlit** para predecir diabetes tipo 2 utilizando múltiples algoritmos de machine learning.

## 📋 Descripción

Esta aplicación permite cargar datasets, limpiar datos, entrenar modelos de clasificación y realizar predicciones individuales de diabetes tipo 2. Incluye visualizaciones interactivas y métricas de evaluación para comparar el rendimiento de diferentes algoritmos.

## ✨ Características

- 📂 **Carga de archivos**: Soporte para formatos CSV y ARFF
- 🧹 **Limpieza de datos**: Reemplazo automático de valores ceros por medianas
- 📊 **Visualizaciones**: Gráficos de distribución de clases y matrices de correlación
- 🤖 **Múltiples algoritmos**: Regresión Logística, Árbol de Decisión, Random Forest y SVM
- 📈 **Métricas detalladas**: Reportes de clasificación y matrices de confusión
- 🔮 **Predicción individual**: Interfaz para predicciones en tiempo real
- 💾 **Descarga de datos**: Exportación de datasets limpios

## 🛠️ Tecnologías Utilizadas

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-learn
- **Visualización**: Matplotlib, Seaborn
- **Manipulación de datos**: Pandas, NumPy
- **Formato de datos**: SciPy (para archivos ARFF)

## 📦 Instalación

### Prerrequisitos
- Python 3.8 o superior
- pip

### Pasos de instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/Moltraxpa/PROYECTO_2B.git
cd PROYECTO_2B
```

2. **Crear un entorno virtual**
```bash
python -m venv proyectoia
```

3. **Activar el entorno virtual**

En Windows:
```bash
.\proyectoia\Scripts\activate
```

En macOS/Linux:
```bash
source proyectoia/bin/activate
```

4. **Instalar dependencias**
```bash
pip install streamlit pandas numpy scipy scikit-learn matplotlib seaborn
```

## 🚀 Uso

### Ejecutar la aplicación

```bash
streamlit run proyecto.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

### Pasos para usar la aplicación

1. **Cargar datos**: Sube un archivo CSV o ARFF con datos de diabetes
2. **Limpiar datos**: Usa el botón de limpieza para reemplazar valores faltantes
3. **Explorar datos**: Revisa las visualizaciones y estadísticas
4. **Entrenar modelos**: Selecciona algoritmos y ajusta parámetros
5. **Evaluar resultados**: Analiza métricas y matrices de confusión
6. **Realizar predicciones**: Introduce valores para obtener predicciones individuales

## 📊 Algoritmos Incluidos

| Algoritmo | Descripción | Uso recomendado |
|-----------|-------------|-----------------|
| **Regresión Logística** | Modelo lineal probabilístico | Baseline, interpretabilidad |
| **Árbol de Decisión** | Modelo basado en reglas | Interpretabilidad, datos categóricos |
| **Random Forest** | Ensemble de árboles | Balance precisión-velocidad |
| **SVM** | Máquinas de vectores de soporte | Datos complejos, alta precisión |

## 📁 Estructura del Proyecto

```
PROYECTO_2B/
├── proyecto.py          # Aplicación principal
├── requirements.txt     # Dependencias
├── README.md           # Documentación
└── proyectoia/         # Entorno virtual (local)
```

## 🎯 Características del Dataset

El sistema espera datos con las siguientes columnas:

- `preg`: Número de embarazos
- `plas`: Concentración de glucosa
- `pres`: Presión arterial diastólica
- `skin`: Grosor del pliegue cutáneo
- `insu`: Insulina sérica
- `mass`: Índice de masa corporal
- `pedi`: Función de pedigrí de diabetes
- `age`: Edad
- `class`: Resultado (tested_positive/tested_negative)

## 📈 Métricas de Evaluación

- **Precisión**: Proporción de predicciones correctas
- **Recall**: Proporción de casos positivos identificados
- **F1-Score**: Media armónica de precisión y recall
- **Matriz de Confusión**: Visualización detallada de predicciones



