# Análisis y Predicción de Precios de Equipos (Series de Tiempo)

Este proyecto contiene el análisis exploratorio y el modelado predictivo de precios de equipos industriales en función de tres materias primas (`Price_X`, `Price_Y`, `Price_Z`). El trabajo se estructura en dos fases: análisis exploratorio de datos (EDA) y modelado predictivo con comparación de algoritmos, culminando en una aplicación interactiva con arquitectura multi-agente.

##  Estructura del Proyecto

```text
Gestión_de_costos/
├── README.md
├── requirements.txt
├── Datos/
│   └── historico_equipos.csv
│   └── X.csv
│   └── Y.csv
│   └── Z.csv
├── notebooks/
│   ├── EDA_series_de_tiempo.ipynb       # Análisis exploratorio y validación estadística
│   └── Prediccion.ipynb      # Feature engineering, entrenamiento y evaluación
└── app/
    └── agente.py                        # Aplicación Streamlit + CrewAI (Orquestación)
```

## Notebooks

### 1. Análisis Exploratorio (`EDA_series_de_tiempo.ipynb`)
- Carga, limpieza y validación de tipos de datos.
- Estadísticas descriptivas y boxplots para detección de outliers.
- Resampling temporal (mensual y anual) para suavizar ruido y observar tendencias estructurales.
- Matriz de correlación de Pearson y gráficos de dispersión con ajuste lineal.
- **Hallazgo clave**: Se identificó una relación determinista con correlación de 1.0 entre `Price_Y` y `Price_Equipo1`, y de 0.98 entre `Price_Z` y `Price_Equipo2`. `Price_X` no muestra correlación significativa.

### 2. Predicción y Comparación de Modelos (`Predicción.ipynb`)
- **Baseline**: Media Móvil, estableciendo el punto de referencia mínimo de error.
- **ARIMA (1,1,1)**: Modelo estadístico clásico, precedido por la prueba de Dickey-Fuller aumentada para validar estacionariedad y definir el parámetro de diferenciación ($d$).
- **Random Forest**: Modelo de Machine Learning con feature engineering específico para series de tiempo (creación de lags y ventanas móviles), transformando el problema en regresión supervisada.
- **Evaluación**: Split cronológico (últimos 30 días como test) evaluado con RMSE y MAE. Random Forest obtuvo el mejor desempeño (RMSE 16.19 vs 24.88 del baseline), capturando interacciones no lineales que ARIMA no modela.

## Aplicación Multi-Agente (CrewAI)

La carpeta `app/` contiene una interfaz Streamlit que orquesta un equipo de dos agentes:

- **Analista de Costos**: Ejecuta una herramienta personalizada de Python que realiza una Regresión Lineal Múltiple (OLS) con NumPy. Calcula pesos relativos, estadísticas descriptivas.
- **Gerente de Estrategia**: Sintetiza los resultados cuantitativos en lenguaje ejecutivo, marcando explícitamente qué cifras son datos verificados y cuáles carecen de fuente externa.

El LLM se ejecuta de forma local mediante Ollama (`llama3.1`) con `temperature=0`, garantizando reproducibilidad y eliminando el riesgo de alucinaciones numéricas, ya que el modelo solo redacta sobre los outputs matemáticos validados por la herramienta.

## Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/MSaladoM/Gestion_de_costos.git
   cd Gestion_de_costos

   

2. Crear y activar un entorno virtual (ejemplo con Conda):
   ```bash
   conda create -n series-mlops python=3.11
   conda activate series-mlops

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt

4. (Opcional, solo para la app) Configurar el LLM local:
Descargar e instalar Ollama.
Descargar el modelo: ollama pull llama3.1

## Ejecución
Para la Aplicación Streamlit:
   ```bash
  cd app
  streamlit run agente.py



