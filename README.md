# 🚢 Titanic Analytics & Predictor

> De un dataset histórico a un producto de datos interactivo: análisis visual, *feature engineering* y predicción de supervivencia en una aplicación web creada con Streamlit.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Data%20App-FF4B4B?logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)

## ✨ Sobre el proyecto

**Titanic Analytics & Predictor** es una aplicación de analítica y *machine learning* que transforma el dataset del Titanic en una experiencia interactiva orientada a la toma de decisiones.

El proyecto no se limita a entrenar un modelo: conecta limpieza de datos, creación de variables, visualización interactiva y un simulador predictivo dentro de una interfaz clara y accesible. Es una muestra práctica de cómo convertir un análisis exploratorio en un prototipo funcional listo para ser compartido, validado y evolucionado.

## 🎯 Qué demuestra

- **Análisis de datos:** exploración de patrones de supervivencia, perfiles de pasajeros, clase social, familia, puerto de embarque y tarifas.
- **Feature engineering:** extracción de títulos sociales, tamaño familiar, pasajeros que viajan solos, cubierta y coste individual por billete.
- **Machine learning aplicado:** uso de un modelo previamente entrenado para estimar la probabilidad de supervivencia a partir de datos introducidos por el usuario.
- **Data storytelling:** gráficos interactivos y métricas que permiten pasar de datos históricos a conclusiones comprensibles.
- **Prototipado rápido:** desarrollo de una aplicación web funcional con Streamlit.
- **Diseño mantenible:** separación entre interfaz (`app.py`) y lógica de negocio/procesamiento (`logic.py`).

## 🧭 Funcionalidades

### 📌 Resumen ejecutivo

- KPIs dinámicos de pasajeros analizados, fallecidos, tasa de supervivencia, tarifa individual promedio y edad promedio.
- Filtros globales por clase, género y puerto de embarque.
- Visualización de supervivencia por clase y género.
- Distribución de pasajeros por puerto de embarque.

### 🧐 Historias ocultas

- Análisis de supervivencia por perfil social: Señor, Señora, Señorita, Joven/niño y Título nobiliario.
- Exploración del impacto del tamaño familiar en la supervivencia.

### 💰 Economía e inflación

- Cálculo de tarifa individual a partir de billetes compartidos.
- Conversión estimada de tarifas históricas a euros actuales.
- Comparación de precios por clase y cubierta del barco.

### 🔮 Simulador de supervivencia

- Formulario para introducir clase, género, edad, título social, acompañantes y tarifa.
- Predicción de probabilidad de supervivencia en tiempo real.
- Explicación contextual de los factores relevantes para el resultado.

## 🚀 Demo

[👉 **Probar la aplicación en Streamlit**](https://proyecto-titanic.streamlit.app/)

## ▶️ Youtube

[👉 **Ver el video del proyecto**](https://youtu.be/lN9bZSHNfTU)

## 🏗️ Arquitectura

```text
proyecto_titanic/
├── app.py                            # Interfaz, visualizaciones y experiencia Streamlit
├── logic.py                          # Preprocesamiento, filtros, carga del modelo y predicción
├── titanic.csv                       # Dataset de entrada
├── model_titanic.pkl                 # Modelo entrenado
├── model_columns.pkl                 # Columnas esperadas por el modelo
├── requirements.txt                  # Dependencias de Python
├── diccionario_de_datos_titanic.txt  # Traduccion de algunos nombres
└── Titanic.ipynb                     # Trabajo exploratorio y entrenamiento del modelo
```

La separación de responsabilidades permite modificar el procesamiento o el modelo sin mezclar esa lógica con el código de presentación.

## 🛠️ Stack tecnológico

| Área | Tecnologías |
| --- | --- |
| Lenguaje | Python |
| Procesamiento de datos | Pandas |
| Visualización | Plotly Express |
| Aplicación web | Streamlit |
| Machine Learning | scikit-learn |
| Persistencia del modelo | Joblib |

## 🚀 Ejecución local

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd proyecto_titanic
```

### 2. Crear y activar un entorno virtual

```bash
python -m venv .venv
```

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Iniciar la aplicación

```bash
streamlit run app.py
```

La aplicación estará disponible normalmente en `http://localhost:8501`.

## 🔍 Decisiones de análisis destacadas

- La tarifa (`Fare`) se transforma en **tarifa por persona** al dividirla entre las personas que comparten el mismo billete.
- Las tarifas se convierten a una estimación de euros actuales para facilitar su interpretación.
- Los títulos extraídos del nombre permiten aproximar la estructura social de los pasajeros.
- El tamaño familiar y la variable de viaje en solitario incorporan contexto adicional al análisis y a la predicción.
- Los filtros globales mantienen sincronizadas las métricas y visualizaciones para explorar segmentos concretos de pasajeros.

## 🧪 Posibles siguientes pasos

- Añadir métricas de evaluación y comparación entre modelos.
- Incorporar explicabilidad local de predicciones con SHAP o técnicas similares.
- Publicar la aplicación en Streamlit Community Cloud.
- Añadir pruebas automatizadas para las funciones de `logic.py`.
- Incorporar capturas de pantalla o un enlace a una demo desplegada.

## 👤 Autoría

Proyecto desarrollado como ejercicio de portafolio de **análisis de datos, machine learning y prototipado rápido de productos de datos**.

---

Si te interesa conversar sobre el enfoque técnico, decisiones de modelado o cómo llevar un análisis a una aplicación interactiva, estaré encantado de hacerlo.
