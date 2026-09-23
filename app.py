import pandas as pd
import plotly.express as px
import streamlit as st

# Importar funciones de lógica de negocio
from logic import (
    filter_dataframe,
    fmt_num,
    load_model_artifacts,
    predict_survival,
    preprocess_data,
)

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA Y CARGA DE RECURSOS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Titanic Analytics & Simulador",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def get_data():
    df_raw = pd.read_csv('titanic.csv')
    return preprocess_data(df_raw)

@st.cache_resource
def get_model():
    return load_model_artifacts()

# Cargar DataFrame
df = get_data()

# Cargar Modelo de Inteligencia Artificial
model, model_columns = None, None
try:
    model, model_columns = get_model()
    model_loaded = True
except (FileNotFoundError, OSError, ValueError, ImportError):
    model_loaded = False
    st.error("⚠️ No se pudieron cargar los archivos 'model_titanic.pkl' y 'model_columns.pkl'. Asegúrate de haber ejecutado el entrenamiento primero.")

# -----------------------------------------------------------------------------
# 2. BARRA LATERAL (FILTROS GLOBALES)
# -----------------------------------------------------------------------------
st.sidebar.title("🚢 Filtros Globales")

pclass_filter = st.sidebar.multiselect(
    "Clase del Pasajero:",
    options=[1, 2, 3],
    default=[1, 2, 3]
)

sex_filter = st.sidebar.multiselect(
    "Género:",
    options=['male', 'female'],
    format_func=lambda x: "Hombre" if x == 'male' else "Mujer",
    default=['male', 'female']
)

embarked_filter = st.sidebar.multiselect(
    "Puerto de Embarque:",
    options=['S', 'C', 'Q'],
    format_func=lambda x: {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}[x],
    default=['S', 'C', 'Q']
)

# Aplicar filtros
df_filtered = filter_dataframe(df, pclass_filter, sex_filter, embarked_filter)

# -----------------------------------------------------------------------------
# 3. ESTRUCTURA PRINCIPAL (PESTAÑAS)
# -----------------------------------------------------------------------------
st.title("🚢 Titanic: Análisis de Datos & Simulador Predictivo")
st.markdown("Exploración interactiva de datos históricos e inteligencia artificial para evaluar la supervivencia en el desastre de 1912.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Resumen Ejecutivo",
    "🧐 Historias Ocultas",
    "💰 Economía e Inflación",
    "🔮 Simulador de Supervivencia"
])

# =============================================================================
# PESTAÑA 1: RESUMEN EJECUTIVO
# =============================================================================
with tab1:
    st.header("📌 Resumen Ejecutivo y Métricas Globales")

    col1, col2, col3, col4, col5 = st.columns(5)
    total_pasajeros = len(df_filtered)
    total_fallecidos = int((df_filtered['Survived'] == 0).sum())
    tasa_supervivencia = (df_filtered['Survived'].mean() * 100) if total_pasajeros > 0 else 0
    tarifa_promedio = df_filtered['Fare_Per_Person_EUR'].mean() if total_pasajeros > 0 else 0
    edad_promedio = df_filtered['Age'].mean() if total_pasajeros > 0 else 0

    col1.metric("Pasajeros Analizados", fmt_num(total_pasajeros, 0))
    col2.metric("Fallecidos", fmt_num(total_fallecidos, 0))
    col3.metric("Tasa de Supervivencia", f"{fmt_num(tasa_supervivencia, 1)}%")
    col4.metric("Tarifa Prom. Individual", f"€{fmt_num(tarifa_promedio, 2)}")
    col5.metric("Edad Promedio", f"{fmt_num(edad_promedio, 1)} años")

    st.markdown("---")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("Supervivencia por Clase y Género")
        df_plot_survival = df_filtered.copy()
        df_plot_survival['Estado'] = [
            'Superviviente' if survived == 1 else 'Fallecido'
            for survived in df_plot_survival['Survived']
        ]
        df_plot_survival['Género'] = [
            'Hombre' if sex == 'male' else 'Mujer'
            for sex in df_plot_survival['Sex']
        ]

        fig_class = px.histogram(
            df_plot_survival, x="Pclass", color="Estado", barmode="group",
            facet_col="Género",
            labels={"Pclass": "Clase", "count": "Pasajeros", "Género": "Género"},
            color_discrete_map={'Fallecido': '#EF553B', 'Superviviente': '#00CC96'}
        )
        fig_class.update_layout(separators=",.")
        st.plotly_chart(fig_class, use_container_width=True)

    with col_g2:
        st.subheader("Distribución de Pasajeros por Puerto")
        df_plot_embarked = df_filtered.copy()
        puertos_map = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
        df_plot_embarked['Puerto'] = [
            puertos_map[embarked]
            for embarked in df_plot_embarked['Embarked']
        ]

        fig_embarked = px.pie(
            df_plot_embarked, names='Puerto', title="Porcentaje de Pasajeros por Puerto",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_embarked.update_layout(separators=",.")
        st.plotly_chart(fig_embarked, use_container_width=True)

# =============================================================================
# PESTAÑA 2: HISTORIAS OCULTAS
# =============================================================================
with tab2:
    st.header("🧐 Historias Ocultas y Estructura Social")
    st.markdown("Análisis del impacto de la jerarquía social y las redes familiares en las probabilidades de rescate.")

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.subheader("Jerarquía Social mediante Títulos")
        df_plot_title = df_filtered.copy()
        title_labels = {
            'Mr': 'Señor', 'Miss': 'Señorita', 'Mrs': 'Señora',
            'Master': 'Joven/niño', 'Rare': 'Título nobiliario'
        }
        df_plot_title['Individuo'] = [
            title_labels[title]
            for title in df_plot_title['Title']
        ]
        df_plot_title['Estado'] = [
            'Superviviente' if survived == 1 else 'Fallecido'
            for survived in df_plot_title['Survived']
        ]

        fig_title = px.histogram(
            df_plot_title, x="Individuo", color="Estado", barmode="stack",
            labels={"Individuo": "Individuo", "Estado": "Estado"},
            title="Distribución de Supervivencia por Individuo"
        )
        fig_title.update_layout(separators=",.")
        st.plotly_chart(fig_title, use_container_width=True)

    with col_t2:
        st.subheader("Impacto del Tamaño Familiar")
        df_plot_family = df_filtered.copy()
        df_plot_family['Estado'] = [
            'Superviviente' if survived == 1 else 'Fallecido'
            for survived in df_plot_family['Survived']
        ]

        fig_family = px.histogram(
            df_plot_family, x="FamilySize", color="Estado", barmode="group",
            labels={"FamilySize": "Integrantes de la Familia", "Estado": "Estado"},
            title="Supervivencia según Tamaño de la Familia"
        )
        fig_family.update_layout(separators=",.")
        st.plotly_chart(fig_family, use_container_width=True)

    st.info("💡 **Dato clave:** Viajar en familias pequeñas (2 a 4 miembros) incrementó significativamente las posibilidades de rescate respecto a viajar solo o en familias numerosas (>5 miembros).")

# =============================================================================
# PESTAÑA 3: ECONOMÍA E INFLACIÓN
# =============================================================================
with tab3:
    st.header("💰 Economía del Titanic e Inflación Histórica")
    st.markdown("Ajuste de las tarifas grupales para obtener el costo individual real convertido a Euros actuales (€).")

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.subheader("Tarifa Individual Actual (€) por Clase")
        fig_box = px.box(
            df_filtered, x="Pclass", y="Fare_Per_Person_EUR", color="Pclass",
            labels={"Pclass": "Clase", "Fare_Per_Person_EUR": "Tarifa Individual (€)"},
            points="outliers"
        )
        fig_box.update_layout(separators=",.")
        st.plotly_chart(fig_box, use_container_width=True)

    with col_e2:
        st.subheader("Distribución de Precios por Cubierta (Deck)")
        df_plot_deck = df_filtered[df_filtered['Deck'] != 'U'].copy()
        df_plot_deck['Estado'] = [
            'Superviviente' if survived == 1 else 'Fallecido'
            for survived in df_plot_deck['Survived']
        ]

        fig_deck = px.box(
            df_plot_deck, x="Deck", y="Fare_Per_Person_EUR", color="Estado",
            labels={"Deck": "Cubierta", "Fare_Per_Person_EUR": "Tarifa Individual (€)", "Estado": ""},
            title="Precios de Cubiertas Conocidas (A-G)"
        )
        fig_deck.update_layout(separators=",.", legend_title_text="")
        st.plotly_chart(fig_deck, use_container_width=True)

    st.info(
        "💡 **Cómo interpretar los gráficos:** cada importe representa el coste estimado por pasajero "
        "en euros actuales, calculado a partir de la tarifa histórica. En los diagramas de caja, la línea central "
        "indica la mediana, la caja reúne el rango más habitual de precios y los puntos muestran valores atípicos. "
        "Las clases y cubiertas con importes más altos reflejan, en general, alojamientos con mayor nivel de comodidad "
        "y ubicación en el barco."
    )

# =============================================================================
# PESTAÑA 4: SIMULADOR INTERACTIVO
# =============================================================================
with tab4:
    st.header("🔮 Simulador Interactivo: ¿Habrías Sobrevivido?")
    st.markdown("Introduce tus datos de viaje para que nuestro modelo de **Random Forest** evalúe tus probabilidades de supervivencia.")

    if not model_loaded or model is None or model_columns is None:
        st.warning("El simulador no está disponible porque falta el modelo entrenado (.pkl).")
    else:
        col_s1, col_s2 = st.columns([1, 1])

        with col_s1:
            st.subheader("Ingresa tus Datos de Pasajero")

            pclass_input = st.selectbox("Clase en la que viajarías:", [1, 2, 3], index=2)
            sex_input = st.radio("Género:", ["male", "female"], format_func=lambda x: "Hombre" if x == 'male' else "Mujer")
            age_input = st.slider("Edad:", min_value=1, max_value=80, value=25)

            title_input = st.selectbox(
                "Título social:",
                ["Mr", "Mrs", "Miss", "Master", "Rare"],
                format_func=lambda title: {
                    "Mr": "Señor", "Mrs": "Señora", "Miss": "Señorita",
                    "Master": "Joven/niño", "Rare": "Título nobiliario"
                }[title]
            )

            family_members = st.number_input("Número de familiares a bordo (cónyuge, hermanos, padres, hijos):", min_value=0, max_value=10, value=0)
            fare_input = st.slider("Tarifa estimada a pagar por persona (€ actuales):", min_value=500, max_value=25000, value=1200, step=500)

            btn_predict = st.button("🚀 Calcular Probabilidad", type="primary")

        with col_s2:
            st.subheader("Resultado de la Predicción")

            if btn_predict:
                # Realizar predicción desde la capa de lógica
                prob_survived = predict_survival(
                    model, model_columns,
                    pclass_input, sex_input, age_input,
                    title_input, family_members, fare_input
                )

                st.markdown("---")
                prob_str = fmt_num(prob_survived, 1)

                if prob_survived >= 50:
                    st.success(f"🎉 **¡Habrías Sobrevivido!** Probabilidad estimada: **{prob_str}%**")
                else:
                    st.error(f"⚠️ **Habrías Fallecido.** Probabilidad estimada de supervivencia: **{prob_str}%**")

                st.progress(int(prob_survived))

                st.markdown("---")
                st.subheader("Explicabilidad de tu Resultado")
                st.write(f"- **Clase elegida:** {pclass_input}ª Clase")
                st.write(f"- **Protocolo marítimo:** Se priorizó a mujeres y niños. ({'Favorecido por el protocolo' if sex_input == 'female' or age_input < 12 else 'Desfavorecido por el protocolo'})")
                st.write(f"- **Acompañantes:** {'Viajas solo' if family_members == 0 else f'Viajas acompañado ({family_members + 1} miembros)'}")
