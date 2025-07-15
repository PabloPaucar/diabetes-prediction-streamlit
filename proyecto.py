import streamlit as st
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ──────────────────────────── UTILIDAD ───────────────────────────────
def fig_to_png(fig):
    """Convierte una figura Matplotlib a bytes PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

# ───────────────────── CONFIGURACIÓN DE LA PÁGINA ────────────────────
st.set_page_config(page_title="Predicción Diabetes", layout="wide")
st.title("🧬 Predicción de Diabetes Tipo 2")

# ───────────────────────────── ESTADO ────────────────────────────────
if "datos_limpios" not in st.session_state:
    st.session_state["datos_limpios"] = False

# ─────────────────────── SUBIDA DE ARCHIVO ───────────────────────────
uploaded_file = st.file_uploader("📂 Cargar archivo (.csv o .arff)", type=["csv", "arff"])
df = None

if uploaded_file:
    try:
        ext = uploaded_file.name.split('.')[-1].lower()
        if ext == 'csv':
            df = pd.read_csv(uploaded_file)
        elif ext == 'arff':
            text, _ = uploaded_file.read().decode('utf-8'), uploaded_file.seek(0)
            data, _ = arff.loadarff(io.StringIO(text))
            df = pd.DataFrame(data)
            df['class'] = df['class'].str.decode('utf-8')
        st.success("✅ Archivo cargado correctamente.")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Error al cargar archivo: {e}")

# ────────────────────────── LIMPIEZA DE DATOS ───────────────────────
if df is not None:
    st.subheader("🧹 Limpieza de Datos")
    if st.button("Limpiar columnas con ceros"):
        for col in ['plas', 'pres', 'skin', 'insu', 'mass']:
            if col in df.columns:
                mediana = df[df[col] != 0][col].median()
                df[col] = df[col].replace(0, mediana)
        st.session_state["datos_limpios"] = True
        st.success("✔️ Datos limpiados correctamente")

        # Descargar CSV limpio
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Descargar CSV limpio", data=csv,
                           file_name="datos_limpios.csv", mime="text/csv")

    # ───────────────────────── VISUALIZACIÓN ─────────────────────────
    st.subheader("📊 Distribución de Clases")
    if "class" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        df['class'].value_counts().plot(kind='bar', ax=ax,
                                        color=['#3498db', '#e74c3c'])
        ax.set_title("Distribución de Clases")
        ax.set_xlabel("")
        ax.set_ylabel("")

        col_left, col_mid, col_right = st.columns([1, 2, 1])
        with col_mid:
            st.image(fig_to_png(fig), width=600)

    st.subheader("📈 Estadísticas")
    st.dataframe(df.describe())

    st.subheader("📌 Matriz de Correlación")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(df.select_dtypes(include=np.number).corr(),
                annot=True, cmap='coolwarm', ax=ax2)
    ax2.set_title("Correlación")

    col_left, col_mid, col_right = st.columns([1, 2, 1])
    with col_mid:
        st.image(fig_to_png(fig2), width=600)

    # ─────────────────────── ENTRENAMIENTO ───────────────────────────
    st.subheader("🧠 Entrenamiento de Modelos")
    if not st.session_state["datos_limpios"]:
        st.warning("⚠️ Primero debes limpiar los datos antes de entrenar.")
    else:
        test_size    = st.slider("Proporción prueba", 0.1, 0.5, 0.2)
        random_state = st.number_input("Semilla", 1, 9999, 42)
        modelos_sel  = st.multiselect("Selecciona modelos",
                          ["Regresión Logística", "Árbol de Decisión",
                           "Random Forest", "SVM"],
                          default=["Regresión Logística", "Árbol de Decisión"])

        if st.button("Entrenar"):
            X = df.drop('class', axis=1)
            y = df['class'].map({'tested_negative': 0,
                                 'tested_positive': 1}) \
                if df['class'].dtype == object else df['class']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size,
                random_state=random_state, stratify=y)

            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

            st.session_state["scaler"]  = scaler
            st.session_state["modelos"] = {}

            for nombre in modelos_sel:
                if nombre == "Regresión Logística":
                    modelo = LogisticRegression(max_iter=1000)
                elif nombre == "Árbol de Decisión":
                    modelo = DecisionTreeClassifier()
                elif nombre == "Random Forest":
                    modelo = RandomForestClassifier()
                elif nombre == "SVM":
                    modelo = SVC(probability=True)
                else:
                    continue

                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)

                st.markdown(f"### 🔍 {nombre}")
                st.text(classification_report(y_test, y_pred))

                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_xlabel("Predicción")
                ax_cm.set_ylabel("Real")

                col_left, col_mid, col_right = st.columns([1, 2, 1])
                with col_mid:
                    st.image(fig_to_png(fig_cm), width=600)

                st.session_state["modelos"][nombre] = modelo

    # ──────────────────── PREDICCIÓN INDIVIDUAL ──────────────────────
    st.subheader("🔮 Predicción Individual")
    with st.form("form_prediccion"):
        col1, col2 = st.columns(2)
        with col1:
            preg = st.number_input("Embarazos", step=1)
            plas = st.number_input("Glucosa")
            pres = st.number_input("Presión arterial")
            skin = st.number_input("Pliegue cutáneo")
        with col2:
            insu = st.number_input("Insulina")
            mass = st.number_input("IMC")
            pedi = st.number_input("Pedigrí")
            age  = st.number_input("Edad", step=1)

        modelo_sel = st.selectbox(
            "Modelo", list(st.session_state.get("modelos", {}).keys()))
        submit = st.form_submit_button("Predecir")

    if submit:
        try:
            modelo  = st.session_state["modelos"].get(modelo_sel)
            scaler  = st.session_state["scaler"]
            entrada = np.array([[preg, plas, pres, skin,
                                 insu, mass, pedi, age]])
            entrada = scaler.transform(entrada)
            pred    = modelo.predict(entrada)[0]
            proba   = (modelo.predict_proba(entrada)[0][1]
                       if hasattr(modelo, "predict_proba") else None)
            resultado = ("🟥 POSITIVO (riesgo de diabetes)"
                         if pred == 1 else
                         "🟩 NEGATIVO (sin riesgo)")
            st.success(f"Resultado: {resultado}")
            if proba is not None:
                st.info(f"Probabilidad de diabetes: {proba:.2%}")
        except Exception as e:
            st.error(f"❌ Error en predicción: {e}")
else:
    st.info("⬆️ Carga un archivo para comenzar.")
