import io
import numpy as np
import pandas as pd
import joblib
import altair as alt
import streamlit as st
 
st.set_page_config(page_title="Predicción de Churn", page_icon="📉", layout="wide")
 
NUM = ["Tenure", "NumberOfOrders", "AvgOrderValue", "TotalSpent",
       "LastPurchaseDays", "DiscountUsed", "SupportTickets"]
CAT = ["DeviceType", "Newsletter"]
REQUERIDAS = NUM + CAT
COLOR_RIESGO = alt.Scale(domain=["En riesgo", "Bajo riesgo"], range=["#d9534f", "#5b8def"])
 
 
@st.cache_resource
def cargar():
    return joblib.load("modelo_churn.joblib")
 
 
def limpiar(df):
    """Mismas reglas de limpieza usadas al entrenar: valores imposibles -> nulo
    (el pipeline del modelo los imputa con la mediana)."""
    d = df.copy()
    for c in NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in CAT:
        d[c] = d[c].where(d[c].isna(), d[c].astype(str).str.strip())
    d.loc[(d.Tenure < 0) | (d.Tenure > 120), "Tenure"] = np.nan
    d.loc[(d.TotalSpent <= 0) | (d.TotalSpent > 1e5), "TotalSpent"] = np.nan
    d.loc[(d.LastPurchaseDays < 0) | (d.LastPurchaseDays > 365), "LastPurchaseDays"] = np.nan
    return d
 
 
def predecir(df, modelo, umbral):
    d = limpiar(df)
    prob = modelo.predict_proba(d[REQUERIDAS])[:, 1]
    out = df.copy()
    out["Prob_Churn"] = np.round(prob, 3)
    out["Riesgo"] = np.where(prob >= umbral, "En riesgo", "Bajo riesgo")
    ids = [c for c in ["RecordID", "FullName"] if c in out.columns]
    resto = [c for c in out.columns if c not in ids + ["Prob_Churn", "Riesgo"]]
    return out[ids + ["Prob_Churn", "Riesgo"] + resto].sort_values("Prob_Churn", ascending=False)
 
 
def leer_archivo(archivo):
    if archivo.name.lower().endswith(".xlsx"):
        return pd.read_excel(archivo)
    try:
        return pd.read_csv(archivo, sep=None, engine="python")
    except UnicodeDecodeError:
        archivo.seek(0)
        return pd.read_csv(archivo, sep=None, engine="python", encoding="latin-1")
 
 
modelo = cargar()
 
st.title("📉 Predicción de Churn en E-commerce")
st.caption("Modelo: Regresión Logística (class_weight='balanced')")
 
with st.sidebar:
    st.header("Configuración")
    umbral = st.slider(
        "Umbral de decisión", 0.1, 0.9, 0.5, 0.05,
        help="Un cliente se marca 'en riesgo' si su probabilidad de churn es igual o mayor al umbral. "
             "Más bajo = detecta más clientes en riesgo pero con más falsas alarmas.")
    st.caption("Referencia en pruebas: 0.5 detecta ~57% del churn; 0.6 detecta ~50% con mejor precisión.")
 
tab1, tab2 = st.tabs(["👤 Cliente individual", "📂 Cargar archivo de clientes"])
 
# ------------------------------------------------------------------ individual
with tab1:
    st.subheader("Datos del cliente")
    c1, c2 = st.columns(2)
    with c1:
        tenure = st.number_input("Antigüedad (Tenure)", 0, 120, 38)
        orders = st.number_input("Número de órdenes", 1, 20, 9)
        avg = st.number_input("Valor promedio de orden", 0.0, 250.0, 45.0)
        total = st.number_input("Total gastado", 1.0, 100000.0, 400.0)
        last = st.number_input("Días desde la última compra", 0, 365, 16)
    with c2:
        disc = st.slider("Uso de descuentos", 0.0, 1.0, 0.26)
        tickets = st.number_input("Tickets de soporte", 0, 10, 1)
        device = st.selectbox("Dispositivo", ["Desktop", "Mobile", "Tablet"])
        news = st.selectbox("Newsletter", ["Yes", "No"])
 
    if st.button("Predecir"):
        x = pd.DataFrame([{"Tenure": tenure, "NumberOfOrders": orders, "AvgOrderValue": avg,
                           "TotalSpent": total, "LastPurchaseDays": last, "DiscountUsed": disc,
                           "SupportTickets": tickets, "DeviceType": device, "Newsletter": news}])
        p = modelo.predict_proba(limpiar(x)[REQUERIDAS])[0, 1]
        st.metric("Probabilidad de churn", f"{p:.1%}")
        if p >= umbral:
            st.error("⚠️ Cliente en riesgo de churn: conviene una acción de retención.")
        else:
            st.success("✅ Riesgo bajo de churn.")
 
# ------------------------------------------------------------------ archivo
with tab2:
    st.subheader("Predicción para una base de clientes")
    st.write("Sube un archivo **CSV o Excel (.xlsx)** con una fila por cliente. "
             "Debe incluir estas columnas: " + ", ".join(f"`{c}`" for c in REQUERIDAS) +
             ". Si trae `RecordID` o `FullName`, se usarán para identificar a cada cliente.")
 
    plantilla = pd.DataFrame([
        {"RecordID": 1, "Tenure": 24, "NumberOfOrders": 5, "AvgOrderValue": 40.0, "TotalSpent": 200.0,
         "LastPurchaseDays": 120, "DiscountUsed": 0.3, "SupportTickets": 3, "DeviceType": "Mobile", "Newsletter": "Yes"},
        {"RecordID": 2, "Tenure": 60, "NumberOfOrders": 15, "AvgOrderValue": 55.0, "TotalSpent": 825.0,
         "LastPurchaseDays": 10, "DiscountUsed": 0.1, "SupportTickets": 0, "DeviceType": "Desktop", "Newsletter": "No"}])
    st.download_button("⬇️ Descargar plantilla de ejemplo", plantilla.to_csv(index=False).encode("utf-8"),
                       file_name="plantilla_clientes.csv", mime="text/csv")
 
    archivo = st.file_uploader("Archivo de clientes", type=["csv", "xlsx"])
 
    if archivo is not None:
        try:
            df = leer_archivo(archivo)
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
            st.stop()
 
        faltan = [c for c in REQUERIDAS if c not in df.columns]
        if faltan:
            st.error("Al archivo le faltan estas columnas: " + ", ".join(faltan))
            st.stop()
 
        resultado = predecir(df, modelo, umbral)
        en_riesgo = int((resultado["Riesgo"] == "En riesgo").sum())
 
        m1, m2, m3 = st.columns(3)
        m1.metric("Clientes analizados", f"{len(resultado):,}")
        m2.metric("Clientes en riesgo", f"{en_riesgo:,}")
        m3.metric("% en riesgo", f"{en_riesgo / len(resultado):.1%}")
 
        incompletos = int(limpiar(df)[REQUERIDAS].isna().any(axis=1).sum())
        if incompletos:
            st.info(f"{incompletos} cliente(s) tenían datos faltantes o inválidos; "
                    "se completaron automáticamente con la mediana/moda del entrenamiento.")
 
        st.subheader("Clientes con mayor probabilidad de churn")
        st.dataframe(resultado[resultado["Riesgo"] == "En riesgo"].head(100))
 
        with st.expander("Ver todos los clientes"):
            st.dataframe(resultado)
 
        st.download_button("⬇️ Descargar resultados (CSV)",
                           resultado.to_csv(index=False).encode("utf-8-sig"),
                           file_name="resultados_churn.csv", mime="text/csv")
 
        st.subheader("Diagrama de dispersión")
        g1, g2 = st.columns(2)
        ejex = g1.selectbox("Eje X", NUM, index=NUM.index("LastPurchaseDays"))
        ejey = g2.selectbox("Eje Y", NUM, index=NUM.index("NumberOfOrders"))
 
        plot = limpiar(resultado)
        plot["Prob_Churn"] = resultado["Prob_Churn"].values
        plot["Riesgo"] = resultado["Riesgo"].values
        id_col = "RecordID" if "RecordID" in plot.columns else None
        tooltip = ([id_col] if id_col else []) + [ejex, ejey, "Prob_Churn", "Riesgo"]
        grafico = (alt.Chart(plot.dropna(subset=[ejex, ejey]))
                   .mark_circle(size=70, opacity=0.7)
                   .encode(x=alt.X(ejex, title=ejex), y=alt.Y(ejey, title=ejey),
                           color=alt.Color("Riesgo:N", scale=COLOR_RIESGO),
                           tooltip=tooltip)
                   .properties(height=420)
                   .interactive())
        st.altair_chart(grafico)
        st.caption("Rojo = cliente en riesgo según el umbral elegido. Puedes hacer zoom y mover el gráfico.")
