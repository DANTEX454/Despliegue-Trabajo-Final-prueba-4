import streamlit as st, pandas as pd, joblib

st.set_page_config(page_title="Predicción de Churn", page_icon="📉")
st.title("📉 Predicción de Churn en E-commerce")
st.caption("Modelo: Regresión Logística (class_weight='balanced')")

@st.cache_resource
def cargar():
    return joblib.load("modelo_churn.joblib")
modelo = cargar()

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

umbral = st.slider("Umbral de decisión", 0.1, 0.9, 0.5, 0.05,
                   help="Más bajo = detecta más clientes en riesgo (más recall) pero con más falsas alarmas.")

if st.button("Predecir"):
    x = pd.DataFrame([{"Tenure": tenure, "NumberOfOrders": orders, "AvgOrderValue": avg,
                       "TotalSpent": total, "LastPurchaseDays": last, "DiscountUsed": disc,
                       "SupportTickets": tickets, "DeviceType": device, "Newsletter": news}])
    p = modelo.predict_proba(x)[0, 1]
    st.metric("Probabilidad de churn", f"{p:.1%}")
    if p >= umbral:
        st.error("⚠️ Cliente en riesgo de churn: conviene una acción de retención.")
    else:
        st.success("✅ Riesgo bajo de churn.")
