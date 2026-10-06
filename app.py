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
