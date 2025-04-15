import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import seaborn as sns

# Set page configuration
st.set_page_config(layout="wide")
st.title("Prediksi Volume Sampah Harian - Kota Magelang")

# ================================
# Fungsi: Load model LSTM
# ================================
@st.cache_resource
def load_lstm_model():
    return load_model("model_lstm_volume_sampah_terbaik.h5", compile=False)

# ================================
# Upload File dan Pilih Tahun
# ================================
st.header("Upload Dataset dan Pilih Tahun untuk Prediksi")

uploaded_file = st.file_uploader("Unggah File Dataset (.xlsx)", type=["xlsx"])

# Memastikan file diupload
if uploaded_file is not None:
    # Membaca dataset
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()
    df['tanggal'] = pd.to_datetime(df['tanggal'], dayfirst=True, errors='coerce')
    df = df.sort_values("tanggal")
    
    # Menampilkan ringkasan data
    st.header("Ringkasan Data")
    st.write(df.describe())

    # Menampilkan supplier sampah (asumsi kolom 'supplier' ada)
    if 'supplier' in df.columns:
        st.subheader("Supplier Sampah")
        supplier_counts = df['supplier'].value_counts()
        st.write(supplier_counts)
    else:
        st.warning("Kolom 'supplier' tidak ditemukan dalam dataset.")

    # Pilih Tahun untuk Prediksi
    years = df['tanggal'].dt.year.unique()
    selected_year = st.selectbox("Pilih Tahun untuk Prediksi", ['ALL'] + list(map(str, years)))

    # ================================
    # Filter Data Berdasarkan Tahun
    # ================================
    if selected_year != 'ALL':
        df = df[df['tanggal'].dt.year == int(selected_year)]

    # ================================
    # Preprocessing dan Prediksi
    # ================================
    df_selected_year = df.copy()

    # Preprocessing
    data_selected_year = df_selected_year['netto_kg'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_selected_year)

    # Buat input sequence untuk LSTM (window 30 hari)
    window_size = 30
    X = []
    y = []
    for i in range(window_size, len(data_scaled)):
        X.append(data_scaled[i-window_size:i])
        y.append(data_scaled[i])
    X, y = np.array(X), np.array(y)

    # Load Model LSTM
    model = load_lstm_model()

    # Prediksi
    y_pred_scaled = model.predict(X)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_actual = scaler.inverse_transform(y)

    # ================================
    # Menambahkan kolom aktual dan prediksi ke DataFrame
    # ================================
    df_pred = df_selected_year.iloc[window_size:].copy()
    df_pred["aktual"] = y_actual.flatten()
    df_pred["prediksi"] = y_pred.flatten()

    # ================================
    # Filter untuk melihat data per Bulan, Tahun, atau Hari
    # ================================
    st.sidebar.header("Pilih Periode Tampilan Data")
    period_choice = st.sidebar.selectbox("Pilih Periode", ["Bulan", "Tahun", "Hari"])

    # ================================
    # Visualisasi berdasarkan pilihan periode
    # ================================
    if period_choice == "Bulan":
        df_pred_monthly = df_pred.copy()
        df_pred_monthly['bulan'] = df_pred_monthly['tanggal'].dt.month

        monthly_actual = df_pred_monthly.groupby('bulan')['aktual'].sum()
        monthly_pred = df_pred_monthly.groupby('bulan')['prediksi'].sum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_actual.index, y=monthly_actual.values,
            mode='lines+markers', name='Aktual', line=dict(color='blue', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=monthly_pred.index, y=monthly_pred.values,
            mode='lines+markers', name='Prediksi', line=dict(color='orange', width=2)
        ))

        fig.update_layout(
            title=f"Prediksi vs Aktual Volume Sampah per Bulan (Tahun {selected_year})",
            xaxis=dict(tickmode='array', tickvals=np.arange(1, 13), ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]),
            xaxis_title="Bulan",
            yaxis_title="Volume Sampah (kg)",
            showlegend=True,
            template="plotly_dark"
        )
        st.plotly_chart(fig)

    elif period_choice == "Tahun":
        df_pred_yearly = df_pred.copy()
        df_pred_yearly['tahun'] = df_pred_yearly['tanggal'].dt.year

        yearly_actual = df_pred_yearly.groupby('tahun')['aktual'].sum()
        yearly_pred = df_pred_yearly.groupby('tahun')['prediksi'].sum()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(yearly_actual.index, yearly_actual.values, label="Aktual", marker='o', linestyle='-', color='blue')
        ax.plot(yearly_pred.index, yearly_pred.values, label="Prediksi", marker='o', linestyle='-', color='orange')

        ax.set_title(f"Prediksi vs Aktual Volume Sampah per Tahun ({selected_year})")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Volume Sampah (kg)")
        ax.legend()
        st.pyplot(fig)

    elif period_choice == "Hari":
        df_pred_daily = df_pred.copy()
        df_pred_daily['hari'] = df_pred_daily['tanggal'].dt.date

        daily_actual = df_pred_daily.groupby('hari')['aktual'].sum()
        daily_pred = df_pred_daily.groupby('hari')['prediksi'].sum()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(daily_actual.index, daily_actual.values, label="Aktual", color='blue')
        ax.plot(daily_pred.index, daily_pred.values, label="Prediksi", color='orange')

        ax.set_title(f"Prediksi vs Aktual Volume Sampah per Hari ({selected_year})")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Volume Sampah (kg)")
        ax.legend()
        st.pyplot(fig)

    # ================================
    # Evaluasi Model
    # ================================
    mse = mean_squared_error(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)

    st.header("Evaluasi Model")
    st.metric("MSE (Mean Squared Error)", f"{mse:.2f}")
    st.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")

    # ================================
    # Klasifikasi Musim (Kemarau & Hujan)
    # ================================
    df_pred['musim'] = df_pred['bulan'].apply(lambda x: 'Hujan' if x in [11, 12, 1, 2, 3, 4] else 'Kemarau')

    seasonal_actual = df_pred.groupby('musim')['aktual'].sum()
    seasonal_pred = df_pred.groupby('musim')['prediksi'].sum()

    heatmap_data = pd.DataFrame({
        'Aktual': seasonal_actual,
        'Prediksi': seasonal_pred
    })

    st.header("Heatmap Perbandingan Volume Sampah (Musim Hujan vs Kemarau)")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(heatmap_data.T, annot=True, cmap='YlGnBu', fmt='.2f', cbar=True, ax=ax)
    ax.set_title(f"Perbandingan Volume Sampah (Aktual vs Prediksi) per Musim - Tahun {selected_year}")
    st.pyplot(fig)
