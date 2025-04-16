import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Set page configuration
st.set_page_config(page_title="EDA Sampah Dashboard", layout="wide", page_icon="♻️")

# Custom styling for Streamlit
st.markdown(
    """
    <style>
    .metric-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .sidebar-title {
        font-size: 20px;
        font-weight: bold;
        color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.markdown('<div class="sidebar-title">EDA Sampah Dashboard</div>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload File Excel (.xlsx)", type="xlsx")
model_type = st.sidebar.selectbox("Pilih Model", ["LSTM", "RNN", "XGBoost", "Random Forest"])
prediction_years = st.sidebar.multiselect("Tahun Prediksi", options=[2022, 2023, 2024, 2025, 2026], default=[2025])
months_to_predict = st.sidebar.slider("Jumlah Bulan Prediksi", min_value=1, max_value=24, value=12)
run_prediction = st.sidebar.button("Jalankan Prediksi")

# Function to preprocess the data
def preprocess_data(data):
    required_cols = ["Tanggal", "Hari", "Bulan", "Tahun", "No_Polisi", "jenis_sampah", "Suplier", 
                     "Netto_kg", "Jam", "Sopir", "Admin", "Kecamatan", "Musim"]
    if not all(col in data.columns for col in required_cols):
        st.error("Kolom tidak lengkap.")
        return None
    
    try:
        data["Tanggal"] = pd.to_datetime(data["Tanggal"], format="%d/%m/%Y")
    except ValueError:
        data["Tanggal"] = pd.to_datetime(data["Tanggal"], dayfirst=True)
    
    data["Tahun"] = data["Tahun"].astype(str)
    data["Bulan"] = data["Bulan"].astype(int)
    data["Kecamatan"] = data["Kecamatan"].str.strip()
    data["Musim"] = data["Musim"].str.strip()
    data["jenis_sampah"] = data["jenis_sampah"].str.strip()
    return data.dropna()

# Function to filter data
def filter_data(data, years, jenis_sampah):
    if years and "ALL" not in years:
        data = data[data["Tahun"].isin(years)]
    if jenis_sampah and "ALL" not in jenis_sampah:
        data = data[data["jenis_sampah"].isin(jenis_sampah)]
    return data

# Function to load Keras model
def load_keras_model(model_type):
    model_path = "best_lstm_model.h5" if model_type == "LSTM" else "best_rnn_model.h5"
    model = load_model(model_path, compile=False)
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])
    return model

# Main content
if uploaded_file:
    # Load and preprocess data
    df = pd.read_excel(uploaded_file)
    df = preprocess_data(df)
    
    if df is not None:
        # Update sidebar filters
        year_filter = st.sidebar.multiselect("Filter Tahun", options=["ALL"] + sorted(df["Tahun"].unique().tolist()), default=["ALL"])
        jenis_sampah_filter = st.sidebar.multiselect("Filter Jenis Sampah", options=["ALL"] + sorted(df["jenis_sampah"].unique().tolist()), default=["ALL"])
        
        # Filter data
        filtered_data = filter_data(df, year_filter, jenis_sampah_filter)

        # Summary metrics
        st.markdown("### Ringkasan Data")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Sampah (kg)", f"{filtered_data['Netto_kg'].sum():,.0f}")
        with col2:
            st.metric("Rata-rata Netto (kg)", f"{filtered_data['Netto_kg'].mean():,.2f}")
        with col3:
            st.metric("Netto Tertinggi (kg)", f"{filtered_data['Netto_kg'].max():,.0f}")
        with col4:
            st.metric("Jumlah Kecamatan", filtered_data["Kecamatan"].nunique())
        with col5:
            st.metric("Jumlah Suplier", filtered_data["Suplier"].nunique())

        # Heatmap of Contributions by Kecamatan and Musim for All Years (Displayed Horizontally)
        st.markdown("### Heatmap Musim, Kecamatan, dan Total Sumbangsih Sampah per Tahun (Horizontal)")
        heatmap_data = (
            filtered_data.groupby(["Tahun", "Kecamatan", "Musim"], as_index=False)
            .agg({"Netto_kg": "sum"})
        )
        fig_heatmap = px.density_heatmap(
            heatmap_data,
            x="Musim",
            y="Kecamatan",
            z="Netto_kg",
            facet_col="Tahun",
            color_continuous_scale="Viridis",
            title="Total Sumbangsih Sampah Berdasarkan Musim dan Kecamatan per Tahun",
            labels={"Netto_kg": "Total Sampah (kg)", "Musim": "Musim", "Kecamatan": "Kecamatan"}
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Barplot per Kecamatan
        st.markdown("### Total Sampah per Kecamatan")
        kecamatan_data = (
            filtered_data.groupby(["Kecamatan", "Musim", "Tahun"], as_index=False)
            .agg({"Netto_kg": "sum"})
        )
        fig_bar_kecamatan = px.bar(
            kecamatan_data, x="Kecamatan", y="Netto_kg", color="Musim", 
            barmode="group", facet_col="Tahun", 
            title="Total Sampah per Kecamatan",
            labels={"Netto_kg": "Total Sampah (kg)", "Kecamatan": "Kecamatan"}
        )
        st.plotly_chart(fig_bar_kecamatan, use_container_width=True)

        # Distribusi musim
        st.markdown("### Distribusi Sampah per Musim")
        musim_data = filtered_data.groupby(["Musim"], as_index=False).agg({"Netto_kg": "sum"})
        fig_musim = px.pie(
            musim_data, names="Musim", values="Netto_kg", 
            title="Distribusi Sampah per Musim", 
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig_musim, use_container_width=True)

        # Boxplot per jenis_sampah
        st.markdown("### Boxplot per Jenis Sampah dan Suplier")
        fig_boxplot = px.box(
            filtered_data, x="jenis_sampah", y="Netto_kg", color="Suplier", 
            facet_col="Tahun", title="Boxplot per Jenis Sampah dan Suplier",
            labels={"jenis_sampah": "Jenis Sampah", "Netto_kg": "Netto (kg)"}
        )
        st.plotly_chart(fig_boxplot, use_container_width=True)

        # Prediction
        if run_prediction:
            st.markdown("### Prediksi Volume Sampah")
            prediction_data = (
                filtered_data.groupby(["Tahun", "Bulan"], as_index=False)
                .agg({"Netto_kg": "sum"})
            )
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(prediction_data[["Netto_kg"]])

            # Prepare input sequence
            window_size = 12
            if len(scaled_data) >= window_size:
                input_sequence = np.array([scaled_data[-window_size:]])
                model = load_keras_model(model_type)
                predictions = []
                for _ in range(months_to_predict):
                    pred = model.predict(input_sequence)
                    predictions.append(pred[0][0])
                    pred_reshaped = np.reshape(pred, (1, 1, -1))
                    input_sequence = np.append(input_sequence[:, 1:, :], pred_reshaped, axis=1)

                # Rescale predictions
                predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
                
                # Generate valid future dates
                start_date = pd.Timestamp(year=prediction_years[0], month=1, day=1)
                pred_dates = pd.date_range(start=start_date, periods=months_to_predict, freq="M")

                # Create prediction DataFrame
                prediction_df = pd.DataFrame({"Tanggal": pred_dates, "Prediksi_Netto_kg": predictions_rescaled.flatten()})
                
                # Show predictions
                fig_prediction = px.line(
                    prediction_df, x="Tanggal", y="Prediksi_Netto_kg", 
                    title="Prediksi Volume Sampah",
                    labels={"Prediksi_Netto_kg": "Prediksi Netto (kg)", "Tanggal": "Tanggal"},
                    color_discrete_sequence=["#17BECF"]
                )
                st.plotly_chart(fig_prediction, use_container_width=True)
                st.table(prediction_df)
            else:
                st.warning("Data tidak cukup untuk melakukan prediksi.")
