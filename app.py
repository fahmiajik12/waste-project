import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

st.set_page_config(page_title="EDA Sampah Dashboard", layout="wide", page_icon="♻️")

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

# Prediction years - Updated to include more years and ALL option
prediction_years = st.sidebar.multiselect(
    "Tahun Prediksi", 
    options=["ALL", 2024, 2025, 2026], 
    default=["ALL"]
)

months_to_predict = st.sidebar.slider("Jumlah Bulan Prediksi", min_value=1, max_value=36, value=12)
run_prediction = st.sidebar.button("Jalankan Prediksi")

# Pilih mode tampilan prediksi
prediction_display = st.sidebar.radio(
    "Tampilan Prediksi",
    options=["Terpisah per Tahun", "Gabungan Semua Tahun"]
)

# Pilih model prediksi
model_choices = ["LSTM", "RNN", "GRU", "ALL"]
selected_models = st.sidebar.multiselect("Pilih Model untuk Prediksi", options=model_choices, default=["ALL"])

# Preprocess the data
def preprocess_data(data):
    required_cols = ["Tanggal", "Hari", "Bulan", "Tahun", "No_Polisi", "jenis_sampah", "Suplier", 
                     "Netto_kg", "Jam", "Sopir", "Admin", "Kecamatan", "Musim", "Jml_curah_hujan", "Kategori_Curah_Hujan"]
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
    data["Kategori_Curah_Hujan"] = data["Kategori_Curah_Hujan"].str.strip()
    return data.dropna()

# Filter data
def filter_data(data, years, jenis_sampah, curah_hujan):
    if years and "ALL" not in years:
        data = data[data["Tahun"].isin(years)]
    if jenis_sampah and "ALL" not in jenis_sampah:
        data = data[data["jenis_sampah"].isin(jenis_sampah)]
    if curah_hujan and "ALL" not in curah_hujan:
        data = data[data["Kategori_Curah_Hujan"].isin(curah_hujan)]
    return data

# Load Keras model
def load_keras_model(model_type):
    if model_type == "LSTM":
        model_path = "bestLSTM.h5"
    elif model_type == "RNN":
        model_path = "bestRNN.h5"
    elif model_type == "GRU":
        model_path = "bestGRU.h5"  
    else:
        st.error("Model tidak dikenali.")
        return None
    model = load_model(model_path, compile=False)
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])
    return model

# Generate predictions for specific years
def generate_predictions(model, input_sequence, scaler, months_to_predict):
    predictions = []
    current_sequence = input_sequence.copy()
    
    for _ in range(months_to_predict):
        pred = model.predict(current_sequence, verbose=0)
        predictions.append(pred[0][0])
        pred_reshaped = np.reshape(pred, (1, 1, -1))
        current_sequence = np.append(current_sequence[:, 1:, :], pred_reshaped, axis=1)
    
    # Convert predictions back to original scale
    # Create a DataFrame with the same column name as was used for fitting
    predictions_df = pd.DataFrame(
        np.array(predictions).reshape(-1, 1), 
        columns=["Netto_kg"]
    )
    
    # Transform using the scaler
    inverse_predictions = scaler.inverse_transform(predictions_df)
    
    return inverse_predictions.flatten()

# Main 
if uploaded_file:
    # Load and preprocess data
    df = pd.read_excel(uploaded_file)
    df = preprocess_data(df)
    
    if df is not None:
        # Sidebar filter
        year_filter = st.sidebar.multiselect("Filter Tahun", options=["ALL"] + sorted(df["Tahun"].unique().tolist()), default=["ALL"])
        jenis_sampah_filter = st.sidebar.multiselect("Filter Jenis Sampah", options=["ALL"] + sorted(df["jenis_sampah"].unique().tolist()), default=["ALL"])
        curah_hujan_filter = st.sidebar.multiselect("Filter Kategori Curah Hujan", options=["ALL"] + sorted(df["Kategori_Curah_Hujan"].unique().tolist()), default=["ALL"])
        
        # Filter data
        filtered_data = filter_data(df, year_filter, jenis_sampah_filter, curah_hujan_filter)

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

        # Heatmap 
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

        # Total Sampah per Kategori Curah Hujan
        st.markdown("### Total Sampah per Kategori Curah Hujan")
        curah_hujan_data = (
            filtered_data.groupby(["Kategori_Curah_Hujan", "Tahun"], as_index=False)
            .agg({"Netto_kg": "sum"})
        )
        fig_curah_hujan = px.bar(
            curah_hujan_data, x="Kategori_Curah_Hujan", y="Netto_kg", color="Tahun", 
            title="Total Sampah per Kategori Curah Hujan",
            labels={"Netto_kg": "Total Sampah (kg)", "Kategori_Curah_Hujan": "Kategori Curah Hujan"}
        )
        st.plotly_chart(fig_curah_hujan, use_container_width=True)

        # Distribusi Sampah per Kategori Curah Hujan
        st.markdown("### Distribusi Sampah per Kategori Curah Hujan")
        kategori_curah_hujan_data = filtered_data.groupby(["Kategori_Curah_Hujan"], as_index=False).agg({"Netto_kg": "sum"})
        fig_pie_curah_hujan = px.pie(
            kategori_curah_hujan_data, names="Kategori_Curah_Hujan", values="Netto_kg", 
            title="Distribusi Sampah per Kategori Curah Hujan", 
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig_pie_curah_hujan, use_container_width=True)

        # Boxplot per jenis_sampah
        st.markdown("### Boxplot per Jenis Sampah dan Suplier")
        fig_boxplot = px.box(
            filtered_data, x="jenis_sampah", y="Netto_kg", color="Suplier", 
            facet_col="Tahun", title="Boxplot per Jenis Sampah dan Suplier",
            labels={"jenis_sampah": "Jenis Sampah", "Netto_kg": "Netto (kg)"}
        )
        st.plotly_chart(fig_boxplot, use_container_width=True)

        # Prediksi 
        if run_prediction:
            st.markdown("### Prediksi Volume Sampah")
            
            # Prepare time series data for prediction
            prediction_data = (
                filtered_data.groupby(["Tahun", "Bulan"], as_index=False)
                .agg({"Netto_kg": "sum"})
            )
            
            # Create a DataFrame with appropriate column names for scaling
            netto_df = prediction_data[["Netto_kg"]]
            
            # Fit the scaler on the DataFrame with column names
            scaler = MinMaxScaler()
            scaler.fit(netto_df)
            
            # Transform the data using the fitted scaler
            scaled_data = scaler.transform(netto_df)
        
            window_size = 12
            if len(scaled_data) >= window_size:
                input_sequence = np.array([scaled_data[-window_size:]])  # Last window_size elements
                
                # Determine which years to predict
                years_to_predict = [2024, 2025, 2026] if "ALL" in prediction_years else prediction_years
                
                # Create storage for all predictions
                all_predictions = {}
                
                # Generate predictions for each model
                for model_type in (["LSTM", "RNN", "GRU"] if "ALL" in selected_models else selected_models):
                    model = load_keras_model(model_type)
                    if model:
                        model_predictions = {}
                        current_sequence = input_sequence.copy()
                        
                        for year in years_to_predict:
                            # Define prediction months for this year
                            months_in_year = 12
                            monthly_predictions = generate_predictions(
                                model, 
                                current_sequence.copy(), 
                                scaler, 
                                months_in_year
                            )
                            
                            # Create date range for predictions
                            date_range = pd.date_range(start=f"{year}-01-01", periods=months_in_year, freq='MS')
                            
                            # Store predictions for this year
                            model_predictions[year] = pd.DataFrame({
                                "Tanggal": date_range,
                                "Prediksi_Netto_kg": monthly_predictions,
                                "Tahun": str(year)
                            })
                            
                            # Update sequence for next year prediction
                            # Create a proper DataFrame for the latest data
                            latest_data_df = pd.DataFrame(
                                monthly_predictions[-window_size:].reshape(-1, 1), 
                                columns=["Netto_kg"]
                            )
                            # Scale it properly
                            updated_sequence = scaler.transform(latest_data_df)
                            # Reshape for model input
                            current_sequence = np.array([updated_sequence])
                        
                        all_predictions[model_type] = model_predictions
                
                # Display predictions based on selected view mode
                if prediction_display == "Terpisah per Tahun":
                    # Display separate charts for each year
                    for year in years_to_predict:
                        st.subheader(f"Prediksi Tahun {year}")
                        
                        for model_name, model_data in all_predictions.items():
                            if year in model_data:
                                model_idx = ["LSTM", "RNN", "GRU"].index(model_name) % len(px.colors.qualitative.Plotly)
                                fig = px.line(
                                    model_data[year], 
                                    x="Tanggal", 
                                    y="Prediksi_Netto_kg",
                                    color_discrete_sequence=[px.colors.qualitative.Plotly[model_idx]],
                                    labels={"Prediksi_Netto_kg": "Prediksi Netto (kg)", "Tanggal": "Tanggal"},
                                    title=f"Prediksi {model_name} untuk Tahun {year}"
                                )
                                fig.update_layout(
                                    xaxis=dict(
                                        title="Bulan",
                                        tickformat="%B",
                                        dtick="M1",
                                    ),
                                    yaxis=dict(
                                        title="Prediksi Netto (kg)"
                                    )
                                )
                                st.plotly_chart(fig, use_container_width=True)
                
                else:  # Combined view
                    # Create combined dataframe for all predictions
                    for model_name, model_data in all_predictions.items():
                        combined_df = pd.concat([df for df in model_data.values()])
                        
                        fig = px.line(
                            combined_df,
                            x="Tanggal",
                            y="Prediksi_Netto_kg",
                            color="Tahun",
                            title=f"Prediksi {model_name} untuk Tahun 2024-2026",
                            labels={"Prediksi_Netto_kg": "Prediksi Netto (kg)", "Tanggal": "Tanggal"}
                        )
                        fig.update_layout(
                            xaxis=dict(
                                title="Tanggal",
                                tickformat="%b %Y",
                                dtick="M3",
                            ),
                            yaxis=dict(
                                title="Prediksi Netto (kg)"
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display prediction data table
                        st.subheader(f"Tabel Prediksi {model_name}")
                        formatted_df = combined_df.copy()
                        formatted_df["Tanggal"] = formatted_df["Tanggal"].dt.strftime("%b %Y")
                        formatted_df["Prediksi_Netto_kg"] = formatted_df["Prediksi_Netto_kg"].round(2)
                        st.dataframe(formatted_df[["Tanggal", "Tahun", "Prediksi_Netto_kg"]])
                
                # Compare all models in one chart for each year
                if len(selected_models) > 1 or "ALL" in selected_models:
                    st.subheader("Perbandingan Model per Tahun")
                    
                    for year in years_to_predict:
                        comparison_data = []
                        
                        for model_name, model_data in all_predictions.items():
                            if year in model_data:
                                temp_df = model_data[year].copy()
                                temp_df["Model"] = model_name
                                comparison_data.append(temp_df)
                        
                        if comparison_data:
                            comparison_df = pd.concat(comparison_data)
                            
                            fig = px.line(
                                comparison_df,
                                x="Tanggal",
                                y="Prediksi_Netto_kg",
                                color="Model",
                                title=f"Perbandingan Model untuk Tahun {year}",
                                labels={"Prediksi_Netto_kg": "Prediksi Netto (kg)", "Tanggal": "Tanggal"}
                            )
                            fig.update_layout(
                                xaxis=dict(
                                    title="Bulan",
                                    tickformat="%B",
                                    dtick="M1",
                                ),
                                yaxis=dict(
                                    title="Prediksi Netto (kg)"
                                )
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                # Add a combined view of all models across all years if multiple models and years are selected
                if (len(selected_models) > 1 or "ALL" in selected_models) and len(years_to_predict) > 1:
                    st.subheader("Perbandingan Model untuk Semua Tahun")
                    
                    all_comparison_data = []
                    for model_name, model_data in all_predictions.items():
                        for year, year_data in model_data.items():
                            temp_df = year_data.copy()
                            temp_df["Model"] = f"{model_name} ({year})"
                            all_comparison_data.append(temp_df)
                    
                    if all_comparison_data:
                        all_comparison_df = pd.concat(all_comparison_data)
                        
                        fig = px.line(
                            all_comparison_df,
                            x="Tanggal",
                            y="Prediksi_Netto_kg",
                            color="Model",
                            title="Perbandingan Semua Model untuk Semua Tahun",
                            labels={"Prediksi_Netto_kg": "Prediksi Netto (kg)", "Tanggal": "Tanggal"}
                        )
                        fig.update_layout(
                            xaxis=dict(
                                title="Tanggal",
                                tickformat="%b %Y",
                                dtick="M3",
                            ),
                            yaxis=dict(
                                title="Prediksi Netto (kg)"
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
