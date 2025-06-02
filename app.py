import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import numpy as np
import re

st.set_page_config(page_title="Rainfall Data Analysis", layout="wide")
st.title("📊 TCE 415 - Rainfall Data Analysis App")

st.markdown("""
Upload your rainfall dataset in the same format (LGA in first column, monthly columns from Jan-20 to Nov-23).
""")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File Uploaded Successfully!")

        group = st.sidebar.selectbox("Choose Study Group Objective", [
            "Group 1: Wet/Dry Seasons & Rainfall Variability",
            "Group 2: Trend Analysis & Clustering",
            "Group 3: Planting Season & Water Availability",
            "Group 4: Drought/Flood Risk & Rainfall Map Profiles"
        ])

        valid_month_cols = [col for col in df.columns if re.match(r'^[A-Za-z]{3}-\d{2}$', col.strip())]
        df_melted = df.melt(id_vars=['LG'], value_vars=valid_month_cols, var_name='Month-Year', value_name='Rainfall')
        df_melted['Month-Year'] = pd.to_datetime(df_melted['Month-Year'].str.strip(), format='%b-%y', errors='coerce')
        df_melted = df_melted.dropna(subset=['Month-Year'])
        df_melted['Month'] = df_melted['Month-Year'].dt.month
        df_melted['Year'] = df_melted['Month-Year'].dt.year

        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                     7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Data Overview")
        if st.sidebar.checkbox("Show Raw Data"):
            st.subheader("📂 Uploaded Dataset")
            st.dataframe(df)

        if group.startswith("Group 1"):
            st.header("🌧️ Wet and Dry Seasons Identification")
            lga_selected = st.selectbox("Select LGA", df['LG'].unique())
            lga_df = df_melted[df_melted['LG'] == lga_selected]
            monthly_avg = lga_df.groupby('Month')['Rainfall'].mean()

            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(x=[month_map[m] for m in monthly_avg.index], y=monthly_avg.values, ax=ax, palette="Blues_d")
            ax.set_title(f"Average Monthly Rainfall in {lga_selected} (2020-2023)")
            ax.set_xlabel("Month")
            ax.set_ylabel("Rainfall (mm)")
            st.pyplot(fig)

            wet_months_raw = [5, 6, 7, 8, 9, 10]
            dry_months_raw = [1, 2, 3, 4, 11, 12]

            wet_months = [f"{month_map[m]} ({monthly_avg[m]:.1f} mm)" for m in wet_months_raw]
            dry_months = [f"{month_map[m]} ({monthly_avg[m]:.1f} mm)" for m in dry_months_raw]

            st.markdown(f"""
            ### 🤖 AI Rainfall Insight for **{lga_selected}**
            From the data collected between **2020 and 2023**, here are the seasonal patterns:

            - 🌧️ **Wet Season (Rainy Months)**: These are months with significant rainfall that support agriculture and water supply.
            - ☀️ **Dry Season (Little or No Rain)**: These months have low rainfall, indicating the dry season.

            📍 **Breakdown by Month for {lga_selected}:**
            """)

            st.info(f"🌧️ **Wet Season Months:** {', '.join(wet_months)}")
            st.info(f"☀️ **Dry Season Months:** {', '.join(dry_months)}")

            st.header("📈 Rainfall Variability Across LGAs")
            variability = df.set_index('LG')[valid_month_cols].std(axis=1).sort_values(ascending=False)
            st.bar_chart(variability, height=300, use_container_width=True)

        elif group.startswith("Group 2"):
            st.header("📈 Rainfall Trends Over Time and Prediction")
            lga_selected = st.selectbox("Select LGA", df['LG'].unique())
            lga_df = df_melted[df_melted['LG'] == lga_selected]
            yearly_avg = lga_df.groupby('Year')['Rainfall'].mean().reset_index()

            model = LinearRegression()
            X = yearly_avg['Year'].values.reshape(-1, 1)
            y = yearly_avg['Rainfall'].values
            model.fit(X, y)
            future_years = np.arange(2024, 2029).reshape(-1, 1)
            future_pred = model.predict(future_years)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(X.flatten(), y, marker='o', label='Historical')
            ax.plot(future_years.flatten(), future_pred, marker='x', linestyle='--', color='r', label='Predicted')
            ax.set_title(f"Rainfall Trend and Forecast for {lga_selected}")
            ax.set_xlabel("Year")
            ax.set_ylabel("Average Rainfall")
            ax.legend()
            st.pyplot(fig)

            st.markdown(f"""
            ### 📘 Explanation:
            For **{lga_selected}**, we analyzed the **average annual rainfall** from **2020 to 2023**. We then applied a **Linear Regression model** to detect the trend in rainfall over time. 

            - The blue line shows the actual historical average rainfall per year.
            - The red dashed line is the **forecast**, calculated using the best-fit line from the past data.
            - The predictions for **2024 to 2028** were computed by extending this trend line forward based on the pattern observed in recent years.
            
            This allows us to:
            - Detect **increasing or decreasing trends** in rainfall.
            - Support **planning decisions** in agriculture and water management based on expected future rainfall.
            - Understand the potential **impact of climate change** on local rainfall patterns.
            """)

            st.header("🌍 Cluster LGAs into Agro-Climatic Zones")
            mean_rainfall = df[valid_month_cols].mean(axis=1)
            kmeans = KMeans(n_clusters=3, n_init=10, random_state=0).fit(mean_rainfall.values.reshape(-1, 1))
            df['Zone'] = kmeans.labels_
            st.dataframe(df[['LG', 'Zone']].sort_values('Zone'))

            st.markdown("""
            ### 📘 Explanation:
            This analysis uses **KMeans clustering** to classify LGAs into three agro-climatic zones based on average rainfall:

            - **Zone 0 (Low Rainfall)**: Suitable for drought-tolerant crops and livestock farming.
            - **Zone 1 (Moderate Rainfall)**: Ideal for mixed farming and certain staple crops.
            - **Zone 2 (High Rainfall)**: Best for crops that require a lot of water, such as rice or vegetables.

            Clustering helps in **targeted agricultural policies**, **efficient resource allocation**, and **risk assessment**.
            """)

        elif group.startswith("Group 3"):
            st.header("🌱 Suitable Planting Season By LGA")
            lga_selected = st.selectbox("Select LGA", df['LG'].unique())
            lga_df = df_melted[df_melted['LG'] == lga_selected]
            monthly_avg = lga_df.groupby('Month')['Rainfall'].mean()
            chart_data = monthly_avg.rename(index=month_map)
            st.bar_chart(chart_data)
            best_months = [month_map[m] for m in monthly_avg[monthly_avg.between(80, 200)].index.tolist()]
            st.success(f"🌾 Recommended Planting Months: {', '.join(best_months)}")

            st.header("💧 Monthly Water Availability Potential")
            month_avg = df_melted.groupby(['Year', 'Month'])['Rainfall'].mean().unstack()
            month_avg.columns = [month_map[m] for m in month_avg.columns]
            st.line_chart(month_avg.T)

        elif group.startswith("Group 4"):
            st.header("🚨 Drought or Flood Prone Areas")
            total_rain = df[valid_month_cols].sum(axis=1)
            df['TotalRainfall'] = total_rain
            drought = df[df['TotalRainfall'] < total_rain.mean() * 0.7]
            flood = df[df['TotalRainfall'] > total_rain.mean() * 1.3]

            st.subheader("LGAs Prone to Drought")
            st.dataframe(drought[['LG', 'TotalRainfall']])

            st.subheader("LGAs Prone to Flood")
            st.dataframe(flood[['LG', 'TotalRainfall']])

            st.header("🗺️ Rainfall Distribution Profiles")
            selected_lga = st.selectbox("Select LGA", df['LG'].unique(), key="map")
            profile = df[df['LG'] == selected_lga][valid_month_cols].T
            profile.index = pd.to_datetime(profile.index.str.strip(), format='%b-%y')
            profile.sort_index(inplace=True)
            st.line_chart(profile.values.flatten())

    except Exception as e:
        st.error("❌ The uploaded file is not in the correct format. Please check the column names and structure. Make sure you have a 'LG' column and monthly columns like Jan-20, Feb-20, etc.")
else:
    st.info("👈 Upload a CSV file to begin analysis.")