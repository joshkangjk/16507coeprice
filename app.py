import streamlit as st
import pandas as pd
import requests
from urllib3.util.retry import Retry
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="COE Analytics", layout="wide")

# ==========================================
# 2. PREDICTION RANGE UTILITY
# ==========================================
def predict_with_range(model, X_input):
    preds = [tree.predict(X_input) for tree in model.estimators_]
    pred_low = np.percentile(preds, 25)
    pred_high = np.percentile(preds, 75)
    pred_mean = np.mean(preds)
    return pred_low, pred_mean, pred_high

# ==========================================
# 3. DATA LAYER
# ==========================================
@st.cache_data
def load_data():
    s = requests.Session()
    
    # 1. Disguise the Python script as a standard Mac web browser
    s.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })

    # 2. Set up smart retries (if we hit a 429, wait and try again)
    retry_strategy = Retry(
        total=5,  # Try up to 5 times
        backoff_factor=2,  # Wait times: 2s, 4s, 8s, etc.
        status_forcelist=[429, 500, 502, 503, 504]  # Force retry on these errors
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
    s.mount('https://', adapter)

    try:
        # COE data request
        url_coe = "https://data.gov.sg/api/action/datastore_search"
        params_coe = {"resource_id": "d_69b3380ad7e51aff3a7dcc84eba52b8a", "limit": 15000}
        r_coe = s.get(url_coe, params=params_coe, timeout=15)
        
        if r_coe.status_code != 200:
            st.error(f"COE API Error: Status {r_coe.status_code} - {r_coe.text}")
            return pd.DataFrame()

        data_coe = r_coe.json()

        # CPI data request
        url_cpi = "https://data.gov.sg/api/action/datastore_search"
        params_cpi = {"resource_id": "d_bdaff844e3ef89d39fceb962ff8f0791", "limit": 500}
        r_cpi = s.get(url_cpi, params=params_cpi, timeout=15)
        data_cpi = r_cpi.json()

        # Clean and prepare COE data
        if data_coe.get('success'):
            df = pd.DataFrame(data_coe['result']['records'])
            df['date'] = pd.to_datetime(df['month'])

            # Clean numeric columns
            for col in ['premium', 'quota', 'bids_received', 'bids_success']:
                df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['bids_per_quota'] = df['bids_received'] / df['quota']
            df['category'] = df['vehicle_class'].astype(str).str.strip()
            df = df[df['category'] != 'Category D']

            # Clean and prepare CPI data
            if data_cpi.get('success'):
                cpi_raw = pd.DataFrame(data_cpi['result']['records'])
                id_vars = [c for c in cpi_raw.columns if not c[0].isdigit()]
                cpi_df = cpi_raw.melt(id_vars=id_vars, var_name='month_str', value_name='cpi_val')
                cpi_df['month'] = pd.to_datetime(cpi_df['month_str'], format='%Y%b', errors='coerce')
                cpi_df = cpi_df.dropna(subset=['month']).sort_values('month')
                cpi_df['cpi_val'] = pd.to_numeric(cpi_df['cpi_val'], errors='coerce')

                df['merge_date'] = df['date'].dt.to_period('M').astype(str)
                cpi_df['merge_date'] = cpi_df['month'].dt.to_period('M').astype(str)
                cpi_map = cpi_df.groupby('merge_date')['cpi_val'].mean().to_dict()
                df['real_cpi'] = df['merge_date'].map(cpi_map)
            else:
                st.warning("CPI data failed to load. Proceeding with estimates.")
                df['real_cpi'] = np.nan

            if df['real_cpi'].isnull().sum() > len(df) * 0.9:
                df['real_cpi'] = np.linspace(80, 115, len(df))
            else:
                df['real_cpi'] = df['real_cpi'].ffill()

            return df
        else:
            st.error(f"API Connected, but returned an error: {data_coe}")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Critical Data Connection Error: {e}")
        return pd.DataFrame()
    s = requests.Session()
    retries = requests.adapters.HTTPAdapter(max_retries=3)
    s.mount('https://', retries)

    try:
        # COE data request
        url_coe = "https://data.gov.sg/api/action/datastore_search"
        params_coe = {"resource_id": "d_69b3380ad7e51aff3a7dcc84eba52b8a", "limit": 15000}
        r_coe = s.get(url_coe, params=params_coe, timeout=10)
        
        # ADDED: Check if the website actually responded with a 200 OK status
        if r_coe.status_code != 200:
            st.error(f"COE API Error: Status {r_coe.status_code} - {r_coe.text}")
            return pd.DataFrame()

        data_coe = r_coe.json()

        # CPI data request
        url_cpi = "https://data.gov.sg/api/action/datastore_search"
        params_cpi = {"resource_id": "d_bdaff844e3ef89d39fceb962ff8f0791", "limit": 500}
        r_cpi = s.get(url_cpi, params=params_cpi, timeout=10)
        data_cpi = r_cpi.json()

        # Clean and prepare COE data
        if data_coe.get('success'):
            df = pd.DataFrame(data_coe['result']['records'])
            df['date'] = pd.to_datetime(df['month'])

            # Clean numeric columns that may contain commas or non numeric characters
            for col in ['premium', 'quota', 'bids_received', 'bids_success']:
                df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Derive bids per quota and category, and exclude Category D
            df['bids_per_quota'] = df['bids_received'] / df['quota']
            df['category'] = df['vehicle_class'].astype(str).str.strip()
            df = df[df['category'] != 'Category D']

            # Clean and prepare CPI data, then merge as real_cpi
            if data_cpi.get('success'):
                cpi_raw = pd.DataFrame(data_cpi['result']['records'])
                id_vars = [c for c in cpi_raw.columns if not c[0].isdigit()]
                cpi_df = cpi_raw.melt(id_vars=id_vars, var_name='month_str', value_name='cpi_val')
                cpi_df['month'] = pd.to_datetime(cpi_df['month_str'], format='%Y%b', errors='coerce')
                cpi_df = cpi_df.dropna(subset=['month']).sort_values('month')
                cpi_df['cpi_val'] = pd.to_numeric(cpi_df['cpi_val'], errors='coerce')

                df['merge_date'] = df['date'].dt.to_period('M').astype(str)
                cpi_df['merge_date'] = cpi_df['month'].dt.to_period('M').astype(str)
                cpi_map = cpi_df.groupby('merge_date')['cpi_val'].mean().to_dict()
                df['real_cpi'] = df['merge_date'].map(cpi_map)
            else:
                st.warning("CPI data failed to load. Proceeding with estimates.")
                df['real_cpi'] = np.nan

            # Fill or backfill CPI if missing too often
            if df['real_cpi'].isnull().sum() > len(df) * 0.9:
                df['real_cpi'] = np.linspace(80, 115, len(df))
            else:
                df['real_cpi'] = df['real_cpi'].ffill()

            return df
        else:
            # ADDED: Catch if the API connects but says 'success: false'
            st.error(f"API Connected, but returned an error: {data_coe}")
            return pd.DataFrame()

    except Exception as e:
        # This will catch SSL errors or lack of internet
        st.error(f"Critical Data Connection Error: {e}")
        return pd.DataFrame()
    s = requests.Session()
    retries = requests.adapters.HTTPAdapter(max_retries=3)
    s.mount('https://', retries)

    try:
        # COE data request
        url_coe = "https://data.gov.sg/api/action/datastore_search"
        params_coe = {"resource_id": "d_69b3380ad7e51aff3a7dcc84eba52b8a", "limit": 15000}
        r_coe = s.get(url_coe, params=params_coe, timeout=10)
        data_coe = r_coe.json()

        # CPI data request
        url_cpi = "https://data.gov.sg/api/action/datastore_search"
        params_cpi = {"resource_id": "d_bdaff844e3ef89d39fceb962ff8f0791", "limit": 500}
        r_cpi = s.get(url_cpi, params=params_cpi, timeout=10)
        data_cpi = r_cpi.json()

        # Clean and prepare COE data
        if data_coe.get('success'):
            df = pd.DataFrame(data_coe['result']['records'])
            df['date'] = pd.to_datetime(df['month'])

            # Clean numeric columns that may contain commas or non numeric characters
            for col in ['premium', 'quota', 'bids_received', 'bids_success']:
                df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Derive bids per quota and category, and exclude Category D
            df['bids_per_quota'] = df['bids_received'] / df['quota']
            df['category'] = df['vehicle_class'].astype(str).str.strip()
            df = df[df['category'] != 'Category D']

            # Clean and prepare CPI data, then merge as real_cpi
            if data_cpi.get('success'):
                cpi_raw = pd.DataFrame(data_cpi['result']['records'])
                id_vars = [c for c in cpi_raw.columns if not c[0].isdigit()]
                cpi_df = cpi_raw.melt(id_vars=id_vars, var_name='month_str', value_name='cpi_val')
                cpi_df['month'] = pd.to_datetime(cpi_df['month_str'], format='%Y%b', errors='coerce')
                cpi_df = cpi_df.dropna(subset=['month']).sort_values('month')
                cpi_df['cpi_val'] = pd.to_numeric(cpi_df['cpi_val'], errors='coerce')

                df['merge_date'] = df['date'].dt.to_period('M').astype(str)
                cpi_df['merge_date'] = cpi_df['month'].dt.to_period('M').astype(str)
                cpi_map = cpi_df.groupby('merge_date')['cpi_val'].mean().to_dict()
                df['real_cpi'] = df['merge_date'].map(cpi_map)
            else:
                df['real_cpi'] = np.nan

            # Fill or backfill CPI if missing too often
            if df['real_cpi'].isnull().sum() > len(df) * 0.9:
                df['real_cpi'] = np.linspace(80, 115, len(df))
            else:
                df['real_cpi'] = df['real_cpi'].ffill()

            return df
    except Exception as e:
        st.error(f"Data Connection Error: {e}")
    return pd.DataFrame()

# ==========================================
# 4. MACHINE LEARNING LAYER
# ==========================================
@st.cache_resource
def get_model(df, category):
    sub = df[df['category'] == category].sort_values('date')
    sub = sub.dropna(subset=['real_cpi'])

    if len(sub) > 10:
        features = ['quota', 'bids_per_quota', 'real_cpi']
        model = RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(sub[features].fillna(0), sub['premium'])
        return model, sub.iloc[-1]
    return None, None

# ==========================================
# 5. DASHBOARD UI
# ==========================================
st.title("Singapore COE Analytics")

# Session state for slider reset behavior
if 'reset_id' not in st.session_state:
    st.session_state.reset_id = 0

df_all = load_data()

if not df_all.empty:
    # -----------------------------
    # Sidebar configuration
    # -----------------------------
    st.sidebar.header("Configuration")
    cats = sorted(df_all['category'].unique())
    selected_cat = st.sidebar.selectbox("Select Category", cats)

    min_year = int(df_all['date'].dt.year.min())
    max_year = int(df_all['date'].dt.year.max())
    default_start = max(min_year, max_year - 5)
    selected_years = st.sidebar.slider("Training Period", min_year, max_year, (default_start, max_year))

    df_filtered = df_all[
        (df_all['date'].dt.year >= selected_years[0]) &
        (df_all['date'].dt.year <= selected_years[1])
    ]

    model, last_row = get_model(df_filtered, selected_cat)

    if model:
        st.sidebar.markdown("---")
        st.sidebar.header("Scenario Tools")

        # Anchor values for sliders and ratios
        current_quota = int(last_row['quota'])
        current_bidders = int(last_row['bids_received'])
        current_cpi = float(last_row['real_cpi'])
        current_ratio = current_bidders / current_quota

        # CPI bounds with buffer for scenario exploration
        MIN_CPI = float(df_all['real_cpi'].min())
        HISTORICAL_MAX_CPI = float(df_all['real_cpi'].max())
        MAX_CPI = HISTORICAL_MAX_CPI * 1.10  # 10 percent buffer

        # Slider keys tied to reset id so reset button can reinitialize them
        rid = st.session_state.reset_id

        input_quota = st.sidebar.slider(
            "Quota Supply",
            int(current_quota * 0.5), int(current_quota * 1.5), current_quota,
            key=f"quota_{rid}"
        )

        input_bidders = st.sidebar.slider(
            "Total Bidders",
            int(current_bidders * 0.5), int(current_bidders * 1.5), current_bidders,
            key=f"bidders_{rid}"
        )

        # CPI slider uses buffered upper bound
        input_cpi = st.sidebar.slider(
            "Inflation (CPI Index)",
            MIN_CPI, MAX_CPI, current_cpi, 0.5,
            key=f"cpi_{rid}"
        )

        st.sidebar.markdown("---")
        if st.sidebar.button("Reset Sliders"):
            st.session_state.reset_id += 1
            st.rerun()

        # --------------------------------
        # Scenario calculation and scaling
        # --------------------------------
        calculated_ratio = input_bidders / input_quota
        input_data = [[input_quota, calculated_ratio, input_cpi]]

        # Base ensemble prediction from model
        base_low, base_mean, base_high = predict_with_range(model, input_data)

        # Apply scalar adjustments for CPI and demand ratio
        cpi_scalar = input_cpi / current_cpi
        ratio_scalar = calculated_ratio / current_ratio if current_ratio > 0 else 1.0

        final_mean = base_mean * cpi_scalar * (ratio_scalar ** 0.5)
        final_low = base_low * cpi_scalar * (ratio_scalar ** 0.5)
        final_high = base_high * cpi_scalar * (ratio_scalar ** 0.5)

        # -----------------------------
        # Pre calculations for driver bar chart
        # -----------------------------
        corr_series = df_filtered[['premium', 'quota', 'bids_received', 'real_cpi']].corr()['premium'].drop('premium')
        driver_df = pd.DataFrame(corr_series).reset_index()
        driver_df.columns = ['Factor', 'Correlation']
        name_map = {
            'quota': 'Supply (Quota)',
            'bids_received': 'Consumer Demand',
            'real_cpi': 'Inflation (CPI)'
        }
        driver_df['Factor'] = driver_df['Factor'].map(name_map)
        driver_df['Color'] = driver_df['Correlation'].apply(
            lambda x: '#00CC96' if x > 0 else '#FF4B4B'
        )

        fig_net = px.bar(
            driver_df,
            x='Correlation',
            y='Factor',
            orientation='h',
            text_auto=".2f",
            title=None
        )

        fig_net.update_traces(
            marker_color=driver_df['Color'],
            textposition='outside',
            textfont=dict(color='white', size=14, weight='bold'),
            cliponaxis=False
        )
        fig_net.update_layout(
            height=250,
            margin=dict(t=20, b=20, l=150, r=50),
            xaxis=dict(
                title="Correlation Strength (-1 to +1)",
                range=[-1.2, 1.2],
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='white',
                showticklabels=False
            ),
            yaxis=dict(title=None, tickfont=dict(size=14, color='#E0E0E0')),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )

        # ==========================================
        # ROW 1: TREND AND FORECAST
        # ==========================================
        row1_col1, row1_col2 = st.columns([2, 1])

        # ------------------------------------------
        # Row 1, Col 1: Price trend and confidence band
        # ------------------------------------------
        with row1_col1:
            st.subheader(f"Price Trend ({selected_years[0]}-{selected_years[1]})")
            hist_df = df_filtered[df_filtered['category'] == selected_cat]

            features = ['quota', 'bids_per_quota', 'real_cpi']

            # Prepare historical data and predicted band
            plot_df = hist_df.copy().dropna(subset=features)
            plot_df['Predicted'] = model.predict(plot_df[features].fillna(0))

            # Simple volatility based margin for confidence band
            volatility_margin = plot_df['premium'].std() * 0.4
            plot_df['Upper'] = plot_df['Predicted'] + volatility_margin
            plot_df['Lower'] = plot_df['Predicted'] - volatility_margin

            fig = go.Figure()

            # Confidence band fill area
            fig.add_trace(go.Scatter(
                x=plot_df['date'],
                y=plot_df['Upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip',
            ))
            fig.add_trace(go.Scatter(
                x=plot_df['date'],
                y=plot_df['Lower'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0, 100, 80, 0.1)',
                name='Confidence Band'
            ))

            # Actual historical prices
            fig.add_trace(go.Scatter(
                x=plot_df['date'],
                y=plot_df['premium'],
                mode='lines',
                line=dict(color='#00CC96', width=3),
                name='Actual Price'
            ))

            # Scenario marker as star at last date
            fig.add_traces(
                px.scatter(
                    x=[plot_df.iloc[-1]['date']],
                    y=[final_mean]
                ).update_traces(
                    marker=dict(
                        color='blue',
                        size=15,
                        symbol='star',
                        line=dict(width=1, color='white')
                    ),
                    name='Current Scenario'
                ).data
            )

            fig.update_layout(
                height=350,
                margin=dict(t=20, b=20, l=20, r=20),
                xaxis_title=None,
                yaxis_title='COE Premium ($)',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=1.2,
                    xanchor="left",
                    x=0
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            # Driver correlation network
            st.markdown("---")
            st.subheader("Historical Driver Network (Push vs. Pull)")
            st.caption(
                "How do these factors historically affect the price? "
                "Green bars indicate they push price up, red bars indicate they pull price down."
            )
            st.plotly_chart(fig_net, use_container_width=True)

        # ------------------------------------------
        # Row 1, Col 2: Forecast and narrative
        # ------------------------------------------
        with row1_col2:
            st.subheader("Forecast")
            delta_val = final_mean - last_row['premium']

            # Projected price main metric
            st.metric("Projected Price", f"${final_mean:,.0f}")

            # Directional arrow and color for price change
            if delta_val >= 0:
                delta_color = "rgb(6, 178, 93)"  # green
                delta_arrow = "↑"
            else:
                delta_color = "rgb(255, 75, 75)"  # red
                delta_arrow = "↓"

            delta_text = f"{delta_arrow} ${int(abs(delta_val)):,d}"

            st.markdown(
                f"""
                <div style='color: {delta_color}; font-size: 14px; margin-top: -35px; margin-bottom: 10px; text-align: left; font-weight: bold;'>
                    {delta_text}
                </div>
                """,
                unsafe_allow_html=True
            )

            # Likely range metric
            st.metric(
                label="Likely Range (50 percent confidence)",
                value=f"\${final_low:,.0f} – \${final_high:,.0f}",
            )

            st.markdown("---")

            # Percentage changes in each driver
            pct_quota = (input_quota - current_quota) / current_quota
            pct_bids = (input_bidders - current_bidders) / current_bidders
            pct_cpi = (input_cpi - current_cpi) / current_cpi

            # Helper to render driver chips
            def get_chip(name, val, inverse=False):
                if abs(val) < 0.001:
                    return ""
                is_up = (val > 0) if not inverse else (val < 0)
                arrow = "↑" if is_up else "↓"
                color = "#d4edda" if not is_up else "#f8d7da"
                text_color = "#155724" if not is_up else "#721c24"
                return (
                    f"<span style='background:{color}; color:{text_color}; "
                    f"padding:4px 8px; border-radius:15px; font-size:16px; "
                    f"font-weight:bold; margin-right:5px;'>"
                    f"{name} {val:+.1%} ({arrow})</span>"
                )

            chip_quota = get_chip("Quota", pct_quota, inverse=True)
            chip_bids = get_chip("Bidders", pct_bids)
            chip_cpi = get_chip("Inflation", pct_cpi)

            # Typical impact size benchmark
            baseline_mean = final_mean

            # Scenario with 10 percent reduction in quota, holding bidders fixed
            test_quota_cut = input_quota * 0.90
            total_bids_implied = input_bidders
            test_bids_per_quota = total_bids_implied / test_quota_cut

            test_input_raw = [[test_quota_cut, test_bids_per_quota, input_cpi]]
            test_mean_base, _, _ = predict_with_range(model, test_input_raw)

            ratio_scalar_test = test_bids_per_quota / current_ratio if current_ratio > 0 else 1.0
            test_mean = test_mean_base * cpi_scalar * (ratio_scalar_test ** 0.5)

            impact_10pct = abs(test_mean - baseline_mean)

            # Driver summary and impact display
            st.subheader("Plain English Drivers")
            if chip_quota or chip_bids or chip_cpi:
                st.markdown(f"{chip_quota}{chip_bids}{chip_cpi}", unsafe_allow_html=True)
            else:
                st.info("Adjust the sliders to see how key drivers change.")

            st.markdown("---")

            st.subheader("Typical Impact Size")
            st.caption("Based on current market conditions:")
            st.metric(
                label="A 10 percent cut in quota typically raises the price by",
                value=f"${impact_10pct:,.0f}"
            )

            st.markdown("---")

        # ==========================================
        # ROW 2: MECHANICS AND DRIVERS
        # ==========================================
        row2_col1, row2_col2 = st.columns([2, 1])

        # ------------------------------------------
        # Row 2, Col 1: Volatility and validation
        # ------------------------------------------
        with row2_col1:

            # Historical volatility by year (box plot)
            st.subheader("Historical Price Volatility (Annual)")
            st.caption("Each box shows the central 50 percent of winning prices for that year.")

            box_df = df_filtered[df_filtered['category'] == selected_cat].copy()
            box_df['year'] = box_df['date'].dt.year.astype(str)

            fig_box = px.box(
                box_df,
                x='year',
                y='premium',
                color_discrete_sequence=['#00CC96']
            )

            annotation_price = f"Your Scenario: ${final_mean:,.0f}"
            fig_box.add_hline(
                y=final_mean,
                line_width=3,
                line_dash="dash",
                line_color="red",
                annotation_text=annotation_price,
                annotation_position="top right"
            )

            fig_box.update_layout(
                height=380,
                margin=dict(t=20, b=20, l=20, r=20),
                showlegend=False,
                xaxis=dict(title="Year"),
                yaxis=dict(title="Winning COE Price ($)"),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig_box, use_container_width=True)

            # Model validation metrics and fit chart
            st.markdown("---")
            st.subheader("Validation Framework (Historical Fit)")

            features = ['quota', 'bids_per_quota', 'real_cpi']

            val_df = box_df.copy().dropna(subset=['premium', 'quota', 'bids_received', 'real_cpi'])

            if len(val_df) > 10:
                val_df['Predicted'] = model.predict(val_df[features].fillna(0))

                # RMSE of model fit
                rmse = np.sqrt(mean_squared_error(val_df['premium'], val_df['Predicted']))

                # Confidence metric based on one sigma band
                val_df['LowerBound'] = val_df['Predicted'] - rmse
                val_df['UpperBound'] = val_df['Predicted'] + rmse

                count_in_range = (
                    (val_df['premium'] >= val_df['LowerBound']) &
                    (val_df['premium'] <= val_df['UpperBound'])
                ).sum()

                confidence_pct = (count_in_range / len(val_df)) * 100

                st.metric(
                    label="Root Mean Squared Error (RMSE)",
                    value=f"${rmse:,.0f}"
                )

                st.metric(
                    label="Prediction Confidence (Historical one sigma fit)",
                    value=f"{confidence_pct:.1f}%",
                    delta_color='off'
                )

                # Historical actual vs predicted chart
                val_df = val_df[['date', 'premium', 'Predicted']].set_index('date')
                fig_val = px.line(val_df, y=['premium', 'Predicted'], labels={'value': 'Price ($)'})

                fig_val.update_traces(
                    selector=dict(name='premium'),
                    name='Actual Price',
                    line_color='#00CC96'
                )
                fig_val.update_traces(
                    selector=dict(name='Predicted'),
                    name='Predicted Price',
                    line=dict(dash='dash'),
                    line_color='#FF4B4B'
                )

                fig_val.update_layout(
                    height=250,
                    margin=dict(t=10, b=20, l=20, r=20),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(
                        orientation='h',
                        yanchor='top',
                        y=1.2,
                        xanchor='left',
                        x=0
                    ),
                    legend_title_text='Series'
                )
                st.plotly_chart(fig_val, use_container_width=True)
            else:
                st.info("Not enough data to run validation.")

        # ------------------------------------------
        # Row 2, Col 2: Market tension (supply vs demand)
        # ------------------------------------------
        with row2_col2:
            st.subheader("Market Tension (Supply vs. Demand)")

            # Compute relative pressure on supply and demand
            if input_quota > 0:
                supply_pressure = current_quota / input_quota
            else:
                supply_pressure = 2.0

            demand_pressure = (input_bidders / current_bidders) if current_bidders > 0 else 1.0

            # Weight pressures by feature importance from the model
            base_imp = model.feature_importances_
            final_supply_score = base_imp[0] * supply_pressure
            final_demand_score = base_imp[1] * demand_pressure

            total_score = final_supply_score + final_demand_score
            pct_supply = final_supply_score / total_score
            pct_demand = final_demand_score / total_score

            tension_df = pd.DataFrame({
                'Factor': ['Supply (Quota)', 'Consumer Demand'],
                'Score': [pct_supply, pct_demand]
            })

            tension_df['Legend_Label'] = tension_df.apply(
                lambda x: (
                    f"{x['Factor']}<br>"
                    f"<span style='font-size:20px'><b>{x['Score']:.1%}</b></span>"
                ),
                axis=1
            )

            base_colors = {'Supply (Quota)': '#31333F', 'Consumer Demand': '#00CC96'}
            final_color_map = {
                row['Legend_Label']: base_colors[row['Factor']]
                for _, row in tension_df.iterrows()
            }

            fig = px.pie(
                tension_df,
                values='Score',
                names='Legend_Label',
                color='Legend_Label',
                color_discrete_map=final_color_map,
                hole=0
            )

            fig.update_traces(
                textinfo='none',
                marker=dict(line=dict(color='#000000', width=1))
            )

            fig.update_layout(
                height=380,
                margin=dict(t=20, b=20, l=10, r=10),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.05,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=14, color="white"),
                    itemwidth=70
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Not enough data.")
else:
    st.error("Data connection failed.")