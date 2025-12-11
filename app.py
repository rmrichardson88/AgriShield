import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

# ---------------- CONFIG: SHEET + COLUMNS ----------------

SHEET_ID = "1a7VlfeMyunPJO_ycgfC3s7pJEKRb4jRY84SaDJ3kHcg"

READINGS_GID = 0  # first tab
NODES_GID = 1020285433  # second tab

READINGS_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={READINGS_GID}"
NODES_CSV_URL    = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={NODES_GID}"

# column names from your sample
TIMESTAMP_COL = "timestamp"
NODE_ID_COL   = "node_id"

# node-metadata columns (tab 2)
LAST_SEEN_COL = "last_seen"
LAT_COL       = "latitude"
LON_COL       = "longitude"
STATUS_COL    = "status"   # e.g. "running"

# how recent last_seen must be to be considered "online"
ONLINE_THRESHOLD_MINUTES = 10


# ---------------- DATA LOADING ----------------

@st.cache_data(ttl=120)  # ~2 minutes cache; fresh when app is opened, lightweight within session
def load_data():
    readings = pd.read_csv(READINGS_CSV_URL)
    nodes = pd.read_csv(NODES_CSV_URL)

    # Parse timestamps
    readings[TIMESTAMP_COL] = (
        pd.to_datetime(readings[TIMESTAMP_COL], errors="coerce", utc=True)
          .dt.tz_convert(None)   # drop timezone, keep UTC clock time
    )
    
    nodes[LAST_SEEN_COL] = (
        pd.to_datetime(nodes[LAST_SEEN_COL], errors="coerce", utc=True)
          .dt.tz_convert(None)
    )

    # Drop rows with invalid timestamps
    readings = readings.dropna(subset=[TIMESTAMP_COL])
    nodes = nodes.dropna(subset=[LAST_SEEN_COL])

    # Last reading per node (to grab latest battery/sensor state)
    last_readings = (
        readings.sort_values(TIMESTAMP_COL)
        .groupby(NODE_ID_COL)
        .tail(1)
        .set_index(NODE_ID_COL)
    )

    nodes = nodes.set_index(NODE_ID_COL)
    nodes_with_last = nodes.join(last_readings, rsuffix="_last", how="left").reset_index()

    return readings, nodes_with_last


def compute_online_status(nodes_df):
    """Compute online/offline based on LAST_SEEN_COL."""
    if LAST_SEEN_COL not in nodes_df.columns:
        nodes_df["computed_status"] = "unknown"
        return nodes_df

    latest_seen = nodes_df[LAST_SEEN_COL].max()
    cutoff = latest_seen - pd.Timedelta(minutes=ONLINE_THRESHOLD_MINUTES)

    nodes_df["computed_status"] = np.where(
        nodes_df[LAST_SEEN_COL] >= cutoff,
        "online",
        "offline"
    )
    return nodes_df


# ---------------- STREAMLIT LAYOUT ----------------

st.set_page_config(
    page_title="Wildfire Sensor Prototype",
    layout="wide"
)

st.sidebar.title("Wildfire Sensor Dashboard")
page = st.sidebar.radio("View", ["Overview", "Node Explorer", "Feature Lab"])

readings, nodes = load_data()
nodes = compute_online_status(nodes)

# For convenience
numeric_cols_all = readings.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols_all = [c for c in numeric_cols_all if c != NODE_ID_COL]


# ---------------- PAGE 1: OVERVIEW ----------------

if page == "Overview":
    st.title("Network Overview")

    total_nodes = len(nodes)
    online_nodes = (nodes["computed_status"] == "online").sum()
    offline_nodes = total_nodes - online_nodes

    # Simple headline metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total nodes", total_nodes)
    col2.metric("Online (recent last_seen)", int(online_nodes))
    col3.metric("Offline", int(offline_nodes))

    st.subheader("Sensor Map")

    if LAT_COL in nodes.columns and LON_COL in nodes.columns:
        # Use full node info (including last readings) for tooltips
        map_df = nodes.dropna(subset=[LAT_COL, LON_COL]).copy()
        map_df["lat"] = map_df[LAT_COL]
        map_df["lon"] = map_df[LON_COL]

        # Ensure expected columns exist so tooltip placeholders don't break
        for col in ["timestamp", "battery_v", "battery_pct"]:
            if col not in map_df.columns:
                map_df[col] = pd.NA

        # Compute a reasonable initial view (center of all nodes)
        if not map_df.empty:
            center_lat = float(map_df["lat"].mean())
            center_lon = float(map_df["lon"].mean())
        else:
            center_lat, center_lon = 35.21, -101.83  # fallback: Amarillo

        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=7,
            pitch=0,
        )

        # Scatterplot layer for node markers
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_radius=800,
            get_fill_color=[0, 153, 255, 160],
            pickable=True,             # REQUIRED for hover tooltips
            auto_highlight=True,
        )

        # Tooltip shows the most recent reading (joined from readings tab)
        tooltip = {
            "html": (
                "<b>Node:</b> {node_id}<br/>"
                "<b>Status:</b> {computed_status}<br/>"
                "<b>Last seen (node):</b> {last_seen}<br/>"
                "<b>Last reading ts:</b> {timestamp}<br/>"
                "<b>Battery:</b> {battery_v} V ({battery_pct}%)"
            ),
            "style": {
                "backgroundColor": "rgba(0, 0, 0, 0.8)",
                "color": "white",
            },
        }

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style=None,  # let Streamlit theme choose the base map
            tooltip=tooltip,
        )

        st.pydeck_chart(deck)

    else:
        st.info("Latitude/longitude columns not found; adjust LAT_COL / LON_COL if needed.")

    st.subheader("Node table (with last sensor reading)")

    # Nice subset of columns for humans
    cols_to_show = [
        "node_id",
        LAST_SEEN_COL,
        "computed_status",
        STATUS_COL,
        "deployment_id",
        "site_id",
        "battery_v",
        "battery_pct",
        "battery_status",
    ]
    cols_to_show = [c for c in cols_to_show if c in nodes.columns]

    st.dataframe(
        nodes[cols_to_show].sort_values("node_id"),
        use_container_width=True,
    )


# ---------------- PAGE 2: NODE EXPLORER ----------------

elif page == "Node Explorer":
    st.title("Node Explorer")

    node_ids = sorted(readings[NODE_ID_COL].dropna().unique())
    selected_node = st.sidebar.selectbox("Node", node_ids)

    node_df = (
        readings[readings[NODE_ID_COL] == selected_node]
        .sort_values(TIMESTAMP_COL)
        .copy()
    )

    st.subheader(f"Time series for node {selected_node}")

    numeric_cols_node = node_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_node = [c for c in numeric_cols_node if c != NODE_ID_COL]

    # Use a list for default
    default_metrics = ["battery_v"] if "battery_v" in numeric_cols_node else numeric_cols_node[:1]

    metrics_to_plot = st.multiselect(
        "Metrics to plot",
        numeric_cols_node,
        default=default_metrics,
    )

    if metrics_to_plot:
        plot_df = node_df.set_index(TIMESTAMP_COL)[metrics_to_plot]
        st.line_chart(plot_df)
    else:
        st.info("Select one or more numeric metrics to plot.")

    st.subheader("Recent raw data")
    st.dataframe(node_df.tail(200), use_container_width=True)


# ---------------- PAGE 3: FEATURE LAB ----------------

else:  # "Feature Lab"
    st.title("Feature Lab")

    if readings.empty:
        st.warning("No readings yet.")
    else:
        min_ts = readings[TIMESTAMP_COL].min()
        max_ts = readings[TIMESTAMP_COL].max()

        st.markdown("Compare variance & correlations across two time windows")

        colA, colB = st.columns(2)
        with colA:
            st.subheader("Window A (baseline)")
            start_A = st.date_input(
                "Start A",
                min_ts.date(),
                min_value=min_ts.date(),
                max_value=max_ts.date(),
                key="start_A"
            )
            end_A = st.date_input(
                "End A",
                max_ts.date(),
                min_value=min_ts.date(),
                max_value=max_ts.date(),
                key="end_A"
            )

        with colB:
            st.subheader("Window B (event / comparison)")
            start_B = st.date_input(
                "Start B",
                min_ts.date(),
                min_value=min_ts.date(),
                max_value=max_ts.date(),
                key="start_B"
            )
            end_B = st.date_input(
                "End B",
                max_ts.date(),
                min_value=min_ts.date(),
                max_value=max_ts.date(),
                key="end_B"
            )

        metrics = st.multiselect(
            "Metrics to analyze",
            numeric_cols_all,
            default=[
                c for c in numeric_cols_all
                if any(
                    key in c
                    for key in [
                        "bme688_temperature_c",
                        "bme688_humidity_pct",
                        "bme688_pressure_hpa",
                        "anemometer_wind_speed_ms",
                        "anemometer_wind_gust_ms",
                        "rain_gauge_rain_interval_mm",
                        "rain_gauge_rain_hourly_mm",
                        "battery_v",
                        "battery_pct",
                    ]
                )
            ][:6]  # keep default reasonably small
        )

        if metrics:
            ts_start_A = pd.to_datetime(start_A)
            ts_end_A   = pd.to_datetime(end_A) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            ts_start_B = pd.to_datetime(start_B)
            ts_end_B   = pd.to_datetime(end_B) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

            mask_A = (readings[TIMESTAMP_COL] >= ts_start_A) & (readings[TIMESTAMP_COL] <= ts_end_A)
            mask_B = (readings[TIMESTAMP_COL] >= ts_start_B) & (readings[TIMESTAMP_COL] <= ts_end_B)

            df_A = readings.loc[mask_A, metrics]
            df_B = readings.loc[mask_B, metrics]

            st.subheader("Variance comparison")
            var_A = df_A.var()
            var_B = df_B.var()
            var_ratio = (var_B / (var_A + 1e-9)).sort_values(ascending=False)

            var_df = pd.DataFrame({
                "var_A": var_A,
                "var_B": var_B,
                "var_ratio_B_over_A": var_ratio
            }).sort_values("var_ratio_B_over_A", ascending=False)

            st.dataframe(var_df, use_container_width=True)

            st.subheader("Correlation matrix (Window B)")
            corr = df_B.corr()
            st.dataframe(corr.style.background_gradient(axis=None), use_container_width=True)

        else:
            st.info("Select at least one metric to analyze.")
