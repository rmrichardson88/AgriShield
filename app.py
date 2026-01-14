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

# Common keys
TIMESTAMP_COL = "timestamp"
NODE_ID_COL   = "node_id"

# Nodes tab (metadata)
LAST_SEEN_COL = "last_seen"
LAT_COL       = "latitude"
LON_COL       = "longitude"
STATUS_COL    = "status"

# Online definition: within 1 hour of current time (UTC-naive baseline)
ONLINE_THRESHOLD_MINUTES = 60

# Display timezone (CST/CDT rules handled by America/Chicago)
DISPLAY_TZ = "America/Chicago"


# ---------------- HELPERS ----------------

def _to_display_tz_str(ts: pd.Timestamp) -> str:
    """
    Convert a naive UTC timestamp to a formatted string in DISPLAY_TZ.
    Safe for NaT/NA. Assumes input is naive UTC (as produced by our parsing).
    """
    if pd.isna(ts):
        return ""
    try:
        return (
            pd.Timestamp(ts)
            .tz_localize("UTC")
            .tz_convert(DISPLAY_TZ)
            .strftime("%Y-%m-%d %H:%M:%S %Z")
        )
    except Exception:
        return str(ts)

def _safe_read_csv(url: str) -> pd.DataFrame:
    """
    Read CSV from Google Sheets export.
    - If a sheet is headers-only, pandas returns an empty DF with columns (desired).
    - If the fetch fails (permissions/network), return an empty DF so the app still runs.
    """
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.warning(f"Could not read data from Google Sheet URL. Details: {e}")
        return pd.DataFrame()


# ---------------- DATA LOADING ----------------

@st.cache_data(ttl=120)  # cache ~2 minutes
def load_data():
    readings = _safe_read_csv(READINGS_CSV_URL)
    nodes    = _safe_read_csv(NODES_CSV_URL)

    # Ensure key columns exist to avoid KeyErrors if schema changes or sheet is blanked
    for df, required_cols in [
        (readings, [TIMESTAMP_COL, NODE_ID_COL]),
        (nodes,    [NODE_ID_COL, LAST_SEEN_COL]),
    ]:
        for c in required_cols:
            if c not in df.columns:
                df[c] = pd.NA

    # Normalize node_id types (avoid join mismatches)
    readings[NODE_ID_COL] = readings[NODE_ID_COL].astype("string")
    nodes[NODE_ID_COL]    = nodes[NODE_ID_COL].astype("string")

    # Parse timestamps as UTC, then drop timezone => naive UTC baseline for comparisons
    readings[TIMESTAMP_COL] = (
        pd.to_datetime(readings[TIMESTAMP_COL], errors="coerce", utc=True)
          .dt.tz_convert(None)
    )
    nodes[LAST_SEEN_COL] = (
        pd.to_datetime(nodes[LAST_SEEN_COL], errors="coerce", utc=True)
          .dt.tz_convert(None)
    )

    # Drop invalid timestamps per your preference
    readings = readings.dropna(subset=[TIMESTAMP_COL])
    nodes    = nodes.dropna(subset=[LAST_SEEN_COL])

    # Last reading per node (for tooltip/table enrichment)
    if not readings.empty:
        last_readings = (
            readings.sort_values(TIMESTAMP_COL)
            .dropna(subset=[NODE_ID_COL])
            .groupby(NODE_ID_COL, dropna=True)
            .tail(1)
            .set_index(NODE_ID_COL)
        )
    else:
        last_readings = pd.DataFrame().set_index(pd.Index([], name=NODE_ID_COL))

    # Join: nodes + last sensor row
    nodes = nodes.set_index(NODE_ID_COL)
    nodes_with_last = nodes.join(last_readings, rsuffix="_last", how="left").reset_index()

    return readings, nodes_with_last


def compute_online_status(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Online/offline relative to current time in UTC:
      online if last_seen >= (now_utc - threshold)

    Always compares tz-aware UTC to tz-aware UTC (avoids pandas invalid comparison errors).
    """
    if LAST_SEEN_COL not in nodes_df.columns:
        nodes_df["computed_status"] = "unknown"
        return nodes_df

    # Parse as tz-aware UTC no matter what the incoming dtype is (string/naive/tz-aware)
    last_seen_utc = pd.to_datetime(nodes_df[LAST_SEEN_COL], errors="coerce", utc=True)

    # Current time as tz-aware UTC
    now_utc = pd.Timestamp.now(tz="UTC")
    cutoff = now_utc - pd.Timedelta(minutes=ONLINE_THRESHOLD_MINUTES)

    # Optional: clamp future timestamps (clock skew) so they don't distort status
    last_seen_effective = last_seen_utc.mask(last_seen_utc > now_utc, now_utc)

    nodes_df["computed_status"] = np.where(
        last_seen_effective >= cutoff,
        "online",
        "offline",
    )

    # Keep your existing convention downstream: store last_seen as naive UTC
    nodes_df[LAST_SEEN_COL] = last_seen_utc.dt.tz_convert(None)

    return nodes_df

# ---------------- STREAMLIT LAYOUT ----------------

st.set_page_config(page_title="Wildfire Sensor Prototype", layout="wide")

st.sidebar.title("Wildfire Sensor Dashboard")
page = st.sidebar.radio("View", ["Overview", "Node Explorer", "Feature Lab"])

readings, nodes = load_data()
nodes = compute_online_status(nodes)

# Numeric columns available in readings (may be empty if no data yet)
numeric_cols_all = readings.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols_all = [c for c in numeric_cols_all if c != NODE_ID_COL]


# ---------------- PAGE 1: OVERVIEW ----------------

if page == "Overview":
    st.title("Network Overview")

    total_nodes = len(nodes)
    online_nodes = int((nodes.get("computed_status", pd.Series(dtype="string")) == "online").sum()) if total_nodes else 0
    offline_nodes = total_nodes - online_nodes

    col1, col2, col3 = st.columns(3)
    col1.metric("Total nodes", total_nodes)
    col2.metric("Online (last 1 hour)", online_nodes)
    col3.metric("Offline", offline_nodes)

    st.subheader("Sensor Map")

    if LAT_COL in nodes.columns and LON_COL in nodes.columns:
        map_df = nodes.dropna(subset=[LAT_COL, LON_COL]).copy()
        map_df["lat"] = pd.to_numeric(map_df[LAT_COL], errors="coerce")
        map_df["lon"] = pd.to_numeric(map_df[LON_COL], errors="coerce")
        map_df = map_df.dropna(subset=["lat", "lon"])

        # Ensure any columns referenced in tooltip exist (avoid pydeck key errors)
        tooltip_cols = [
            # node metadata
            "node_id",
            "hostname",
            "ip_addresses",
            "version",
            "uptime_hours",
            "sensors_count",
            "deployment_id",
            "site_id",
            STATUS_COL,
            "computed_status",
            LAST_SEEN_COL,

            # last reading timestamp (joined from readings)
            TIMESTAMP_COL,

            # batteries (new names)
            "battery_voltage_v",
            "battery_percentage",
            "battery_status",

            # high-interest air metrics
            "scd41_co2_ppm",
            "ze03_co_ppm",
            "sps30_pm2_5",
            "sps30_pm10",

            # environment + weather kit
            "bme688_temperature_c",
            "bme688_humidity_pct",
            "bme688_pressure_hpa",
            "weather_kit_anemometer_wind_speed_ms",
            "weather_kit_anemometer_wind_gust_ms",
            "weather_kit_rain_gauge_rain_interval_mm",
            "weather_kit_rain_gauge_rain_hourly_mm",
        ]
        for c in tooltip_cols:
            if c not in map_df.columns:
                map_df[c] = pd.NA

        # Display-friendly timestamp strings (CST) without affecting core computations
        map_df["last_seen_cst"] = map_df[LAST_SEEN_COL].apply(_to_display_tz_str)
        map_df["last_reading_cst"] = map_df[TIMESTAMP_COL].apply(_to_display_tz_str)

        # Initial view centered on nodes
        if not map_df.empty:
            center_lat = float(map_df["lat"].mean())
            center_lon = float(map_df["lon"].mean())
        else:
            center_lat, center_lon = 35.21, -101.83  # fallback: Amarillo

        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=7, pitch=0)

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_radius=800,
            get_fill_color=[0, 153, 255, 160],
            pickable=True,
            auto_highlight=True,
        )

        tooltip = {
            "html": (
                "<b>Node:</b> {node_id} ({hostname})<br/>"
                "<b>IP(s):</b> {ip_addresses}<br/>"
                "<b>Status:</b> {computed_status} (meta: {status})<br/>"
                "<b>Last seen:</b> {last_seen_cst}<br/>"
                "<b>Last reading:</b> {last_reading_cst}<br/>"
                "<b>Deployment/Site:</b> {deployment_id} / {site_id}<br/>"
                "<b>Version:</b> {version} &nbsp; <b>Uptime:</b> {uptime_hours} hrs &nbsp; <b>Sensors:</b> {sensors_count}"
                "<hr style='border:0;border-top:1px solid rgba(255,255,255,0.2);margin:6px 0;'/>"
                "<b>CO2:</b> {scd41_co2_ppm} ppm<br/>"
                "<b>CO:</b> {ze03_co_ppm} ppm<br/>"
                "<b>PM2.5:</b> {sps30_pm2_5}<br/>"
                "<b>PM10:</b> {sps30_pm10}<br/>"
                "<hr style='border:0;border-top:1px solid rgba(255,255,255,0.2);margin:6px 0;'/>"
                "<b>Battery:</b> {battery_voltage_v} V ({battery_percentage}%) &nbsp; <b>State:</b> {battery_status}<br/>"
                "<b>Temp:</b> {bme688_temperature_c} °C<br/>"
                "<b>Humidity:</b> {bme688_humidity_pct} %<br/>"
                "<b>Pressure:</b> {bme688_pressure_hpa} hPa<br/>"
                "<b>Wind speed:</b> {weather_kit_anemometer_wind_speed_ms} m/s<br/>"
                "<b>Wind gust:</b> {weather_kit_anemometer_wind_gust_ms} m/s<br/>"
                "<b>Rain (interval):</b> {weather_kit_rain_gauge_rain_interval_mm} mm<br/>"
                "<b>Rain (hourly):</b> {weather_kit_rain_gauge_rain_hourly_mm} mm"
            ),
            "style": {"backgroundColor": "rgba(0, 0, 0, 0.8)", "color": "white"},
        }

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style=None,
            tooltip=tooltip,
        )
        st.pydeck_chart(deck)

    else:
        st.info("Latitude/longitude columns not found on nodes sheet; adjust LAT_COL / LON_COL if needed.")

    st.subheader("Node table (with last sensor reading if available)")

    # Human-friendly columns: node metadata + high-signal last-reading fields
    preferred_cols = [
        "node_id",
        "hostname",
        "ip_addresses",
        "deployment_id",
        "site_id",
        STATUS_COL,
        "computed_status",
        LAST_SEEN_COL,
        "version",
        "uptime_hours",
        "sensors_count",
        "scd41_co2_ppm",
        "ze03_co_ppm",
        "sps30_pm2_5",
        "sps30_pm10",
        "bme688_temperature_c",
        "battery_voltage_v",
        "battery_percentage",
        "battery_status",
        TIMESTAMP_COL,
    ]
    cols_to_show = [c for c in preferred_cols if c in nodes.columns]

    display_df = nodes.copy()

    # Add CST display columns for table readability
    if LAST_SEEN_COL in display_df.columns:
        display_df["last_seen_cst"] = display_df[LAST_SEEN_COL].apply(_to_display_tz_str)
    if TIMESTAMP_COL in display_df.columns:
        display_df["last_reading_cst"] = display_df[TIMESTAMP_COL].apply(_to_display_tz_str)

    # Swap raw timestamp cols for display versions
    if "last_seen_cst" in display_df.columns and LAST_SEEN_COL in cols_to_show:
        cols_to_show = [("last_seen_cst" if c == LAST_SEEN_COL else c) for c in cols_to_show]
    if "last_reading_cst" in display_df.columns and TIMESTAMP_COL in cols_to_show:
        cols_to_show = [("last_reading_cst" if c == TIMESTAMP_COL else c) for c in cols_to_show]

    if cols_to_show:
        st.dataframe(display_df[cols_to_show].sort_values("node_id"), use_container_width=True)
    else:
        st.info("No displayable columns found yet—confirm nodes sheet headers.")


# ---------------- PAGE 2: NODE EXPLORER ----------------

elif page == "Node Explorer":
    st.title("Node Explorer")

    if readings.empty:
        st.warning("No sensor readings yet (readings tab is currently headers-only). This page will populate once data arrives.")
        st.subheader("Current nodes (metadata)")
        meta_cols = [c for c in ["node_id", "hostname", LAST_SEEN_COL, "computed_status", STATUS_COL, "deployment_id", "site_id"] if c in nodes.columns]
        if meta_cols:
            df = nodes.copy()
            if LAST_SEEN_COL in df.columns:
                df["last_seen_cst"] = df[LAST_SEEN_COL].apply(_to_display_tz_str)
                meta_cols = [("last_seen_cst" if c == LAST_SEEN_COL else c) for c in meta_cols]
            st.dataframe(df[meta_cols].sort_values("node_id"), use_container_width=True)
        st.stop()

    node_ids = sorted(readings[NODE_ID_COL].dropna().astype("string").unique().tolist())
    if not node_ids:
        st.warning("No node IDs found in readings yet.")
        st.stop()

    selected_node = st.sidebar.selectbox("Node", node_ids)

    node_df = (
        readings[readings[NODE_ID_COL] == selected_node]
        .sort_values(TIMESTAMP_COL)
        .copy()
    )

    st.subheader(f"Time series for node {selected_node}")

    numeric_cols_node = node_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_node = [c for c in numeric_cols_node if c != NODE_ID_COL]

    # Default plot preference: CO2 first, then battery voltage/percentage
    default_candidates = [c for c in ["scd41_co2_ppm", "battery_voltage_v", "battery_percentage"] if c in numeric_cols_node]
    default_metrics = default_candidates[:1] if default_candidates else (numeric_cols_node[:1] if numeric_cols_node else [])

    metrics_to_plot = st.multiselect("Metrics to plot", numeric_cols_node, default=default_metrics)

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
        st.warning("No readings yet. This page will populate once sensor data arrives.")
        st.stop()

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
            key="start_A",
        )
        end_A = st.date_input(
            "End A",
            max_ts.date(),
            min_value=min_ts.date(),
            max_value=max_ts.date(),
            key="end_A",
        )

    with colB:
        st.subheader("Window B (event / comparison)")
        start_B = st.date_input(
            "Start B",
            min_ts.date(),
            min_value=min_ts.date(),
            max_value=max_ts.date(),
            key="start_B",
        )
        end_B = st.date_input(
            "End B",
            max_ts.date(),
            min_value=min_ts.date(),
            max_value=max_ts.date(),
            key="end_B",
        )

    default_metric_order = [
        "scd41_co2_ppm",
        "ze03_co_ppm",
        "sps30_pm2_5",
        "sps30_pm10",
        "bme688_temperature_c",
        "bme688_humidity_pct",
        "bme688_pressure_hpa",
        "weather_kit_anemometer_wind_speed_ms",
        "weather_kit_anemometer_wind_gust_ms",
        "weather_kit_rain_gauge_rain_interval_mm",
        "weather_kit_rain_gauge_rain_hourly_mm",
        "battery_voltage_v",
        "battery_percentage",
    ]
    default_metrics = [c for c in default_metric_order if c in numeric_cols_all][:6]

    metrics = st.multiselect("Metrics to analyze", numeric_cols_all, default=default_metrics)

    if not metrics:
        st.info("Select at least one metric to analyze.")
        st.stop()

    ts_start_A = pd.to_datetime(start_A)
    ts_end_A   = pd.to_datetime(end_A) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    ts_start_B = pd.to_datetime(start_B)
    ts_end_B   = pd.to_datetime(end_B) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    mask_A = (readings[TIMESTAMP_COL] >= ts_start_A) & (readings[TIMESTAMP_COL] <= ts_end_A)
    mask_B = (readings[TIMESTAMP_COL] >= ts_start_B) & (readings[TIMESTAMP_COL] <= ts_end_B)

    df_A = readings.loc[mask_A, metrics]
    df_B = readings.loc[mask_B, metrics]

    st.subheader("Variance comparison")
    var_A = df_A.var(numeric_only=True)
    var_B = df_B.var(numeric_only=True)
    var_ratio = (var_B / (var_A + 1e-9)).sort_values(ascending=False)

    var_df = pd.DataFrame({"var_A": var_A, "var_B": var_B, "var_ratio_B_over_A": var_ratio})
    var_df = var_df.sort_values("var_ratio_B_over_A", ascending=False)

    st.dataframe(var_df, use_container_width=True)

    st.subheader("Correlation matrix (Window B)")
    corr = df_B.corr(numeric_only=True)
    st.dataframe(corr.style.background_gradient(axis=None), use_container_width=True)
