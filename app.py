from urllib.parse import quote

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

# ---------------- CONFIG: SHEET + COLUMNS ----------------

SHEET_ID = "1a7VlfeMyunPJO_ycgfC3s7pJEKRb4jRY84SaDJ3kHcg"

READINGS_SHEET_NAME = "Readings"
NODES_SHEET_NAME = "Nodes"
SYSTEM_HEALTH_SHEET_NAME = "System_Health"

# Common keys
TIMESTAMP_COL = "timestamp"
NODE_ID_COL = "node_id"

# Nodes tab (metadata)
LAST_SEEN_COL = "last_seen"
LAT_COL = "latitude"
LON_COL = "longitude"
STATUS_COL = "status"
MINUTES_SINCE_ACTIVE_COL = "minutes_since_active"

SYSTEM_HEALTH_TIMESTAMP_COL = f"{TIMESTAMP_COL}_health"

SYSTEM_HEALTH_BATTERY_ALIASES = {
    "battery_voltage_v": ["sensor_health_battery_voltage_v"],
    "battery_percentage": ["sensor_health_battery_percentage"],
    "battery_status": ["sensor_health_battery_status"],
}

# Online definition: within 1 hour of current time (UTC-naive baseline)
ONLINE_THRESHOLD_MINUTES = 60

# Display timezone (CST/CDT rules handled by America/Chicago)
DISPLAY_TZ = "America/Chicago"


# ---------------- HELPERS ----------------


def _sheet_csv_url(sheet_name: str) -> str:
    return (
        f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq"
        f"?tqx=out:csv&sheet={quote(sheet_name, safe='')}"
    )


READINGS_CSV_URL = _sheet_csv_url(READINGS_SHEET_NAME)
NODES_CSV_URL = _sheet_csv_url(NODES_SHEET_NAME)
SYSTEM_HEALTH_CSV_URL = _sheet_csv_url(SYSTEM_HEALTH_SHEET_NAME)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = []
    rename_map = {}

    for original in df.columns:
        cleaned = "" if original is None else str(original).strip()
        if not cleaned or cleaned.lower().startswith("unnamed:"):
            continue
        keep_cols.append(original)
        if cleaned != original:
            rename_map[original] = cleaned

    return df.loc[:, keep_cols].rename(columns=rename_map)


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


def _safe_read_csv(url: str, label: str) -> pd.DataFrame:
    """
    Read CSV from Google Sheets export.
    - If a sheet is headers-only, pandas returns an empty DF with columns (desired).
    - If the fetch fails (permissions/network), return an empty DF so the app still runs.
    """
    try:
        return _normalize_columns(pd.read_csv(url))
    except Exception as e:
        st.warning(f"Could not read '{label}' from Google Sheets. Details: {e}")
        return pd.DataFrame()


def _coalesce_columns(
    df: pd.DataFrame, target_col: str, source_cols: list[str]
) -> pd.DataFrame:
    available_cols = [col for col in [target_col, *source_cols] if col in df.columns]
    if not available_cols:
        df[target_col] = pd.NA
        return df

    stacked = df[available_cols].copy()
    for col in stacked.columns:
        if pd.api.types.is_object_dtype(stacked[col]) or pd.api.types.is_string_dtype(
            stacked[col]
        ):
            stacked[col] = stacked[col].replace(r"^\s*$", pd.NA, regex=True)

    df[target_col] = stacked.bfill(axis=1).iloc[:, 0]
    return df


def _latest_by_node(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame().set_index(pd.Index([], name=NODE_ID_COL))

    return (
        df.sort_values(timestamp_col)
        .dropna(subset=[NODE_ID_COL])
        .groupby(NODE_ID_COL, dropna=True)
        .tail(1)
        .set_index(NODE_ID_COL)
    )


# ---------------- DATA LOADING ----------------


@st.cache_data(ttl=120)  # cache ~2 minutes
def load_data():
    readings = _safe_read_csv(READINGS_CSV_URL, READINGS_SHEET_NAME)
    nodes = _safe_read_csv(NODES_CSV_URL, NODES_SHEET_NAME)
    system_health = _safe_read_csv(SYSTEM_HEALTH_CSV_URL, SYSTEM_HEALTH_SHEET_NAME)

    # Ensure key columns exist to avoid KeyErrors if schema changes or sheet is blanked
    for df, required_cols in [
        (readings, [TIMESTAMP_COL, NODE_ID_COL]),
        (nodes, [NODE_ID_COL, LAST_SEEN_COL]),
        (system_health, [TIMESTAMP_COL, NODE_ID_COL]),
    ]:
        for col in required_cols:
            if col not in df.columns:
                df[col] = pd.NA

    # Normalize node_id types (avoid join mismatches)
    readings[NODE_ID_COL] = readings[NODE_ID_COL].astype("string")
    nodes[NODE_ID_COL] = nodes[NODE_ID_COL].astype("string")
    system_health[NODE_ID_COL] = system_health[NODE_ID_COL].astype("string")

    # Parse timestamps as UTC, then drop timezone => naive UTC baseline for comparisons
    readings[TIMESTAMP_COL] = (
        pd.to_datetime(readings[TIMESTAMP_COL], errors="coerce", utc=True)
        .dt.tz_convert(None)
    )
    nodes[LAST_SEEN_COL] = (
        pd.to_datetime(nodes[LAST_SEEN_COL], errors="coerce", utc=True).dt.tz_convert(
            None
        )
    )
    system_health[TIMESTAMP_COL] = (
        pd.to_datetime(system_health[TIMESTAMP_COL], errors="coerce", utc=True)
        .dt.tz_convert(None)
    )

    # Drop invalid timestamps per your preference
    readings = readings.dropna(subset=[TIMESTAMP_COL])
    nodes = nodes.dropna(subset=[LAST_SEEN_COL])
    system_health = system_health.dropna(subset=[TIMESTAMP_COL])

    # Last reading per node (for tooltip/table enrichment)
    last_readings = _latest_by_node(readings, TIMESTAMP_COL)
    last_system_health = _latest_by_node(system_health, TIMESTAMP_COL)

    # Join: nodes + last sensor row + latest system health row
    nodes = nodes.set_index(NODE_ID_COL)
    nodes_with_last = nodes.join(last_readings, rsuffix="_last", how="left")
    nodes_with_last = nodes_with_last.join(
        last_system_health, rsuffix="_health", how="left"
    ).reset_index()

    # Preserve existing UI field names even after battery metrics moved to System_Health
    for target_col, source_cols in SYSTEM_HEALTH_BATTERY_ALIASES.items():
        nodes_with_last = _coalesce_columns(nodes_with_last, target_col, source_cols)
        system_health = _coalesce_columns(system_health, target_col, source_cols)

    return readings, nodes_with_last, system_health


def compute_online_status(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer minutes_since_active when available; otherwise fall back to last_seen.
    Online/offline relative to current time in UTC:
      online if last_seen >= (now_utc - threshold)

    Always compares tz-aware UTC to tz-aware UTC (avoids pandas invalid comparison errors).
    """
    computed_status = pd.Series("unknown", index=nodes_df.index, dtype="string")

    if MINUTES_SINCE_ACTIVE_COL in nodes_df.columns:
        minutes_since_active = pd.to_numeric(
            nodes_df[MINUTES_SINCE_ACTIVE_COL], errors="coerce"
        )
        has_minutes = minutes_since_active.notna()
        computed_status.loc[has_minutes] = np.where(
            minutes_since_active.loc[has_minutes] <= ONLINE_THRESHOLD_MINUTES,
            "online",
            "offline",
        )

    if LAST_SEEN_COL in nodes_df.columns:
        last_seen_utc = pd.to_datetime(nodes_df[LAST_SEEN_COL], errors="coerce", utc=True)
        now_utc = pd.Timestamp.now(tz="UTC")
        cutoff = now_utc - pd.Timedelta(minutes=ONLINE_THRESHOLD_MINUTES)
        last_seen_effective = last_seen_utc.mask(last_seen_utc > now_utc, now_utc)

        needs_fallback = computed_status == "unknown"
        computed_status.loc[needs_fallback] = np.where(
            last_seen_effective.loc[needs_fallback] >= cutoff,
            "online",
            "offline",
        )

        # Keep your existing convention downstream: store last_seen as naive UTC
        nodes_df[LAST_SEEN_COL] = last_seen_utc.dt.tz_convert(None)

    nodes_df["computed_status"] = computed_status
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
page = st.sidebar.radio(
    "View", ["Overview", "Node Explorer", "System Health", "Feature Lab"]
)

readings, nodes, system_health = load_data()
nodes = compute_online_status(nodes)

# Numeric columns available in readings (may be empty if no data yet)
numeric_cols_all = readings.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols_all = [col for col in numeric_cols_all if col != NODE_ID_COL]


# ---------------- PAGE 1: OVERVIEW ----------------


if page == "Overview":
    st.title("Network Overview")

    total_nodes = len(nodes)
    online_nodes = (
        int(
            (
                nodes.get("computed_status", pd.Series(dtype="string")) == "online"
            ).sum()
        )
        if total_nodes
        else 0
    )
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
            MINUTES_SINCE_ACTIVE_COL,
            # latest sensor timestamp
            TIMESTAMP_COL,
            # latest system health timestamp
            SYSTEM_HEALTH_TIMESTAMP_COL,
            # batteries
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
            # system health
            "system_health_cpu_temperature_c",
            "system_health_cpu_usage_percent",
            "system_health_memory_usage_percent",
            "system_health_disk_usage_percent",
            "system_health_network_latency_ms",
            "system_health_wifi_signal_level_dbm",
            "system_health_queue_pending_batches",
            "system_health_tailscale_backend_state",
            "system_health_tailscale_active_peers",
            "system_health_tailscale_relay",
        ]
        for col in tooltip_cols:
            if col not in map_df.columns:
                map_df[col] = pd.NA

        # Display-friendly timestamp strings without affecting core computations
        map_df["last_seen_cst"] = map_df[LAST_SEEN_COL].apply(_to_display_tz_str)
        map_df["last_reading_cst"] = map_df[TIMESTAMP_COL].apply(_to_display_tz_str)
        map_df["health_timestamp_cst"] = map_df[SYSTEM_HEALTH_TIMESTAMP_COL].apply(
            _to_display_tz_str
        )

        # Initial view centered on nodes
        if not map_df.empty:
            center_lat = float(map_df["lat"].mean())
            center_lon = float(map_df["lon"].mean())
        else:
            center_lat, center_lon = 35.21, -101.83  # fallback: Amarillo

        view_state = pdk.ViewState(
            latitude=center_lat, longitude=center_lon, zoom=7, pitch=0
        )

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
                "<b>Minutes since active:</b> {minutes_since_active}<br/>"
                "<b>Last reading:</b> {last_reading_cst}<br/>"
                "<b>Last health check:</b> {health_timestamp_cst}<br/>"
                "<b>Deployment/Site:</b> {deployment_id} / {site_id}<br/>"
                "<b>Version:</b> {version} &nbsp; <b>Uptime:</b> {uptime_hours} hrs &nbsp; <b>Sensors:</b> {sensors_count}"
                "<hr style='border:0;border-top:1px solid rgba(255,255,255,0.2);margin:6px 0;'/>"
                "<b>CO2:</b> {scd41_co2_ppm} ppm<br/>"
                "<b>CO:</b> {ze03_co_ppm} ppm<br/>"
                "<b>PM2.5:</b> {sps30_pm2_5}<br/>"
                "<b>PM10:</b> {sps30_pm10}<br/>"
                "<b>Battery:</b> {battery_voltage_v} V ({battery_percentage}%) &nbsp; <b>State:</b> {battery_status}<br/>"
                "<b>Temp:</b> {bme688_temperature_c} °C<br/>"
                "<b>Humidity:</b> {bme688_humidity_pct} %<br/>"
                "<b>Pressure:</b> {bme688_pressure_hpa} hPa<br/>"
                "<b>Wind speed:</b> {weather_kit_anemometer_wind_speed_ms} m/s<br/>"
                "<b>Wind gust:</b> {weather_kit_anemometer_wind_gust_ms} m/s<br/>"
                "<b>Rain (interval):</b> {weather_kit_rain_gauge_rain_interval_mm} mm<br/>"
                "<b>Rain (hourly):</b> {weather_kit_rain_gauge_rain_hourly_mm} mm"
                "<hr style='border:0;border-top:1px solid rgba(255,255,255,0.2);margin:6px 0;'/>"
                "<b>CPU temp:</b> {system_health_cpu_temperature_c} °C<br/>"
                "<b>CPU usage:</b> {system_health_cpu_usage_percent} %<br/>"
                "<b>Memory:</b> {system_health_memory_usage_percent} %<br/>"
                "<b>Disk:</b> {system_health_disk_usage_percent} %<br/>"
                "<b>Latency:</b> {system_health_network_latency_ms} ms<br/>"
                "<b>Wi-Fi signal:</b> {system_health_wifi_signal_level_dbm} dBm<br/>"
                "<b>Queued batches:</b> {system_health_queue_pending_batches}<br/>"
                "<b>Tailscale:</b> {system_health_tailscale_backend_state} / peers={system_health_tailscale_active_peers} / relay={system_health_tailscale_relay}"
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
        st.info(
            "Latitude/longitude columns not found on nodes sheet; adjust LAT_COL / LON_COL if needed."
        )

    st.subheader("Node table (with latest readings and health)")

    preferred_cols = [
        "node_id",
        "hostname",
        "ip_addresses",
        "deployment_id",
        "site_id",
        STATUS_COL,
        "computed_status",
        MINUTES_SINCE_ACTIVE_COL,
        LAST_SEEN_COL,
        "version",
        "uptime_hours",
        "sensors_count",
        "system_health_cpu_temperature_c",
        "system_health_cpu_usage_percent",
        "system_health_memory_usage_percent",
        "system_health_queue_pending_batches",
        "battery_voltage_v",
        "battery_percentage",
        "battery_status",
        "scd41_co2_ppm",
        "ze03_co_ppm",
        "sps30_pm2_5",
        "sps30_pm10",
        "bme688_temperature_c",
        TIMESTAMP_COL,
        SYSTEM_HEALTH_TIMESTAMP_COL,
    ]
    cols_to_show = [col for col in preferred_cols if col in nodes.columns]

    display_df = nodes.copy()

    if LAST_SEEN_COL in display_df.columns:
        display_df["last_seen_cst"] = display_df[LAST_SEEN_COL].apply(_to_display_tz_str)
    if TIMESTAMP_COL in display_df.columns:
        display_df["last_reading_cst"] = display_df[TIMESTAMP_COL].apply(
            _to_display_tz_str
        )
    if SYSTEM_HEALTH_TIMESTAMP_COL in display_df.columns:
        display_df["last_health_cst"] = display_df[SYSTEM_HEALTH_TIMESTAMP_COL].apply(
            _to_display_tz_str
        )

    if "last_seen_cst" in display_df.columns and LAST_SEEN_COL in cols_to_show:
        cols_to_show = [
            "last_seen_cst" if col == LAST_SEEN_COL else col for col in cols_to_show
        ]
    if "last_reading_cst" in display_df.columns and TIMESTAMP_COL in cols_to_show:
        cols_to_show = [
            "last_reading_cst" if col == TIMESTAMP_COL else col for col in cols_to_show
        ]
    if (
        "last_health_cst" in display_df.columns
        and SYSTEM_HEALTH_TIMESTAMP_COL in cols_to_show
    ):
        cols_to_show = [
            "last_health_cst" if col == SYSTEM_HEALTH_TIMESTAMP_COL else col
            for col in cols_to_show
        ]

    if cols_to_show:
        st.dataframe(display_df[cols_to_show].sort_values("node_id"), use_container_width=True)
    else:
        st.info("No displayable columns found yet—confirm nodes sheet headers.")


# ---------------- PAGE 2: NODE EXPLORER ----------------


elif page == "Node Explorer":
    st.title("Node Explorer")

    if readings.empty:
        st.warning(
            "No sensor readings yet (readings tab is currently headers-only). This page will populate once data arrives."
        )
        st.subheader("Current nodes (metadata)")
        meta_cols = [
            col
            for col in [
                "node_id",
                "hostname",
                LAST_SEEN_COL,
                "computed_status",
                STATUS_COL,
                "deployment_id",
                "site_id",
                MINUTES_SINCE_ACTIVE_COL,
            ]
            if col in nodes.columns
        ]
        if meta_cols:
            df = nodes.copy()
            if LAST_SEEN_COL in df.columns:
                df["last_seen_cst"] = df[LAST_SEEN_COL].apply(_to_display_tz_str)
                meta_cols = [
                    "last_seen_cst" if col == LAST_SEEN_COL else col for col in meta_cols
                ]
            st.dataframe(df[meta_cols].sort_values("node_id"), use_container_width=True)
        st.stop()

    node_ids = sorted(readings[NODE_ID_COL].dropna().astype("string").unique().tolist())
    if not node_ids:
        st.warning("No node IDs found in readings yet.")
        st.stop()

    selected_node = st.sidebar.selectbox("Node", node_ids)

    node_df = (
        readings[readings[NODE_ID_COL] == selected_node].sort_values(TIMESTAMP_COL).copy()
    )

    st.subheader(f"Time series for node {selected_node}")

    numeric_cols_node = node_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_node = [col for col in numeric_cols_node if col != NODE_ID_COL]

    default_candidates = [
        col
        for col in ["scd41_co2_ppm", "battery_voltage_v", "battery_percentage"]
        if col in numeric_cols_node
    ]
    default_metrics = (
        default_candidates[:1]
        if default_candidates
        else (numeric_cols_node[:1] if numeric_cols_node else [])
    )

    metrics_to_plot = st.multiselect(
        "Metrics to plot", numeric_cols_node, default=default_metrics
    )

    if metrics_to_plot:
        plot_df = node_df.set_index(TIMESTAMP_COL)[metrics_to_plot]
        st.line_chart(plot_df)
    else:
        st.info("Select one or more numeric metrics to plot.")

    st.subheader("Recent raw data")
    st.dataframe(node_df.tail(200), use_container_width=True)


# ---------------- PAGE 3: SYSTEM HEALTH ----------------


elif page == "System Health":
    st.title("System Health")

    if system_health.empty:
        st.warning("No system health rows found yet.")
        st.stop()

    node_ids = sorted(
        system_health[NODE_ID_COL].dropna().astype("string").unique().tolist()
    )
    selected_node = st.sidebar.selectbox("Health node", node_ids, key="health_node")

    health_df = (
        system_health[system_health[NODE_ID_COL] == selected_node]
        .sort_values(TIMESTAMP_COL)
        .copy()
    )

    numeric_cols_health = health_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_health = [col for col in numeric_cols_health if col != NODE_ID_COL]

    default_health_metrics = [
        col
        for col in [
            "system_health_cpu_temperature_c",
            "system_health_cpu_usage_percent",
            "system_health_memory_usage_percent",
            "system_health_disk_usage_percent",
            "system_health_network_latency_ms",
            "system_health_queue_pending_batches",
            "battery_voltage_v",
            "battery_percentage",
        ]
        if col in numeric_cols_health
    ]

    metrics_to_plot = st.multiselect(
        "Health metrics to plot",
        numeric_cols_health,
        default=default_health_metrics[:4],
        key="health_metrics",
    )

    if metrics_to_plot:
        plot_df = health_df.set_index(TIMESTAMP_COL)[metrics_to_plot]
        st.line_chart(plot_df)
    else:
        st.info("Select one or more system health metrics to plot.")

    latest_health = health_df.tail(1)
    if not latest_health.empty:
        st.subheader("Latest health snapshot")
        st.dataframe(latest_health, use_container_width=True)

    st.subheader("Recent system health data")
    st.dataframe(health_df.tail(200), use_container_width=True)


# ---------------- PAGE 4: FEATURE LAB ----------------


else:  # "Feature Lab"
    st.title("Feature Lab")

    if readings.empty:
        st.warning("No readings yet. This page will populate once sensor data arrives.")
        st.stop()

    min_ts = readings[TIMESTAMP_COL].min()
    max_ts = readings[TIMESTAMP_COL].max()

    st.markdown("Compare variance & correlations across two time windows")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Window A (baseline)")
        start_a = st.date_input(
            "Start A",
            min_ts.date(),
            min_value=min_ts.date(),
            max_value=max_ts.date(),
            key="start_A",
        )
        end_a = st.date_input(
            "End A",
            max_ts.date(),
            min_value=min_ts.date(),
            max_value=max_ts.date(),
            key="end_A",
        )

    with col_b:
        st.subheader("Window B (event / comparison)")
        start_b = st.date_input(
            "Start B",
            min_ts.date(),
            min_value=min_ts.date(),
            max_value=max_ts.date(),
            key="start_B",
        )
        end_b = st.date_input(
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
    default_metrics = [
        col for col in default_metric_order if col in numeric_cols_all
    ][:6]

    metrics = st.multiselect("Metrics to analyze", numeric_cols_all, default=default_metrics)

    if not metrics:
        st.info("Select at least one metric to analyze.")
        st.stop()

    ts_start_a = pd.to_datetime(start_a)
    ts_end_a = pd.to_datetime(end_a) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    ts_start_b = pd.to_datetime(start_b)
    ts_end_b = pd.to_datetime(end_b) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    mask_a = (readings[TIMESTAMP_COL] >= ts_start_a) & (
        readings[TIMESTAMP_COL] <= ts_end_a
    )
    mask_b = (readings[TIMESTAMP_COL] >= ts_start_b) & (
        readings[TIMESTAMP_COL] <= ts_end_b
    )

    df_a = readings.loc[mask_a, metrics]
    df_b = readings.loc[mask_b, metrics]

    st.subheader("Variance comparison")
    var_a = df_a.var(numeric_only=True)
    var_b = df_b.var(numeric_only=True)
    var_ratio = (var_b / (var_a + 1e-9)).sort_values(ascending=False)

    var_df = pd.DataFrame(
        {"var_A": var_a, "var_B": var_b, "var_ratio_B_over_A": var_ratio}
    ).sort_values("var_ratio_B_over_A", ascending=False)

    st.dataframe(var_df, use_container_width=True)

    st.subheader("Correlation matrix (Window B)")
    corr = df_b.corr(numeric_only=True)
    st.dataframe(corr.style.background_gradient(axis=None), use_container_width=True)
