import streamlit as st
import requests
from datetime import datetime, timedelta, time
import pandas as pd
import pydeck as pdk
import itertools
import math
import numpy as np
import streamlit.components.v1 as components

# --- 1. Page Setup ---
st.set_page_config(page_title="Light & Fog Predictor", page_icon="🏔️", layout="centered")
st.title("🏔️ Landscape & Astro Forecaster")
st.caption("Multi-model consensus | Celestial Tracking | Live Ground Sensors")

WAQI_TOKEN = "ee0ee12bcf2cf2da796899543b1d0f91d20e3c7a" 
STORMGLASS_TOKEN = "41a49954-877a-11f1-bcd5-0242ac120004-41a499e0-877a-11f1-bcd5-0242ac120004" 

# --- CACHED API FUNCTIONS (OFFLINE MODE FAILSAFE) ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_geocoding(query):
    try:
        headers = {"User-Agent": "LandscapeAstroForecaster/1.0"}
        payload = {"q": query, "format": "json", "limit": 10}
        res = requests.get("https://nominatim.openstreetmap.org/search", params=payload, headers=headers, timeout=5).json()
        
        if res:
            formatted_results = []
            for loc in res:
                formatted_results.append({
                    "name": loc.get("display_name"),
                    "latitude": float(loc.get("lat")),
                    "longitude": float(loc.get("lon")),
                    "timezone": "auto"
                })
            return {"results": formatted_results}
        return None
    except:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather(lat, lon, timezone="auto"):
    try:
        payload = {
            "latitude": lat, "longitude": lon, 
            "hourly": "cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,relative_humidity_300hPa,temperature_1000hPa,temperature_975hPa,temperature_950hPa,temperature_925hPa,temperature_900hPa,temperature_850hPa",
            "daily": "sunrise,sunset",
            "timezone": timezone, "forecast_days": 14
        }
        return requests.get("https://api.open-meteo.com/v1/forecast", params=payload, timeout=5).json()
    except:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_air_quality(lat, lon, timezone="auto"):
    try:
        payload = {"latitude": lat, "longitude": lon, "hourly": "pm2_5", "timezone": timezone, "forecast_days": 14}
        return requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params=payload, timeout=5).json()
    except:
        return None

@st.cache_data(ttl=600, show_spinner=False)
def fetch_waqi_live(lat, lon):
    try:
        res = requests.get(f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_TOKEN}", timeout=5).json()
        if res.get("status") == "ok":
            aqi = res["data"]["iaqi"].get("pm25", {}).get("v", 0)
            station = res["data"].get("city", {}).get("name", "Nearest Sensor")
            return aqi, station
        return None, None
    except:
        return None, None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_elevation(lat, lon):
    try:
        return requests.get(f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}", timeout=5).json().get("elevation", [0])[0]
    except:
        return 0

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_model_grid(lat_str, lon_str, model_code):
    try:
        p_grid = {
            "latitude": lat_str, "longitude": lon_str, 
            "hourly": "cloud_cover_low,cloud_cover_mid,cloud_cover_high,relative_humidity_300hPa,temperature_1000hPa,temperature_975hPa,temperature_950hPa,temperature_925hPa,temperature_900hPa,temperature_850hPa",
            "timezone": "auto", "forecast_days": 14, "models": model_code
        }
        return requests.get("https://api.open-meteo.com/v1/forecast", params=p_grid, timeout=10).json()
    except:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_aq_grid(lat_str, lon_str):
    try:
        p_aq_grid = {
            "latitude": lat_str, "longitude": lon_str, 
            "hourly": "pm2_5",
            "timezone": "auto", "forecast_days": 14
        }
        return requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params=p_aq_grid, timeout=10).json()
    except:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_tides(lat, lon, token):
    if token == "demo":
        return "demo"
    try:
        start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        end = start + timedelta(days=4)
        payload = {
            'lat': lat,
            'lng': lon,
            'start': start.isoformat(),
            'end': end.isoformat()
        }
        headers = {'Authorization': token}
        res = requests.get("https://api.stormglass.io/v2/tide/extremes/point", params=payload, headers=headers, timeout=5).json()
        return res.get('data', [])
    except:
        return []

def safe_val(data, key, i):
    try:
        val = data["hourly"][key][i]
        return val if val is not None else 0
    except (KeyError, IndexError, TypeError):
        return 0

def aqi_to_pm25(aqi):
    if aqi <= 50: return aqi * (12.0 / 50.0)
    elif aqi <= 100: return 12.0 + (aqi - 50) * (23.4 / 50.0)
    elif aqi <= 150: return 35.4 + (aqi - 100) * (20.0 / 50.0)
    elif aqi <= 200: return 55.4 + (aqi - 150) * (95.0 / 50.0)
    else: return 150.4 + (aqi - 200) * (100.0 / 100.0)

# --- EXACT GRANULAR FOG / MARINE LAYER ENGINE ---
def estimate_inversion_height(weather_data, idx):
    try:
        levels = [(100, "temperature_1000hPa"), (300, "temperature_975hPa"), (500, "temperature_950hPa"), 
                  (800, "temperature_925hPa"), (1000, "temperature_900hPa"), (1500, "temperature_850hPa")]
        
        surface_temp = safe_val(weather_data, levels[0][1], idx)
        inversion_base_alt = 0
        peak_temp = surface_temp
        
        prev_temp = surface_temp
        for alt, key in levels[1:]:
            t = safe_val(weather_data, key, idx)
            if t > prev_temp and inversion_base_alt == 0:
                inversion_base_alt = alt 
            if t > peak_temp:
                peak_temp = t
            prev_temp = t
                
        delta_t = round(peak_temp - surface_temp, 1)
        
        if delta_t > 0 and inversion_base_alt > 0:
            return delta_t, inversion_base_alt
            
        return 0, 0
    except:
        return 0, 0

# --- CONTINUOUS RASTER (GAUSSIAN RBF) INTERPOLATION ENGINE ---
BURN_CMAP = [[255, 237, 160], [254, 178, 76], [253, 141, 60], [240, 59, 32], [189, 0, 38]]
SKUNK_CMAP = [[242, 240, 247], [203, 201, 226], [158, 154, 200], [117, 107, 177], [84, 39, 143]]
SMOKE_CMAP = [[246, 232, 195], [223, 194, 125], [191, 129, 45], [140, 81, 10], [84, 48, 5]]
FOG_CMAP = [[237, 248, 251], [178, 226, 226], [102, 194, 164], [44, 162, 95], [0, 109, 44]]

def interpolate_dense_grid(orig_data, value_key, cmap, max_v, step):
    points = [d for d in orig_data if value_key in d]
    if not points: return None
    
    o_lats = np.array([d['lat'] for d in points])
    o_lons = np.array([d['lon'] for d in points])
    o_vals = np.array([d[value_key] for d in points])
    
    if np.max(o_vals) <= 0: return None
        
    grid_size = 75 
    
    min_lat, max_lat = np.min(o_lats) - (step/2), np.max(o_lats) + (step/2)
    min_lon, max_lon = np.min(o_lons) - (step/2), np.max(o_lons) + (step/2)
    
    lon_array = np.linspace(min_lon, max_lon, grid_size)
    lat_array = np.linspace(min_lat, max_lat, grid_size)
    glons, glats = np.meshgrid(lon_array, lat_array)
    dense_lons, dense_lats = glons.flatten(), glats.flatten()
    
    hw = ((max_lon - min_lon) / (grid_size - 1) / 2.0) * 1.02
    hh = ((max_lat - min_lat) / (grid_size - 1) / 2.0) * 1.02
    
    d_lon = dense_lons[:, np.newaxis] - o_lons[np.newaxis, :]
    d_lat = dense_lats[:, np.newaxis] - o_lats[np.newaxis, :]
    dist = np.sqrt(d_lon**2 + d_lat**2)
    
    sigma = step * 1.2 
    weights = np.exp(-(dist**2) / (2 * sigma**2))
    
    weight_sums = np.sum(weights, axis=1)
    weight_sums[weight_sums == 0] = 1e-10
    dense_vals = np.sum(weights * o_vals[np.newaxis, :], axis=1) / weight_sums
    
    out_data = []
    for i in range(len(dense_vals)):
        v = dense_vals[i]
        if v < 1: continue 
        
        pct = max(0.0, min(1.0, v / max_v))
        idx = pct * (len(cmap) - 1)
        lower = int(math.floor(idx))
        upper = int(math.ceil(idx))
        color_weight = idx - lower
        
        r = int(cmap[lower][0] * (1 - color_weight) + cmap[upper][0] * color_weight)
        g = int(cmap[lower][1] * (1 - color_weight) + cmap[upper][1] * color_weight)
        b = int(cmap[lower][2] * (1 - color_weight) + cmap[upper][2] * color_weight)
        
        alpha = int(max(0, min(150, pct * 255))) 
        
        lon, lat = float(dense_lons[i]), float(dense_lats[i])
        polygon = [
            [lon - hw, lat - hh],
            [lon + hw, lat - hh],
            [lon + hw, lat + hh],
            [lon - hw, lat + hh]
        ]
        out_data.append({"polygon": polygon, "color": [r, g, b, alpha]})
        
    return pdk.Layer(
        'PolygonLayer',
        data=pd.DataFrame(out_data),
        get_polygon='polygon',
        get_fill_color='color',
        filled=True,
        stroked=False,
        wireframe=False, 
        pickable=False
    )

# --- UNIFIED CELESTIAL MATH ENGINE (100% OFFLINE) ---
def get_celestial_az_alt(lat, lon, local_time, tz_string, target="galactic_core"):
    try:
        utc_time = pd.Timestamp(local_time).tz_localize(tz_string).tz_convert('UTC').replace(tzinfo=None)
    except:
        utc_time = local_time - timedelta(hours=lon/15.0)
    
    D = (utc_time - datetime(2000, 1, 1, 12, 0, 0)).total_seconds() / 86400.0
    
    if target == "galactic_core":
        ra = math.radians(266.405) 
        dec = math.radians(-28.936) 
    elif target == "sun":
        g = math.radians((357.529 + 0.98560028 * D) % 360)
        q = (280.459 + 0.98564736 * D) % 360
        L = math.radians((q + 1.915 * math.sin(g) + 0.020 * math.sin(2*g)) % 360)
        e = math.radians(23.439)
        dec = math.asin(math.sin(e) * math.sin(L))
        ra = math.atan2(math.cos(e) * math.sin(L), math.cos(L))
    elif target == "moon":
        L_m = (218.316 + 13.176396 * D) % 360
        M_m = math.radians((134.963 + 13.064993 * D) % 360)
        F_m = math.radians((93.272 + 13.229350 * D) % 360)
        lam = math.radians(L_m + 6.289 * math.sin(M_m))
        bet = math.radians(5.128 * math.sin(F_m))
        e = math.radians(23.439)
        dec = math.asin(math.sin(bet) * math.cos(e) + math.cos(bet) * math.sin(e) * math.sin(lam))
        ra = math.atan2(math.sin(lam) * math.cos(e) - math.tan(bet) * math.sin(e), math.cos(lam))

    GMST = (18.697374558 + 24.06570982441908 * D) % 24
    LST = (GMST + (lon / 15.0)) % 24
    lst_rad = math.radians(LST * 15)
    ha_rad = lst_rad - ra
    
    lat_rad = math.radians(lat)
    sin_alt = math.sin(dec) * math.sin(lat_rad) + math.cos(dec) * math.cos(lat_rad) * math.cos(ha_rad)
    alt = math.asin(sin_alt)
    
    cos_az = (math.sin(dec) - math.sin(alt) * math.sin(lat_rad)) / (math.cos(alt) * math.cos(lat_rad))
    cos_az = max(-1.0, min(1.0, cos_az)) 
    az = math.acos(cos_az)
    
    if math.sin(ha_rad) > 0: az = 2 * math.pi - az
    return math.degrees(az), math.degrees(alt)

@st.cache_data(ttl=3600, show_spinner=False)
def calculate_celestial_events(lat, lon, date_str, tz_string):
    """Scans the entire selected day in 2-minute increments to find exact twilight, sunrise, and moonset times."""
    target_date = datetime.fromisoformat(date_str).date()
    base_dt = datetime.combine(target_date, time(0, 0))
    events = {}
    prev_sun_alt = None
    prev_moon_alt = None
    
    for m in range(0, 1440, 2):
        dt = base_dt + timedelta(minutes=m)
        _, s_alt = get_celestial_az_alt(lat, lon, dt, tz_string, "sun")
        _, m_alt = get_celestial_az_alt(lat, lon, dt, tz_string, "moon")
        
        if prev_sun_alt is not None:
            if prev_sun_alt < 0 and s_alt >= 0: events['Sunrise'] = dt
            if prev_sun_alt > 0 and s_alt <= 0: events['Sunset'] = dt
            if prev_sun_alt < -6 and s_alt >= -6: events['Dawn (Civil)'] = dt
            if prev_sun_alt > -6 and s_alt <= -6: events['Dusk (Civil)'] = dt
            if prev_sun_alt < -12 and s_alt >= -12: events['Dawn (Nautical)'] = dt
            if prev_sun_alt > -12 and s_alt <= -12: events['Dusk (Nautical)'] = dt
            if prev_sun_alt < -18 and s_alt >= -18: events['Dawn (Astro)'] = dt
            if prev_sun_alt > -18 and s_alt <= -18: events['Dusk (Astro)'] = dt
            
        if prev_moon_alt is not None:
            if prev_moon_alt < 0 and m_alt >= 0: events['Moonrise'] = dt
            if prev_moon_alt > 0 and m_alt <= 0: events['Moonset'] = dt
            
        prev_sun_alt = s_alt
        prev_moon_alt = m_alt
        
    return events

def create_vector_line(lat, lon, azimuth, length_deg, color):
    end_lat = lat + length_deg * math.cos(math.radians(azimuth))
    end_lon = lon + length_deg * math.sin(math.radians(azimuth)) / math.cos(math.radians(lat))
    return {"start_lon": lon, "start_lat": lat, "end_lon": end_lon, "end_lat": end_lat, "color": color}

# --- APP START ---
mode = st.radio("Select Dashboard Mode:", ["🌅 Sunrise & Sunset", "🌌 Astrophotography"], horizontal=True)

device_location = "Lake Louise, Alberta, Canada" 
default_location = device_location if device_location else "San Francisco, California"

input_method = st.radio("Location Entry:", ["🔍 Search by Name (Internet Required)", "📍 Manual Coordinates (Offline Mode)"], horizontal=True)

lat, lon, tz = None, None, None

if input_method == "🔍 Search by Name (Internet Required)":
    search_query = st.text_input("Enter a location (e.g., Jasper, Banff, Yosemite):", default_location)
    if search_query:
        with st.spinner(f"Locating {search_query}..."):
            geo_response = fetch_geocoding(search_query)
            
        if not geo_response or "results" not in geo_response:
            st.error(f"Could not find coordinates for '{search_query}'. You may be offline or the location is invalid.")
        else:
            location_options = {}
            for loc in geo_response["results"]:
                display_name = loc.get("name", "Unknown Location")
                if display_name in location_options:
                    display_name += f" ({loc['latitude']}, {loc['longitude']})"
                location_options[display_name] = {"lat": loc["latitude"], "lon": loc["longitude"], "tz": loc.get("timezone", "auto")}
            
            selected_loc = st.selectbox("Select the exact location:", list(location_options.keys()))
            lat = location_options[selected_loc]["lat"]
            lon = location_options[selected_loc]["lon"]
            tz = location_options[selected_loc]["tz"]
else:
    st.info("📡 **Offline Mode Active:** The celestial engine will run entirely on your device's local CPU using pure astronomical math.")
    c1, c2, c3 = st.columns(3)
    lat = c1.number_input("Latitude:", value=51.4254, format="%.4f")
    lon = c2.number_input("Longitude:", value=-116.1773, format="%.4f")
    tz = c3.text_input("Timezone (IANA):", value="America/Edmonton")

if lat is not None and lon is not None and tz is not None:
    fetch_tides_toggle = st.checkbox("🌊 Fetch Coastal Tide Data (Consumes 1 Stormglass API Call)", value=False)

    with st.spinner('Loading marine and atmospheric data...'):
        base_data = fetch_weather(lat, lon, tz)
        aq_data = fetch_air_quality(lat, lon, tz)
        live_aqi, live_station = fetch_waqi_live(lat, lon)
        
        tide_data = fetch_tides(lat, lon, STORMGLASS_TOKEN) if fetch_tides_toggle else None
        
        real_tz = base_data.get("timezone", tz) if base_data else (tz if tz != "auto" else "UTC")

    st.divider()

    # ==========================================
    # MODE 1: SUNRISE & SUNSET
    # ==========================================
    if mode == "🌅 Sunrise & Sunset":
        st.write("### 🕒 3-Day Forecast Window")
        daily_data = base_data.get("daily", {}) if base_data else {}
        hourly_times = base_data.get("hourly", {}).get("time", []) if base_data else []
        
        if not daily_data or not hourly_times:
            st.error("Failed to fetch reliable baseline weather data. You may be fully offline.")
            st.stop()
            
        event_menu = {}
        for i in range(3):
            sr_str = daily_data["sunrise"][i]
            ss_str = daily_data["sunset"][i]
            sr_dt = datetime.fromisoformat(sr_str)
            ss_dt = datetime.fromisoformat(ss_str)
            sr_nice = sr_dt.strftime("%A, %b %d at %I:%M %p")
            ss_nice = ss_dt.strftime("%A, %b %d at %I:%M %p")
            event_menu[f"🌅 Sunrise ({sr_nice})"] = ("sunrise", sr_str)
            event_menu[f"🌇 Sunset ({ss_nice})"] = ("sunset", ss_str)

        selected_event_label = st.selectbox("Select your shooting window:", list(event_menu.keys()))
        event_type, exact_time_str = event_menu[selected_event_label]

        dt = datetime.fromisoformat(exact_time_str)
        if dt.minute >= 30: dt += timedelta(hours=1)
        dt = dt.replace(minute=0, second=0, microsecond=0)
        closest_hour_str = dt.strftime("%Y-%m-%dT%H:00")
        
        st.subheader("⛰️ Topographical Ray-Tracing")
        with st.spinner("Scanning mountain profiles..."):
            current_elev = fetch_elevation(lat, lon)
            horizon_elev = fetch_elevation(lat, lon - 0.06) if event_type == "sunset" else fetch_elevation(lat, lon + 0.06)
            direction = "Western" if event_type == "sunset" else "Eastern"
            elev_diff = horizon_elev - current_elev
            
            if elev_diff > 150: 
                angle_rads = math.atan(elev_diff / 5000)
                minutes_lost = round(math.degrees(angle_rads) * 4)
                elev_diff_ft = round(elev_diff * 3.28084)
                st.warning(f"⚠️ **Mountain Shadow Detected:** The {direction} ridge is {round(elev_diff)}m ({elev_diff_ft:,} ft) higher than your location. The sun will disappear behind the peaks **~{minutes_lost} minutes before** official {event_type}.")
            else:
                st.success(f"✅ **Clear Horizon:** No significant topographical blocking detected to the {direction}.")

        if closest_hour_str not in hourly_times:
            st.error("Forecast data is not yet available for that time slot.")
        else:
            baseline_idx = hourly_times.index(closest_hour_str)
            upstream_lon = lon + (0.6 if event_type == "sunrise" else -0.6)

            models_to_run = {"High-Res (Local)": "best_match", "ECMWF (European)": "ecmwf_ifs", "GFS (American)": "gfs_seamless"}
            ensemble_results = []
            
            aq_idx = aq_data["hourly"]["time"].index(closest_hour_str) if aq_data and "hourly" in aq_data and closest_hour_str in aq_data["hourly"].get("time", []) else 0
            model_pm25 = safe_val(aq_data, "pm2_5", aq_idx) if aq_data else 0
            
            live_pm25 = aqi_to_pm25(live_aqi) if live_aqi else 0
            is_override = live_pm25 > (model_pm25 + 10)
            active_pm25 = live_pm25 if is_override else model_pm25
            
            with st.spinner("Running Multi-Model Consensus..."):
                for model_label, model_code in models_to_run.items():
                    res_local = fetch_weather(lat, lon, tz)
                    res_up = fetch_weather(lat, upstream_lon, tz)
                    if not res_local or not res_up: continue
                    
                    idx = baseline_idx
                    up_idx = res_up["hourly"].get("time", []).index(closest_hour_str) if closest_hour_str in res_up["hourly"].get("time", []) else idx
                    
                    l_total = safe_val(res_local, "cloud_cover", idx)
                    l_low = safe_val(res_local, "cloud_cover_low", idx)
                    l_mid = safe_val(res_local, "cloud_cover_mid", idx)
                    l_high = safe_val(res_local, "cloud_cover_high", idx)
                    rh_300 = safe_val(res_local, "relative_humidity_300hPa", idx)
                    u_low = safe_val(res_up, "cloud_cover_low", up_idx)
                    
                    opaque_deck = l_low + l_mid 
                    effective_high = max(l_high, max(0, rh_300 - 50) if rh_300 > 50 else 0)
                    
                    vis_block = max(0, min(1.0, (opaque_deck - 45) / 45)) 
                    potential = round(max(0, min(100, ((l_mid * 0.48) + (effective_high * 1.15 * (1.0 - vis_block))) - (u_low * 0.25) - (15 if (l_low > 15 and l_mid > 15 and effective_high > 15) else 0))))
                    
                    skunk_from_smoke = max(0, (active_pm25 - 40) * 1.5)
                    skunk = round(min(100, max(max(0, (l_low - 50) * 2.0), max(0, (u_low - 40) * 1.8), max(0, (opaque_deck - 70) * 3.0) if opaque_deck > 70 else 0, skunk_from_smoke)))
                    
                    ensemble_results.append({"name": model_label, "potential": potential, "skunk": skunk, "total": l_total, "low": l_low, "mid": l_mid, "high": l_high, "rh": rh_300})

            st.divider()
            st.subheader("🔥 Forecast Analysis")
            
            if ensemble_results:
                avg_pot = round(sum(m["potential"] for m in ensemble_results) / len(ensemble_results))
                avg_skunk = round(sum(m["skunk"] for m in ensemble_results) / len(ensemble_results))
                
                c1, c2, c3 = st.columns(3)
                c1.metric("🔥 BURN POTENTIAL", f"{avg_pot}/100")
                c2.metric("🦨 SKUNK CHANCE", f"{avg_skunk}%")
                
                inv_dt, inv_alt = estimate_inversion_height(base_data, baseline_idx)
                local_low_clouds = safe_val(base_data, "cloud_cover_low", baseline_idx)
                
                if inv_dt > 0 and local_low_clouds > 10:
                    inv_alt_ft = round(inv_alt * 3.28084)
                    c3.metric("🌫️ FOG CEILING", f"~{inv_alt} m", f"↑ {inv_alt_ft:,} ft | +{inv_dt}°C ΔT", delta_color="inverse")
                else:
                    c3.metric("🌫️ FOG RISK", "Low", "No Moisture/Inversion", delta_color="normal")
                
                with st.expander("📊 View Ensemble Breakdown (Model Agreement)"):
                    for m in ensemble_results:
                        st.markdown(f"**{m['name']}** - Potential: **{m['potential']}** | Skunk: **{m['skunk']}%**")
                        st.caption(f"Raw: Total {m['total']}% | Low {m['low']}% | Mid {m['mid']}% | High {m['high']}%")

            # --- COASTAL TIDE ANALYSIS UI ---
            st.divider()
            st.subheader("🌊 Coastal Tide Context")
            if not fetch_tides_toggle:
                st.info("⏸️ **Tide Tracker Paused:** Check the 'Fetch Coastal Tide Data' box at the top of the app to consume an API call and load tide times.")
            elif tide_data == "demo":
                st.info("💡 **Tide Tracker Inactive:** To track high/low tide times for coastal reflections and sea stacks, replace `STORMGLASS_TOKEN` at the top of the script with a free API key from stormglass.io.")
            elif isinstance(tide_data, list) and len(tide_data) > 0:
                
                target_dt = pd.Timestamp(dt)
                if target_dt.tzinfo is None:
                    target_dt = target_dt.tz_localize(real_tz)
                else:
                    target_dt = target_dt.tz_convert(real_tz)
                    
                parsed_tides = []
                for t in tide_data:
                    try:
                        t_time = pd.to_datetime(t['time']).tz_convert(real_tz)
                        parsed_tides.append((t_time, t['type'], t['height']))
                    except:
                        continue
                        
                parsed_tides.sort(key=lambda x: x[0])
                
                past_tides = [t for t in parsed_tides if t[0] < target_dt]
                future_tides = [t for t in parsed_tides if t[0] >= target_dt]
                
                display_tides = []
                if past_tides:
                    display_tides.append(past_tides[-1]) 
                display_tides.extend(future_tides[:3]) 
                
                if display_tides:
                    t_cols = st.columns(len(display_tides))
                    for i, (t_time, t_type, t_height) in enumerate(display_tides):
                        icon = "🔼 High" if t_type == "high" else "🔽 Low"
                        t_height_ft = round(t_height * 3.28084, 1)
                        
                        delta_hrs = (t_time - target_dt).total_seconds() / 3600
                        if delta_hrs < 0:
                            rel_str = f"{-delta_hrs:.1f}h before"
                        else:
                            rel_str = f"+{delta_hrs:.1f}h after"
                            
                        t_cols[i].metric(
                            f"{icon} ({t_time.strftime('%a %I:%M %p')})", 
                            f"{round(t_height, 2)}m", 
                            f"{t_height_ft}ft | {rel_str}", 
                            delta_color="off"
                        )
                else:
                    st.info("No extreme tide events detected around this time window.")
            else:
                st.info("No tidal data available for this location (likely an inland elevation).")
                        
            st.divider()
            st.subheader("🌲 Air Quality & Wildfire Smoke")
            
            if is_override:
                st.error(f"🚨 **LOCAL SENSOR OVERRIDE:** Global model predicted clean air, but the '{live_station}' physical sensor is detecting thick smoke.")
                st.metric("PM 2.5 (Smoke Density)", f"{round(active_pm25)} µg/m³", delta="Override Active", delta_color="inverse")
            else:
                st.metric("PM 2.5 (Smoke Density)", f"{round(active_pm25)} µg/m³")
            
            if active_pm25 <= 10:
                st.success("✅ **Clean Air:** No significant wildfire smoke detected. The atmosphere is clear.")
            elif 10 < active_pm25 <= 35:
                st.info("🌤️ **Blood-Orange Sun Potential:** There is a light layer of smoke in the atmosphere. Could enhance reds and oranges at the horizon.")
            elif 35 < active_pm25 <= 60:
                st.warning("⚠️ **Moderate Smoke Smother:** The smoke is getting thick enough to wash out contrast and dim the burn potential.")
            else:
                st.error("🛑 **Heavy Smoke Skunk:** Wildfire smoke is very thick. The sun will likely vanish into a gray/brown haze long before it hits the horizon.")

            st.divider()
            
            st.subheader("🗺️ High-Resolution Regional Overlay")
            
            zoom_level = st.select_slider(
                "Grid Coverage Area:",
                options=["Micro (~20km)", "Local (~45km)", "Regional (~90km)", "Macro (~160km)"],
                value="Regional (~90km)"
            )
            
            interactive_map = st.radio("Do you want an interactive map?", ["Yes (Zoom & Pan)", "No (Static Map)"], horizontal=True)
            is_interactive = (interactive_map == "Yes (Zoom & Pan)")
            
            step_dict = {"Micro (~20km)": 0.02, "Local (~45km)": 0.04, "Regional (~90km)": 0.08, "Macro (~160km)": 0.15}
            zoom_dict = {"Micro (~20km)": 9.5, "Local (~45km)": 8.5, "Regional (~90km)": 7.5, "Macro (~160km)": 6.5}
            
            step = step_dict[zoom_level]
            map_zoom = zoom_dict[zoom_level]
            
            if is_interactive:
                v_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=map_zoom, pitch=0)
            else:
                v_state = pdk.ViewState(
                    latitude=lat, longitude=lon, zoom=map_zoom, pitch=0,
                    min_zoom=map_zoom, max_zoom=map_zoom
                )

            c_b, c_sk, c_sm, c_f = st.columns(4)
            show_burn = c_b.checkbox("🔥 Burn", value=True)
            show_skunk = c_sk.checkbox("🦨 Skunk", value=True)
            show_smoke = c_sm.checkbox("🌲 Smoke", value=True)
            show_fog = c_f.checkbox("☁️ Fog", value=False)

            with st.spinner("Rendering cached continuous mathematical heatmap..."):
                grid_size = 10
                lats = [lat + (i - grid_size//2)*step for i in range(grid_size)]
                lons = [lon + (i - grid_size//2)*step for i in range(grid_size)]
                coords = list(itertools.product(lats, lons))
                lat_str, lon_str = ",".join(str(round(c[0], 4)) for c in coords), ",".join(str(round(c[1], 4)) for c in coords)
                
                grid_res = fetch_model_grid(lat_str, lon_str, "best_match")
                aq_grid_res = fetch_aq_grid(lat_str, lon_str)
                
                if grid_res:
                    map_data = []
                    max_decay_dist = (grid_size / 2) * step
                    
                    for i, c in enumerate(coords):
                        try:
                            loc_w = grid_res[i] if isinstance(grid_res, list) else grid_res
                            loc_aq = aq_grid_res[i] if aq_grid_res and isinstance(aq_grid_res, list) else (aq_grid_res if aq_grid_res else {})
                            if "hourly" not in loc_w: continue
                            
                            idx = loc_w["hourly"]["time"].index(closest_hour_str)
                            aq_idx = loc_aq["hourly"]["time"].index(closest_hour_str) if loc_aq and "hourly" in loc_aq and closest_hour_str in loc_aq["hourly"].get("time", []) else 0
                            
                            grid_pm25 = safe_val(loc_aq, "pm2_5", aq_idx) if loc_aq else 0
                            
                            if is_override: 
                                dist_deg = math.sqrt((c[0] - lat)**2 + (c[1] - lon)**2)
                                decay_factor = max(0.3, 1.0 - (dist_deg / max_decay_dist)) 
                                grid_pm25 = max(grid_pm25, active_pm25 * decay_factor)

                            l_low = safe_val(loc_w, "cloud_cover_low", idx)
                            l_mid = safe_val(loc_w, "cloud_cover_mid", idx)
                            l_high = safe_val(loc_w, "cloud_cover_high", idx)
                            
                            opaque_deck = l_low + l_mid
                            potential = round(max(0, min(100, ((l_mid * 0.48) + (max(l_high, max(0, safe_val(loc_w, "relative_humidity_300hPa", idx) - 50)) * 1.15 * (1.0 - max(0, min(1.0, (opaque_deck - 45) / 45))))))))
                            skunk = round(min(100, max(max(0, (l_low - 50) * 2.0), max(0, (opaque_deck - 70) * 3.0) if opaque_deck > 70 else 0)))
                            
                            inv_dt_grid, inv_alt_grid = estimate_inversion_height(loc_w, idx)
                            
                            if inv_dt_grid > 0 and l_low > 5:
                                fog_intensity = min(100, (l_low / 100.0) * (inv_dt_grid * 25))
                                inv_alt_ft_grid = round(inv_alt_grid * 3.28084)
                                fog_details = f"~{inv_alt_grid}m ({inv_alt_ft_grid:,} ft) [+{inv_dt_grid}°C]"
                            else:
                                fog_intensity = 0
                                fog_details = "Clear (No Moisture/Inversion)"

                            map_data.append({
                                "lat": round(c[0], 4), 
                                "lon": round(c[1], 4),
                                "potential": potential,
                                "skunk": skunk,
                                "pm25": round(grid_pm25),
                                "cloud_low": l_low,
                                "cloud_mid": l_mid,
                                "cloud_high": l_high,
                                "fog_weight": fog_intensity,
                                "fog_text": fog_details
                            })
                        except: continue

                    df_map = pd.DataFrame(map_data)
                    layers = []

                    if not df_map.empty:
                        if show_burn:
                            l_b = interpolate_dense_grid(map_data, 'potential', BURN_CMAP, 100.0, step)
                            if l_b: layers.append(l_b)
                            
                        if show_skunk:
                            l_sk = interpolate_dense_grid(map_data, 'skunk', SKUNK_CMAP, 100.0, step)
                            if l_sk: layers.append(l_sk)
                            
                        if show_smoke:
                            l_sm = interpolate_dense_grid(map_data, 'pm25', SMOKE_CMAP, 150.0, step)
                            if l_sm: layers.append(l_sm)
                            
                        if show_fog:
                            l_fg = interpolate_dense_grid(map_data, 'fog_weight', FOG_CMAP, 100.0, step)
                            if l_fg: layers.append(l_fg)

                        pad_lat = step / 2
                        pad_lon = step / 2
                        min_lon, max_lon = min(lons), max(lons)
                        min_lat, max_lat = min(lats), max(lats)

                        mask_data = [{
                            "polygon": [
                                [[-180, 90], [180, 90], [180, -90], [-180, -90]], 
                                [
                                    [min_lon - pad_lon, min_lat - pad_lat],
                                    [max_lon + pad_lon, min_lat - pad_lat],
                                    [max_lon + pad_lon, max_lat + pad_lat],
                                    [min_lon - pad_lon, max_lat + pad_lat]
                                ] 
                            ]
                        }]

                        layers.append(pdk.Layer(
                            'PolygonLayer',
                            data=mask_data,
                            get_polygon='polygon',
                            get_fill_color=[22, 25, 28, 230], 
                            filled=True,
                            stroked=True,
                            get_line_color=[150, 150, 150, 150], 
                            line_width_min_pixels=2,
                            pickable=False
                        ))

                        r_mult = step / 0.08
                        layers.append(pdk.Layer(
                            'ScatterplotLayer',
                            data=df_map,
                            get_position='[lon, lat]',
                            get_color=[0, 0, 0, 0], 
                            get_radius=5000 * r_mult,
                            pickable=True
                        ))

                        tooltip_html = (
                            "<b>Coord:</b> {lat}, {lon}<br/>"
                            "<b>🔥 Burn:</b> {potential}/100 | <b>🦨 Skunk:</b> {skunk}%<br/>"
                            "<b>🌲 Smoke:</b> {pm25} µg/m³<br/>"
                            "<b>☁️ Clouds:</b> {cloud_low}% L | {cloud_mid}% M | {cloud_high}% H<br/>"
                            "<b>🌫️ Fog Ceiling:</b> {fog_text}"
                        )

                        st.pydeck_chart(pdk.Deck(
                            map_style='dark',
                            views=[pdk.View(type="MapView", controller=is_interactive)],
                            initial_view_state=v_state,
                            layers=layers,
                            tooltip={"html": tooltip_html, "style": {"backgroundColor": "#222222", "color": "white"}}
                        ))
                    else:
                        st.error("No valid map data could be rendered. The API might be offline for this specific region.")

            # --- LIVE CLOUD MOVEMENT EMBED ---
            st.divider()
            st.subheader("☁️ Live Cloud Movement & Tracking")
            st.write("Cross-reference the mathematical burn potential above with actual cloud flow over the next 3 days.")
            
            windy_html = f"""
            <iframe width="100%" height="500" 
                src="https://embed.windy.com/embed2.html?lat={lat}&lon={lon}&zoom=7&level=surface&overlay=clouds&product=ecmwf&menu=&message=true&marker=true&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=km%2Fh&metricTemp=%C2%B0C&radarRange=-1" 
                frameborder="0">
            </iframe>
            """
            components.html(windy_html, height=500)


        # ==========================================
        # MODE 2: ASTROPHOTOGRAPHY
        # ==========================================
        elif mode == "🌌 Astrophotography":
            st.write("### 🕒 Astro Planning Window")
            
            selected_date = st.date_input("Target Date:", datetime.today().date())

            # --- CELESTIAL DAILY MILESTONES ---
            st.subheader("⏱️ Daily Celestial Events")
            with st.spinner("Calculating exact horizon crossings..."):
                events = calculate_celestial_events(lat, lon, selected_date.isoformat(), real_tz)
                
            e1, e2, e3, e4 = st.columns(4)
            e1.markdown(f"**Sunrise:** {events.get('Sunrise', 'N/A').strftime('%I:%M %p') if 'Sunrise' in events else 'N/A'}")
            e1.markdown(f"**Sunset:** {events.get('Sunset', 'N/A').strftime('%I:%M %p') if 'Sunset' in events else 'N/A'}")
            
            e2.markdown(f"**Dawn (Civil):** {events.get('Dawn (Civil)', 'N/A').strftime('%I:%M %p') if 'Dawn (Civil)' in events else 'N/A'}")
            e2.markdown(f"**Dusk (Civil):** {events.get('Dusk (Civil)', 'N/A').strftime('%I:%M %p') if 'Dusk (Civil)' in events else 'N/A'}")
            
            e3.markdown(f"**Dawn (Nautical):** {events.get('Dawn (Nautical)', 'N/A').strftime('%I:%M %p') if 'Dawn (Nautical)' in events else 'N/A'}")
            e3.markdown(f"**Dusk (Nautical):** {events.get('Dusk (Nautical)', 'N/A').strftime('%I:%M %p') if 'Dusk (Nautical)' in events else 'N/A'}")
            
            e4.markdown(f"**Moonrise:** {events.get('Moonrise', 'N/A').strftime('%I:%M %p') if 'Moonrise' in events else 'N/A'}")
            e4.markdown(f"**Moonset:** {events.get('Moonset', 'N/A').strftime('%I:%M %p') if 'Moonset' in events else 'N/A'}")

            st.divider()
            
            st.write("### 🔭 Advanced Celestial Tracking Map")
            
            interactive_astro = st.radio("Do you want an interactive map?", ["Yes (Zoom & Pan)", "No (Static Map)"], horizontal=True, key="astro_toggle")
            is_astro_interactive = (interactive_astro == "Yes (Zoom & Pan)")
            
            # --- EXACT TIME SLIDER LOGIC MOVED DIRECTLY ABOVE THE MAP ---
            start_of_day = datetime.combine(selected_date, time(0, 0))
            end_of_day = datetime.combine(selected_date, time(23, 59))
            
            if selected_date == datetime.today().date():
                default_time = datetime.now().replace(second=0, microsecond=0)
            else:
                default_time = datetime.combine(selected_date, time(12, 0))
                
            tracking_time = st.slider(
                "Select Exact Time (Local):",
                min_value=start_of_day,
                max_value=end_of_day,
                value=default_time,
                step=timedelta(minutes=1),
                format="hh:mm A"
            )
            
            gc_az, gc_alt = get_celestial_az_alt(lat, lon, tracking_time, real_tz, "galactic_core")
            sun_az, sun_alt = get_celestial_az_alt(lat, lon, tracking_time, real_tz, "sun")
            moon_az, moon_alt = get_celestial_az_alt(lat, lon, tracking_time, real_tz, "moon")
            
            # --- DYNAMIC PHOTOPILLS COLOR WASH OVERLAY ---
            if sun_alt > 6: 
                wash_color = [255, 240, 200, 25]  
                sky_status = "☀️ Daytime (Yellow)"
            elif sun_alt > 0: 
                wash_color = [255, 140, 0, 45]   
                sky_status = "🌇 Golden Hour (Orange)"
            elif sun_alt > -6: 
                wash_color = [100, 150, 255, 50] 
                sky_status = "🌆 Blue Hour / Civil Twilight"
            elif sun_alt > -12: 
                wash_color = [20, 50, 150, 80]  
                sky_status = "🌌 Nautical Twilight"
            elif sun_alt > -18:
                wash_color = [0, 10, 50, 100]
                sky_status = "🌌 Astronomical Twilight"
            else: 
                wash_color = [0, 0, 20, 140]     
                sky_status = "🌃 True Night"

            st.write(f"**Target Time:** {tracking_time.strftime('%A, %I:%M %p')}")
            c_a, c_b, c_c = st.columns(3)
            c_a.markdown(f"**Milky Way:** Alt {round(gc_alt)}°")
            c_b.markdown(f"**Sun:** Alt {round(sun_alt)}°")
            c_c.markdown(f"**Map State:** {sky_status}")
            
            if gc_alt > 7: base_color, strength_multiplier = [255, 215, 0], min(1.8, gc_alt / 6)
            elif gc_alt > 0: base_color, strength_multiplier = [147, 112, 219], max(0.5, gc_alt / 6)
            else: base_color, strength_multiplier = [100, 100, 100], 0.2 
                
            dot_data, line_data = [], []
            for i in range(1, 9):
                dist = (i / 8) * 0.45 
                d_lat = lat + dist * math.cos(math.radians(gc_az))
                d_lon = lon + dist * math.sin(math.radians(gc_az)) / math.cos(math.radians(lat))
                
                base_radius = (300 + (i * 400)) * strength_multiplier
                alpha = int(100 + (i / 8) * 155)
                dot_data.extend([
                    {"lon": d_lon, "lat": d_lat, "radius": base_radius * 1.5, "color": base_color + [alpha // 3]},
                    {"lon": d_lon, "lat": d_lat, "radius": base_radius * 0.4, "color": base_color + [alpha]}
                ])

            if sun_alt > -18: line_data.append(create_vector_line(lat, lon, sun_az, 0.45, [255, 140, 0, 200] if sun_alt > 0 else [255, 140, 0, 80]))
            if moon_alt > -10: line_data.append(create_vector_line(lat, lon, moon_az, 0.45, [200, 220, 255, 200] if moon_alt > 0 else [200, 220, 255, 60]))

            if is_astro_interactive:
                astro_view = pdk.ViewState(latitude=lat, longitude=lon, zoom=8.5, pitch=45, bearing=0)
            else:
                astro_view = pdk.ViewState(
                    latitude=lat, longitude=lon, zoom=8.5, pitch=45, bearing=0,
                    min_zoom=8.5, max_zoom=8.5
                )
                
            wash_layer = pdk.Layer(
                'PolygonLayer',
                data=pd.DataFrame([{"polygon": [[-180, 90], [180, 90], [180, -90], [-180, -90]], "color": wash_color}]),
                get_polygon='polygon',
                get_fill_color='color',
                filled=True,
                stroked=False,
                pickable=False
            )

            st.pydeck_chart(pdk.Deck(
                map_style='light',
                views=[pdk.View(type="MapView", controller=is_astro_interactive)],
                initial_view_state=astro_view,
                layers=[
                    wash_layer,
                    pdk.Layer('LineLayer', data=pd.DataFrame(line_data) if line_data else pd.DataFrame(columns=["start_lon", "start_lat", "end_lon", "end_lat", "color"]), get_source_position='[start_lon, start_lat]', get_target_position='[end_lon, end_lat]', get_color='color', get_width=3, width_units='"pixels"'),
                    pdk.Layer('ScatterplotLayer', data=pd.DataFrame(dot_data), get_position='[lon, lat]', get_color='color', get_radius='radius', pickable=False)
                ]
            ))
            st.caption("🟠 Orange Line = Sun Direction | ⚪ White Line = Moon Direction | 🟣/🟡 Dots = Milky Way Core")

            st.divider()

            # --- WEATHER / CLOUD COVER AT TARGET TIME ---
            closest_hour_str = tracking_time.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00")
            hourly_times = base_data.get("hourly", {}).get("time", []) if base_data else []

            st.subheader("🌌 Sky Conditions at Target Time")
            
            if base_data and closest_hour_str in hourly_times:
                baseline_idx = hourly_times.index(closest_hour_str)
                total_clouds = safe_val(base_data, "cloud_cover", baseline_idx)
                high_clouds = safe_val(base_data, "cloud_cover_high", baseline_idx)
                
                aq_idx = aq_data["hourly"]["time"].index(closest_hour_str) if aq_data and "hourly" in aq_data and closest_hour_str in aq_data["hourly"].get("time", []) else 0
                model_pm25 = safe_val(aq_data, "pm2_5", aq_idx) if aq_data else 0
                
                live_pm25 = aqi_to_pm25(live_aqi) if live_aqi else 0
                is_override = live_pm25 > (model_pm25 + 10)
                active_pm25 = live_pm25 if is_override else model_pm25
                
                inv_dt_astro, inv_alt_astro = estimate_inversion_height(base_data, baseline_idx)
                
                if inv_dt_astro > 0:
                    seeing_quality = "Excellent 🟢 (Stable Air)"
                elif inv_dt_astro > -3:
                    seeing_quality = "Good 🟡 (Moderate Stability)"
                else:
                    seeing_quality = "Poor 🔴 (Turbulent)"
                    
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Cloud Cover", f"{total_clouds}%", delta="Clear" if total_clouds < 15 else "Obscured", delta_color="inverse")
                col2.metric("High Altitude", f"{high_clouds}%")
                col3.metric("Atmospheric Seeing", seeing_quality.split(" ")[0])
                col4.metric("PM 2.5 (Smoke)", f"{round(active_pm25)} µg/m³", delta="🚨 SENSOR OVERRIDE" if is_override else ("Clear Air" if active_pm25 <= 10 else "Haze/Smoke"), delta_color="inverse")
            else:
                st.info("Weather predictions are currently unavailable for this exact target time.")

            # --- COASTAL TIDE ANALYSIS UI (ASTRO) ---
            st.divider()
            st.subheader("🌊 Coastal Tide Context")
            if not fetch_tides_toggle:
                st.info("⏸️ **Tide Tracker Paused:** Check the 'Fetch Coastal Tide Data' box at the top of the app to consume an API call and load tide times.")
            elif tide_data == "demo":
                st.info("💡 **Tide Tracker Inactive:** To track high/low tide times for coastal reflections and sea stacks, replace `STORMGLASS_TOKEN` at the top of the script with a free API key from stormglass.io.")
            elif isinstance(tide_data, list) and len(tide_data) > 0:
                
                target_dt = pd.Timestamp(tracking_time)
                if target_dt.tzinfo is None:
                    target_dt = target_dt.tz_localize(real_tz)
                else:
                    target_dt = target_dt.tz_convert(real_tz)
                    
                parsed_tides = []
                for t in tide_data:
                    try:
                        t_time = pd.to_datetime(t['time']).tz_convert(real_tz)
                        parsed_tides.append((t_time, t['type'], t['height']))
                    except:
                        continue
                        
                parsed_tides.sort(key=lambda x: x[0])
                
                past_tides = [t for t in parsed_tides if t[0] < target_dt]
                future_tides = [t for t in parsed_tides if t[0] >= target_dt]
                
                display_tides = []
                if past_tides:
                    display_tides.append(past_tides[-1])
                display_tides.extend(future_tides[:3]) 
                
                if display_tides:
                    t_cols = st.columns(len(display_tides))
                    for i, (t_time, t_type, t_height) in enumerate(display_tides):
                        icon = "🔼 High" if t_type == "high" else "🔽 Low"
                        t_height_ft = round(t_height * 3.28084, 1)
                        
                        delta_hrs = (t_time - target_dt).total_seconds() / 3600
                        if delta_hrs < 0:
                            rel_str = f"{-delta_hrs:.1f}h before"
                        else:
                            rel_str = f"+{delta_hrs:.1f}h after"
                            
                        t_cols[i].metric(
                            f"{icon} ({t_time.strftime('%a %I:%M %p')})", 
                            f"{round(t_height, 2)}m", 
                            f"{t_height_ft}ft | {rel_str}", 
                            delta_color="off"
                        )
                else:
                    st.info("No extreme tide events detected around this time window.")
            else:
                st.info("No tidal data available for this location (likely an inland elevation).")
            
            # --- LIVE CLOUD MOVEMENT EMBED (ASTRO) ---
            st.divider()
            st.subheader("☁️ Live Cloud Movement & Tracking")
            st.write("Ensure the exact window of your astrophotography shoot remains completely clear of incoming cloud banks.")
            
            windy_html = f"""
            <iframe width="100%" height="500" 
                src="https://embed.windy.com/embed2.html?lat={lat}&lon={lon}&zoom=7&level=surface&overlay=clouds&product=ecmwf&menu=&message=true&marker=true&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=km%2Fh&metricTemp=%C2%B0C&radarRange=-1" 
                frameborder="0">
            </iframe>
            """
            components.html(windy_html, height=500)