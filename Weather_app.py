import streamlit as st
import requests
from datetime import datetime, timedelta, time
import pandas as pd
import pydeck as pdk
import itertools
import math
import numpy as np
import streamlit.components.v1 as components
import json
import os

# --- 1. Page Setup ---
st.set_page_config(page_title="Light & Fog Predictor", page_icon="🏔️", layout="centered")

WAQI_TOKEN = "ee0ee12bcf2cf2da796899543b1d0f91d20e3c7a" 
STORMGLASS_TOKEN = "41a49954-877a-11f1-bcd5-0242ac120004-41a499e0-877a-11f1-bcd5-0242ac120004" 
CACHE_FILE = "saved_locations.json"

# --- LOCAL FILE STORAGE ENGINE ---
def load_saved_locations():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_location_to_cache(name, lat, lon, tz):
    locations = load_saved_locations()
    locations[name] = {"lat": float(lat), "lon": float(lon), "tz": tz}
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(locations, f, indent=4)
    except Exception:
        pass

# --- DUAL-ENGINE GEOCODER (OpenStreetMap + Open-Meteo Fallback) ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_geocoding(query):
    # Engine 1: OpenStreetMap (Excellent for remote trailheads, lakes, parks)
    try:
        headers = {"User-Agent": "AstroFieldApp_Mobile/2.0 (contact@example.com)"}
        payload = {"q": query, "format": "json", "limit": 5}
        res_osm = requests.get("https://nominatim.openstreetmap.org/search", params=payload, headers=headers, timeout=5).json()
        
        if res_osm and isinstance(res_osm, list) and len(res_osm) > 0:
            return {"results": [{"name": loc.get("display_name"), "latitude": float(loc.get("lat")), "longitude": float(loc.get("lon")), "timezone": "auto"} for loc in res_osm]}
    except Exception:
        pass

    # Engine 2: Open-Meteo Fallback (Excellent for major cities like Chicago to bypass OSM rate limits)
    try:
        payload = {"name": query, "count": 5, "language": "en", "format": "json"}
        res_meteo = requests.get("https://geocoding-api.open-meteo.com/v1/search", params=payload, timeout=5).json()
        
        if res_meteo and "results" in res_meteo:
            formatted = []
            for loc in res_meteo["results"]:
                parts = [loc.get("name"), loc.get("admin1"), loc.get("country")]
                display_name = ", ".join([p for p in parts if p])
                formatted.append({"name": display_name, "latitude": float(loc.get("latitude")), "longitude": float(loc.get("longitude")), "timezone": loc.get("timezone", "auto")})
            return {"results": formatted}
    except Exception:
        pass

    return None

# --- CACHED API FUNCTIONS ---
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
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_air_quality(lat, lon, timezone="auto"):
    try:
        payload = {"latitude": lat, "longitude": lon, "hourly": "pm2_5", "timezone": timezone, "forecast_days": 14}
        return requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params=payload, timeout=5).json()
    except Exception:
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
    except Exception:
        return None, None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_elevation(lat, lon):
    try:
        return requests.get(f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}", timeout=5).json().get("elevation", [0])[0]
    except Exception:
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
    except Exception:
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
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_tides(lat, lon, token):
    if token == "demo":
        return "demo"
    try:
        start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        end = start + timedelta(days=4)
        payload = {'lat': lat, 'lng': lon, 'start': start.isoformat(), 'end': end.isoformat()}
        headers = {'Authorization': token}
        res = requests.get("https://api.stormglass.io/v2/tide/extremes/point", params=payload, headers=headers, timeout=5).json()
        return res.get('data', [])
    except Exception:
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
    except Exception:
        return 0, 0

# --- CONTINUOUS RASTER INTERPOLATION ENGINE ---
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
        polygon = [[lon - hw, lat - hh], [lon + hw, lat - hh], [lon + hw, lat + hh], [lon - hw, lat + hh]]
        out_data.append({"polygon": polygon, "color": [r, g, b, alpha]})
        
    return pdk.Layer('PolygonLayer', data=pd.DataFrame(out_data), get_polygon='polygon', get_fill_color='color', filled=True, stroked=False, wireframe=False, pickable=False)

# --- UNIFIED CELESTIAL MATH ENGINE (100% OFFLINE) ---
def get_celestial_az_alt(lat, lon, local_time, tz_string, target="galactic_core"):
    try:
        utc_time = pd.Timestamp(local_time).tz_localize(tz_string).tz_convert('UTC').replace(tzinfo=None)
    except Exception:
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

# --- CHECK FOR INCOMING URL GPS PARAMETERS ---
query_params = st.query_params
has_gps_url = "gps_lat" in query_params and "gps_lon" in query_params

# --- MOBILE COMPACT SIDEBAR UI ---
with st.sidebar:
    st.title("⚙️ Setup & Location")
    mode = st.radio("Dashboard Mode:", ["🌅 Sunrise & Sunset", "🌌 Astrophotography"])
    st.divider()
    
    # Automatically switch to the GPS tab if the URL parameter hack triggered
    input_method = st.radio(
        "Location Entry:", 
        ["🔍 Online Search", "📂 Saved Spots (Offline Storage)", "🛰️ Live GPS Sensor"],
        index=2 if has_gps_url else 0
    )
    
    lat, lon, tz = None, None, None
    cached_db = load_saved_locations()
    
    if input_method == "🔍 Online Search":
        search_query = st.text_input("Enter location name:", placeholder="e.g., Chicago, Lake Louise")
        
        if search_query:
            with st.spinner("Searching dual-engine global map database..."):
                geo_response = fetch_geocoding(search_query)
                
            if not geo_response or "results" not in geo_response or not geo_response["results"]:
                st.error("Location not found or offline.")
            else:
                location_options = {}
                for loc in geo_response["results"]:
                    display_name = loc.get("name", "Unknown Location")
                    if display_name in location_options:
                        display_name += f" ({loc['latitude']}, {loc['longitude']})"
                    location_options[display_name] = {"lat": loc["latitude"], "lon": loc["longitude"], "tz": loc.get("timezone", "auto")}
                
                selected_loc = st.selectbox("Confirm location:", list(location_options.keys()))
                lat = location_options[selected_loc]["lat"]
                lon = location_options[selected_loc]["lon"]
                tz = location_options[selected_loc]["tz"]
                
                save_location_to_cache(selected_loc, lat, lon, tz)

    elif input_method == "📂 Saved Spots (Offline Storage)":
        st.info("📡 **Offline Storage Active:** Select any location previously searched on this device.")
        if cached_db:
            selected_offline_spot = st.selectbox("Select Saved Spot:", list(cached_db.keys()))
            lat = cached_db[selected_offline_spot]["lat"]
            lon = cached_db[selected_offline_spot]["lon"]
            tz = cached_db[selected_offline_spot]["tz"]
            st.success(f"Loaded: {selected_offline_spot}")
        else:
            st.warning("No spots cached in local storage yet. Search a few locations while online first!")
            
    elif input_method == "🛰️ Live GPS Sensor":
        if has_gps_url:
            lat = float(query_params["gps_lat"])
            lon = float(query_params["gps_lon"])
            tz = "auto"
            st.success(f"✅ GPS Lock Acquired: {lat}, {lon}")
            
            if st.button("❌ Clear GPS Lock & Search Manually"):
                st.query_params.clear()
                st.rerun()
        else:
            st.info("Tap below to ask your phone's hardware GPS chip for your exact coordinates. The app will generate a secure link to load your spot!")
            
            # --- THE SANDBOX ESCAPE HACK ---
            # Instead of forcing a redirect, it generates an <a> tag button. 
            # The browser allows the user-initiated click to pass the data to the parent window seamlessly!
            gps_html = """
            <div style="font-family: sans-serif; color: white; text-align: center;">
                <button id="ping-btn" onclick="getLocation()" style="background-color: #4CAF50; color: white; padding: 10px 15px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; width: 100%;">📍 Ping GPS Satellites</button>
                <p id="gps-data" style="margin-top: 15px; font-size: 14px; font-weight: bold;"></p>
                <div id="link-container"></div>
            </div>
            <script>
            function getLocation() {
                var x = document.getElementById("gps-data");
                var btn = document.getElementById("ping-btn");
                btn.style.display = "none";
                x.innerHTML = "<i>Pinging satellites... Allow location access if prompted.</i>";
                
                if (navigator.geolocation) {
                    navigator.geolocation.getCurrentPosition(function(position) {
                        var lat = position.coords.latitude.toFixed(5);
                        var lon = position.coords.longitude.toFixed(5);
                        x.innerHTML = "✅ GPS Lock Acquired: " + lat + ", " + lon;
                        
                        var link = document.createElement("a");
                        link.href = "?gps_lat=" + lat + "&gps_lon=" + lon;
                        link.target = "_parent"; 
                        link.innerHTML = "<button style='background-color: #008CBA; color: white; padding: 10px 15px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; width: 100%; margin-top: 10px;'>🚀 Load Map with GPS</button>";
                        
                        document.getElementById("link-container").appendChild(link);
                    }, function(error) {
                        x.innerHTML = "❌ Error: " + error.message;
                        btn.style.display = "block";
                    }, {enableHighAccuracy: true, timeout: 10000, maximumAge: 0});
                } else {
                    x.innerHTML = "Geolocation not supported.";
                }
            }
            </script>
            """
            components.html(gps_html, height=180)
        
    st.divider()
    fetch_tides_toggle = st.checkbox("🌊 Fetch Coastal Tides", value=False, help="Consumes 1 Stormglass API Call")

# --- MAIN DASHBOARD START ---
st.title("🏔️ Landscape & Astro Forecaster")

if lat is None or lon is None or tz is None:
    st.info("👈 Open the sidebar menu to search for a location or auto-feed your GPS.")
else:
    with st.spinner('Loading environment data...'):
        base_data = fetch_weather(lat, lon, tz)
        aq_data = fetch_air_quality(lat, lon, tz)
        live_aqi, live_station = fetch_waqi_live(lat, lon)
        tide_data = fetch_tides(lat, lon, STORMGLASS_TOKEN) if fetch_tides_toggle else None
        real_tz = base_data.get("timezone", tz) if base_data else (tz if tz != "auto" else "UTC")

    # ==========================================
    # MODE 1: SUNRISE & SUNSET
    # ==========================================
    if mode == "🌅 Sunrise & Sunset":
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

        selected_event_label = st.selectbox("Select Target Window:", list(event_menu.keys()))
        event_type, exact_time_str = event_menu[selected_event_label]

        dt = datetime.fromisoformat(exact_time_str)
        if dt.minute >= 30: dt += timedelta(hours=1)
        dt = dt.replace(minute=0, second=0, microsecond=0)
        closest_hour_str = dt.strftime("%Y-%m-%dT%H:00")
        
        baseline_idx = hourly_times.index(closest_hour_str) if closest_hour_str in hourly_times else 0
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

        tab_map, tab_details, tab_clouds = st.tabs(["🗺️ Radar Map", "📊 Forecast Details", "☁️ Live Clouds"])
        
        with tab_map:
            if ensemble_results:
                avg_pot = round(sum(m["potential"] for m in ensemble_results) / len(ensemble_results))
                avg_skunk = round(sum(m["skunk"] for m in ensemble_results) / len(ensemble_results))
                c1, c2 = st.columns(2)
                c1.metric("🔥 BURN POTENTIAL", f"{avg_pot}/100")
                c2.metric("🦨 SKUNK CHANCE", f"{avg_skunk}%")
                
            with st.expander("🛠️ Map Controls (Zoom & Overlays)", expanded=False):
                zoom_level = st.select_slider("Grid Coverage Area:", options=["Micro (~20km)", "Local (~45km)", "Regional (~90km)", "Macro (~160km)"], value="Regional (~90km)")
                interactive_map = st.radio("Interactive Map?", ["Yes (Zoom & Pan)", "No (Static Map)"], horizontal=True)
                is_interactive = (interactive_map == "Yes (Zoom & Pan)")
                c_b, c_sk, c_sm, c_f = st.columns(4)
                show_burn = c_b.checkbox("🔥 Burn", value=True)
                show_skunk = c_sk.checkbox("🦨 Skunk", value=True)
                show_smoke = c_sm.checkbox("🌲 Smoke", value=True)
                show_fog = c_f.checkbox("☁️ Fog", value=False)
                
            step_dict = {"Micro (~20km)": 0.02, "Local (~45km)": 0.04, "Regional (~90km)": 0.08, "Macro (~160km)": 0.15}
            zoom_dict = {"Micro (~20km)": 9.5, "Local (~45km)": 8.5, "Regional (~90km)": 7.5, "Macro (~160km)": 6.5}
            step, map_zoom = step_dict[zoom_level], zoom_dict[zoom_level]
            
            if is_interactive:
                v_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=map_zoom, pitch=0)
            else:
                v_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=map_zoom, pitch=0, min_zoom=map_zoom, max_zoom=map_zoom)

            with st.spinner("Rendering continuous mathematical heatmap..."):
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

                            l_low, l_mid, l_high = safe_val(loc_w, "cloud_cover_low", idx), safe_val(loc_w, "cloud_cover_mid", idx), safe_val(loc_w, "cloud_cover_high", idx)
                            opaque_deck = l_low + l_mid
                            potential = round(max(0, min(100, ((l_mid * 0.48) + (max(l_high, max(0, safe_val(loc_w, "relative_humidity_300hPa", idx) - 50)) * 1.15 * (1.0 - max(0, min(1.0, (opaque_deck - 45) / 45))))))))
                            skunk = round(min(100, max(max(0, (l_low - 50) * 2.0), max(0, (opaque_deck - 70) * 3.0) if opaque_deck > 70 else 0)))
                            
                            inv_dt_grid, inv_alt_grid = estimate_inversion_height(loc_w, idx)
                            if inv_dt_grid > 0 and l_low > 5:
                                fog_intensity = min(100, (l_low / 100.0) * (inv_dt_grid * 25))
                                fog_details = f"~{inv_alt_grid}m ({round(inv_alt_grid * 3.28084):,} ft) [+{inv_dt_grid}°C]"
                            else:
                                fog_intensity, fog_details = 0, "Clear (No Moisture/Inversion)"

                            map_data.append({"lat": round(c[0], 4), "lon": round(c[1], 4), "potential": potential, "skunk": skunk, "pm25": round(grid_pm25), "cloud_low": l_low, "cloud_mid": l_mid, "cloud_high": l_high, "fog_weight": fog_intensity, "fog_text": fog_details})
                        except Exception: continue

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

                        pad_lat, pad_lon = step / 2, step / 2
                        min_lon, max_lon = min(lons), max(lons)
                        min_lat, max_lat = min(lats), max(lats)

                        mask_data = [{"polygon": [[[-180, 90], [180, 90], [180, -90], [-180, -90]], [[min_lon - pad_lon, min_lat - pad_lat], [max_lon + pad_lon, min_lat - pad_lat], [max_lon + pad_lon, max_lat + pad_lat], [min_lon - pad_lon, max_lat + pad_lat]]]}]
                        layers.append(pdk.Layer('PolygonLayer', data=mask_data, get_polygon='polygon', get_fill_color=[22, 25, 28, 230], filled=True, stroked=True, get_line_color=[150, 150, 150, 150], line_width_min_pixels=2, pickable=False))
                        layers.append(pdk.Layer('ScatterplotLayer', data=df_map, get_position='[lon, lat]', get_color=[0, 0, 0, 0], get_radius=5000 * (step / 0.08), pickable=True))

                        tooltip_html = "<b>Coord:</b> {lat}, {lon}<br/><b>🔥 Burn:</b> {potential}/100 | <b>🦨 Skunk:</b> {skunk}%<br/><b>🌲 Smoke:</b> {pm25} µg/m³<br/><b>☁️ Clouds:</b> {cloud_low}% L | {cloud_mid}% M | {cloud_high}% H<br/><b>🌫️ Fog Ceiling:</b> {fog_text}"
                        st.pydeck_chart(pdk.Deck(map_style='dark', views=[pdk.View(type="MapView", controller=is_interactive)], initial_view_state=v_state, layers=layers, tooltip={"html": tooltip_html, "style": {"backgroundColor": "#222222", "color": "white"}}))
                    else:
                        st.error("No valid map data could be rendered.")

        with tab_details:
            st.subheader("⛰️ Topographical Ray-Tracing")
            current_elev = fetch_elevation(lat, lon)
            horizon_elev = fetch_elevation(lat, lon - 0.06) if event_type == "sunset" else fetch_elevation(lat, lon + 0.06)
            direction = "Western" if event_type == "sunset" else "Eastern"
            elev_diff = horizon_elev - current_elev
            
            if elev_diff > 150: 
                minutes_lost = round(math.degrees(math.atan(elev_diff / 5000)) * 4)
                st.warning(f"⚠️ **Mountain Shadow:** The {direction} ridge is {round(elev_diff)}m ({round(elev_diff * 3.28084):,} ft) higher. Sun will disappear **~{minutes_lost} mins early**.")
            else:
                st.success(f"✅ **Clear Horizon:** No topographical blocking to the {direction}.")
                
            st.subheader("🌫️ Fog & Inversion Risk")
            inv_dt, inv_alt = estimate_inversion_height(base_data, baseline_idx)
            local_low_clouds = safe_val(base_data, "cloud_cover_low", baseline_idx)
            if inv_dt > 0 and local_low_clouds > 10:
                st.metric("FOG CEILING", f"~{inv_alt} m", f"↑ {round(inv_alt * 3.28084):,} ft | +{inv_dt}°C ΔT", delta_color="inverse")
            else:
                st.metric("FOG RISK", "Low", "No Moisture/Inversion", delta_color="normal")

            with st.expander("📊 View Ensemble Breakdown (Model Agreement)"):
                for m in ensemble_results:
                    st.markdown(f"**{m['name']}** - Potential: **{m['potential']}** | Skunk: **{m['skunk']}%**")
                    st.caption(f"Raw: Total {m['total']}% | Low {m['low']}% | Mid {m['mid']}% | High {m['high']}%")

            st.subheader("🌲 Air Quality & Smoke")
            if is_override:
                st.error(f"🚨 **SENSOR OVERRIDE:** '{live_station}' sensor detects thick smoke.")
                st.metric("PM 2.5", f"{round(active_pm25)} µg/m³", delta="Override Active", delta_color="inverse")
            else:
                st.metric("PM 2.5", f"{round(active_pm25)} µg/m³")
                
            if active_pm25 <= 10: st.success("✅ Clean Air")
            elif active_pm25 <= 35: st.info("🌤️ Light Smoke (Blood-Orange Sun)")
            elif active_pm25 <= 60: st.warning("⚠️ Moderate Smoke Smother")
            else: st.error("🛑 Heavy Smoke Skunk")

            st.subheader("🌊 Coastal Tide Context")
            if not fetch_tides_toggle:
                st.info("⏸️ **Tide Tracker Paused:** Check the box in the sidebar to load tide times.")
            elif tide_data == "demo":
                st.info("💡 **Tide Tracker Inactive:** Replace STORMGLASS_TOKEN with API key.")
            elif isinstance(tide_data, list) and len(tide_data) > 0:
                target_dt = pd.Timestamp(dt).tz_localize(real_tz) if pd.Timestamp(dt).tzinfo is None else pd.Timestamp(dt).tz_convert(real_tz)
                parsed_tides = sorted([(pd.to_datetime(t['time']).tz_convert(real_tz), t['type'], t['height']) for t in tide_data], key=lambda x: x[0])
                past_tides, future_tides = [t for t in parsed_tides if t[0] < target_dt], [t for t in parsed_tides if t[0] >= target_dt]
                display_tides = (past_tides[-1:] if past_tides else []) + future_tides[:3]
                
                if display_tides:
                    for t_time, t_type, t_height in display_tides:
                        icon = "🔼 High" if t_type == "high" else "🔽 Low"
                        delta_hrs = (t_time - target_dt).total_seconds() / 3600
                        rel_str = f"{-delta_hrs:.1f}h before" if delta_hrs < 0 else f"+{delta_hrs:.1f}h after"
                        st.metric(f"{icon} ({t_time.strftime('%a %I:%M %p')})", f"{round(t_height, 2)}m", f"{round(t_height * 3.28084, 1)}ft | {rel_str}", delta_color="off")
                else:
                    st.info("No extreme tide events detected.")
            else:
                st.info("No tidal data available (inland elevation).")

        with tab_clouds:
            st.write("Ensure exact window remains clear of incoming cloud banks.")
            windy_html = f'<iframe width="100%" height="500" src="https://embed.windy.com/embed2.html?lat={lat}&lon={lon}&zoom=7&level=surface&overlay=clouds&product=ecmwf&menu=&message=true&marker=true&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=km%2Fh&metricTemp=%C2%B0C&radarRange=-1" frameborder="0"></iframe>'
            components.html(windy_html, height=500)

    # ==========================================
    # MODE 2: ASTROPHOTOGRAPHY
    # ==========================================
    elif mode == "🌌 Astrophotography":
        selected_date = st.date_input("Target Date:", datetime.today().date())
        
        tab_astro, tab_events, tab_weather = st.tabs(["🔭 Celestial Tracking", "⏱️ Milestones", "☁️ Conditions & Tides"])
        
        with tab_astro:
            start_of_day = datetime.combine(selected_date, time(0, 0))
            end_of_day = datetime.combine(selected_date, time(23, 59))
            default_time = datetime.now().replace(second=0, microsecond=0) if selected_date == datetime.today().date() else datetime.combine(selected_date, time(12, 0))
                
            tracking_time = st.slider("Select Exact Time (Local):", min_value=start_of_day, max_value=end_of_day, value=default_time, step=timedelta(minutes=1), format="hh:mm A")
            
            gc_az, gc_alt = get_celestial_az_alt(lat, lon, tracking_time, real_tz, "galactic_core")
            sun_az, sun_alt = get_celestial_az_alt(lat, lon, tracking_time, real_tz, "sun")
            moon_az, moon_alt = get_celestial_az_alt(lat, lon, tracking_time, real_tz, "moon")
            
            if sun_alt > 6: wash_color, sky_status = [255, 240, 200, 25], "☀️ Daytime (Yellow)"
            elif sun_alt > 0: wash_color, sky_status = [255, 140, 0, 45], "🌇 Golden Hour (Orange)"
            elif sun_alt > -6: wash_color, sky_status = [100, 150, 255, 50], "🌆 Blue Hour / Civil"
            elif sun_alt > -12: wash_color, sky_status = [20, 50, 150, 80], "🌌 Nautical Twilight"
            elif sun_alt > -18: wash_color, sky_status = [0, 10, 50, 100], "🌌 Astro Twilight"
            else: wash_color, sky_status = [0, 0, 20, 140], "🌃 True Night"

            c_a, c_b, c_c = st.columns(3)
            c_a.markdown(f"**Milky Way:** Alt {round(gc_alt)}°")
            c_b.markdown(f"**Sun:** Alt {round(sun_alt)}°")
            c_c.markdown(f"**Map:** {sky_status}")
            
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
                dot_data.extend([{"lon": d_lon, "lat": d_lat, "radius": base_radius * 1.5, "color": base_color + [alpha // 3]}, {"lon": d_lon, "lat": d_lat, "radius": base_radius * 0.4, "color": base_color + [alpha]}])

            if sun_alt > -18: line_data.append(create_vector_line(lat, lon, sun_az, 0.45, [255, 140, 0, 200] if sun_alt > 0 else [255, 140, 0, 80]))
            if moon_alt > -10: line_data.append(create_vector_line(lat, lon, moon_az, 0.45, [200, 220, 255, 200] if moon_alt > 0 else [200, 220, 255, 60]))

            with st.expander("🛠️ Map Controls"):
                interactive_astro = st.radio("Interactive Map?", ["Yes (Zoom & Pan)", "No (Static Map)"], horizontal=True, key="astro_toggle")
            
            astro_view = pdk.ViewState(latitude=lat, longitude=lon, zoom=8.5, pitch=45, bearing=0) if interactive_astro == "Yes (Zoom & Pan)" else pdk.ViewState(latitude=lat, longitude=lon, zoom=8.5, pitch=45, bearing=0, min_zoom=8.5, max_zoom=8.5)
            wash_layer = pdk.Layer('PolygonLayer', data=pd.DataFrame([{"polygon": [[-180, 90], [180, 90], [180, -90], [-180, -90]], "color": wash_color}]), get_polygon='polygon', get_fill_color='color', filled=True, stroked=False, pickable=False)

            st.pydeck_chart(pdk.Deck(map_style='light', views=[pdk.View(type="MapView", controller=(interactive_astro == "Yes (Zoom & Pan)"))], initial_view_state=astro_view, layers=[wash_layer, pdk.Layer('LineLayer', data=pd.DataFrame(line_data) if line_data else pd.DataFrame(columns=["start_lon", "start_lat", "end_lon", "end_lat", "color"]), get_source_position='[start_lon, start_lat]', get_target_position='[end_lon, end_lat]', get_color='color', get_width=3, width_units='"pixels"'), pdk.Layer('ScatterplotLayer', data=pd.DataFrame(dot_data), get_position='[lon, lat]', get_color='color', get_radius='radius', pickable=False)]))
            st.caption("🟠 Sun Direction | ⚪ Moon Direction | 🟣/🟡 Milky Way Core")

        with tab_events:
            with st.spinner("Calculating exact horizon crossings..."):
                events = calculate_celestial_events(lat, lon, selected_date.isoformat(), real_tz)
                
            e1, e2 = st.columns(2)
            e1.markdown(f"**Sunrise:** {events.get('Sunrise', 'N/A').strftime('%I:%M %p') if 'Sunrise' in events else 'N/A'}")
            e1.markdown(f"**Sunset:** {events.get('Sunset', 'N/A').strftime('%I:%M %p') if 'Sunset' in events else 'N/A'}")
            e1.markdown(f"**Moonrise:** {events.get('Moonrise', 'N/A').strftime('%I:%M %p') if 'Moonrise' in events else 'N/A'}")
            e1.markdown(f"**Moonset:** {events.get('Moonset', 'N/A').strftime('%I:%M %p') if 'Moonset' in events else 'N/A'}")
            
            e2.markdown(f"**Dawn/Dusk (Civil):** {events.get('Dawn (Civil)', 'N/A').strftime('%I:%M %p') if 'Dawn (Civil)' in events else 'N/A'} / {events.get('Dusk (Civil)', 'N/A').strftime('%I:%M %p') if 'Dusk (Civil)' in events else 'N/A'}")
            e2.markdown(f"**Dawn/Dusk (Nautical):** {events.get('Dawn (Nautical)', 'N/A').strftime('%I:%M %p') if 'Dawn (Nautical)' in events else 'N/A'} / {events.get('Dusk (Nautical)', 'N/A').strftime('%I:%M %p') if 'Dusk (Nautical)' in events else 'N/A'}")
            e2.markdown(f"**Dawn/Dusk (Astro):** {events.get('Dawn (Astro)', 'N/A').strftime('%I:%M %p') if 'Dawn (Astro)' in events else 'N/A'} / {events.get('Dusk (Astro)', 'N/A').strftime('%I:%M %p') if 'Dusk (Astro)' in events else 'N/A'}")

        with tab_weather:
            closest_hour_str = tracking_time.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00")
            hourly_times = base_data.get("hourly", {}).get("time", []) if base_data else []

            st.subheader("🌌 Conditions at Target Time")
            if base_data and closest_hour_str in hourly_times:
                baseline_idx = hourly_times.index(closest_hour_str)
                total_clouds, high_clouds = safe_val(base_data, "cloud_cover", baseline_idx), safe_val(base_data, "cloud_cover_high", baseline_idx)
                
                aq_idx = aq_data["hourly"]["time"].index(closest_hour_str) if aq_data and "hourly" in aq_data and closest_hour_str in aq_data["hourly"].get("time", []) else 0
                model_pm25 = safe_val(aq_data, "pm2_5", aq_idx) if aq_data else 0
                live_pm25 = aqi_to_pm25(live_aqi) if live_aqi else 0
                is_override = live_pm25 > (model_pm25 + 10)
                active_pm25 = live_pm25 if is_override else model_pm25
                
                inv_dt_astro, inv_alt_astro = estimate_inversion_height(base_data, baseline_idx)
                seeing_quality = "Excellent 🟢" if inv_dt_astro > 0 else ("Good 🟡" if inv_dt_astro > -3 else "Poor 🔴")
                    
                col1, col2 = st.columns(2)
                col1.metric("Cloud Cover", f"{total_clouds}%", delta="Clear" if total_clouds < 15 else "Obscured", delta_color="inverse")
                col2.metric("High Altitude", f"{high_clouds}%")
                col1.metric("Atmospheric Seeing", seeing_quality)
                col2.metric("PM 2.5 (Smoke)", f"{round(active_pm25)} µg/m³", delta="🚨 OVERRIDE" if is_override else ("Clear" if active_pm25 <= 10 else "Haze"), delta_color="inverse")
            else:
                st.info("Weather unavailable for this exact future time.")

            st.subheader("🌊 Coastal Tide Context")
            if not fetch_tides_toggle:
                st.info("⏸️ **Tide Tracker Paused:** Check the sidebar to load tide times.")
            elif tide_data == "demo":
                st.info("💡 **Tide Tracker Inactive:** Replace STORMGLASS_TOKEN.")
            elif isinstance(tide_data, list) and len(tide_data) > 0:
                target_dt = pd.Timestamp(tracking_time).tz_localize(real_tz) if pd.Timestamp(tracking_time).tzinfo is None else pd.Timestamp(tracking_time).tz_convert(real_tz)
                parsed_tides = sorted([(pd.to_datetime(t['time']).tz_convert(real_tz), t['type'], t['height']) for t in tide_data], key=lambda x: x[0])
                past_tides, future_tides = [t for t in parsed_tides if t[0] < target_dt], [t for t in parsed_tides if t[0] >= target_dt]
                display_tides = (past_tides[-1:] if past_tides else []) + future_tides[:3]
                
                if display_tides:
                    for t_time, t_type, t_height in display_tides:
                        icon = "🔼 High" if t_type == "high" else "🔽 Low"
                        delta_hrs = (t_time - target_dt).total_seconds() / 3600
                        rel_str = f"{-delta_hrs:.1f}h before" if delta_hrs < 0 else f"+{delta_hrs:.1f}h after"
                        st.metric(f"{icon} ({t_time.strftime('%a %I:%M %p')})", f"{round(t_height, 2)}m", f"{round(t_height * 3.28084, 1)}ft | {rel_str}", delta_color="off")
                else:
                    st.info("No extreme tide events detected.")
            else:
                st.info("No tidal data available (inland elevation).")
            
            st.write("☁️ **Live Cloud Movement & Tracking**")
            windy_html = f'<iframe width="100%" height="500" src="https://embed.windy.com/embed2.html?lat={lat}&lon={lon}&zoom=7&level=surface&overlay=clouds&product=ecmwf&menu=&message=true&marker=true&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=km%2Fh&metricTemp=%C2%B0C&radarRange=-1" frameborder="0"></iframe>'
            components.html(windy_html, height=500)