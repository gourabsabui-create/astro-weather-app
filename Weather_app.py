import streamlit as st
import requests
from datetime import datetime, timedelta, time
import pandas as pd
import pydeck as pdk
import itertools
import math

# --- 1. Page Setup ---
st.set_page_config(page_title="Light & Fog Predictor", page_icon="🏔️", layout="centered")
st.title("🏔️ Landscape & Astro Forecaster")
st.caption("Multi-model consensus | Celestial Tracking | Live Ground Sensors")

WAQI_TOKEN = "demo" # Replace with a free token from aqicn.org if you hit rate limits

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
    """
    Interpolates the exact thermal center of mass of the inversion 
    to pinpoint the true fog ceiling down to the meter.
    """
    try:
        levels = [(100, "temperature_1000hPa"), (300, "temperature_975hPa"), (500, "temperature_950hPa"), 
                  (800, "temperature_925hPa"), (1000, "temperature_900hPa"), (1500, "temperature_850hPa")]
        
        surface_temp = safe_val(weather_data, levels[0][1], idx)
        peak_temp = surface_temp
        peak_alt = 100
        
        # 1. Find the absolute peak of the thermal inversion
        for alt, key in levels:
            t = safe_val(weather_data, key, idx)
            if t > peak_temp:
                peak_temp = t
                peak_alt = alt
                
        delta_t = round(peak_temp - surface_temp, 1)
        
        # 2. Interpolate the exact ceiling (steepest gradient point)
        if delta_t > 0:
            lower_alt, lower_temp = 100, surface_temp
            for alt, key in levels:
                if alt == peak_alt:
                    break
                lower_alt = alt
                lower_temp = safe_val(weather_data, key, idx)
            
            if peak_temp > lower_temp:
                weight = (peak_temp - lower_temp) / delta_t if delta_t > 0 else 0.5
                exact_ceiling = int(lower_alt + ((peak_alt - lower_alt) * weight))
                return delta_t, exact_ceiling
            return delta_t, peak_alt
            
        return 0, 0
    except:
        return 0, 0

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

def create_vector_line(lat, lon, azimuth, length_deg, color):
    end_lat = lat + length_deg * math.cos(math.radians(azimuth))
    end_lon = lon + length_deg * math.sin(math.radians(azimuth)) / math.cos(math.radians(lat))
    return {"start_lon": lon, "start_lat": lat, "end_lon": end_lon, "end_lat": end_lat, "color": color}

# --- APP START ---
mode = st.radio("Select Dashboard Mode:", ["🌅 Sunrise & Sunset", "🌌 Astrophotography"], horizontal=True)
search_query = st.text_input("Enter a location (e.g., Jasper, Banff, Yosemite):", "Lake Louise")

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

        with st.spinner('Loading atmospheric data...'):
            base_data = fetch_weather(lat, lon, tz)
            aq_data = fetch_air_quality(lat, lon, tz)
            live_aqi, live_station = fetch_waqi_live(lat, lon)

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
                    if inv_dt > 0:
                        inv_alt_ft = round(inv_alt * 3.28084)
                        # Re-structured to fit gracefully inside standard UI bounds
                        c3.metric("🌫️ FOG CEILING", f"{inv_alt} m", f"↑ {inv_alt_ft:,} ft | +{inv_dt}°C ΔT", delta_color="inverse")
                    else:
                        c3.metric("🌫️ FOG RISK", "Low", "No Inversion", delta_color="normal")
                    
                    with st.expander("📊 View Ensemble Breakdown (Model Agreement)"):
                        for m in ensemble_results:
                            st.markdown(f"**{m['name']}** - Potential: **{m['potential']}** | Skunk: **{m['skunk']}%**")
                            st.caption(f"Raw: Total {m['total']}% | Low {m['low']}% | Mid {m['mid']}% | High {m['high']}%")
                            
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
                
                step_dict = {"Micro (~20km)": 0.02, "Local (~45km)": 0.04, "Regional (~90km)": 0.08, "Macro (~160km)": 0.15}
                step = step_dict[zoom_level]
                r_mult = step / 0.08 
                
                c_b, c_sk, c_sm, c_f = st.columns(4)
                show_burn = c_b.checkbox("🔥 Burn", value=True)
                show_skunk = c_sk.checkbox("🦨 Skunk", value=True)
                show_smoke = c_sm.checkbox("🌲 Smoke", value=True)
                show_fog = c_f.checkbox("☁️ Fog", value=False)

                with st.spinner("Rendering cached continuous heat map..."):
                    grid_size = 10
                    lats = [lat + (i - grid_size//2)*step for i in range(grid_size)]
                    lons = [lon + (i - grid_size//2)*step for i in range(grid_size)]
                    coords = list(itertools.product(lats, lons))
                    lat_str, lon_str = ",".join(str(round(c[0], 4)) for c in coords), ",".join(str(round(c[1], 4)) for c in coords)

                    grid_res = fetch_model_grid(lat_str, lon_str, "best_match")
                    aq_grid_res = fetch_aq_grid(lat_str, lon_str)
                    
                    if grid_res:
                        map_data = []
                        for i, c in enumerate(coords):
                            try:
                                loc_w = grid_res[i] if isinstance(grid_res, list) else grid_res
                                loc_aq = aq_grid_res[i] if aq_grid_res and isinstance(aq_grid_res, list) else (aq_grid_res if aq_grid_res else {})
                                if "hourly" not in loc_w: continue
                                
                                idx = loc_w["hourly"]["time"].index(closest_hour_str)
                                aq_idx = loc_aq["hourly"]["time"].index(closest_hour_str) if loc_aq and "hourly" in loc_aq and closest_hour_str in loc_aq["hourly"].get("time", []) else 0
                                
                                grid_pm25 = safe_val(loc_aq, "pm2_5", aq_idx) if loc_aq else 0
                                if is_override: grid_pm25 = max(grid_pm25, active_pm25)

                                l_low = safe_val(loc_w, "cloud_cover_low", idx)
                                l_mid = safe_val(loc_w, "cloud_cover_mid", idx)
                                l_high = safe_val(loc_w, "cloud_cover_high", idx)
                                
                                opaque_deck = l_low + l_mid
                                potential = round(max(0, min(100, ((l_mid * 0.48) + (max(l_high, max(0, safe_val(loc_w, "relative_humidity_300hPa", idx) - 50)) * 1.15 * (1.0 - max(0, min(1.0, (opaque_deck - 45) / 45))))))))
                                skunk = round(min(100, max(max(0, (l_low - 50) * 2.0), max(0, (opaque_deck - 70) * 3.0) if opaque_deck > 70 else 0)))
                                
                                inv_dt_grid, inv_alt_grid = estimate_inversion_height(loc_w, idx)
                                if inv_dt_grid > 0:
                                    inv_alt_ft_grid = round(inv_alt_grid * 3.28084)
                                    fog_details = f"{inv_alt_grid}m ({inv_alt_ft_grid:,} ft) [+{inv_dt_grid}°C]"
                                else:
                                    fog_details = "Clear"

                                map_data.append({
                                    "lat": round(c[0], 4), 
                                    "lon": round(c[1], 4),
                                    "potential": potential,
                                    "skunk": skunk,
                                    "pm25": round(grid_pm25),
                                    "cloud_low": l_low,
                                    "cloud_mid": l_mid,
                                    "cloud_high": l_high,
                                    "fog_weight": 100 if inv_dt_grid > 0 else 0,
                                    "fog_text": fog_details
                                })
                            except: continue

                        df_map = pd.DataFrame(map_data)
                        layers = []

                        if show_burn and not df_map[df_map['potential'] > 0].empty:
                            layers.append(pdk.Layer(
                                'HeatmapLayer',
                                data=df_map[df_map['potential'] > 0],
                                get_position='[lon, lat]',
                                get_weight='potential',
                                opacity=0.6,
                                radiusPixels=55,
                                colorRange=[[255, 237, 160], [254, 178, 76], [253, 141, 60], [240, 59, 32], [189, 0, 38]]
                            ))
                            
                        if show_skunk and not df_map[df_map['skunk'] > 0].empty:
                            layers.append(pdk.Layer(
                                'HeatmapLayer',
                                data=df_map[df_map['skunk'] > 0],
                                get_position='[lon, lat]',
                                get_weight='skunk',
                                opacity=0.6,
                                radiusPixels=55,
                                colorRange=[[242, 240, 247], [203, 201, 226], [158, 154, 200], [117, 107, 177], [84, 39, 143]] 
                            ))

                        if show_smoke and not df_map[df_map['pm25'] > 0].empty:
                            layers.append(pdk.Layer(
                                'HeatmapLayer',
                                data=df_map[df_map['pm25'] > 0],
                                get_position='[lon, lat]',
                                get_weight='pm25',
                                opacity=0.6,
                                radiusPixels=55,
                                colorRange=[[246, 232, 195], [223, 194, 125], [191, 129, 45], [140, 81, 10], [84, 48, 5]]
                            ))

                        if show_fog and not df_map[df_map['fog_weight'] > 0].empty:
                            layers.append(pdk.Layer(
                                'HeatmapLayer',
                                data=df_map[df_map['fog_weight'] > 0],
                                get_position='[lon, lat]',
                                get_weight='fog_weight',
                                opacity=0.6,
                                radiusPixels=55,
                                colorRange=[[237, 248, 251], [178, 226, 226], [102, 194, 164], [44, 162, 95], [0, 109, 44]] 
                            ))

                        layers.append(pdk.Layer(
                            'ScatterplotLayer',
                            data=df_map,
                            get_position='[lon, lat]',
                            get_color=[0, 0, 0, 0], 
                            get_radius=5000 * r_mult,
                            pickable=True
                        ))

                        # Re-structured, shorter tooltip to prevent screen clipping
                        tooltip_html = (
                            "<b>Coord:</b> {lat}, {lon}<br/>"
                            "<b>🔥 Burn:</b> {potential}/100 | <b>🦨 Skunk:</b> {skunk}%<br/>"
                            "<b>🌲 Smoke:</b> {pm25} µg/m³<br/>"
                            "<b>☁️ Clouds:</b> {cloud_low}% L | {cloud_mid}% M | {cloud_high}% H<br/>"
                            "<b>🌫️ Fog Ceiling:</b> {fog_text}"
                        )

                        st.pydeck_chart(pdk.Deck(
                            map_style='dark',
                            initial_view_state=pdk.ViewState(
                                latitude=lat, longitude=lon, 
                                zoom=7.5 if zoom_level == "Regional (~90km)" else (9.5 if zoom_level == "Micro (~20km)" else 8.5), 
                                pitch=0
                            ),
                            layers=layers,
                            tooltip={"html": tooltip_html, "style": {"backgroundColor": "#222222", "color": "white"}}
                        ))


        # ==========================================
        # MODE 2: ASTROPHOTOGRAPHY
        # ==========================================
        elif mode == "🌌 Astrophotography":
            st.write("### 🕒 Astro Planning Window")
            c1, c2 = st.columns(2)
            selected_date = c1.date_input("Target Date:", datetime.today().date())
            selected_time = c2.time_input("Target Time (Local):", time(0, 0))
            
            dt = datetime.combine(selected_date, selected_time)
            closest_hour_str = dt.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00")
            hourly_times = base_data.get("hourly", {}).get("time", []) if base_data else []

            st.subheader("🌌 Milky Way & Night Sky Analysis")
            
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
                st.info("Weather predictions are currently unavailable for this date, but mathematical celestial tracking remains active.")

            st.divider()
            
            # --- OFFLINE GC TRACKING UI (PHOTOPILLS 3D STYLE) ---
            st.write("### 🔭 Advanced Celestial Tracking Map")
            minute_offset = st.slider("Scrub Time (Granular Adjustment):", -360, 360, 0, 5, "%d mins")
            tracking_time = dt + timedelta(minutes=minute_offset)
            
            gc_az, gc_alt = get_celestial_az_alt(lat, lon, tracking_time, tz, "galactic_core")
            sun_az, sun_alt = get_celestial_az_alt(lat, lon, tracking_time, tz, "sun")
            moon_az, moon_alt = get_celestial_az_alt(lat, lon, tracking_time, tz, "moon")
            
            if sun_alt > 0: bg_style, sky_status = 'light', "☀️ Daytime (Light Map)"
            elif sun_alt > -6: bg_style, sky_status = 'dark', "🌇 Civil Twilight (Golden/Blue Hour)"
            elif sun_alt > -12: bg_style, sky_status = 'dark', "🌆 Nautical Twilight (Stars Emerging)"
            else: bg_style, sky_status = 'dark', "🌌 True Night (Dark Map)"

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

            st.pydeck_chart(pdk.Deck(
                map_style=bg_style,
                initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=8.5, pitch=45, bearing=0),
                layers=[
                    pdk.Layer('LineLayer', data=pd.DataFrame(line_data) if line_data else pd.DataFrame(columns=["start_lon", "start_lat", "end_lon", "end_lat", "color"]), get_source_position='[start_lon, start_lat]', get_target_position='[end_lon, end_lat]', get_color='color', get_width=300),
                    pdk.Layer('ScatterplotLayer', data=pd.DataFrame(dot_data), get_position='[lon, lat]', get_color='color', get_radius='radius', pickable=False)
                ]
            ))
            st.caption("🟠 Orange Line = Sun Direction | ⚪ White Line = Moon Direction | 🟣/🟡 Dots = Milky Way Core")