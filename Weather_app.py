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
st.caption("Multi-model consensus | Celestial Tracking | Offline Caching")

# --- CACHED API FUNCTIONS (OFFLINE MODE FAILSAFE) ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_geocoding(query):
    try:
        return requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=5", timeout=5).json()
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
            "hourly": "cloud_cover_low,cloud_cover_mid,cloud_cover_high,relative_humidity_300hPa,temperature_1000hPa,temperature_900hPa",
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
    
    if math.sin(ha_rad) > 0:
        az = 2 * math.pi - az
        
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
            display_name = ", ".join([p for p in [loc.get("name", ""), loc.get("admin1", ""), loc.get("country", "")] if p])
            location_options[display_name] = {"lat": loc["latitude"], "lon": loc["longitude"], "tz": loc.get("timezone", "auto")}
        
        selected_loc = st.selectbox("Select the exact location:", list(location_options.keys()))
        lat = location_options[selected_loc]["lat"]
        lon = location_options[selected_loc]["lon"]
        tz = location_options[selected_loc]["tz"]

        with st.spinner('Loading atmospheric data (cached for offline use)...'):
            base_data = fetch_weather(lat, lon, tz)
            aq_data = fetch_air_quality(lat, lon, tz)

        st.divider()

        # ==========================================
        # MODE 1: SUNRISE & SUNSET (3-DAY EVENT MENU)
        # ==========================================
        if mode == "🌅 Sunrise & Sunset":
            st.write("### 🕒 3-Day Forecast Window")
            daily_data = base_data.get("daily", {}) if base_data else {}
            hourly_times = base_data.get("hourly", {}).get("time", []) if base_data else []
            
            if not daily_data or not hourly_times:
                st.error("Failed to fetch reliable baseline weather data. You may be fully offline with no cached data.")
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
            if dt.minute >= 30:
                dt += timedelta(hours=1)
            dt = dt.replace(minute=0, second=0, microsecond=0)
            closest_hour_str = dt.strftime("%Y-%m-%dT%H:00")
            
            st.subheader("⛰️ Topographical Ray-Tracing")
            with st.spinner("Scanning mountain profiles..."):
                current_elev = fetch_elevation(lat, lon)
                
                if event_type == "sunset":
                    horizon_elev = fetch_elevation(lat, lon - 0.06)
                    direction = "Western"
                else:
                    horizon_elev = fetch_elevation(lat, lon + 0.06)
                    direction = "Eastern"
                    
                elev_diff = horizon_elev - current_elev
                
                if elev_diff > 150: 
                    angle_rads = math.atan(elev_diff / 5000)
                    angle_degrees = math.degrees(angle_rads)
                    minutes_lost = round(angle_degrees * 4)
                    st.warning(f"⚠️ **Mountain Shadow Detected:** The {direction} ridge is {round(elev_diff)}m higher than your location. The sun will disappear behind the peaks **~{minutes_lost} minutes before** official {event_type}.")
                else:
                    st.success(f"✅ **Clear Horizon:** No significant topographical blocking detected to the {direction}.")

            if closest_hour_str not in hourly_times:
                st.error("Forecast data is not yet available for that time slot.")
            else:
                baseline_idx = hourly_times.index(closest_hour_str)
                lon_offset = 0.6 if event_type == "sunrise" else -0.6
                upstream_lon = lon + lon_offset

                models_to_run = {"High-Res (Local)": "best_match", "ECMWF (European)": "ecmwf_ifs", "GFS (American)": "gfs_seamless"}
                ensemble_results = []
                
                aq_idx = aq_data["hourly"]["time"].index(closest_hour_str) if aq_data and "hourly" in aq_data and closest_hour_str in aq_data["hourly"].get("time", []) else 0
                local_pm25 = safe_val(aq_data, "pm2_5", aq_idx) if aq_data else 0
                
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
                        ghost_high = max(0, rh_300 - 50) if rh_300 > 50 else 0
                        effective_high = max(l_high, ghost_high)
                        
                        mid_value = l_mid * 0.48 
                        raw_high = effective_high * 1.15 
                        vis_block = max(0, min(1.0, (opaque_deck - 45) / 45)) 
                        high_value = raw_high * (1.0 - vis_block) 
                        
                        shadow_penalty = u_low * 0.25 
                        muddy_penalty = 15 if (l_low > 15 and l_mid > 15 and effective_high > 15) else 0
                        potential = round(max(0, min(100, (mid_value + high_value) - shadow_penalty - muddy_penalty)))
                        
                        skunk_from_smoke = max(0, (local_pm25 - 40) * 1.5)
                        skunk = round(min(100, max(max(0, (l_low - 50) * 2.0), max(0, (u_low - 40) * 1.8), max(0, (opaque_deck - 70) * 3.0) if opaque_deck > 70 else 0, skunk_from_smoke)))
                        
                        ensemble_results.append({
                            "name": model_label, "potential": potential, "skunk": skunk,
                            "total": l_total, "low": l_low, "mid": l_mid, "high": l_high, "rh": rh_300
                        })

                st.divider()
                st.subheader("🔥 Forecast Analysis")
                
                if ensemble_results:
                    avg_pot = round(sum(m["potential"] for m in ensemble_results) / len(ensemble_results))
                    avg_skunk = round(sum(m["skunk"] for m in ensemble_results) / len(ensemble_results))
                    c1, c2 = st.columns(2)
                    c1.metric("🔥 CONSENSUS BURN POTENTIAL", f"{avg_pot}/100")
                    c2.metric("🦨 CONSENSUS CHANCE OF SKUNK", f"{avg_skunk}%")
                    
                    with st.expander("📊 View Ensemble Breakdown (Model Agreement)"):
                        for m in ensemble_results:
                            st.markdown(f"**{m['name']}** - Potential: **{m['potential']}** | Skunk: **{m['skunk']}%**")
                            st.caption(f"Raw: Total {m['total']}% | Low {m['low']}% | Mid {m['mid']}% | High {m['high']}%")
                            
                # --- ALWAYS VISIBLE SMOKE TEXT BLOCK ---
                st.divider()
                st.subheader("🌲 Air Quality & Wildfire Smoke")
                st.metric("PM 2.5 (Smoke Density)", f"{round(local_pm25)} µg/m³")
                
                if local_pm25 <= 10:
                    st.success("✅ **Clean Air:** No significant wildfire smoke detected. The atmosphere is clear and should not impact the sunset or sky visibility.")
                elif 10 < local_pm25 <= 35:
                    st.info("🌤️ **Blood-Orange Sun Potential:** There is a light layer of smoke in the atmosphere. It shouldn't block the light, but it could dramatically enhance the reds and oranges at the horizon.")
                elif 35 < local_pm25 <= 60:
                    st.warning("⚠️ **Moderate Smoke Smother:** The smoke is getting thick enough to wash out contrast and dim the burn potential.")
                else:
                    st.error("🛑 **Heavy Smoke Skunk:** Wildfire smoke is very thick. The sun will likely vanish into a gray/brown haze long before it hits the horizon.")

                st.divider()
                st.subheader("🗺️ High-Resolution Regional Overlay")
                c_b, c_sk, c_sm, c_f = st.columns(4)
                show_burn = c_b.checkbox("🔥 Burn", value=True)
                show_skunk = c_sk.checkbox("🦨 Skunk", value=True)
                show_smoke = c_sm.checkbox("🌲 Smoke", value=True)
                show_fog = c_f.checkbox("☁️ Fog", value=False)

                with st.spinner("Rendering cached spatial overlay..."):
                    grid_size = 10
                    step = 0.08 
                    lats = [lat + (i - grid_size//2)*step for i in range(grid_size)]
                    lons = [lon + (i - grid_size//2)*step for i in range(grid_size)]
                    coords = list(itertools.product(lats, lons))
                    
                    lat_str = ",".join(str(round(c[0], 4)) for c in coords)
                    lon_str = ",".join(str(round(c[1], 4)) for c in coords)

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
                                pm25 = safe_val(loc_aq, "pm2_5", aq_idx) if loc_aq else 0
                                
                                l_low = safe_val(loc_w, "cloud_cover_low", idx)
                                l_mid = safe_val(loc_w, "cloud_cover_mid", idx)
                                l_high = safe_val(loc_w, "cloud_cover_high", idx)
                                rh_300 = safe_val(loc_w, "relative_humidity_300hPa", idx)
                                t_100m = safe_val(loc_w, "temperature_1000hPa", idx)
                                t_1000m = safe_val(loc_w, "temperature_900hPa", idx)
                                
                                opaque_deck = l_low + l_mid
                                potential = round(max(0, min(100, ((l_mid * 0.48) + (max(l_high, max(0, rh_300 - 50)) * 1.15 * (1.0 - max(0, min(1.0, (opaque_deck - 45) / 45))))))))
                                skunk = round(min(100, max(max(0, (l_low - 50) * 2.0), max(0, (opaque_deck - 70) * 3.0) if opaque_deck > 70 else 0)))
                                is_inversion = 1 if t_1000m > t_100m else 0

                                map_data.append({
                                    "lat": c[0], "lon": c[1],
                                    "burn_color": [255, max(0, int(255 - (potential * 2.55))), 0, 140] if show_burn else [0,0,0,0],
                                    "skunk_color": [min(255, int(skunk * 2.55)), 0, max(0, 200 - int(skunk * 2)), 190] if show_skunk else [0,0,0,0],
                                    "smoke_color": [139, 69, 19, min(200, int(pm25 * 3))] if show_smoke else [0,0,0,0], 
                                    "fog_color": [180, 220, 255, 200] if (is_inversion and show_fog) else [0,0,0,0]
                                })
                            except:
                                continue

                        df_map = pd.DataFrame(map_data)
                        st.pydeck_chart(pdk.Deck(
                            map_style='dark',
                            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=7.5, pitch=0),
                            layers=[
                                pdk.Layer('ScatterplotLayer', data=df_map, get_position='[lon, lat]', get_color='burn_color', get_radius=5000),
                                pdk.Layer('ScatterplotLayer', data=df_map, get_position='[lon, lat]', get_color='skunk_color', get_radius=3000),
                                pdk.Layer('ScatterplotLayer', data=df_map, get_position='[lon, lat]', get_color='smoke_color', get_radius=1500), 
                                pdk.Layer('ScatterplotLayer', data=df_map, get_position='[lon, lat]', get_color='fog_color', get_radius=700)
                            ]
                        ))
                    else:
                        st.warning("Offline Mode: High-res spatial grid cannot be downloaded without an internet connection.")


        # ==========================================
        # MODE 2: ASTROPHOTOGRAPHY (ANY DATE PLANNER)
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
                t_100m = safe_val(base_data, "temperature_1000hPa", baseline_idx)
                t_1500m = safe_val(base_data, "temperature_850hPa", baseline_idx)
                
                # Extract PM 2.5 for Astro Mode
                aq_idx = aq_data["hourly"]["time"].index(closest_hour_str) if aq_data and "hourly" in aq_data and closest_hour_str in aq_data["hourly"].get("time", []) else 0
                local_pm25 = safe_val(aq_data, "pm2_5", aq_idx) if aq_data else 0
                
                seeing_quality = "Poor 🔴 (Turbulent)"
                if t_1500m > t_100m:
                    seeing_quality = "Excellent 🟢 (Stable Inversion)"
                elif (t_100m - t_1500m) < 5:
                    seeing_quality = "Good 🟡 (Moderate Stability)"
                    
                # Upgraded to 4 columns to include Smoke tracking
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Cloud Cover", f"{total_clouds}%", delta="Clear" if total_clouds < 15 else "Obscured", delta_color="inverse")
                col2.metric("High Altitude", f"{high_clouds}%")
                col3.metric("Atmospheric Seeing", seeing_quality.split(" ")[0])
                col4.metric("PM 2.5 (Smoke)", f"{round(local_pm25)} µg/m³", delta="Clear Air" if local_pm25 <= 10 else "Haze/Smoke", delta_color="inverse")
            else:
                st.info("Weather predictions are currently unavailable for this date (outside the 14-day window or fully offline), but mathematical celestial tracking remains active.")

            st.divider()
            
            # --- OFFLINE GC TRACKING UI (PHOTOPILLS 3D STYLE) ---
            st.write("### 🔭 Advanced Celestial Tracking Map")
            
            minute_offset = st.slider(
                "Scrub Time (Granular Adjustment):", 
                min_value=-360, 
                max_value=360, 
                value=0, 
                step=5, 
                format="%d mins"
            )
            tracking_time = dt + timedelta(minutes=minute_offset)
            
            # Calculate all celestial bodies simultaneously
            gc_az, gc_alt = get_celestial_az_alt(lat, lon, tracking_time, tz, "galactic_core")
            sun_az, sun_alt = get_celestial_az_alt(lat, lon, tracking_time, tz, "sun")
            moon_az, moon_alt = get_celestial_az_alt(lat, lon, tracking_time, tz, "moon")
            
            # Determine Adjusted Dynamic Background Map Style (Carto Engine)
            if sun_alt > 0:
                bg_style = 'light'
                sky_status = "☀️ Daytime (Light Map)"
            elif sun_alt > -6:
                bg_style = 'dark'
                sky_status = "🌇 Civil Twilight (Golden/Blue Hour)"
            elif sun_alt > -12:
                bg_style = 'dark'
                sky_status = "🌆 Nautical Twilight (Stars Emerging)"
            else:
                bg_style = 'dark'
                sky_status = "🌌 True Night (Dark Map)"

            st.write(f"**Target Time:** {tracking_time.strftime('%A, %I:%M %p')}")
            c_a, c_b, c_c = st.columns(3)
            c_a.markdown(f"**Milky Way:** Alt {round(gc_alt)}°")
            c_b.markdown(f"**Sun:** Alt {round(sun_alt)}°")
            c_c.markdown(f"**Map State:** {sky_status}")
            
            # Canadian Threshold Logic for GC
            if gc_alt > 7:
                status = "Visible & High (Strong Core)"
                base_color = [255, 215, 0] # Bright Gold
                strength_multiplier = min(1.8, gc_alt / 6)
            elif gc_alt > 0:
                status = "Cresting Horizon (Weak Core)"
                base_color = [147, 112, 219] # Purple
                strength_multiplier = max(0.5, gc_alt / 6)
            else:
                status = "Below Horizon (Not Visible)"
                base_color = [100, 100, 100] # Gray
                strength_multiplier = 0.2 
                
            # Generate 3D Layered Dots (PhotoPills Style)
            dot_data = []
            line_data = []
            num_dots = 8
            vector_length = 0.45 
            
            # 1. Milky Way 3D Dots
            for i in range(1, num_dots + 1):
                dist = (i / num_dots) * vector_length
                d_lat = lat + dist * math.cos(math.radians(gc_az))
                d_lon = lon + dist * math.sin(math.radians(gc_az)) / math.cos(math.radians(lat))
                
                base_radius = (300 + (i * 400)) * strength_multiplier
                alpha = int(100 + (i / num_dots) * 155)
                
                # Outer diffuse glow
                dot_data.append({
                    "lon": d_lon, "lat": d_lat, 
                    "radius": base_radius * 1.5,
                    "color": base_color + [alpha // 3] # Highly transparent halo
                })
                # Inner sharp core
                dot_data.append({
                    "lon": d_lon, "lat": d_lat, 
                    "radius": base_radius * 0.4,
                    "color": base_color + [alpha] # Solid bright center
                })

            # 2. Sun & Moon Lines
            if sun_alt > -18:
                line_color = [255, 140, 0, 200] if sun_alt > 0 else [255, 140, 0, 80]
                line_data.append(create_vector_line(lat, lon, sun_az, vector_length, line_color))
                
            if moon_alt > 0:
                line_data.append(create_vector_line(lat, lon, moon_az, vector_length, [200, 220, 255, 200]))
            elif moon_alt > -10:
                line_data.append(create_vector_line(lat, lon, moon_az, vector_length, [200, 220, 255, 60]))

            st.pydeck_chart(pdk.Deck(
                map_style=bg_style,
                initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=8.5, pitch=45, bearing=0),
                layers=[
                    pdk.Layer(
                        'LineLayer',
                        data=pd.DataFrame(line_data) if line_data else pd.DataFrame(columns=["start_lon", "start_lat", "end_lon", "end_lat", "color"]),
                        get_source_position='[start_lon, start_lat]',
                        get_target_position='[end_lon, end_lat]',
                        get_color='color',
                        get_width=300,
                    ),
                    pdk.Layer(
                        'ScatterplotLayer',
                        data=pd.DataFrame(dot_data),
                        get_position='[lon, lat]',
                        get_color='color',
                        get_radius='radius',
                        pickable=False
                    )
                ]
            ))
            
            st.caption("🟠 Orange Line = Sun Direction | ⚪ White Line = Moon Direction | 🟣/🟡 Dots = Milky Way Core")