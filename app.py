import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import datetime
import hashlib
import time
import math
import calendar
import requests
from streamlit_lottie import st_lottie
from streamlit_js_eval import get_geolocation

# Attempt import for Azure/OpenAI
try:
    from openai import AzureOpenAI, OpenAI
except ImportError:
    AzureOpenAI = None
    OpenAI = None

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================

# --- AZURE / OPENAI CREDENTIALS ---
AZURE_ENDPOINT = "https://healthcareassistant.openai.azure.com/"
AZURE_KEY = "9nZfZjY7lNvIibkaqQmFAVPJGzisfZI1DurHsiOxNbUaxp9bj5faJQQJ99BIACYeBjFXJ3w3AAABACOGl1vk"  # <-- replace with your key or store in env vars
AZURE_MODEL = "gpt-4o"
AZURE_API_VERSION = "2024-02-01"

# --- FILE PATHS ---
USERS_FILE = "users.csv"
DISEASE_FILE = "disease_reports_v2.csv"
PROFILES_FILE = "user_profiles.csv"

# --- ASSETS ---
LOTTIE_HEALTH_URL = "https://lottie.host/9c339797-4011-4475-a044-6a9cb57b7f43/4WnLqV9x6F.json"
LOTTIE_AI_URL = "https://lottie.host/575a7bc0-2183-4318-9d41-3945a6396b99/jQ8P9xW2x2.json"

# --- DATA LISTS ---
DISEASE_OPTIONS = [
    "Dengue", "Malaria", "Covid-19", "Flu (Influenza)", "Typhoid",
    "Chikungunya", "Common Cold", "Food Poisoning", "Migraine",
    "Chickenpox", "Conjunctivitis (Pink Eye)", "Allergies",
    "Gastroenteritis", "Pneumonia", "Tuberculosis"
]

CITY_COORDINATES = {
    "New Delhi": (28.6139, 77.2090), "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777), "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707), "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867), "Ahmedabad": (23.0225, 72.5714),
    "Pune": (18.5204, 73.8567), "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462), "Kanpur": (26.4499, 80.3319),
    "Nagpur": (21.1458, 79.0882), "Indore": (22.7196, 75.8577),
    "Patna": (25.5941, 85.1376), "Bhopal": (23.2599, 77.4126),
    "Visakhapatnam": (17.6868, 83.2185), "Vadodara": (22.3072, 73.1812),
    "Ludhiana": (30.9010, 75.8573), "Agra": (27.1767, 78.0081),
    "Nashik": (19.9975, 73.7898), "Guwahati": (26.1445, 91.7362),
    "Chandigarh": (30.7333, 76.7794), "Thiruvananthapuram": (8.5241, 76.9366),
    "Bhubaneswar": (20.2961, 85.8245), "Raipur": (21.2514, 81.6296)
}

SYMPTOMS_MAP = {
    "fever": ["Dengue", "Malaria", "Covid-19", "Flu", "Typhoid", "Chikungunya", "Pneumonia", "Chickenpox"],
    "headache": ["Dengue", "Flu", "Covid-19", "Migraine", "Common Cold", "Typhoid"],
    "joint pain": ["Dengue", "Chikungunya", "Flu"],
    "cough": ["Covid-19", "Flu", "Common Cold", "Pneumonia", "Tuberculosis"],
    "breathing": ["Covid-19", "Pneumonia", "Asthma", "Tuberculosis"],
    "chills": ["Malaria", "Flu", "Pneumonia"],
    "rash": ["Dengue", "Chickenpox", "Allergies"],
    "sneezing": ["Common Cold", "Flu", "Allergies"],
    "runny nose": ["Common Cold", "Flu", "Allergies"],
    "nausea": ["Food Poisoning", "Migraine", "Malaria", "Gastroenteritis"],
    "vomiting": ["Food Poisoning", "Typhoid", "Gastroenteritis"],
    "stomach pain": ["Food Poisoning", "Typhoid", "Gastroenteritis"],
    "itchy eyes": ["Conjunctivitis", "Allergies"],
    "sore throat": ["Common Cold", "Flu", "Covid-19"]
}

# --- CSS STYLING ---
CUSTOM_CSS = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    
    /* FORCE BACKGROUND & TEXT COLORS */
    .stApp { background-color: #f0f2f6; }
    
    /* Sidebar Fix: White Background, Dark Text */
    section[data-testid="stSidebar"] { 
        background-color: #ffffff; 
        border-right: 1px solid #e9ecef;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #333333 !important;
    }

    /* Metric Cards Fix */
    .metric-card {
        background-color: white; 
        padding: 20px; 
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); 
        text-align: center;
        transition: transform 0.2s;
        color: #333333 !important;
    }
    .metric-card h4, .metric-card span, .metric-card small, .metric-card b {
        color: #333333 !important;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 12px rgba(0, 0, 0, 0.1); }

    /* Risk Cards */
    .risk-container {
        padding: 20px; border-radius: 15px; color: white !important;
        text-align: center; margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .risk-container h1, .risk-container h2, .risk-container h3, .risk-container p {
        color: white !important;
    }
    .risk-safe { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .risk-moderate { background: linear-gradient(135deg, #f09819 0%, #edde5d 100%); }
    .risk-high { background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%); }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: white;
        border-radius: 8px; padding: 10px 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        font-weight: 600;
        color: #333333;
    }
    .stTabs [aria-selected="true"] { background-color: #e3f2fd; color: #0d47a1; border: 1px solid #bbdefb; }
    
    .stChatMessage { background-color: white; border-radius: 12px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); border: 1px solid #f0f0f0; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; height: 45px; border: none; transition: all 0.3s ease; }
    
    [data-testid="stMarkdownContainer"] p { color: inherit; }
    </style>
"""

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=2)
        if r.status_code != 200: return None
        return r.json()
    except Exception:
        return None


def get_nearest_city_name(lat, lng):
    closest_city = "Detected Location"
    min_dist = float('inf')
    for city, (c_lat, c_lng) in CITY_COORDINATES.items():
        dist = math.sqrt((lat - c_lat)**2 + (lng - c_lng)**2)
        if dist < min_dist:
            min_dist = dist
            closest_city = city
    if min_dist < 0.5: return closest_city
    return "GPS Location"


def get_weather_health_analysis(lat, lng):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&daily=uv_index_max&timezone=auto"
        response = requests.get(url, timeout=3)
        data = response.json()
        current = data.get("current", {})
        daily = data.get("daily", {})
        temp = current.get("temperature_2m", 0)
        humidity = current.get("relative_humidity_2m", 0)
        uv = daily.get("uv_index_max", [0])[0] if daily.get("uv_index_max") else 0
        
        health_risks = []
        if temp > 35: health_risks.append(("🔥 High Heat", "Stay hydrated."))
        elif temp < 10: health_risks.append(("❄️ Cold Front", "Keep warm."))
        if humidity > 80: health_risks.append(("💧 High Humidity", "Asthma warning."))
        elif humidity < 20: health_risks.append(("🏜️ Dry Air", "Throat irritation."))
        if uv > 7: health_risks.append(("☀️ High UV", "Wear sunscreen."))

        return {"temp": temp, "humidity": humidity, "uv": uv, "risks": health_risks}
    except Exception:
        return None

# ==========================================
# 3. AUTH & DATA MANAGERS
# ==========================================

class AuthManager:
    def __init__(self):
        self.users_file = USERS_FILE
        # users.csv now stores username, password_hash, age, gender
        if not os.path.exists(self.users_file):
            pd.DataFrame(columns=["username", "password_hash", "age", "gender"]).to_csv(self.users_file, index=False)

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def signup(self, username, password, age=None, gender=None):
        try:
            df = pd.read_csv(self.users_file)
            if username in df["username"].values:
                return False, "Username already exists."
            new_user = pd.DataFrame([{"username": username, "password_hash": self.hash_password(password), "age": age, "gender": gender}])
            new_user.to_csv(self.users_file, mode="a", header=False, index=False)
            return True, "Account created! Please login."
        except Exception as e:
            return False, f"Error creating account: {e}"

    def login(self, username, password):
        try:
            df = pd.read_csv(self.users_file)
            user_row = df[df["username"] == username]
            if not user_row.empty:
                if user_row.iloc[0]["password_hash"] == self.hash_password(password):
                    return True
        except Exception:
            pass
        return False

    def get_user_info(self, username):
        try:
            df = pd.read_csv(self.users_file)
            user_row = df[df["username"] == username]
            if not user_row.empty:
                row = user_row.iloc[0]
                return {"age": int(row.get("age")) if not pd.isna(row.get("age")) else None,
                        "gender": row.get("gender") if not pd.isna(row.get("gender")) else None}
        except Exception:
            pass
        return {"age": None, "gender": None}

    def update_user_info(self, username, age=None, gender=None):
        try:
            df = pd.read_csv(self.users_file)
            if username in df["username"].values:
                if age is not None:
                    df.loc[df["username"] == username, "age"] = age
                if gender is not None:
                    df.loc[df["username"] == username, "gender"] = gender
                df.to_csv(self.users_file, index=False)
                return True
            else:
                return False
        except Exception as e:
            print(f"Error updating user info: {e}")
            return False

class DataManager:
    def __init__(self):
        self.csv_file = DISEASE_FILE
        self.profiles_file = PROFILES_FILE
        if not os.path.exists(self.csv_file): self.init_csv()
        if not os.path.exists(self.profiles_file): self.init_profiles_db()

    def init_csv(self):
        df = pd.DataFrame(columns=["lat", "lng", "disease", "weight", "user_id", "timestamp"])
        df.to_csv(self.csv_file, index=False)

    def init_profiles_db(self):
        pd.DataFrame(columns=["username", "medical_history", "last_updated"]).to_csv(self.profiles_file, index=False)

    def get_user_profile(self, username):
        try:
            df = pd.read_csv(self.profiles_file)
            user_data = df[df["username"] == username]
            if not user_data.empty: return user_data.iloc[0]["medical_history"]
        except Exception:
            return None
        return None

    def update_user_profile(self, username, history_text):
        new_row = {"username": username, "medical_history": history_text, "last_updated": datetime.datetime.now()}
        try:
            if os.path.exists(self.profiles_file):
                df = pd.read_csv(self.profiles_file)
                if "username" not in df.columns:
                    df["username"] = ""
                usernames = df["username"].fillna("").astype(str)
                if username in usernames.values:
                    df.loc[df["username"].fillna("").astype(str) == username, "medical_history"] = history_text
                    df.loc[df["username"].fillna("").astype(str) == username, "last_updated"] = datetime.datetime.now()
                else:
                    new_df = pd.DataFrame([new_row])
                    df = pd.concat([df, new_df], ignore_index=True)
            else:
                df = pd.DataFrame([new_row])
            df.to_csv(self.profiles_file, index=False)
            return True
        except PermissionError:
            return False
        except Exception as e:
            print(f"Error updating user profile: {e}")
            return False

    def add_report(self, lat, lng, disease, weight, user_id):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_data = pd.DataFrame([{
            "lat": float(lat), "lng": float(lng), "disease": str(disease),
            "weight": int(weight), "user_id": str(user_id), "timestamp": ts
        }])
        try:
            header = not os.path.exists(self.csv_file)
            # Write atomically and flush to disk
            with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
                new_data.to_csv(f, mode="a", header=header, index=False)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
            return True, "Report submitted."
        except PermissionError:
            return False, "❌ Error: Please close the CSV file if open."
        except Exception as e:
            return False, f"Error: {e}"

    def import_bulk_data(self, uploaded_file):
        try:
            raw_df = pd.read_csv(uploaded_file)
            converted_data = []
            for _, row in raw_df.iterrows():
                city = str(row.get('City', '')).strip()
                disease = row.get('Disease', 'Unknown')
                cases = row.get('Cases', 1)
                lat, lng = 0.0, 0.0
                if city in CITY_COORDINATES:
                    lat, lng = CITY_COORDINATES[city]
                if lat != 0.0:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    converted_data.append({
                        'lat': float(lat), 'lng': float(lng), 'disease': str(disease),
                        'weight': int(cases), 'user_id': 'bulk_import', 'timestamp': ts
                    })
            if converted_data:
                new_df = pd.DataFrame(converted_data)
                try:
                    header = not os.path.exists(self.csv_file)
                    new_df.to_csv(self.csv_file, mode='a', header=header, index=False)
                except PermissionError:
                    return False, "❌ Error: Unable to write to file. Please ensure the file is not open in another program."
                return True, f"Imported {len(new_df)}."
            else:
                return False, "No valid cities."
        except Exception as e:
            return False, f"Error: {e}"

    def get_data(self, user_filter=None):
        if not os.path.exists(self.csv_file): return pd.DataFrame()
        try:
            df = pd.read_csv(self.csv_file)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
                # If conversion failed, fill missing timestamps with now so recent reports are included
                df.loc[df["timestamp"].isna(), "timestamp"] = datetime.datetime.now()
            if "lat" in df.columns and "lng" in df.columns:
                df["lat"] = pd.to_numeric(df["lat"], errors='coerce')
                df["lng"] = pd.to_numeric(df["lng"], errors='coerce')
                df = df.dropna(subset=["lat", "lng"])
            if "weight" in df.columns:
                df["weight"] = pd.to_numeric(df["weight"], errors='coerce').fillna(1)
            if user_filter:
                return df[df["user_id"] == user_filter]
            return df
        except Exception as e:
            print(f"Error reading data file: {e}")
            return pd.DataFrame()

    def get_filtered_data(self, user_lat, user_lng, radius_km, mode, lookback_days=30, selected_year=None, selected_month=None):
        df = self.get_data()
        if df.empty: return pd.DataFrame(), 0, "Unknown", pd.DataFrame()

        if mode == "Historical Archive" and selected_year and selected_month:
            df_filtered = df[(df['timestamp'].dt.year == selected_year) & (df['timestamp'].dt.month == selected_month)].copy()
        else:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=lookback_days)
            df_filtered = df[df["timestamp"] >= cutoff_date].copy()

        if df_filtered.empty: return pd.DataFrame(), 0, "Safe", df_filtered

        df_filtered["distance_km"] = df_filtered.apply(lambda row: calculate_distance(user_lat, user_lng, row["lat"], row["lng"]), axis=1)
        nearby_cases = df_filtered[df_filtered["distance_km"] <= radius_km].copy()

        total_risk = nearby_cases["weight"].sum()
        risk_level = "Safe"
        if total_risk > 0: risk_level = "Moderate"
        if total_risk > 15: risk_level = "High"

        return nearby_cases, round(total_risk, 1), risk_level, df_filtered

# ==========================================
# 4. AI LOGIC
# ==========================================

def get_ai_client():
    if not AZURE_KEY or AZURE_KEY.startswith("<REDACTED>"): return None
    try:
        if AzureOpenAI and "azure" in AZURE_ENDPOINT:
            return AzureOpenAI(api_version=AZURE_API_VERSION, azure_endpoint=AZURE_ENDPOINT, api_key=AZURE_KEY)
        elif OpenAI:
            return OpenAI(base_url=AZURE_ENDPOINT, api_key=AZURE_KEY)
    except Exception:
        return None


def generate_daily_briefing(weather_data, nearby_df, user_profile):
    client = get_ai_client()
    if not client: return "AI Configuration Missing."

    disease_context = "No outbreaks."
    if nearby_df is not None and not nearby_df.empty:
        counts = nearby_df['disease'].value_counts().to_string()
        disease_context = f"Outbreaks nearby:{counts}"

    weather_context = f"Temp: {weather_data['temp']}C" if weather_data else "No weather data."
    profile_context = user_profile if user_profile else "None."

    system_prompt = """You are a Proactive Medical Intelligence System. 
    Provide a 'Daily Health Briefing'.
    Structure:
    1. 🛡️ **Immune Outlook**: One sentence risk summary.
    2. ⚠️ **Primary Threat**: Biggest risk (weather/disease).
    3. ✅ **Action Plan**: 3 short bullet points.
    Keep it concise."""

    user_prompt = f"Profile: {profile_context} Weather: {weather_context} Diseases: {disease_context}"


    try:
        completion = client.chat.completions.create(
            model=AZURE_MODEL, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], max_tokens=300
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"


def generate_trend_analysis(daily_counts_dict):
    client = get_ai_client()
    if not client: return "AI Missing."
    if not daily_counts_dict: return "Not enough data."
    data_str = "".join([f"{k}: {v}" for k, v in daily_counts_dict.items()])
    system_prompt = "You are an Epidemiologist. Analyze the daily case counts. 1. Trend (Rising/Falling). 2. Prediction (Next 3 days). 3. One recommendation. Max 50 words."
    try:
        completion = client.chat.completions.create(
            model=AZURE_MODEL, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Data: {data_str}"}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"


def generate_ai_response(user_input, risk_level, data_db):
    client = get_ai_client()
    if not client:
        return "AI Offline."
    profile = data_db.get_user_profile(st.session_state.get('username'))
    context = f"Profile: {profile} Risk Level: {risk_level}"
    try:
        completion = client.chat.completions.create(
            model=AZURE_MODEL,
            messages=[
                {"role": "system", "content": f"You are Vita AI, a medical assistant. Context: {context}. Be professional. Disclaimer: See a doctor."},
                {"role": "user", "content": user_input}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"

# ==========================================
# 5. UI COMPONENTS
# ==========================================

def login_user(username):
    st.session_state.logged_in = True
    st.session_state.username = username
    st.session_state.messages = []
    if "daily_briefing" in st.session_state: del st.session_state["daily_briefing"]


def logout_user():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.messages = []
    if "daily_briefing" in st.session_state: del st.session_state["daily_briefing"]
    st.rerun()


def show_login_page(auth_db):
    lottie_health = load_lottieurl(LOTTIE_HEALTH_URL)
    _, col2, _ = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if lottie_health:
            st_lottie(lottie_health, height=150, key="login_anim")
        st.markdown("<h1 style='text-align: center; color: #2c3e50;'>Vita AI</h1>", unsafe_allow_html=True)
        tabs = st.tabs(["🔐 Login", "📝 Sign Up"])
        with tabs[0]:
            with st.form("login_form"):
                user = st.text_input("Username")
                pw = st.text_input("Password", type="password")
                if st.form_submit_button("Access Dashboard"):
                    if auth_db.login(user, pw):
                        login_user(user)
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
        with tabs[1]:
            with st.form("signup_form"):
                new_user = st.text_input("New Username")
                new_pw = st.text_input("New Password", type="password")
                new_age = st.number_input("Age", min_value=0, max_value=120, value=25)
                new_gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
                if st.form_submit_button("Create Account"):
                    success, msg = auth_db.signup(new_user, new_pw, age=new_age, gender=new_gender)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)


def show_dashboard(data_db):
    # Force reload DataManager from disk so we always read fresh CSV data
    st.session_state.data_db = DataManager()
    data_db = st.session_state.data_db

    # Sidebar
    with st.sidebar:
        lottie_ai = load_lottieurl(LOTTIE_AI_URL)
        if lottie_ai: st_lottie(lottie_ai, height=100, key="sidebar_anim")
        st.markdown(f"### Hello, {st.session_state.username} 👋")
        
        st.markdown("#### 📍 Location Settings")
        location_mode = st.radio("Locate Me By:", ["Auto (GPS)", "Manual / City"]) 
        user_lat, user_lng = 28.6139, 77.2090
        display_city_name = "Unknown"

        if location_mode == "Auto (GPS)":
            loc_data = get_geolocation(component_key='get_geo')
            if loc_data:
                user_lat = loc_data['coords']['latitude']
                user_lng = loc_data['coords']['longitude']
                display_city_name = get_nearest_city_name(user_lat, user_lng)
                st.caption("✅ GPS Active")
            else:
                st.caption("GPS not available or permission denied.")
                if st.button("🔄 Refresh"): st.rerun()
        else:
            selected_city = st.selectbox("Select City", list(CITY_COORDINATES.keys()))
            if selected_city:
                user_lat, user_lng = CITY_COORDINATES[selected_city]
                display_city_name = selected_city
            with st.expander("Fine-tune Coordinates"):
                user_lat = st.number_input("Lat", value=user_lat, format="%.4f")
                user_lng = st.number_input("Lng", value=user_lng, format="%.4f")

        st.markdown("---")
        # Map scope: Nearby or Global
        scope = st.radio("Map Scope", ["Nearby (limited)", "Global"], index=0)
        radius_km = 5.0
        if scope == "Nearby (limited)":
            radius_km = st.slider("Radius (km)", 1.0, 200.0, 5.0, step=1.0)

        st.markdown("---")
        view_mode = st.radio("Data Source", ["Live / Recent", "Historical Archive"])
        lookback_days = 30
        selected_year, selected_month = None, None
        if view_mode == "Live / Recent":
            lookback_days = st.slider("Lookback (Days)", 1, 365, 30)
        else:
            col_y, col_m = st.columns(2)
            with col_y:
                current_year = datetime.datetime.now().year
                selected_year = st.selectbox("Year", range(current_year, current_year - 6, -1))
            with col_m:
                month_names = list(calendar.month_name)[1:]
                selected_month_name = st.selectbox("Month", month_names)
                selected_month = month_names.index(selected_month_name) + 1
        
        if st.button("Logout", type="secondary"): logout_user()

    # Recompute filtered data using fresh DataManager and current filters
    # If scope is global, we don't filter by distance; else we use radius_km
    if scope == "Global":
        # Use either all data or apply time-based filtering based on view_mode
        all_df = data_db.get_data()
        if all_df.empty:
            nearby_df, risk_score, risk_label, map_df = pd.DataFrame(), 0, "Safe", pd.DataFrame()
        else:
            if view_mode == "Historical Archive" and selected_year and selected_month:
                map_df = all_df[(all_df['timestamp'].dt.year == selected_year) & (all_df['timestamp'].dt.month == selected_month)].copy()
            else:
                cutoff_date = datetime.datetime.now() - datetime.timedelta(days=lookback_days)
                map_df = all_df[all_df['timestamp'] >= cutoff_date].copy()
            # For global view, show everything (map_df) and nearby_df is subset within radius for informational purposes
            if not map_df.empty:
                map_df['distance_km'] = map_df.apply(lambda row: calculate_distance(user_lat, user_lng, row['lat'], row['lng']), axis=1)
                nearby_df = map_df[map_df['distance_km'] <= radius_km].copy()
            else:
                nearby_df = pd.DataFrame()
            total_risk = nearby_df['weight'].sum() if not nearby_df.empty else 0
            risk_label = 'Safe' if total_risk == 0 else ('High' if total_risk > 15 else 'Moderate')
            risk_score = round(total_risk,1)
    else:
        nearby_df, risk_score, risk_label, df_filtered = data_db.get_filtered_data(
            user_lat, user_lng, radius_km, view_mode, lookback_days, selected_year, selected_month
        )
        map_df = df_filtered.copy() if not df_filtered.empty else nearby_df.copy()

    weather_data = get_weather_health_analysis(user_lat, user_lng)

    # Auto AI Briefing
    if "daily_briefing" not in st.session_state:
        user_profile = data_db.get_user_profile(st.session_state.username)
        if weather_data:
            with st.spinner("✨ AI is analyzing local disease risks..."):
                st.session_state.daily_briefing = generate_daily_briefing(weather_data, nearby_df, user_profile)
        else: st.session_state.daily_briefing = "Waiting for weather data..."

    # Main UI
    st.markdown("## 🛡️ Health Surveillance Dashboard")

    with st.container():
        c_head, c_btn = st.columns([6, 1])
        with c_head: st.markdown("### 📋 Daily Health Outlook")
        with c_btn:
            if st.button("🔄", help="Regenerate Briefing"):
                if "daily_briefing" in st.session_state: del st.session_state["daily_briefing"]
                st.rerun()
        st.markdown(f"""
        <div style="background-color: white; padding: 25px; border-radius: 12px; border-left: 6px solid #11998e; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
            <div style="color: #333; line-height: 1.6;">{st.session_state.get('daily_briefing', 'Loading...')}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    with c1:
        st.markdown(f"""<div class="risk-container risk-{risk_label.lower()}"><h3>{risk_label} Risk</h3><h1>{risk_score}</h1></div>""", unsafe_allow_html=True)
    with c2:
        if weather_data:
            st.markdown(f"""<div class="metric-card" style="text-align:left; padding:15px;"><h4>🌤️ Biometeorology</h4><p>📍 {display_city_name}</p><div style="display:flex; justify-content:space-between;"><span><b>{weather_data['temp']}°C</b><br><small>Temp</small></span><span><b>{weather_data['humidity']}%</b><br><small>Hum</small></span><span><b>{weather_data['uv']}</b><br><small>UV</small></span></div></div>""", unsafe_allow_html=True)
        else: st.markdown('<div class="metric-card">Weather Unavailable</div>', unsafe_allow_html=True)
    with c3:
        active_reports_count = len(nearby_df) if scope == "Nearby (limited)" else (len(map_df) if map_df is not None else 0)
        st.markdown(f"""<div class="metric-card"><h4>Active Reports</h4><h1>{active_reports_count}</h1></div>""", unsafe_allow_html=True)

    # Tabs
    tab_chat, tab_map, tab_trends, tab_alerts, tab_report, tab_profile = st.tabs(["🤖 AI Doctor", "📍 Nearby Map", "📈 Trends", "⚠️ Alerts", "📢 Report", "👤 Profile"])

    # Chat Tab
    with tab_chat:
        col_main, col_info = st.columns([3, 1])
        with col_info:
            st.info("Quick Actions")
            if st.button("🌡️ Fever?"):
                st.session_state.messages.append({"role": "user", "content": "I have fever."})
                st.rerun()
        with col_main:
            # Show chat history
            for msg in st.session_state.get('messages', []):
                if msg['role'] == 'user': st.chat_message('user').write(msg['content'])
                else: st.chat_message('assistant').write(msg['content'])

            prompt = st.chat_input("Symptoms...")
            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)
                resp = generate_ai_response(prompt, risk_label, data_db)
                st.session_state.messages.append({"role": "assistant", "content": resp})
                st.chat_message("assistant").write(resp)

    # Map Tab
    with tab_map:
        user_layer = pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([{"lat": user_lat, "lng": user_lng, "disease": "You"}]),
            get_position="[lng, lat]",
            get_color="[0, 128, 255, 200]",
            get_radius=200,
            pickable=False
        )
        layers = [user_layer]
        if not map_df.empty:
            if scope == "Global":
                # render global with smaller radii for clarity
                map_df['render_radius'] = map_df['weight'].fillna(1).clip(1,5) * 200
                layers.append(pdk.Layer("ScatterplotLayer", data=map_df, get_position="[lng, lat]", get_color="[200, 30, 0, 160]", get_radius="render_radius", pickable=True))
            else:
                map_df['render_radius'] = map_df['weight'].fillna(1)*60
                layers.append(pdk.Layer("ScatterplotLayer", data=map_df, get_position="[lng, lat]", get_color="[200, 30, 0, 160]", get_radius="render_radius", pickable=True))

        # Set view: zoom out for global, zoom closer for nearby
        if scope == "Global":
            if not map_df.empty:
                center_lat = float(map_df['lat'].mean())
                center_lng = float(map_df['lng'].mean())
            else:
                center_lat, center_lng = user_lat, user_lng
            view = pdk.ViewState(latitude=center_lat, longitude=center_lng, zoom=2)
        else:
            view = pdk.ViewState(latitude=user_lat, longitude=user_lng, zoom=12)

        # Use CSV mtime in the key so the chart re-renders when file changes, include scope
        map_mtime = 0
        try:
            map_mtime = int(os.path.getmtime(data_db.csv_file))
        except Exception:
            pass

        st.pydeck_chart(
            pdk.Deck(layers=layers, initial_view_state=view, tooltip={"text": "{disease}"}),
            key=f"map_{scope}_{len(map_df)}_{map_mtime}"
        )

    # Trends Tab
    with tab_trends:
        if not nearby_df.empty:
            nearby_df['date'] = nearby_df['timestamp'].dt.date
            daily_counts = nearby_df.groupby('date').size()
            c_g, c_a = st.columns([2,1])
            with c_g: st.line_chart(daily_counts)
            with c_a:
                st.markdown("### AI Forecast")
                if "trend_analysis" not in st.session_state:
                    st.session_state.trend_analysis = generate_trend_analysis({str(k):v for k,v in daily_counts.items()})
                st.info(st.session_state.trend_analysis)
                if st.button("Update Forecast"):
                    if "trend_analysis" in st.session_state: del st.session_state["trend_analysis"]
                    st.rerun()
        else: st.info("Need more data for trends.")

    # Alerts Tab
    with tab_alerts:
        if not nearby_df.empty:
            st.dataframe(nearby_df[["disease", "timestamp", "weight", "distance_km"]].sort_values("distance_km"), use_container_width=True)
        else:
            st.success("No alerts nearby.")

    # Report Tab
    with tab_report:
        c1, c2 = st.columns(2)
        with c1:
            with st.form("report_form"):
                lat_input = st.number_input("Lat", value=user_lat, format="%.4f")
                lng_input = st.number_input("Lng", value=user_lng, format="%.4f")
                disease = st.selectbox("Disease", DISEASE_OPTIONS)
                severity = st.slider("Severity", min_value=1, max_value=10, value=5, step=1)
                if st.form_submit_button("Submit Report"):
                    success, msg = data_db.add_report(lat_input, lng_input, disease, severity, st.session_state.username)
                    if success:
                        st.success("Submitted!")
                        # Refresh the DataManager so subsequent reads are fresh
                        st.session_state.data_db = DataManager()
                        # Optional debug: show last rows
                        try:
                            st.dataframe(pd.read_csv(data_db.csv_file).tail(5))
                        except Exception as e:
                            st.warning(f"Could not read CSV for preview: {e}")
                        time.sleep(0.6)
                        st.rerun()
                    else:
                        st.error(msg)
        with c2:
            st.info(f"Reporting for: **{display_city_name}**")

    # Profile Tab
    with tab_profile:
        c1, c2 = st.columns(2)
        with c1:
            hist = data_db.get_user_profile(st.session_state.username)
            with st.form("prof"):
                txt = st.text_area("Conditions", value=hist if hist else "")
                age_input = st.number_input("Age", min_value=0, max_value=120, value=25)
                gender_input = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"]) 
                if st.form_submit_button("Save"):
                    data_db.update_user_profile(st.session_state.username, txt)
                    st.session_state.auth_db.update_user_info(st.session_state.username, age=age_input, gender=gender_input)
                    st.success("Saved")
                    st.rerun()
        with c2:
            my = data_db.get_data(user_filter=st.session_state.username)
            if not my.empty:
                st.dataframe(my[["disease", "timestamp"]], key=f"my_t_{len(my)}")
            else:
                st.info("No reports.")

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    st.set_page_config(page_title="Vita AI", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    if "auth_db" not in st.session_state: st.session_state.auth_db = AuthManager()
    if "data_db" not in st.session_state: st.session_state.data_db = DataManager()
    if "logged_in" not in st.session_state: st.session_state.logged_in = False
    if "messages" not in st.session_state: st.session_state.messages = []

    if st.session_state.logged_in:
        show_dashboard(st.session_state.data_db)
    else:
        show_login_page(st.session_state.auth_db)
