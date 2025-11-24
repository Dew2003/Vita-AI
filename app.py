import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import datetime
import hashlib
from streamlit_js_eval import get_geolocation

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EpiMap: Real-Time Disease Surveillance",
    layout="wide",
    page_icon="🦠"
)

# --- CSS STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { width: 100%; border-radius: 5px; }
    .auth-box { padding: 40px; background: white; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 1. AUTHENTICATION MANAGER ---
class AuthManager:
    def __init__(self):
        self.users_file = 'users.csv'
        self.init_users_db()

    def init_users_db(self):
        """Initialize users CSV if it doesn't exist"""
        if not os.path.exists(self.users_file):
            df = pd.DataFrame(columns=['username', 'password_hash'])
            df.to_csv(self.users_file, index=False)

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def signup(self, username, password):
        df = pd.read_csv(self.users_file)
        if username in df['username'].values:
            return False, "Username already exists."
        
        new_user = pd.DataFrame([{
            'username': username, 
            'password_hash': self.hash_password(password)
        }])
        new_user.to_csv(self.users_file, mode='a', header=False, index=False)
        return True, "Account created! Please login."

    def login(self, username, password):
        if not os.path.exists(self.users_file):
            return False
        df = pd.read_csv(self.users_file)
        user_row = df[df['username'] == username]
        
        if not user_row.empty:
            stored_hash = user_row.iloc[0]['password_hash']
            if stored_hash == self.hash_password(password):
                return True
        return False

# --- 2. DATA MANAGER (Updated for User IDs) ---
class DataManager:
    def __init__(self):
        self.csv_file = 'disease_reports_v2.csv' # v2 for multi-user support
        self.init_csv()

    def init_csv(self):
        """Initialize CSV with 'user_id' column"""
        if not os.path.exists(self.csv_file):
            dummy_data = {
                'lat': [28.6139, 19.0760, 12.9716],
                'lng': [77.2090, 72.8777, 77.5946],
                'disease': ["Dengue", "Malaria", "Flu"],
                'weight': [5, 8, 3],
                'user_id': ['admin', 'admin', 'admin'], # Tagged as admin
                'timestamp': [datetime.datetime.now()] * 3
            }
            df = pd.DataFrame(dummy_data)
            df.to_csv(self.csv_file, index=False)

    def add_report(self, lat, lng, disease, weight, user_id):
        """Add data tagged with the specific user_id"""
        new_data = pd.DataFrame([{
            'lat': lat, 'lng': lng, 
            'disease': disease, 'weight': weight,
            'user_id': user_id,
            'timestamp': datetime.datetime.now()
        }])
        new_data.to_csv(self.csv_file, mode='a', header=False, index=False)

    def get_data(self, user_filter=None):
        """
        Fetch data. 
        If user_filter is None -> Returns Global Data.
        If user_filter is set -> Returns only that user's data.
        """
        if os.path.exists(self.csv_file):
            df = pd.read_csv(self.csv_file)
            if user_filter:
                return df[df['user_id'] == user_filter]
            return df
        return pd.DataFrame()

# Initialize Managers
auth_db = AuthManager()
data_db = DataManager()

# --- 3. SESSION STATE MANAGEMENT ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

def login_user(username):
    st.session_state.logged_in = True
    st.session_state.username = username

def logout_user():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.rerun()

# --- 4. VIEW: LOGIN PAGE ---
def show_login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 EpiMap Secure Access")
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                user = st.text_input("Username")
                pw = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")
                if submit:
                    if auth_db.login(user, pw):
                        login_user(user)
                        st.success("Welcome back!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

        with tab2:
            with st.form("signup_form"):
                new_user = st.text_input("New Username")
                new_pw = st.text_input("New Password", type="password")
                submit_signup = st.form_submit_button("Create Account")
                if submit_signup:
                    if new_user and new_pw:
                        success, msg = auth_db.signup(new_user, new_pw)
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)
                    else:
                        st.warning("Please fill all fields.")

# --- 5. VIEW: MAIN DASHBOARD ---
def show_dashboard():
    # --- Sidebar (User Profile) ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=50)
        st.write(f"Hello, **{st.session_state.username}**! 👋")
        if st.button("Logout", type="secondary"):
            logout_user()
        
        st.divider()
        st.header("📢 Report Case")
        
        # Geolocation
        location = get_geolocation()
        default_lat, default_lng = 28.6139, 77.2090 
        if location:
            default_lat = location['coords']['latitude']
            default_lng = location['coords']['longitude']
            st.success("📍 Location Active")

        with st.form("report_form"):
            lat_input = st.number_input("Lat", value=default_lat, format="%.4f")
            lng_input = st.number_input("Lng", value=default_lng, format="%.4f")
            disease = st.selectbox("Disease", ["Dengue", "Malaria", "Covid-19", "Flu"])
            severity = st.slider("Severity", 1, 10, 5)
            
            if st.form_submit_button("🔴 Submit Report"):
                data_db.add_report(lat_input, lng_input, disease, severity, st.session_state.username)
                st.success("Report added to global & personal records!")
                st.rerun()

    # --- Main Content Tabs ---
    st.title("🌍 Real-Time Disease Surveillance")
    
    tab_global, tab_personal = st.tabs(["🌐 Global Heatmap", "👤 My Data"])
    
    # TAB 1: GLOBAL DATA (Everyone's data)
    with tab_global:
        df_global = data_db.get_data(user_filter=None)
        st.markdown(f"**Global Live Feed:** {len(df_global)} reports from all users.")
        
        if not df_global.empty:
            view_state = pdk.ViewState(
                latitude=df_global["lat"].mean(),
                longitude=df_global["lng"].mean(),
                zoom=4, pitch=45
            )
            heatmap = pdk.Layer(
                "HeatmapLayer", data=df_global,
                get_position='[lng, lat]', get_weight='weight',
                opacity=0.8, pickable=False
            )
            scatter = pdk.Layer(
                "ScatterplotLayer", data=df_global,
                get_position='[lng, lat]', get_color='[200, 30, 0, 160]',
                get_radius=20000, pickable=True
            )
            r = pdk.Deck(
                layers=[heatmap, scatter],
                initial_view_state=view_state,
                tooltip={"text": "Disease: {disease}\nReported By: {user_id}"},
                map_style=None
            )
            st.pydeck_chart(r)
    
    # TAB 2: PERSONAL DATA (Only my data)
    with tab_personal:
        df_my = data_db.get_data(user_filter=st.session_state.username)
        st.subheader("Your Contribution History")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("My Reports", len(df_my))
        with col_b:
            if not df_my.empty:
                st.metric("Most Reported", df_my['disease'].mode()[0])
        
        if not df_my.empty:
            st.dataframe(df_my[['timestamp', 'disease', 'weight', 'lat', 'lng']], use_container_width=True)
            
            # Simple Map for Personal Data
            st.caption("Your Reported Locations")
            # FIX: Rename 'lng' to 'lon' so Streamlit recognizes the column
            st.map(df_my[['lat', 'lng']].rename(columns={'lng': 'lon'}))
        else:
            st.info("You haven't submitted any reports yet.")

# --- MAIN APP FLOW ---
if st.session_state.logged_in:
    show_dashboard()
else:
    show_login_page()