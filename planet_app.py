from datetime import datetime
import glob
import json
import os
import random
import sys
import urllib.parse
import pandas as pd
from PIL import Image
import requests
import sqlite3
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# -------------------------
# 📌 VERSION & GITHUB CONFIG
# -------------------------
CURRENT_VERSION = "12.7.0"
VERSION_URL = "https://raw.githubusercontent.com/Major309Robin/Plant-Disease-DetectorUpdater/refs/heads/main/version.txt"
GITHUB_API_URL = "https://api.github.com/repos/Major309Robin/Plant-Disease-DetectorUpdater/contents"

# إعدادات الصفحة
st.set_page_config(
    page_title="Plant Disease Detector", page_icon="🌿", layout="wide"
)

# إنشاء مجلد الأرشيف للصور
ARCHIVE_DIR = "scans_archive"
if not os.path.exists(ARCHIVE_DIR):
    os.makedirs(ARCHIVE_DIR)

# إنشاء مجلد لقاعدة البيانات
DB_FOLDER = "Database"
if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

# -------------------------
# 🔑 API KEY & USER DB FILE HANDLING
# -------------------------
API_KEY_FILE = "Groq API Key.txt"
USER_DB_PATH = os.path.join(DB_FOLDER, "users.db")


def init_user_db():
    conn = sqlite3.connect(USER_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            username TEXT,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()


init_user_db()


def save_user_credentials(email, username, password):
    conn = sqlite3.connect(USER_DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO users (email, username, password)
            VALUES (?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET username=excluded.username, password=excluded.password
        """,
            (email, username, password),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving user: {e}")
        conn.close()
        return False


def load_saved_api_key():
    if os.path.exists(API_KEY_FILE):
        try:
            with open(API_KEY_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        except:
            return ""
    return ""


def save_api_key_to_file(key):
    try:
        with open(API_KEY_FILE, "w", encoding="utf-8") as f:
            f.write(key.strip())
    except Exception as e:
        print(f"Error saving API key: {e}")


# -------------------------
# 🚪 ONBOARDING / WELCOME GATE (MANDATORY USER CREDENTIALS FIRST)
# -------------------------
if "auth_status" not in st.session_state:
    st.session_state.auth_status = False

if "onboarding_mode" not in st.session_state:
    st.session_state.onboarding_mode = None

if "sent_code" not in st.session_state:
    st.session_state.sent_code = None

if "code_sent_status" not in st.session_state:
    st.session_state.code_sent_status = False

saved_key_init = load_saved_api_key()

if not st.session_state.auth_status and not saved_key_init:
    st.markdown(
        """
        <style>
        .stApp { background-color: #0e1117; color: #ffffff; }
        .welcome-card {
            background-color: #161a25;
            padding: 40px;
            border-radius: 15px;
            border: 1px solid #262730;
            text-align: center;
            max-width: 650px;
            margin: 40px auto;
        }
        .groq-link-btn {
            display: inline-block;
            background-color: #f55036;
            color: #ffffff !important;
            padding: 10px 20px;
            font-weight: bold;
            border-radius: 8px;
            text-decoration: none !important;
            margin-top: 10px;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="welcome-card">
            <h1>🌿 Welcome to Plant Disease Detector</h1>
            <p>Please enter your account details and choose how to connect your Groq AI Assistant:</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # خانات تسجيل الإيميل واليوزرنيم والباسورد الإجبارية في البداية لكل الحالات
    st.markdown("### 🔐 User Credentials Registration (Required)")
    with st.form(key="mandatory_user_signup_form"):
        user_email = st.text_input("Email Address:")
        username = st.text_input("Username:")
        password = st.text_input("Password:", type="password")
        submit_user_data = st.form_submit_button(label="Save Credentials 💾")

    if submit_user_data:
        if user_email and "@" in user_email and username and password:
            save_user_credentials(user_email.strip(), username.strip(), password.strip())
            st.success(f"✅ User credentials saved successfully for: {username}")
        else:
            st.error("⚠️ Please fill in all user credential fields correctly (Email, Username, Password).")

    st.markdown("---")
    st.markdown("### Choose Groq Connection Mode:")
    col_g1, col_g2, col_g3 = st.columns(3)

    with col_g1:
        if st.button("🔑 I have Groq Key", use_container_width=True):
            st.session_state.onboarding_mode = "have_key"
            st.rerun()

    with col_g2:
        if st.button("📧 I don't have Groq Key", use_container_width=True):
            st.session_state.onboarding_mode = "no_key"
            st.session_state.code_sent_status = False
            st.rerun()

    with col_g3:
        if st.button("⏭️ Skip", use_container_width=True):
            st.warning("⚠️ Warning: The app will run without full AI chat features!")
            st.session_state.auth_status = True
            st.rerun()

    if st.session_state.onboarding_mode == "have_key":
        st.markdown("---")
        st.markdown("### Enter your Groq API Key:")
        entered_key = st.text_input("API Key", type="password", key="input_groq_k")
        if st.button("Verify & Continue"):
            if entered_key.strip():
                try:
                    test_headers = {
                        "Authorization": f"Bearer {entered_key.strip()}",
                        "Content-Type": "application/json",
                    }
                    test_payload = {
                        "model": "openai/gpt-oss-20b",
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                    res = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=test_headers,
                        json=test_payload,
                        timeout=5,
                    )
                    if res.status_code in [200, 400]:
                        save_api_key_to_file(entered_key.strip())
                        st.success("✅ Key verified successfully!")
                        st.session_state.auth_status = True
                        st.rerun()
                    else:
                        st.error("❌ Invalid API Key. Please check and try again.")
                except Exception as e:
                    save_api_key_to_file(entered_key.strip())
                    st.session_state.auth_status = True
                    st.rerun()
            else:
                st.error("Please enter a valid key.")

    elif st.session_state.onboarding_mode == "no_key":
        st.markdown("---")
        if not st.session_state.code_sent_status:
            if st.button("Generate Verification Code"):
                generated_code = str(random.randint(1000, 9999))
                st.session_state.sent_code = generated_code
                st.session_state.code_sent_status = True
                st.success("📥 Verification code generated successfully!")
                st.rerun()
        else:
            st.info(f"🔐 Your Secure Verification Code is: **{st.session_state.sent_code}**")
            entered_code = st.text_input("Enter 4-digit Verification Code", key="user_code_input")
            if st.button("Confirm Code & Setup Key"):
                if entered_code.strip() == str(st.session_state.sent_code):
                    st.success("✅ Verification successful!")
                    st.markdown("Get your official Groq API key from Groq Console:")
                    st.markdown('<a href="https://console.groq.com/keys" target="_blank" class="groq-link-btn">👉 Get Real Groq Key</a>', unsafe_allow_html=True)
                    
                    final_groq_input = st.text_input("Paste your Groq Key here (starts with gsk_):", type="password", key="final_gsk")
                    if st.button("Save & Start Application"):
                        if final_groq_input.strip().startswith("gsk_"):
                            save_api_key_to_file(final_groq_input.strip())
                            st.success("✅ Groq Key Saved Successfully!")
                            st.session_state.auth_status = True
                            st.rerun()
                        else:
                            st.error("Please enter a valid Groq key starting with gsk_")
                else:
                    st.error("❌ Incorrect verification code. Please try again.")

    st.stop()

# -------------------------
# 🗄️ SQLITE DATABASE SETUP
# -------------------------
DB_NAME = os.path.join(DB_FOLDER, "plant_scans.db")


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            plant_name TEXT,
            disease_name TEXT,
            plant_conf REAL,
            disease_conf REAL,
            status TEXT,
            image_path TEXT,
            report_text TEXT,
            chat_history TEXT,
            weather_info TEXT,
            season_info TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()


def save_scan_to_db(
    timestamp,
    plant,
    disease,
    p_conf,
    d_conf,
    status,
    img_path,
    report,
    chat_history="[]",
    weather_info="",
    season_info="",
):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE scans ADD COLUMN weather_info TEXT")
    except:
        pass
    try:
        cursor.execute("ALTER TABLE scans ADD COLUMN season_info TEXT")
    except:
        pass

    cursor.execute(
        """
        INSERT INTO scans (timestamp, plant_name, disease_name, plant_conf, disease_conf, status, image_path, report_text, chat_history, weather_info, season_info)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            timestamp,
            plant,
            disease,
            p_conf,
            d_conf,
            status,
            img_path,
            report,
            chat_history,
            weather_info,
            season_info,
        ),
    )
    conn.commit()
    conn.close()


def update_scan_chat_in_db(scan_id, chat_history):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE scans SET chat_history = ? WHERE id = ?
    """,
        (chat_history, scan_id),
    )
    conn.commit()
    conn.close()


def get_db_stats():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM scans")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM scans WHERE status = 'Healthy'")
    healthy = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM scans WHERE status = 'Diseased'")
    diseased = cursor.fetchone()[0]
    conn.close()
    return total, healthy, diseased


def get_recent_scans(limit=3):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT plant_name, disease_name FROM scans ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


# -------------------------
# 🌐 TRANSLATION DICTIONARY
# -------------------------
translations = {
    "English": {
        "settings": "⚙️ Settings",
        "dark_mode": "🌙 Enable Dark Mode",
        "language": "🌐 Language",
        "control": "📊 Control & Analytics",
        "total_scans": "Total Scans",
        "healthy": "Healthy",
        "diseased": "Diseased",
        "reset": "🔄 Reset / Clear Scans & Archive",
        "faq_title": "💡 Photo Capture FAQ",
        "faq_text": (
            "1. Ensure good natural lighting.\n2. Focus clearly on the affected leaf.\n3. Avoid camera shaking.\n4. Upload JPG or PNG formats."
        ),
        "recent_scans": "📜 Recent Scans",
        "no_scans": "No scans yet.",
        "title": "🌿 Plant Disease Detector",
        "desc": (
            "Upload a plant image to identify species, detect diseases, and get treatment tips."
        ),
        "upload": "Upload Plant Image",
        "uploaded_img": "Uploaded Image",
        "analyzing": "⏳ Analyzing plant health...",
        "report_title": "🌱 Smart Diagnosis Report",
        "plant_type": "Plant Type",
        "condition": "Condition / Disease",
        "status": "General Status",
        "status_healthy": "✅ Healthy & Stable",
        "status_attention": "⚠ Attention Needed",
        "treatment": "💡 Recommended Treatment / Action:",
        "medicine_recommended": "💊 Recommended Treatment / Medicine:",
        "buy_btn_text": "Buy 🛒",
        "download_btn": "📥 Download Report (Text)",
        "copy_btn": "📋 Copy Report",
        "copy_success": "Report copied successfully below!",
        "export_csv_btn": "📊 Export Archive as CSV (Excel)",
        "archive_msg": "Scan and report securely saved to database and folder:",
        "detector_tab": "🌿 Plant Detector",
        "archive_tab": "📁 Database Archive & Search",
        "search_label": "🔍 Search Database (by plant or disease)",
        "filter_status": "Filter by Status",
        "all": "All",
        "no_archive_files": "No database records found matching your criteria.",
        "footer": "Developed by Robin John",
        "ai_chat_title": "🤖 Plant Doctor AI Assistant",
        "ai_chat_intro": (
            "Ask me anything about this plant and its condition (e.g., How to treat it? Can it spread?):"
        ),
        "ai_placeholder": "Type your question here...",
        "api_key_label": "🔑 Groq API Key:",
        "api_missing": (
            "Please enter your Groq API Key in the sidebar to chat with the Plant Doctor AI."
        ),
        "weather_section": "🌤️ Weather & Season Inspector",
        "city_label": "Enter City / Location",
        "current_season": "Current Season",
        "weather_status": "Weather Status",
        "planting_guide_title": "🌱 Seasonal Planting Guide",
        "planting_status_ok": (
            "✅ Great news! This plant is suitable for planting/growing right now in this season."
        ),
        "planting_status_no": (
            "⚠️ Note: This plant is typically not grown in the current season ({season}). Best seasons: {best_seasons}"
        ),
        "version_checker_title": "🔄 Version & Update Checker",
        "check_update": "Check Update",
        "current_v": "Current Version",
        "new_v": "New Version",
        "update_prompt": "An update to version **{v}** is available. Would you like to update?",
        "yes_update": "Yes, Update Now",
        "no_update": "No, Keep Current",
        "dont_show_again": "Don't show this update again",
        "up_to_date": "Up to date",
        "updating_msg": "📥 Downloading all files and applying update...",
        "update_success": "✅ Updated successfully! Restarting application...",
        "download_fail": "Failed to download update files.",
    },
    "العربية": {
        "settings": "⚙️ الإعدادات",
        "dark_mode": "🌙 تفعيل الوضع الليلي",
        "language": "🌐 اللغة",
        "control": "📊 التحكم والإحصائيات",
        "total_scans": "إجمالي الفحوصات",
        "healthy": "سليم",
        "diseased": "مصاب",
        "reset": "🔄 إعادة تعيين / مسح السجلات والأسئلة",
        "faq_title": "💡 Photo Capture FAQ",
        "faq_text": (
            "1. Ensure good natural lighting.\n2. Focus clearly on the affected leaf.\n3. Avoid camera shaking.\n4. Upload JPG or PNG formats."
        ),
        "recent_scans": "📜 الفحوصات الأخيرة",
        "no_scans": "لا توجد فحوصات بعد.",
        "title": "🌿 كشف أمراض النباتات",
        "desc": (
            "قم برفـع صورة نبات لتحديد نوعه، اكتشاف الأمراض، والحصول على نصائح العلاج."
        ),
        "upload": "رفع صورة النبات",
        "uploaded_img": "الصورة المرفوعة",
        "analyzing": "⏳ جاري تحليل حالة النبات...",
        "report_title": "🌱 تقرير التشخيص الذكي",
        "plant_type": "نوع النبات",
        "condition": "الحالة / المرض",
        "status": "الحالة العامة",
        "status_healthy": "✅ سليم ومستقر",
        "status_attention": "⚠ يحتاج إلى اهتمام وعلاج",
        "treatment": "💡 العلاج المقترح / الإجراءات المطلوبة:",
        "medicine_recommended": "💊 الدواء / المبيد المقترح للعلاج:",
        "buy_btn_text": "شراء 🛒",
        "download_btn": "📥 تحميل التقرير (ملف نصي)",
        "copy_btn": "📋 نسخ التقرير",
        "copy_success": "تم عرض التقرير للنسخ أدناه بنجاح!",
        "export_csv_btn": "📊 تصدير الأرشيف كملف CSV (Excel)",
        "archive_msg": "تم حفظ الفحص والتقرير بنجاح في قاعدة البيانات والمجلد:",
        "detector_tab": "🌿 كشف النباتات",
        "archive_tab": "📁 أرشيف قاعدة البيانات والبحث",
        "search_label": "🔍 بحث في قاعدة البيانات (بالنبات أو المرض)",
        "filter_status": "تصفية حسب الحالة",
        "all": "الكل",
        "no_archive_files": "لا توجد سجلات مطابقة للبحث.",
        "footer": "تم التطوير بواسطة روبن جون",
        "ai_chat_title": "🤖 Plant Doctor AI Assistant",
        "ai_chat_intro": (
            "Ask me anything about this plant and its condition (e.g., How to treat it? Can it spread?):"
        ),
        "ai_placeholder": "Type your question here...",
        "api_key_label": "🔑 Groq API Key:",
        "api_missing": (
            "Please enter your Groq API Key in the sidebar to chat with the Plant Doctor AI."
        ),
        "weather_section": "🌤️ Weather & Season Inspector",
        "city_label": "Enter City / Location",
        "current_season": "Current Season",
        "weather_status": "Weather Status",
        "planting_guide_title": "🌱 Seasonal Planting Guide",
        "planting_status_ok": (
            "✅ Great news! This plant is suitable for planting/growing right now in this season."
        ),
        "planting_status_no": (
            "⚠️ Note: This plant is typically not grown in the current season ({season}). Best seasons: {best_seasons}"
        ),
        "version_checker_title": "🔄 Version & Update Checker",
        "check_update": "Check Update",
        "current_v": "Current Version",
        "new_v": "New Version",
        "update_prompt": "An update to version **{v}** is available. Would you like to update?",
        "yes_update": "Yes, Update Now",
        "no_update": "No, Keep Current",
        "dont_show_again": "Don't show this update again",
        "up_to_date": "Up to date",
        "updating_msg": "📥 Downloading all files and applying update...",
        "update_success": "✅ Updated successfully! Restarting application...",
        "download_fail": "Failed to download update files.",
    },
}

# -------------------------
# ⚙ SIDEBAR SETTINGS
# -------------------------
st.sidebar.markdown(f"### {translations['English']['settings']}")
dark_mode = st.sidebar.toggle("🌙 Enable Dark Mode", value=True)
selected_lang = st.sidebar.selectbox(
    "🌐 Language", ["English", "العربية"], index=0, key="lang_selector"
)
t = translations[selected_lang]

# -------------------------
# 🔄 AUTOMATIC VERSION CHECKER
# -------------------------
if "ignored_version" not in st.session_state:
    st.session_state.ignored_version = None

if "check_clicked" not in st.session_state:
    st.session_state.check_clicked = False

st.sidebar.markdown("---")
st.sidebar.subheader(t["version_checker_title"])

if st.sidebar.button(t["check_update"]):
    st.session_state.check_clicked = True

if st.session_state.check_clicked:
    try:
        res_v = requests.get(VERSION_URL, timeout=3)
        if res_v.status_code == 200:
            remote_version = res_v.text.strip()
            if (
                remote_version != CURRENT_VERSION
                and remote_version != st.session_state.ignored_version
            ):
                st.sidebar.warning(t["update_prompt"].format(v=remote_version))
                st.sidebar.text(f"{t['current_v']}: {CURRENT_VERSION}")
                st.sidebar.text(f"{t['new_v']}: {remote_version}")

                col_u1, col_u2 = st.sidebar.columns(2)
                with col_u1:
                    if st.button(t["yes_update"]):
                        with st.spinner(t["updating_msg"]):
                            try:
                                repo_res = requests.get(GITHUB_API_URL, timeout=10)
                                if repo_res.status_code == 200:
                                    files_data = repo_res.json()
                                    
                                    current_file_path = os.path.abspath(__file__)
                                    app_dir = os.path.dirname(current_file_path)
                                    root_dir = os.path.dirname(app_dir)
                                    
                                    downloaded_files = []
                                    for file_info in files_data:
                                        file_name = file_info.get("name")
                                        download_url = file_info.get("download_url")
                                        
                                        if file_name and download_url and file_name != "version.txt" and file_name != "Updater.bat":
                                            file_res = requests.get(download_url, timeout=10)
                                            if file_res.status_code == 200:
                                                if file_name == "planet_app.py":
                                                    target_dir = app_dir
                                                else:
                                                    target_dir = root_dir
                                                    
                                                temp_path = os.path.join(target_dir, f"update_{file_name}")
                                                real_target_path = os.path.join(target_dir, file_name)
                                                
                                                if file_name.endswith(".py"):
                                                    raw_code = file_res.text
                                                    cleaned_code = "\n".join(
                                                        [
                                                            line
                                                            for line in raw_code.splitlines()
                                                            if line.strip() != ""
                                                        ]
                                                    )
                                                    with open(temp_path, "w", encoding="utf-8") as f:
                                                        f.write(cleaned_code)
                                                else:
                                                    with open(temp_path, "wb") as f:
                                                        f.write(file_res.content)
                                                        
                                                downloaded_files.append((temp_path, real_target_path))

                                    if downloaded_files:
                                        python_w_path = sys.executable.replace("python.exe", "pythonw.exe")
                                        if not os.path.exists(python_w_path):
                                            python_w_path = sys.executable
                                        
                                        bat_lines = [
                                            "@echo off",
                                            "chcp 65001 >nul",
                                            f'cd /d "{root_dir}"',
                                            "timeout /t 2 /nobreak >nul"
                                        ]
                                        
                                        for temp_p, real_p in downloaded_files:
                                            bat_lines.append(f'if exist "{real_p}" del /f /q "{real_p}"')
                                            bat_lines.append(f'move /y "{temp_p}" "{real_p}"')
                                            
                                        launcher_path = os.path.join(root_dir, "Plant Disease Detector.exe")
                                        if os.path.exists(launcher_path):
                                            bat_lines.append(f'start "" "{launcher_path}"')
                                        else:
                                            app_script_path = os.path.join(app_dir, "planet_app.py")
                                            bat_lines.append(f'start /b "" "{python_w_path}" -m streamlit run "{app_script_path}" --server.headless true')
                                            
                                        bat_lines.append('del "%~f0"')
                                        
                                        updater_bat_content = "\n".join(bat_lines)
                                        updater_bat_path = os.path.join(root_dir, "Updater.bat")
                                        
                                        with open(updater_bat_path, "w", encoding="utf-8") as bat_f:
                                            bat_f.write(updater_bat_content)

                                        st.success(t["update_success"])
                                        
                                        import subprocess
                                        subprocess.Popen(
                                            ['cmd.exe', '/c', updater_bat_path],
                                            creationflags=subprocess.CREATE_NO_WINDOW,
                                            cwd=root_dir
                                        )
                                        sys.exit()
                                    else:
                                        st.error(t["download_fail"])
                                else:
                                    st.error(t["download_fail"])
                            except Exception as e:
                                st.error(f"Update error: {e}")
                with col_u2:
                    if st.button(t["no_update"]):
                        st.session_state.ignored_version = remote_version
                        st.session_state.check_clicked = False
                        st.rerun()

                if st.sidebar.checkbox(t["dont_show_again"]):
                    st.session_state.ignored_version = remote_version
            else:
                st.sidebar.success(t["up_to_date"])
        else:
            st.sidebar.text(t["up_to_date"])
    except Exception:
        st.sidebar.text(t["up_to_date"])

st.sidebar.markdown("---")
saved_key_current = load_saved_api_key()
groq_api_key = st.sidebar.text_input(
    t["api_key_label"], value=saved_key_current, type="password"
)

if groq_api_key and groq_api_key != saved_key_current:
    save_api_key_to_file(groq_api_key)

# -------------------------
# 🎨 CSS STYLING
# -------------------------
if dark_mode:
    dynamic_css = """
    <style>
    header[data-testid="stHeader"] { background-color: #0e1117 !important; }
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #161a25; color: #ffffff; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #262730 !important;
        color: white !important;
        border-color: #444 !important;
    }
    div[data-testid="stFileUploader"] {
        background-color: #1e1e1e !important;
        border: 1px dashed #555 !important;
        border-radius: 10px;
        padding: 10px;
    }
    div[data-testid="stFileUploader"] section {
        background-color: #1e1e1e !important;
        color: white !important;
    }
    div[data-testid="stFileUploader"] button {
        background-color: #333333 !important;
        color: white !important;
        border: 1px solid #555 !important;
    }
    [data-testid="stSidebar"] button {
        background-color: #262730 !important;
        color: #ffffff !important;
        border: 1px solid #444 !important;
        border-radius: 8px;
    }
    div[data-baseweb="input"] {
        margin-top: 24px !important;
    }
    .custom-weather-box {
        background-color: #0b192c;
        border: 1px solid #1e3e62;
        color: #ffffff;
        padding: 11px 16px;
        border-radius: 8px;
        margin-top: 24px;
        font-size: 14px;
        display: flex;
        align-items: center;
        height: 42px;
    }
    .buy-btn {
        display: inline-block;
        background-color: #ff9900;
        color: #111111 !important;
        padding: 10px 22px;
        font-weight: bold;
        border-radius: 8px;
        text-decoration: none !important;
        border: none !important;
        box-shadow: none !important;
        margin-top: 8px;
    }
    .buy-btn:hover {
        background-color: #e68a00;
        color: #000000 !important;
        text-decoration: none !important;
    }
    </style>
    """
else:
    dynamic_css = """
    <style>
    header[data-testid="stHeader"] { background-color: #ffffff !important; }
    .stApp { background-color: #ffffff; color: #000000; }
    [data-testid="stSidebar"] { background-color: #f8f9fa; color: #000000; }
    [data-testid="stSidebar"] * { color: #000000 !important; }
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: black !important;
    }
    div[data-testid="stFileUploader"] {
        background-color: #f0f2f6 !important;
        border: 1px dashed #ccc !important;
        border-radius: 10px;
        padding: 10px;
    }
    div[data-baseweb="input"] {
        margin-top: 24px !important;
    }
    .custom-weather-box {
        background-color: #e8f0fe;
        border: 1px solid #d2e3fc;
        color: #174ea6;
        padding: 11px 16px;
        border-radius: 8px;
        margin-top: 24px;
        font-size: 14px;
        display: flex;
        align-items: center;
        height: 42px;
    }
    .buy-btn {
        display: inline-block;
        background-color: #ff9900;
        color: #111111 !important;
        padding: 10px 22px;
        font-weight: bold;
        border-radius: 8px;
        text-decoration: none !important;
        border: none !important;
        box-shadow: none !important;
        margin-top: 8px;
    }
    .buy-btn:hover {
        background-color: #e68a00;
        color: #000000 !important;
        text-decoration: none !important;
    }
    </style>
    """

st.markdown(dynamic_css, unsafe_allow_html=True)


# -------------------------
# Model Definition & Loading
# -------------------------
class PlantModel(nn.Module):

    def __init__(self, num_plants, num_diseases):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, 256)
        self.plant_head = nn.Linear(256, num_plants)
        self.disease_head = nn.Linear(256, num_diseases)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        plant_out = self.plant_head(x)
        disease_out = self.disease_head(x)
        return plant_out, disease_out


@st.cache_resource
def load_trained_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "model.pth"))

    net = PlantModel(num_plants=14, num_diseases=21)
    if os.path.exists(MODEL_PATH):
        net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    net.eval()
    return net


with st.spinner(t["analyzing"]):
    model = load_trained_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

idx_to_plant = {
    0: "Apple",
    1: "Blueberry",
    2: "Cherry_(including_sour)",
    3: "Corn_(maize)",
    4: "Grape",
    5: "Orange",
    6: "Peach",
    7: "Pepper,_bell",
    8: "Potato",
    9: "Raspberry",
    10: "Soybean",
    11: "Squash",
    12: "Strawberry",
    13: "Tomato",
}

idx_to_disease = {
    0: "Apple_scab",
    1: "Bacterial_spot",
    2: "Black_rot",
    3: "Cedar_apple_rust",
    4: "Cercospora_leaf_spot Gray_leaf_spot",
    5: "Common_rust",
    6: "Early_blight",
    7: "Esca_(Black_Measles)",
    8: "Haunglongbing_(Citrus_greening)",
    9: "Late_blight",
    10: "Leaf_Mold",
    11: "Leaf_blight_(Isariopsis_Leaf_Spot)",
    12: "Leaf_scorch",
    13: "Northern_Leaf_Blight",
    14: "Powdery_mildew",
    15: "Septoria_leaf_spot",
    16: "Spider_mites Two-spotted_spider_mite",
    17: "Target_Spot",
    18: "Tomato_Yellow_Leaf_Curl_Virus",
    19: "Tomato_mosaic_virus",
    20: "healthy",
}

plant_images_map = {
    "Apple": "https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?w=500",
    "Blueberry": "https://images.unsplash.com/photo-1498557850523-fd3d118b962e?w=500",
    "Cherry_(including_sour)": "https://images.unsplash.com/photo-1528825871115-3581a5387919?w=500",
    "Corn_(maize)": "https://images.unsplash.com/photo-1551754655-cd27e38d2076?w=500",
    "Grape": "https://images.unsplash.com/photo-1537640538966-79f369143f8f?w=500",
    "Orange": "https://images.unsplash.com/photo-1611080626919-7cf5a9dbab5b?w=500",
    "Peach": "https://images.unsplash.com/photo-1643142371900-3331b255474a?w=500",
    "Pepper,_bell": "https://images.unsplash.com/photo-1563565375-f3fdfdbefa83?w=500",
    "Potato": "https://images.unsplash.com/photo-1518977676601-b53f82aba655?w=500",
    "Raspberry": "https://images.unsplash.com/photo-1577069861033-55d04cec4ef5?w=500",
    "Soybean": "https://images.unsplash.com/photo-1599599810694-b5b37304c041?w=500",
    "Squash": "https://images.unsplash.com/photo-1506917728037-b6af01a7d403?w=500",
    "Strawberry": "https://images.unsplash.com/photo-1464965911861-746a04b4bca6?w=500",
    "Tomato": "https://images.unsplash.com/photo-1592924357228-91a4daadcfea?w=500",
}

disease_medicines_en = {
    "Apple_scab": {
        "name": "Copper Fungicide Spray",
        "image": "https://images.unsplash.com/photo-1585314062340-f1a5a7c9328d?w=500",
        "search_term": "Fungicide Spray for Plants"
    },
    "Bacterial_spot": {
        "name": "Copper Bactericide Spray",
        "image": "https://images.unsplash.com/photo-1585314062340-f1a5a7c9328d?w=500",
        "search_term": "Copper Fungicide Spray"
    },
    "Black_rot": {
        "name": "Liquid Copper Fungicide",
        "image": "https://images.unsplash.com/photo-1585314062340-f1a5a7c9328d?w=500",
        "search_term": "Fungicide Spray Plants"
    },
    "Cedar_apple_rust": {
        "name": "Plant Rust Fungicide Spray",
        "image": "https://images.unsplash.com/photo-1615485290382-441e4d049cb5?w=500",
        "search_term": "Plant Fungicide Spray"
    },
    "Cercospora_leaf_spot Gray_leaf_spot": {
        "name": "Chlorothalonil Fungicide Spray",
        "image": "https://images.unsplash.com/photo-1585314062340-f1a5a7c9328d?w=500",
        "search_term": "Plant Fungicide Spray"
    },
    "Common_rust": {
        "name": "Neem Oil Plant Fungicide",
        "image": "https://images.unsplash.com/photo-1615485290382-441e4d049cb5?w=500",
        "search_term": "Neem Oil Spray for Plants"
    },
    "Early_blight": {
        "name": "Copper Fungicide Spray",
        "image": "https://images.unsplash.com/photo-1585314062340-f1a5a7c9328d?w=500",
        "search_term": "Copper Fungicide Spray"
    },
    "Esca_(Black_Measles)": {
        "name": "Systemic Fungicide Spray",
        "image": "https://images.unsplash.com/photo-1585314062340-f1a5a7c9328d?w=500",
        "search_term": "Fungicide Spray"
    },
    "Haunglongbing_(Citrus_greening)": {
        "name": "Citrus Fertilizer & Insecticide",
        "image": "https://images.unsplash.com/photo-1615485290382-441e4d049cb5?w=500",
        "search_term": "Citrus Plant Fertilizer"
    },
    "Late_blight": {
        "name": "Mancozeb Copper Bio-Fungicide",
        "image": "https://images.unsplash.com/photo-1585314062340-f1a5a7c9328d?w=500",
        "search_term": "Plant Fungicide Spray"
    },
    "Leaf_Mold": {
        "name": "Organic Neem Oil Spray",
        "image": "https://images.unsplash.com/photo-1615485290382-441e4d049cb5?w=500",
        "search_term": "Organic Neem Oil Spray"
    },
    "Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "name": "Copper Bio-Fungicide",
        "image": "https://images.unsplash.com/photo-1585314062340-f1a5a7c9328d?w=500",
        "search_term": "Copper Fungicide Spray"
    },
    "Leaf_scorch": {
        "name": "Plant Micronutrient Booster Spray",
        "image": "https://images.unsplash.com/photo-1615485290382-441e4d049cb5?w=500",
        "search_term": "Plant Micronutrient Spray"
    },
    "Northern_Leaf_Blight": {
        "name": "Garden Fungicide Spray",
        "image": "https://images.unsplash.com/photo-1585314062340-f1a5a7c9328d?w=500",
        "search_term": "Garden Fungicide Spray"
    },
    "Powdery_mildew": {
        "name": "Powdery Mildew Neem Oil Spray",
        "image": "https://images.unsplash.com/photo-1615485290382-441e4d049cb5?w=500",
        "search_term": "Neem Oil Spray for Plants"
    },
    "Septoria_leaf_spot": {
        "name": "Liquid Copper Fungicide Tomato",
        "image": "https://images.unsplash.com/photo-1585314062340-f1a5a7c9328d?w=500",
        "search_term": "Copper Fungicide Spray"
    },
    "Spider_mites Two-spotted_spider_mite": {
        "name": "Insecticidal Soap / Miticide Spray",
        "image": "https://images.unsplash.com/photo-1615485290382-441e4d049cb5?w=500",
        "search_term": "Insecticidal Soap Spray"
    },
    "Target_Spot": {
        "name": "Plant Fungicide Spot Treatment",
        "image": "https://images.unsplash.com/photo-1585314062340-f1a5a7c9328d?w=500",
        "search_term": "Plant Fungicide Spray"
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Whitefly Insecticidal Soap",
        "image": "https://images.unsplash.com/photo-1615485290382-441e4d049cb5?w=500",
        "search_term": "Insecticidal Soap Neem Oil"
    },
    "Tomato_mosaic_virus": {
        "name": "Plant Immune Booster Spray",
        "image": "https://images.unsplash.com/photo-1615485290382-441e4d049cb5?w=500",
        "search_term": "Plant Booster Spray"
    }
}

disease_medicines_ar = disease_medicines_en

plant_seasons = {
    "Apple": ["Spring", "Autumn"],
    "Blueberry": ["Spring", "Summer"],
    "Cherry_(including_sour)": ["Spring"],
    "Corn_(maize)": ["Summer", "Spring"],
    "Grape": ["Spring", "Summer"],
    "Orange": ["Spring", "Autumn"],
    "Peach": ["Spring"],
    "Pepper,_bell": ["Spring", "Summer"],
    "Potato": ["Spring", "Autumn"],
    "Raspberry": ["Spring", "Summer"],
    "Soybean": ["Summer"],
    "Squash": ["Spring", "Summer"],
    "Strawberry": ["Autumn", "Spring"],
    "Tomato": ["Spring", "Summer"],
}

treatment_tips_en = {
    "healthy": (
        "The plant looks completely healthy! Maintain regular watering and standard nutrient care."
    ),
    "Early_blight": (
        "Remove infected lower leaves, avoid overhead watering, and apply a copper-based fungicide."
    ),
    "Late_blight": (
        "Isolate or destroy infected plants immediately to prevent spread; apply appropriate systemic fungicides."
    ),
    "Bacterial_spot": (
        "Use disease-free seeds/plants, apply copper sprays, and avoid working with wet plants."
    ),
    "Powdery_mildew": (
        "Improve air circulation, apply sulfur-based fungicides or neem oil."
    ),
    "Spider_mites Two-spotted_spider_mite": (
        "Spray affected leaves with strong water jets or use insecticidal soap/neem oil."
    ),
}

treatment_tips_ar = {
    "healthy": (
        "النبات يبدو سليماً تماماً! حافظ على الري المنتظم والعناية القياسية بالمغذيات."
    ),
    "Early_blight": (
        "قم بإزالة الأوراق السفلى المصابة، وتجنب الري العلوي، واستخدم مبيد فطري نحاسي."
    ),
    "Late_blight": (
        "عزل أو التخلص من النباتات المصابة فوراً لمنع الانتشار؛ وتطبيق مبيدات فطرية جهازية مناسبة."
    ),
    "Bacterial_spot": (
        "استخدم بذور/نباتات خالية من الأمراض، ورش مبيدات نحاسية، وتجنب التعامل مع النباتات وهي مبللة."
    ),
    "Powdery_mildew": (
        "تحسين التهوية، وتطبيق مبيدات فطرية تعتمد على الكبريت أو زيت النيم."
    ),
    "Spider_mites Two-spotted_spider_mite": (
        "رش الأوراق المصابة بتيارات مياه قوية أو استخدام صابون مبيد للحشرات أو زيت النيم."
    ),
}

total_scans, healthy_count, diseased_count = get_db_stats()

# -------------------------
# Sidebar Controls & Analytics
# -------------------------
st.sidebar.markdown("---")
st.sidebar.title(t["control"])
st.sidebar.markdown(f"**{t['total_scans']}:** {total_scans}")
st.sidebar.markdown(f"🟢 {t['healthy']}: {healthy_count}")
st.sidebar.markdown(f"🔴 {t['diseased']}: {diseased_count}")

if st.sidebar.button(t["reset"]):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM scans")
    conn.commit()
    conn.close()

    files = glob.glob(os.path.join(ARCHIVE_DIR, "*"))
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Error deleting file {f}: {e}")

    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader(t["faq_title"])
st.sidebar.info(t["faq_text"])

st.sidebar.markdown("---")
st.sidebar.subheader(t["recent_scans"])
recent_items = get_recent_scans(3)
if recent_items:
    for plant, disease in recent_items:
        st.sidebar.text(f"• {plant} | {disease}")
else:
    st.sidebar.text(t["no_scans"])

# -------------------------
# Tabs Layout
# -------------------------
tab1, tab2 = st.tabs([t["detector_tab"], t["archive_tab"]])

with tab1:
    st.title(t["title"])
    st.write(t["desc"])

    # -------------------------
    # 🌤️ Weather & Location Widget
    # -------------------------
    st.markdown(f"### {t['weather_section']}")
    col_w1, col_w2 = st.columns([2, 3])

    current_month = datetime.now().month
    if selected_lang == "العربية":
        if current_month in [3, 4, 5]:
            current_season_name = "Spring (الربيع)"
        elif current_month in [6, 7, 8]:
            current_season_name = "Summer (الصيف)"
        elif current_month in [9, 10, 11]:
            current_season_name = "Autumn (الخريف)"
        else:
            current_season_name = "Winter (الشتاء)"
    else:
        if current_month in [3, 4, 5]:
            current_season_name = "Spring"
        elif current_month in [6, 7, 8]:
            current_season_name = "Summer"
        elif current_month in [9, 10, 11]:
            current_season_name = "Autumn"
        else:
            current_season_name = "Winter"

    if "city_input_val" not in st.session_state:
        st.session_state.city_input_val = "Cairo"

    with col_w1:
        city_input = st.text_input(t["city_label"], value="Cairo")

    if city_input:
        try:
            weather_url = f"https://wttr.in/{city_input}?format=%C+%t"
            res = requests.get(weather_url, timeout=3)
            if res.status_code == 200:
                weather_info_text = res.text.strip()
            else:
                weather_info_text = "Clear, 25°C"
        except:
            weather_info_text = "Sunny, 26°C"
    else:
        weather_info_text = "Sunny, 25°C"

    with col_w2:
        st.markdown(
            f"""
        <div class="custom-weather-box">
            📍 <b>Location:</b>&nbsp;{city_input}&nbsp;&nbsp;|&nbsp;&nbsp;🌡️ <b>{t['weather_status']}:</b>&nbsp;{weather_info_text}&nbsp;&nbsp;|&nbsp;&nbsp;🍂 <b>{t['current_season']}:</b>&nbsp;{current_season_name}
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    file = st.file_uploader(t["upload"], type=["jpg", "png", "jpeg"])

    if file is not None:
        img = Image.open(file).convert("RGB")
        st.image(img, caption=t["uploaded_img"], use_container_width=True)

        with st.spinner(t["analyzing"]):
            img_t = transform(img).unsqueeze(0)

            with torch.no_grad():
                plant_out, disease_out = model(img_t)

                plant_prob = F.softmax(plant_out, dim=1)
                disease_prob = F.softmax(disease_out, dim=1)

                plant_idx = plant_prob.argmax().item()
                disease_idx = disease_prob.argmax().item()

        detected_plant = idx_to_plant[plant_idx]
        detected_disease = idx_to_disease[disease_idx]
        p_confidence = plant_prob.max().item() * 100
        d_confidence = disease_prob.max().item() * 100

        is_healthy = "healthy" in detected_disease.lower()
        status_icon = (
            t["status_healthy"] if is_healthy else t["status_attention"]
        )
        status_str = "Healthy" if is_healthy else "Diseased"

        tips_dict = (
            treatment_tips_ar
            if selected_lang == "العربية"
            else treatment_tips_en
        )
        tip = tips_dict.get(detected_disease, "Monitor plant closely.")

        medicines_dict = disease_medicines_ar if selected_lang == "العربية" else disease_medicines_en
        recommended_medicine_name = "None"
        if not is_healthy and detected_disease in medicines_dict:
            recommended_medicine_name = medicines_dict[detected_disease]["name"]

        best_seasons_list = plant_seasons.get(detected_plant, ["Spring", "Summer"])
        is_season_match = any(
            s.lower() in current_season_name.lower() for s in best_seasons_list
        )

        if is_season_match:
            season_advice_msg = t["planting_status_ok"]
        else:
            season_advice_msg = t["planting_status_no"].format(
                season=current_season_name,
                best_seasons=", ".join(best_seasons_list),
            )

        plant_img_url = plant_images_map.get(detected_plant, "https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?w=500")

        with st.container(border=True):
            st.markdown(f"### 🌱 {t['report_title']}")
            
            col_rep1, col_rep2 = st.columns([2, 1])
            with col_rep1:
                st.markdown(
                    f"**{t['plant_type']}:** {detected_plant} ({p_confidence:.2f}%)"
                )
                st.markdown(
                    f"**{t['condition']}:** {detected_disease} ({d_confidence:.2f}%)"
                )
                st.markdown(f"**{t['status']}:** {status_icon}")
                st.markdown("---")
                st.markdown(
                    f"**{t['planting_guide_title']}:**\n\n{season_advice_msg}"
                )
                st.markdown("---")
                st.markdown(f"**{t['treatment']}**\n\n{tip}")
                if not is_healthy and recommended_medicine_name != "None":
                    st.markdown(f"**{t['medicine_recommended']}** {recommended_medicine_name}")
            
            with col_rep2:
                st.image(plant_img_url, caption=f"🌿 {detected_plant}", use_container_width=True)

            if not is_healthy and detected_disease in medicines_dict:
                med_info = medicines_dict[detected_disease]
                amazon_search_query = urllib.parse.quote(med_info["search_term"])
                amazon_url = f"https://www.amazon.com/s?k={amazon_search_query}"

                st.markdown("---")
                st.markdown(f"### {t['medicine_recommended']}")
                col_m1, col_m2 = st.columns([1, 2])
                with col_m1:
                    st.image(med_info["image"], caption=med_info["name"], width=180)
                with col_m2:
                    st.markdown(f"**💊 {med_info['name']}**")
                    st.markdown(f'<a href="{amazon_url}" target="_blank" class="buy-btn">{t["buy_btn_text"]}</a>', unsafe_allow_html=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_img_path = os.path.join(ARCHIVE_DIR, f"scan_{timestamp}.png")
        img.save(archive_img_path)

        report_text = f"""--- Plant Diagnosis Report ---
Report Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Location/City: {city_input} | Weather: {weather_info_text}
Current Season: {current_season_name}
Detected Plant: {detected_plant} ({p_confidence:.2f}%)
Detected Disease/Status: {detected_disease} ({d_confidence:.2f}%)
Status: {status_str}
Seasonal Guide: {season_advice_msg}
Treatment Advice: {tip}
Recommended Medicine: {recommended_medicine_name}
------------------------------------
Generated by Robin John
"""
        save_scan_to_db(
            timestamp,
            detected_plant,
            detected_disease,
            p_confidence,
            d_confidence,
            status_str,
            archive_img_path,
            report_text,
            "[]",
            weather_info=f"{city_input}: {weather_info_text}",
            season_info=current_season_name,
        )

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, chat_history FROM scans ORDER BY id DESC LIMIT 1"
        )
        current_scan_id, current_chat_json = cursor.fetchone()
        conn.close()

        st.markdown("<br>", unsafe_allow_html=True)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.download_button(
                label=t["download_btn"],
                data=report_text,
                file_name=f"report_{timestamp}.txt",
                mime="text/plain",
            )
        with col_btn2:
            if st.button(t["copy_btn"]):
                st.success(t["copy_success"])
                st.code(report_text, language="text")

        st.success(f"{t['archive_msg']} `{DB_FOLDER}/`")

        # -------------------------
        # 🤖 PLANT DOCTOR AI CHATBOT
        # -------------------------
        st.markdown("---")
        st.subheader(t["ai_chat_title"])
        st.write(t["ai_chat_intro"])

        if not groq_api_key:
            st.warning(t["api_missing"])
        else:
            try:
                session_messages = (
                    json.loads(current_chat_json) if current_chat_json else []
                )
            except:
                session_messages = []

            for message in session_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if user_query := st.chat_input(t["ai_placeholder"]):
                session_messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            url = "https://api.groq.com/openai/v1/chat/completions"

                            headers = {
                                "Authorization": (
                                    f"Bearer {groq_api_key.strip()}"
                                ),
                                "Content-Type": "application/json",
                            }

                            payload = {
                                "model": "openai/gpt-oss-20b",
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": (
                                            "You are an expert agricultural engineer and plant doctor AI assistant. Detected Plant:"
                                            f" {detected_plant}, Condition:"
                                            f" {detected_disease}, Status:"
                                            f" {status_str}, Weather:"
                                            f" {weather_info_text}, Season:"
                                            f" {current_season_name},"
                                            f" Treatment: {tip}, Medicine:"
                                            f" {recommended_medicine_name}"
                                        ),
                                    }
                                ]
                                + session_messages,
                                "temperature": 0.7,
                            }

                            response = requests.post(
                                url, headers=headers, json=payload
                            )

                            if response.status_code == 200:
                                res_json = response.json()
                                ai_reply = res_json["choices"][0]["message"][
                                    "content"
                                ]
                                st.markdown(ai_reply)
                                session_messages.append({
                                    "role": "assistant",
                                    "content": ai_reply,
                                })
                                update_scan_chat_in_db(
                                    current_scan_id,
                                    json.dumps(
                                        session_messages, ensure_ascii=False
                                    ),
                                )
                            else:
                                st.error(
                                    f"Groq API Error ({response.status_code}): {response.text}"
                                )

                        except Exception as e:
                            st.error(f"Error connecting to Groq API: {e}")

with tab2:
    st.title(t["archive_tab"])

    conn = sqlite3.connect(DB_NAME)
    try:
        df_archive = pd.read_sql_query(
            "SELECT timestamp, plant_name, disease_name, plant_conf, disease_conf, status, weather_info, season_info, image_path FROM scans ORDER BY id DESC",
            conn,
        )
    except:
        df_archive = pd.read_sql_query(
            "SELECT timestamp, plant_name, disease_name, plant_conf, disease_conf, status, image_path FROM scans ORDER BY id DESC",
            conn,
        )

    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, timestamp, plant_name, disease_name, plant_conf, disease_conf, status, image_path, report_text, chat_history FROM scans ORDER BY id DESC"
    )
    rows = cursor.fetchall()
    conn.close()

    if not df_archive.empty:
        csv_data = df_archive.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=t["export_csv_btn"],
            data=csv_data,
            file_name="plant_scans_archive.csv",
            mime="text/csv",
        )
        st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        search_query = st.text_input(t["search_label"], "")
    with col2:
        status_filter = st.selectbox(
            t["filter_status"], [t["all"], t["healthy"], t["diseased"]]
        )

    filtered_rows = []
    for row in rows:
        (
            scan_id_s,
            timestamp_s,
            plant_s,
            disease_s,
            p_conf_s,
            d_conf_s,
            status_s,
            img_p_s,
            report_s,
            chat_h_s,
        ) = row

        matches_search = (
            search_query.lower() in plant_s.lower()
            or search_query.lower() in disease_s.lower()
            or search_query.lower() in report_s.lower()
        )

        matches_status = True
        if status_filter == t["healthy"]:
            matches_status = status_s == "Healthy"
        elif status_filter == t["diseased"]:
            matches_status = status_s == "Diseased"

        if matches_search and matches_status:
            filtered_rows.append(row)

    if filtered_rows:
        st.write(f"📁 Found **{len(filtered_rows)}** saved records in database:")
        for idx, row in enumerate(filtered_rows):
            (
                scan_id_n,
                timestamp_str,
                plant_n,
                disease_n,
                p_c,
                d_c,
                stat,
                img_p,
                rep_t,
                chat_h_val,
            ) = row
            with st.expander(
                f"📄 Scan: {timestamp_str} | Plant: {plant_n} | Status: {stat}"
            ):
                c1, c2 = st.columns([1, 2])
                with c1:
                    if os.path.exists(img_path := img_p):
                        st.image(
                            img_path,
                            caption="Archived Leaf",
                            use_container_width=True,
                        )
                with c2:
                    st.text(rep_t)
                    st.download_button(
                        label=t["download_btn"],
                        data=rep_t,
                        file_name=f"report_{timestamp_str}.txt",
                        mime="text/plain",
                        key=f"dl_{timestamp_str}_{idx}",
                    )

                    if chat_h_val:
                        try:
                            old_chats = json.loads(chat_h_val)
                            if old_chats:
                                st.markdown("---")
                                st.markdown(
                                    "**💬 Archived Chat History for this scan:**"
                                )
                                for chat in old_chats:
                                    role_label = (
                                        "👤 User"
                                        if chat["role"] == "user"
                                        else "🤖 AI"
                                    )
                                    st.markdown(
                                        f"**{role_label}:** {chat['content']}"
                                    )
                        except:
                            pass
    else:
        st.info(t["no_archive_files"])

st.markdown("---")
st.markdown(t["footer"])
