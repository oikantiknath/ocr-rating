import streamlit as st
import os
import json
import glob
import pandas as pd
from PIL import Image
import libsql_client as libsql
import html as _html
import unicodedata
import difflib
import re
from requests_oauthlib import OAuth2Session

# --- 1. AUTHORIZATION AND CONFIGURATION ---

st.set_page_config(layout="wide", page_title="OCR Rating Tool")

# Constants
LANGS = { 'hi': 'Hindi', 'bn': 'Bengali', 'pa': 'Punjabi', 'or': 'Odia', 'ta': 'Tamil', 'te': 'Telugu', 'kn': 'Kannada', 'ml': 'Malayalam', 'mr': 'Marathi', 'gu': 'Gujarati' }
DOC_CATEGORIES = { 'mg': 'magazines', 'tb': 'textbooks', 'nv': 'novels', 'np': 'newspapers', 'rp': 'research-papers', 'br': 'brochures', 'nt': 'notices', 'sy': 'syllabi', 'qp': 'question-papers', 'mn': 'manuals' }
BASE_DIR = 'data'
IMG_DIR, GCP_DIR, GEM_DIR = [f'{BASE_DIR}/{d}' for d in ('ocr_snippets_testing', 'gcp_ocr_snippets', 'gemini_ocr_snippets')]
COLS = ['image_name', 'lang', 'domain', 'image_rating', 'ocr_pred_rating', 'rated_at', 'rated_by_email']
DEFAULT, SKIP = -1, -2

# Google OAuth Configuration from secrets
CLIENT_ID = st.secrets.google_oauth.client_id
CLIENT_SECRET = st.secrets.google_oauth.client_secret
REDIRECT_URI = st.secrets.google_oauth.redirect_uri
AUTHORIZATION_URL = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPE = ["openid", "https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email"]

# --- 2. DATABASE AND HELPER FUNCTIONS ---

@st.cache_resource
def get_db_connection():
    """Establishes and caches a persistent connection to the Turso database."""
    url = st.secrets["TURSO_DB_URL"]
    if url.startswith("libsql://"):
        url = url.replace("libsql://", "https://")
    auth_token = st.secrets["TURSO_AUTH_TOKEN"]
    return libsql.create_client_sync(url=url, auth_token=auth_token)
    
def load_ratings_from_db() -> pd.DataFrame:
    """Loads all ratings from the Turso database and returns them as a DataFrame."""
    conn = get_db_connection()
    rs = conn.execute("SELECT * FROM ratings")
    rows = [dict(zip(rs.columns, row)) for row in rs.rows]
    if not rows: return pd.DataFrame(columns=COLS)
    df = pd.DataFrame(rows)
    df['image_rating'] = pd.to_numeric(df['image_rating'], errors='coerce').fillna(DEFAULT).astype(int)
    df['ocr_pred_rating'] = pd.to_numeric(df['ocr_pred_rating'], errors='coerce').fillna(DEFAULT).astype(int)
    return df

def upsert_rating_to_db(name: str, lang: str, domain: str, img_val: int, ocr_val: int, user_email: str):
    """Inserts or updates a rating, including the timestamp and user email."""
    sql = """
        INSERT INTO ratings (image_name, lang, domain, image_rating, ocr_pred_rating, rated_at, rated_by_email)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
        ON CONFLICT(image_name) DO UPDATE SET
            image_rating = excluded.image_rating, ocr_pred_rating = excluded.ocr_pred_rating,
            rated_at = CURRENT_TIMESTAMP, rated_by_email = excluded.rated_by_email;
    """
    conn = get_db_connection()
    conn.execute(sql, (name, lang, domain, img_val, ocr_val, user_email))

def safe_json(path: str):
    """Safely loads a JSON file, returning None if it doesn't exist."""
    try:
        with open(path) as f: return json.load(f)
    except FileNotFoundError: return None

def get_ocr_text(path: str, source: str) -> str:
    """Extracts OCR text from GCP or Gemini JSON output files."""
    js = safe_json(path)
    if not js: return '—'
    if source == 'gcp': return ' '.join(b.get('block_text', '') for b in js.get('ocr_output', {}).get('blocks', []))
    if source == 'gemini':
        parts = js.get('candidates', [{}])[0].get('content', {}).get('parts')
        if not parts: return '—'
        return ' '.join(p.get('text', '') for p in parts if isinstance(p, dict))
    return '—'

def normalize_text(s: str) -> str:
    """Applies a series of normalization rules to a string for better diffing."""
    if not s: return ""
    quote_map = str.maketrans({'“':'"', '”':'"', '„':'"', '‟':'"', '‘':"'", '’':"'", '‚':"'", '‛':"'"})
    s = s.translate(quote_map)
    s = unicodedata.normalize("NFC", s.replace("\ufeff", ""))
    punct_close, punct_open, digit, dash = r"[\.,;:!?\u0964\u0965\u2026'\")\]\}\»]", r"['\"(\[\{\«]", r"[0-9\u0966-\u096F]", r"[-–—]"
    s = re.sub(rf"\s+({punct_close})", r"\1", s); s = re.sub(rf"({punct_open})\s+", r"\1", s)
    s = re.sub(rf"({digit})\s+(?={digit})", r"\1", s); s = re.sub(r"\s*॥\s*", "॥", s)
    s = re.sub(rf"\s*{dash}\s*", "-", s)
    return " ".join(s.split())

def generate_diff_html(a: str, b: str) -> tuple[str, str]:
    """Compares two strings and returns HTML with colored differences."""
    a, b = normalize_text(a), normalize_text(b)
    sm = difflib.SequenceMatcher(None, a, b); out_a, out_b = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        seg_a, seg_b = _html.escape(a[i1:i2]), _html.escape(b[j1:j2])
        if tag == "equal": out_a.append(seg_a); out_b.append(seg_b)
        else:
            if tag in ("replace", "delete"): out_a.append(f'<span style="background:#ffcccc">{seg_a}</span>')
            if tag in ("replace", "insert"): out_b.append(f'<span style="background:#ffcccc">{seg_b}</span>')
    return "".join(out_a), "".join(out_b)


# --- 3. MAIN APP LOGIC ---

def main_app():
    """This function contains and runs the main part of your Streamlit app."""
    user_email = st.session_state.user_info['email']
    st.sidebar.write(f"Welcome, {st.session_state.user_info['name']}")
    if st.sidebar.button("Logout"):
        del st.session_state.user_info
        st.rerun()
    
    # --- UI STATE & SIDEBAR ---
    ratings_df = load_ratings_from_db()
    
    if "last_lang" not in st.session_state: st.session_state.last_lang = "Hindi"
    if "show_completed" not in st.session_state: st.session_state.show_completed = False

    st.sidebar.title("OCR Rating Tool")
    default_lang_idx = list(LANGS.values()).index(st.session_state.last_lang)
    lang_name = st.sidebar.selectbox('Language', list(LANGS.values()), index=default_lang_idx)
    st.session_state.last_lang = lang_name
    lang_code = next(k for k, v in LANGS.items() if v == lang_name)
    
    total_lang_files = len(glob.glob(os.path.join(IMG_DIR, lang_code, '*')))
    done_lang_files = ratings_df[ratings_df.lang == lang_code].image_name.nunique()
    st.sidebar.markdown(f'**Progress:** {done_lang_files}/{total_lang_files}')
    with st.sidebar.expander('Per-domain progress'):
        for dk, dn in DOC_CATEGORIES.items():
            total = len(glob.glob(os.path.join(IMG_DIR, lang_code, f'{dk}_{lang_code}_*')))
            done_count = ratings_df[(ratings_df.lang == lang_code) & (ratings_df.domain == dk)].image_name.nunique()
            st.write(f'{dn}: {done_count}/{total}')
            
    st.session_state.show_completed = st.sidebar.checkbox('Show completed table', value=st.session_state.show_completed)
    if st.session_state.show_completed:
        st.sidebar.dataframe(ratings_df[ratings_df.lang == lang_code][COLS], use_container_width=True)
        
    st.sidebar.header("Downloads")
    live_ratings_df = load_ratings_from_db()
    csv_bytes = live_ratings_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(label="⬇️ Download Live Ratings", data=csv_bytes, file_name="live_ratings.csv", mime="text/csv")
    
    # --- MAIN PANEL UI ---
    view_mode = st.radio('View:', ['Unfinished', 'Completed'], horizontal=True, label_visibility='collapsed')

    if view_mode == 'Unfinished':
        BATCH_SIZE = 10; tabs = st.tabs(list(DOC_CATEGORIES.values()))
        for (dk, dn), tab in zip(DOC_CATEGORIES.items(), tabs):
            with tab:
                all_imgs = sorted(glob.glob(os.path.join(IMG_DIR, lang_code, f'{dk}_{lang_code}_*')))
                done_imgs = set(ratings_df[(ratings_df.lang == lang_code) & (ratings_df.domain == dk)].image_name.tolist())
                pending_imgs = [p for p in all_imgs if os.path.basename(p) not in done_imgs]
                if not pending_imgs:
                    st.success('All snippets done for this domain!'); continue
                batch = pending_imgs[:BATCH_SIZE]
                with st.form(key=f'form_{dk}'):
                    for file_path in batch:
                        name, stem = os.path.basename(file_path), os.path.splitext(os.path.basename(file_path))[0]
                        st.markdown(f"#### {name}"); c1, c2 = st.columns([1, 2], gap='large')
                        with c1:
                            st.image(Image.open(file_path))
                            st.radio('', [0,1,2,SKIP], format_func=lambda x:{0:'👎',1:'😐',2:'👍',SKIP:'⏭️'}[x],
                                     index=1, horizontal=True, key=f'img_{stem}', label_visibility='collapsed')
                        with c2:
                            gcp_html, gem_html = generate_diff_html(get_ocr_text(os.path.join(GCP_DIR, lang_code, f'{stem}.json'), 'gcp'), get_ocr_text(os.path.join(GEM_DIR, lang_code, f'{stem}.json'), 'gemini'))
                            st.markdown('**GCP OCR**', unsafe_allow_html=True); st.markdown(f'<div class="ocr-text">{gcp_html}</div>', unsafe_allow_html=True)
                            st.markdown('<hr style="margin: 0.5rem 0">', unsafe_allow_html=True)
                            st.markdown('**Gemini OCR**', unsafe_allow_html=True); st.markdown(f'<div class="ocr-text">{gem_html}</div>', unsafe_allow_html=True)
                            st.radio('Which is better?', [0, 1, 2, SKIP], format_func=lambda x: {0:'GCP',1:'Equal',2:'Gemini',SKIP:'⏭️'}[x],
                                     index=1, horizontal=True, key=f'ocr_{stem}', label_visibility='collapsed')
                        st.markdown("---")
                    if st.form_submit_button("💾 Save Batch Ratings"):
                        for file_path in batch:
                            name = os.path.basename(file_path)
                            stem = os.path.splitext(name)[0]
                            domain = name[:2]
                            img_val, ocr_val = st.session_state[f'img_{stem}'], st.session_state[f'ocr_{stem}']
                            is_skipped = (img_val == SKIP) or (ocr_val == SKIP)
                            final_img_val, final_ocr_val = (SKIP, SKIP) if is_skipped else (img_val, ocr_val)
                            upsert_rating_to_db(name, lang_code, domain, final_img_val, final_ocr_val, user_email)
                        st.rerun()

    elif view_mode == 'Completed':
        st.header('✅ Completed Snippets'); badge_map_img = {0: '👎', 1: '😐', 2: '👍', SKIP: '⏭️', DEFAULT: '—'}; badge_map_ocr = {0: 'GCP', 1: 'Equal', 2: 'Gemini', SKIP: 'Skipped', DEFAULT: '—'}
        comp_tabs = st.tabs(list(DOC_CATEGORIES.values()))
        for (dk, dn), tab in zip(DOC_CATEGORIES.items(), comp_tabs):
            with tab:
                done_rows = ratings_df[(ratings_df.lang == lang_code) & (ratings_df.domain == dk)]
                if done_rows.empty: st.info('Nothing completed for this domain yet.'); continue
                for _, row in done_rows.iterrows():
                    name, stem = row['image_name'], os.path.splitext(row['image_name'])[0]
                    file_path = os.path.join(IMG_DIR, lang_code, name)
                    if not os.path.exists(file_path): continue
                    st.markdown(f"#### {name}"); c1, c2 = st.columns([1, 2], gap='large')
                    with c1:
                        st.image(Image.open(file_path))
                        st.markdown(f"**Image Rating:** {badge_map_img.get(row['image_rating'], '—')}")
                        st.caption(f"Rated by {row.get('rated_by_email', 'N/A')} on {pd.to_datetime(row.get('rated_at')).strftime('%d %b %Y')}")
                    with c2:
                        gcp_html, gem_html = generate_diff_html(get_ocr_text(os.path.join(GCP_DIR, lang_code, f'{stem}.json'), 'gcp'), get_ocr_text(os.path.join(GEM_DIR, lang_code, f'{stem}.json'), 'gemini'))
                        st.markdown('**GCP OCR**', unsafe_allow_html=True); st.markdown(f'<div class="ocr-text">{gcp_html}</div>', unsafe_allow_html=True)
                        st.markdown('<hr style="margin: 0.5rem 0">', unsafe_allow_html=True)
                        st.markdown('**Gemini OCR**', unsafe_allow_html=True); st.markdown(f'<div class="ocr-text">{gem_html}</div>', unsafe_allow_html=True)
                        st.success(f"**Comparison Choice:** {badge_map_ocr.get(row['ocr_pred_rating'], '—')}")
                    st.markdown("---")

# --- 4. AUTHENTICATION FLOW ---

if 'user_info' not in st.session_state:
    query_params = st.query_params
    if "code" in query_params:
        # Step 2: User has been redirected back from Google with a code
        code = query_params['code']
        google = OAuth2Session(CLIENT_ID, scope=SCOPE, redirect_uri=REDIRECT_URI)
        try:
            token = google.fetch_token(TOKEN_URL, client_secret=CLIENT_SECRET, code=code)
            google = OAuth2Session(CLIENT_ID, token=token)
            user_info = google.get('https://www.googleapis.com/oauth2/v1/userinfo').json()
            st.session_state.user_info = user_info
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred during token fetch: {e}")
            st.link_button("Try Logging In Again", "/")
    else:
        # Step 1: Show the login button
        st.title("Welcome to the OCR Rating Tool")
        st.write("Please log in with your Google account to continue.")
        google = OAuth2Session(CLIENT_ID, scope=SCOPE, redirect_uri=REDIRECT_URI)
        authorization_url, state = google.authorization_url(AUTHORIZATION_URL, access_type="offline", prompt="select_account")
        st.link_button("Login with Google", authorization_url)
else:
    # If user is already logged in, run the main app
    main_app()