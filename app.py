# base = '/Users/oikantik/expts_check_samples_ocr_quality'
import streamlit as st, os, json, glob, pandas as pd
st.set_page_config(layout="wide")
from PIL import Image

# ───────── CONFIG ────────────────────────────────────────────────────────────
langs_dict = {
    'hi': 'Hindi', 'bn': 'Bengali', 'pa': 'Punjabi', 'or': 'Odia',  'ta': 'Tamil',
    'te': 'Telugu', 'kn': 'Kannada', 'ml': 'Malayalam', 'mr': 'Marathi', 'gu': 'Gujarati'
}
doc_categories = {
    'mg': 'magazines', 'tb': 'textbooks', 'nv': 'novels', 'np': 'newspapers',
    'rp': 'research-papers', 'br': 'brochures', 'nt': 'notices', 'sy': 'syllabi',
    'qp': 'question-papers', 'mn': 'manuals'
}
import os
base = 'data'
# STATE_DIR = "app_state"
# os.makedirs(STATE_DIR, exist_ok=True)
# RATING_FILE   = os.path.join(STATE_DIR, "ratings.csv")
# UI_STATE_FILE = os.path.join(STATE_DIR, "ui_state.json")


img_dir, gcp_dir, gem_dir = [f'{base}/{d}' for d in
    ('ocr_snippets_testing', 'gcp_ocr_snippets', 'gemini_ocr_snippets')]

RATING_FILE   = 'ratings.csv'
UI_STATE_FILE = 'ui_state.json'
COLS          = ['image_name', 'lang', 'domain', 'image_rating', 'ocr_pred_rating']
DEFAULT, SKIP = -1, -2      # -1 = not rated, -2 = skipped

# ───────── HELPERS ───────────────────────────────────────────────────────────
def read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default

def write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def load_ratings():
    if os.path.exists(RATING_FILE):
        return pd.read_csv(RATING_FILE)
    pd.DataFrame(columns=COLS).to_csv(RATING_FILE, index=False)
    return pd.read_csv(RATING_FILE)

def safe_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def gcp_text(path):
    js = safe_json(path)
    if js:
        return ' '.join(
            b['block_text'] for b in js.get('ocr_output', {}).get('blocks', [])
        )
    return '—'

def gem_text(path):
    js = safe_json(path)
    if js:
        parts = (
            js.get('candidates', [{}])[0]
            .get('content', {})
            .get('parts', [])
        )
        if parts:
            return ' '.join(
                p.get('text', '') for p in parts if isinstance(p, dict)
            )
    return '—'

def md15(label, txt):
    st.markdown(
        f'<div style="font-size:15px;"><b>{label}</b><br>{txt}</div>',
        unsafe_allow_html=True,
    )
# ---------- coloured-diff helpers -------------------------------------------
import html as _html, unicodedata, difflib, re

_quote_map = str.maketrans({
    '“':'"', '”':'"', '„':'"', '‟':'"',
    '‘':"'", '’':"'", '‚':"'", '‛':"'",
})
_punct_close = r"[\.,;:!?\u0964\u0965\u2026'\")\]\}\»]"
_punct_open  = r"['\"(\[\{\«]"

SPACE_BEFORE_CLOSE = re.compile(rf"\s+({_punct_close})")
SPACE_AFTER_OPEN  = re.compile(rf"({_punct_open})\s+")

# extra patterns
DIGIT          = r"[0-9\u0966-\u096F]"          # ASCII + Devanagari ०–९
SPACE_IN_DIGIT = re.compile(rf"({DIGIT})\s+(?={DIGIT})")   # 4 9 0  → 490
SPACE_AROUND_DANDA = re.compile(r"\s*॥\s*")                # trim blanks around ॥

DASH          = r"[-–—]"                       # hyphen, en-dash, em-dash
SPACE_AROUND_DASH = re.compile(rf"\s*{DASH}\s*")   # " - "  →  "-"

def _norm(s: str) -> str:
    if not s:                                  # (unchanged pre-checks)
        return ""
    s = s.translate(_quote_map)
    s = unicodedata.normalize("NFC", s.replace("\ufeff", ""))

    # existing rules
    s = SPACE_BEFORE_CLOSE.sub(r"\1", s)
    s = SPACE_AFTER_OPEN.sub(r"\1", s)
    s = SPACE_IN_DIGIT.sub(r"\1", s)            # your ४ ९ ० → ४९० rule
    s = SPACE_AROUND_DANDA.sub("॥", s)          # keep exactly one danda

    # NEW rule
    s = SPACE_AROUND_DASH.sub("-", s)           # remove blanks around - / – / —

    return " ".join(s.split())


def diff_html(a: str, b: str) -> tuple[str, str]:
    a, b = _norm(a), _norm(b)
    sm = difflib.SequenceMatcher(None, a, b)
    out_a, out_b = [], []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        seg_a = _html.escape(a[i1:i2])
        seg_b = _html.escape(b[j1:j2])

        if tag == "equal":
            out_a.append(seg_a)        # plain black
            out_b.append(seg_b)
        elif tag == "replace":
            out_a.append(f'<span style="background:#ffcccc">{seg_a}</span>')
            out_b.append(f'<span style="background:#ffcccc">{seg_b}</span>')
        elif tag == "delete":
            out_a.append(f'<span style="background:#ffcccc">{seg_a}</span>')
        elif tag == "insert":
            out_b.append(f'<span style="background:#ffcccc">{seg_b}</span>')

    return "".join(out_a), "".join(out_b)


# ───────── STATE INIT ────────────────────────────────────────────────────────
ratings_df = load_ratings()

# force numeric ints so comparisons work
ratings_df['image_rating']     = pd.to_numeric(ratings_df['image_rating'],
                                               errors='coerce').fillna(DEFAULT).astype(int)
ratings_df['ocr_pred_rating']  = pd.to_numeric(ratings_df['ocr_pred_rating'],
                                               errors='coerce').fillna(DEFAULT).astype(int)

ui_state   = read_json(
    UI_STATE_FILE,
    {"last_lang": None, "show_completed": False, "view_completed": False},
)

# ───────── SIDEBAR ───────────────────────────────────────────────────────────

# language selector
default_lang = ui_state.get("last_lang")
default_lang_idx = (
    list(langs_dict.values()).index(default_lang)
    if default_lang in langs_dict.values()
    else 0
)
lang_name = st.sidebar.selectbox(
    'Language', list(langs_dict.values()), index=default_lang_idx
)
ui_state["last_lang"] = lang_name                 # remember selection
lang_code = next(k for k, v in langs_dict.items() if v == lang_name)

# overall progress
total_lang = len(glob.glob(os.path.join(img_dir, lang_code, '*')))
done_lang  = ratings_df[ratings_df.lang == lang_code].image_name.nunique()
st.sidebar.markdown(f'**Progress:** {done_lang}/{total_lang}')

# per-domain progress
with st.sidebar.expander('Per-domain progress'):
    for dk, dn in doc_categories.items():
        total = len(glob.glob(os.path.join(img_dir, lang_code, f'{dk}_{lang_code}_*')))
        done  = ratings_df[
            (ratings_df.lang == lang_code) & (ratings_df.domain == dk)
        ].image_name.nunique()
        st.write(f'{dn}: {done}/{total}')

# completed-table toggle
show_tbl = st.sidebar.checkbox(
    'Show completed table',
    value=ui_state.get("show_completed", False)       # safe default
)
ui_state["show_completed"] = show_tbl

if show_tbl:
    st.sidebar.dataframe(
        ratings_df[ratings_df.lang == lang_code][COLS],
        use_container_width=True,
    )

# visual review toggle
view_comp = st.sidebar.checkbox(
    'View completed visually',
    value=ui_state.get("view_completed", False)       # safe default
)
ui_state["view_completed"] = view_comp

# persist sidebar choices immediately
write_json(UI_STATE_FILE, ui_state)

# ───────── DOWNLOAD CURRENT CSV ────────────────────────────────────────────
with open(RATING_FILE, "rb") as f:
    csv_bytes = f.read()

st.sidebar.download_button(
    label="⬇️ Download ratings.csv",
    data=csv_bytes,
    file_name="ratings.csv",
    mime="text/csv",
)

with open(UI_STATE_FILE, "rb") as f:
    st.sidebar.download_button(
        "⬇️ Download ui_state.json",
        f.read(),
        file_name="ui_state.json",
        mime="application/json",
    )


# ───────── MAIN-VIEW TOGGLE ──────────────────────────────────────────────
view_mode = st.radio(
    'Show snippets:',
    ['Unfinished', 'Completed'],
    horizontal=True,
)

# ───────── CSV UPDATE --------------------------------------------------------
from filelock import FileLock


def update_csv(name, img=None, ocr=None, skip=False):
    global ratings_df
    if skip:
        img = ocr = SKIP
    mask = ratings_df.image_name == name
    if mask.any():
        if img is not None:
            ratings_df.loc[mask, 'image_rating'] = img
        if ocr is not None:
            ratings_df.loc[mask, 'ocr_pred_rating'] = ocr
    else:
        ratings_df = pd.concat(
            [
                ratings_df,
                pd.DataFrame(
                    [
                        {
                            'image_name': name,
                            'lang': lang_code,
                            'domain': name[:2],
                            'image_rating': img if img is not None else DEFAULT,
                            'ocr_pred_rating': ocr if ocr is not None else DEFAULT,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    # ratings_df.to_csv(RATING_FILE, index=False)
    with FileLock("ratings.lock"):
        ratings_df.to_csv(RATING_FILE, index=False)


# ───────── MAIN – PENDING SNIPPETS (FORM VERSION) ───────────────────────────
if view_mode == 'Unfinished':
    BATCH = 10
    tabs = st.tabs(list(doc_categories.values()))

    for (dk, dn), tab in zip(doc_categories.items(), tabs):
        with tab:
            all_imgs = sorted(glob.glob(os.path.join(img_dir, lang_code, f'{dk}_{lang_code}_*')))
            done_imgs = ratings_df[(ratings_df.lang == lang_code) & (ratings_df.domain == dk)].image_name.tolist()
            pending = [p for p in all_imgs if os.path.basename(p) not in done_imgs]

            if not pending:
                st.success('All snippets done for this domain!')
                continue

            batch = pending[:BATCH]                       # first 10

            with st.form(key=f'batch_{dk}'):
                for file in batch:
                    name  = os.path.basename(file)
                    stem  = os.path.splitext(name)[0]
                    uid   = '_'.join(name.split('_')[2:-1])
                    region = name.split('_')[-1].split('.')[0]

                    c1, c2 = st.columns([1, 2], gap='large')

                    # -------- image + image-rating radio ------------------------
                    with c1:
                        st.image(Image.open(file))
                        st.markdown(
                            f'**File:** {name}<br>**UID:** {uid}<br>**Region:** {region}',
                            unsafe_allow_html=True,
                        )

                        img_key  = f'img_{stem}'
                        img_choice = st.radio(
                            '',                                  # no label
                            options=[0, 1, 2, SKIP],             # 👎 😐 👍 ⏭️
                            format_func=lambda x: {0:'👎',1:'😐',2:'👍',SKIP:'⏭️'}[x],
                            index=[0,1,2,SKIP].index(st.session_state.get(img_key, 1)),
                            horizontal=True,
                            key=img_key,
                        )

                    # -------- diff + OCR comparison radio -----------------------
                    with c2:
                        gcp_html, gem_html = diff_html(
                            gcp_text(os.path.join(gcp_dir, lang_code, f'{stem}.json')),
                            gem_text(os.path.join(gem_dir, lang_code, f'{stem}.json'))
                        )

                        st.markdown('**GCP OCR**', unsafe_allow_html=True)
                        st.markdown(gcp_html, unsafe_allow_html=True)
                        st.markdown('<hr>', unsafe_allow_html=True)
                        st.markdown('**Gemini OCR**', unsafe_allow_html=True)
                        st.markdown(gem_html, unsafe_allow_html=True)

                        ocr_key  = f'ocr_{stem}'
                        ocr_choice = st.radio(
                            'Which is better?',
                            options=[0, 1, 2, SKIP],             # GCP / Equal / Gemini / Skip
                            format_func=lambda x: {0:'GCP',1:'Equal',2:'Gemini',SKIP:'⏭️'}[x],
                            index=[0,1,2,SKIP].index(st.session_state.get(ocr_key, 1)),
                            horizontal=True,
                            key=ocr_key,
                        )

                    st.markdown('---')

                # ---------- submit once for the whole batch ---------------------
                if st.form_submit_button('💾 Save batch'):
                    for file in batch:
                        name  = os.path.basename(file)
                        stem  = os.path.splitext(name)[0]
                        img_val = st.session_state[f'img_{stem}']
                        ocr_val = st.session_state[f'ocr_{stem}']
                        skip = (img_val == SKIP) or (ocr_val == SKIP)
                        update_csv(name, img_val, ocr_val, skip)
                    st.experimental_rerun()          # next 10 load


# ───────── VISUALISE COMPLETED ───────────────────────────────────────────────
if view_mode == 'Completed':
    st.header('✅ Completed snippets')
    comp_tabs = st.tabs(list(doc_categories.values()))

    badge_map_img = {0: '👎', 1: '😐', 2: '👍', SKIP: '⏭️', DEFAULT: '—'}
    badge_map_ocr = {
        0: 'GCP better',
        1: 'Equal',
        2: 'Gemini better',
        SKIP: 'Skipped',
        DEFAULT: '—',
    }

    for (dk, dn), ctab in zip(doc_categories.items(), comp_tabs):
        with ctab:
            done_rows = ratings_df[
                (ratings_df.lang == lang_code)
                & (ratings_df.domain == dk)
                & (
                    (ratings_df.image_rating != DEFAULT)
                    | (ratings_df.ocr_pred_rating != DEFAULT)
                )
            ]

            if done_rows.empty:
                st.info('Nothing completed here yet.')
                continue

            for _, row in done_rows.iterrows():
                file = os.path.join(img_dir, lang_code, row.image_name)
                stem = os.path.splitext(row.image_name)[0]
                region = row.image_name.split('_')[-1].split('.')[0]
                uid = '_'.join(row.image_name.split('_')[2:-1])

                with st.container():
                    c1, c2 = st.columns([1, 2], gap='large')

                    # image + static badge
                    with c1:
                        st.image(Image.open(file))
                        st.markdown(
                            f'**File:** {row.image_name}<br>'
                            f'**UID:** {uid}<br>'
                            f'**Region:** {region}',
                            unsafe_allow_html=True,
                        )
                        img_badge = badge_map_img.get(int(row.image_rating), '—')
                        st.markdown(f'Image rating: **{img_badge}**')

                    # OCR texts + static badge
                    # with c2:
                    #     md15(
                    #         'GCP OCR',
                    #         gcp_text(os.path.join(gcp_dir, lang_code, f'{stem}.json')),
                    #     )
                    #     st.markdown('<hr>', unsafe_allow_html=True)
                    #     md15(
                    #         'Gemini OCR',
                    #         gem_text(os.path.join(gem_dir, lang_code, f'{stem}.json')),
                    #     )
                    #     ocr_badge = badge_map_ocr.get(int(row.ocr_pred_rating), '—')
                    #     st.success(f'Chosen: {ocr_badge}')

                    # st.markdown('---')

                    with c2:
                        gcp_h, gem_h = diff_html(
                            gcp_text(os.path.join(gcp_dir, lang_code, f'{stem}.json')),
                            gem_text(os.path.join(gem_dir, lang_code, f'{stem}.json'))
                        )
                        st.markdown('**GCP OCR**', unsafe_allow_html=True)
                        st.markdown(gcp_h, unsafe_allow_html=True)
                        st.markdown('<hr>', unsafe_allow_html=True)
                        st.markdown('**Gemini OCR**', unsafe_allow_html=True)
                        st.markdown(gem_h, unsafe_allow_html=True)
                        st.success(f'Chosen: {badge_map_ocr[int(row.ocr_pred_rating)]}')

                    st.markdown('---')

