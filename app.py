# base = '/Users/oikantik/expts_check_samples_ocr_quality'
import streamlit as st, os, json, glob, pandas as pd
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
base = os.environ.get("DATA_DIR", "data/expts_check_samples_ocr_quality")
STATE_DIR = "app_state"
os.makedirs(STATE_DIR, exist_ok=True)
RATING_FILE   = os.path.join(STATE_DIR, "ratings.csv")
UI_STATE_FILE = os.path.join(STATE_DIR, "ui_state.json")


img_dir, gcp_dir, gem_dir = [f'{base}/{d}' for d in
    ('ocr_snippets_testing', 'gcp_ocr_snippets', 'gemini_ocr_snippets')]

# RATING_FILE   = 'ratings.csv'
# UI_STATE_FILE = 'ui_state.json'
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

# ───────── STATE INIT ────────────────────────────────────────────────────────
ratings_df = load_ratings()
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


# ───────── CSV UPDATE --------------------------------------------------------
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
    ratings_df.to_csv(RATING_FILE, index=False)

# ───────── MAIN – PENDING SNIPPETS ───────────────────────────────────────────
tabs = st.tabs(list(doc_categories.values()))

for (dk, dn), tab in zip(doc_categories.items(), tabs):
    with tab:
        all_imgs = sorted(
            glob.glob(os.path.join(img_dir, lang_code, f'{dk}_{lang_code}_*'))
        )
        done_imgs = ratings_df[
            (ratings_df.lang == lang_code) & (ratings_df.domain == dk)
        ].image_name.tolist()
        pending = [p for p in all_imgs if os.path.basename(p) not in done_imgs]

        if not pending:
            st.success('All snippets done for this domain!')
        else:
            for file in pending:
                name = os.path.basename(file)
                stem = os.path.splitext(name)[0]
                region = name.split('_')[-1].split('.')[0]
                uid = '_'.join(name.split('_')[2:-1])

                with st.container():
                    c1, c2 = st.columns([1, 2], gap='large')

                    # image + rating buttons
                    with c1:
                        st.image(Image.open(file))
                        st.markdown(
                            f'**File:** {name}<br>**UID:** {uid}<br>**Region:** {region}',
                            unsafe_allow_html=True,
                        )
                        b1, b2, b3, b4 = st.columns(4)
                        if b1.button('👎', key=f'{stem}_img0'):
                            update_csv(name, img=0)
                        if b2.button('😐', key=f'{stem}_img1'):
                            update_csv(name, img=1)
                        if b3.button('👍', key=f'{stem}_img2'):
                            update_csv(name, img=2)
                        if b4.button('⏭️', key=f'{stem}_skip'):
                            update_csv(name, skip=True)

                    # ocr texts + comparison buttons
                    with c2:
                        md15(
                            'GCP OCR',
                            gcp_text(os.path.join(gcp_dir, lang_code, f'{stem}.json')),
                        )
                        st.markdown('<hr>', unsafe_allow_html=True)
                        md15(
                            'Gemini OCR',
                            gem_text(os.path.join(gem_dir, lang_code, f'{stem}.json')),
                        )
                        st.markdown('<hr>', unsafe_allow_html=True)
                        t1, t2, t3 = st.columns(3)
                        if t1.button(
                            '👍 GCP', key=f'{stem}_ocr0'
                        ):
                            update_csv(name, ocr=0)
                        if t2.button(
                            '😐 Equal', key=f'{stem}_ocr1'
                        ):
                            update_csv(name, ocr=1)
                        if t3.button(
                            '👍 Gemini', key=f'{stem}_ocr2'
                        ):
                            update_csv(name, ocr=2)

                    st.markdown('---')

# ───────── VISUALISE COMPLETED ───────────────────────────────────────────────
if ui_state["view_completed"]:
    st.header('✅ Completed snippets')
    comp_tabs = st.tabs(list(doc_categories.values()))

    for (dk, dn), ctab in zip(doc_categories.items(), comp_tabs):
        with ctab:
            done_rows = ratings_df[
                (ratings_df.lang == lang_code)
                & (ratings_df.domain == dk)
                & (ratings_df.image_rating != DEFAULT)
                & (ratings_df.ocr_pred_rating != DEFAULT)
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
                        img_badge = {0: '👎', 1: '😐', 2: '👍', SKIP: '⏭️'}[
                            row.image_rating
                        ]
                        st.markdown(f'Image rating: **{img_badge}**')

                    # OCR texts + static badge
                    with c2:
                        md15(
                            'GCP OCR',
                            gcp_text(
                                os.path.join(gcp_dir, lang_code, f'{stem}.json')
                            ),
                        )
                        st.markdown('<hr>', unsafe_allow_html=True)
                        md15(
                            'Gemini OCR',
                            gem_text(
                                os.path.join(gem_dir, lang_code, f'{stem}.json')
                            ),
                        )
                        ocr_badge = {
                            0: 'GCP better',
                            1: 'Equal',
                            2: 'Gemini better',
                            SKIP: 'Skipped',
                        }[row.ocr_pred_rating]
                        st.success(f'Chosen: {ocr_badge}')

                    st.markdown('---')
