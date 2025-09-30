# app.py
# pip install streamlit pandas pymysql numpy SQLAlchemy langchain-core langchain-openai
import json
import re
import numpy as np
import pandas as pd
import pymysql
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.caches import BaseCache
from langchain_core.callbacks import Callbacks
from langchain_core.prompts import ChatPromptTemplate
# --- Fix Pydantic forward-ref issue for ChatOpenAI on some stacks ---
try:
    ChatOpenAI.model_rebuild()
except Exception:
    pass
st.set_page_config(page_title="HORECA Arabic Content Generator", layout="wide")
st.title("HORECA Arabic Content Generator")
st.caption("PyMySQL → pandas → LangChain(OpenAI). Tab 1 shows attribute_combined. Tab 2: English source first, then Arabic JSON.")
# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Database Connection")
host = st.sidebar.text_input("Host", value="horecadbdevelopment.c1c86oy8g663.me-south-1.rds.amazonaws.com")
port = st.sidebar.number_input("Port", value=3306, step=1)
user = st.sidebar.text_input("Username", value="horecaDbUAE")
password = st.sidebar.text_input("Password", value="Blackmango2025", type="password")
db_name = st.sidebar.text_input("Database", value="horecadbuae")
st.sidebar.header("OpenAI")
openai_key = st.sidebar.text_input("OPENAI_API_KEY", value="", type="password", help="Paste your OpenAI API key here")
model_name = st.sidebar.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-5"], index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
st.sidebar.divider()
load_btn = st.sidebar.button("Connect & Load Data", type="primary")
# -------------------------
# SQL (row-per-attribute)
# -------------------------
SQL = """
SELECT
  p.id AS id,
  p.sku,
  p.name AS product_name,
  a.name AS attribute_name,
  pa.attribute_value
FROM ec_products AS p
JOIN product_attributes AS pa
  ON pa.product_id = p.id
JOIN attributes AS a
  ON a.id = pa.attribute_id
ORDER BY p.id, a.name;
"""
# -------------------------
# Helpers
# -------------------------
def get_connection(h, u, p, prt, db):
    return pymysql.connect(host=h, user=u, password=p, port=prt, database=db)
@st.cache_data(show_spinner=True, ttl=600)
def load_joined_dataframe(h, u, p, prt, db):
    conn = get_connection(h, u, p, prt, db)
    try:
        df = pd.read_sql(SQL, conn)
        return df
    finally:
        conn.close()
def build_out_dict(df: pd.DataFrame) -> pd.DataFrame:
    """One row per product with attribute_combined (dict), built via pivot (no groupby.apply warnings)."""
    meta = ["id", "sku", "product_name"]
    wide = (
        df.pivot_table(index=meta, columns="attribute_name", values="attribute_value", aggfunc="first")
          .reset_index()
    )
    attr_cols = [c for c in wide.columns if c not in meta]
    wide["attribute_combined"] = wide[attr_cols].apply(
        lambda r: {k: v for k, v in r.items() if pd.notna(v)}, axis=1
    )
    return wide[meta + ["attribute_combined"]]
def get_product_row(out_df: pd.DataFrame, product_id):
    pid = str(product_id).strip()
    mask = out_df["id"].astype(str).str.strip() == pid
    if not mask.any():
        return None
    return out_df.loc[mask].iloc[0]
def to_builtin(obj):
    """Recursively convert NumPy/pandas scalars for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if obj is pd.NA or (isinstance(obj, float) and pd.isna(obj)): return None
    return obj
def build_product_payload(row: pd.Series) -> dict:
    return {
        "id": row["id"],
        "sku": row.get("sku"),
        "product_name": row.get("product_name"),
        "attributes": row.get("attribute_combined") or {},
    }
def choose_k(atr: dict) -> int:
    n = len(atr or {})
    if n <= 4: return 4
    if n <= 8: return 5
    return 6
# -------------------------
# Tabs
# -------------------------
data_tab, generate_tab = st.tabs(["1) Load & View attribute_combined", "2) Generate (English → Arabic)"])
# ----- Tab 1 -----
with data_tab:
    st.subheader("Step 1: Connect and Load")
    if load_btn:
        try:
            df = load_joined_dataframe(host, user, password, port, db_name)
            if df.empty:
                st.warning("The query returned no rows.")
            else:
                out_dict = build_out_dict(df)
                show_cols = ["id", "sku", "product_name", "attribute_combined"]
                st.success(f"Loaded {len(out_dict):,} products. Showing id, sku, product_name, attribute_combined.")
                st.dataframe(out_dict[show_cols], width="stretch", height=620)
                st.session_state["out_dict"] = out_dict
        except Exception as e:
            st.error(f"Error loading data: {e}")
# ----- Tab 2 -----
with generate_tab:
    st.subheader("Step 2: Generate")
    if "out_dict" not in st.session_state:
        st.warning("Load data first (Tab 1).")
    else:
        out_dict = st.session_state["out_dict"]
        c1, c2 = st.columns([2, 1])
        product_id_input = c1.text_input("Enter Product ID", value="")
        run_btn = c2.button("Generate", type="primary", use_container_width=True)
        if run_btn:
            if not product_id_input.strip():
                st.error("Please enter a Product ID."); st.stop()
            if not openai_key.strip():
                st.error("Please paste your OPENAI_API_KEY in the sidebar."); st.stop()
            row = get_product_row(out_dict, product_id_input)
            if row is None:
                st.error(f"Product doesn't exist with associated id: {product_id_input}"); st.stop()
            # Build payload / English source first
            product_payload = build_product_payload(row)
            attribute_json = json.dumps(to_builtin(product_payload), ensure_ascii=False, indent=2)
            k = choose_k(product_payload.get("attributes", {}))
            # Show ENGLISH SOURCE FIRST
            st.markdown("### English Source (attribute_combined)")
            st.code(attribute_json, language="json")
            # PROMPT
            PROMPT_TEMPLATE = """
Here are the product attributes you must use (do not invent anything else):
{attribute_json}

HORECA PRODUCT DESCRIPTION – MASTER INSTRUCTION TEMPLATE

Role
Act as a HORECA content specialist creating product content for a B2B eCommerce platform. Your audience is professional hospitality buyers, chefs, restaurant operators, hotel owners, and commercial kitchen staff.

Tone & Style
- Clear, factual, B2B-focused.
- No jargon, slang, fluff, or marketing hype.
- Use active voice and plain language.
- Only use provided product attributes; no assumptions.
- Always say "commercial" (never "professional") for product use.
- Explain technical terms simply when needed.
- Keep specs exactly as provided; do not rephrase values or units.
- IMPORTANT: All generated output must be in Modern Standard Arabic (MSA), suitable for all Arab regions.

Special Arabic Rules (Critical)
- Brand names must always be transliterated into Arabic (e.g., Rational → راشونال, Nuova Simonelli → نوفا سيمونيلي, Sanremo → سانريمو).
- Do not merge brand name with product model. Keep them separate.
- If multiple colors are listed, translate all (e.g., "Silver, Black" → "فضي وأسود").
- Do not start sentences with filler words like "تعتبر" or "يُعتبر". Start directly with the product introduction.
- Always include all numeric or functional attributes (e.g., 2-Group, 3-Phase).
- SKU must be copied exactly as provided, without adding “N/A” or altering it.
- All features, add-ons, and accessories (e.g., UltraVent) must be transliterated into Arabic while keeping context.
- Units of measurement must be properly localized: "Litre" → "لتر". Do not use incorrect plural forms like "لترات" unless grammatically required.

Hard Rules (Never Break)
- Do NOT mention product weight unless handheld and specified.
- Do NOT include dimensions or weight in benefits or FAQ.
- Dimensions must be in WxDxH or LxDxH format exactly as provided (if given).
- Do NOT invent details not in specs.
- Do NOT reference packaging, shipping, case quantity, or origin if “Made in China”.
- No em dashes (—); use commas or periods instead.
- Do NOT use “fl oz”; always write “oz”.
- Avoid vague adjectives unless backed by a listed feature.

OUTPUT STRUCTURE — RETURN JSON ONLY (no extra prose, no markdown)
{{
  "description": ["paragraph1", "paragraph2", "paragraph3", "paragraph4"],
  "benefits": [  // EXACTLY {k} items
    {{"benefit": "Benefit Title", "feature": "One crisp sentence linking to a listed spec"}}
  ],
  "faqs": [  // EXACTLY 5 items
    {{"question": "Question?", "answer": "Answer."}}
  ]
}}

PRODUCT DESCRIPTION (4 paragraphs; 300–350 characters each)
P1: Start with: "The {brand} {product_name} and SKU {SKU} is ..."
State what it is, where it’s used (commercial kitchens, hotels, catering, etc.), and 1–2 key spec highlights.
P2: Core technical specifications, performance, control features, mechanisms.
P3: Include dimensions in WxDxH or LxDxH format if provided; installation notes; safety/stability features.
P4: Certifications, warranty, commercial suitability; accessories/conversion kits only if listed.
End each paragraph with a period.
** Description should not contain any weights.

BENEFITS & FEATURES — EXACTLY {k} pairs
Format (no bullets; separate with line breaks inside JSON values only):
[Benefit Title]: [One sentence describing the linked feature].
Rules:
- No dimensions or weight here.
- Each benefit must be directly supported by a listed feature in the attributes/description.
- End each sentence with a period.

TECHNICAL FAQ — EXACTLY 5 Q&As
Format:
Q: [Buyer’s realistic technical question]
A: [Answer using only provided specs]
Rules:
- Only technical, product-specific questions.
- Do NOT ask vague questions like “Is it easy to clean?”
- Do NOT use dimensions or weight unless handheld.
- Refer to the product type, not SKU.
- Keep answers short and factual.
- End each sentence with a period.

IMPORTANT:
- Output valid JSON only (no markdown, no explanations).
- Ensure "benefits" has exactly {k} objects and "faqs" has exactly 5 objects.
- Compulsory end the statements with full stop for all.
- All output must be in Modern Standard Arabic (MSA), following the same structure, tone, and style.
""".strip()
            # Build prompt & call model (pass API key explicitly)
            prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            msgs = prompt.format_messages(attribute_json=attribute_json, k=k)
            with st.spinner("Translating to Arabic JSON via OpenAI..."):
                llm = ChatOpenAI(
                    api_key=openai_key, # key from sidebar
                    model=model_name, # e.g., gpt-4o
                    temperature=float(temperature),
                    timeout=120,
                )
                try:
                    resp = llm.invoke(msgs)
                    raw = resp.content.strip()
                    if not raw:
                        raise ValueError("OpenAI returned an empty response.")
                except Exception as e:
                    st.error(f"Error during OpenAI API call: {str(e)}")
                    st.stop()
                # For debugging: Show raw response if parsing fails
                try:
                    result_json = json.loads(raw)
                except json.JSONDecodeError:
                    # Attempt fix for trailing commas
                    fixed = re.sub(r",\s*([}\]])", r"\1", raw)
                    try:
                        result_json = json.loads(fixed)
                    except json.JSONDecodeError:
                        st.error("Failed to parse OpenAI response as JSON. Raw response below:")
                        st.code(raw, language="text")
                        st.stop()
                # Enforce counts
                if not isinstance(result_json.get("benefits"), list) or len(result_json["benefits"]) != k:
                    arr = (result_json.get("benefits") or [])[:k]
                    arr += [{"benefit": ".", "feature": "."}] * (k - len(arr))
                    result_json["benefits"] = arr
                if not isinstance(result_json.get("faqs"), list) or len(result_json["faqs"]) != 5:
                    arr = (result_json.get("faqs") or [])[:5]
                    arr += [{"question": ".", "answer": "."}] * (5 - len(arr))
                    result_json["faqs"] = arr[:5]
            st.markdown("### Arabic JSON (translated output)")
            st.json(result_json, expanded=True)
            # Downloads
            st.download_button(
                "Download English Source JSON",
                data=attribute_json.encode("utf-8"),
                file_name=f"product_{row['id']}_source.json",
                mime="application/json",
            )
            st.download_button(
                "Download Arabic JSON",
                data=json.dumps(result_json, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"product_{row['id']}_arabic.json",
                mime="application/json",
            )
