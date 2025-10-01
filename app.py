# app.py
# pip install streamlit pandas pymysql numpy SQLAlchemy langchain-core langchain-openai textwrap
import json
import re
import textwrap
import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine
import streamlit as st
from langchain_openai import ChatOpenAI
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
# per your ask: use gpt-5 by default, with fallbacks exposed
model_name = st.sidebar.selectbox("Model", ["gpt-5", "gpt-4o", "gpt-4o-mini"], index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
st.sidebar.divider()
load_btn = st.sidebar.button("Connect & Load Data", type="primary")
# -------------------------
# SQL (row-per-attribute) — now with brand
# -------------------------
SQL = textwrap.dedent("""
    SELECT
      p.id AS id,
      p.sku AS sku,
      p.name AS product_name,
      b.name AS brand_name, -- brand from ec_brands
      a.name AS attribute_name,
      pa.attribute_value
    FROM ec_products AS p
    LEFT JOIN ec_brands AS b ON b.id = p.brand_id
    JOIN product_attributes AS pa ON pa.product_id = p.id
    JOIN attributes AS a ON a.id = pa.attribute_id
    ORDER BY p.id, a.name;
""")
# -------------------------
# Helpers
# -------------------------
def get_connection(h, u, p, prt, db):
    return pymysql.connect(host=h, user=u, password=p, port=prt, database=db)
@st.cache_data(show_spinner=True, ttl=600)
def load_joined_dataframe(h, u, p, prt, db):
    engine_str = f"mysql+pymysql://{u}:{p}@{h}:{prt}/{db}"
    engine = create_engine(engine_str)
    try:
        df = pd.read_sql(SQL, engine)
        return df
    finally:
        engine.dispose()
def build_out_dict(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per product with attribute_combined (dict), built via pivot
    to avoid groupby.apply warnings.
    """
    meta = ["id", "sku", "product_name", "brand_name"]
    wide = (
        df.pivot_table(
            index=meta, columns="attribute_name", values="attribute_value", aggfunc="first"
        ).reset_index()
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
        "brand_name": row.get("brand_name"), # include brand in payload
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
                show_cols = ["id", "sku", "product_name", "brand_name", "attribute_combined"]
                st.success(f"Loaded {len(out_dict):,} products. Showing id, sku, product_name, brand_name, attribute_combined.")
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
            # PROMPT — updated with your exact Arabic/brand/units/token rules
            # Note: Ensured all literal braces in examples are properly escaped with double {{ }}
            PROMPT_TEMPLATE = textwrap.dedent("""
            Use ONLY this product JSON as your source of truth:
            {attribute_json}
            ROLE & AUDIENCE
            Act as a HORECA content specialist creating product content for a B2B eCommerce platform. Audience: chefs, commercial kitchen operators, hotels, catering.
            TONE & STYLE
            - Modern Standard Arabic (MSA), clear, factual, B2B.
            - Active voice, no fluff/hype.
            - Use ONLY listed attributes; never invent.
            - Keep units/values exactly as provided.
            CRITICAL NAMING & TOKEN RULES
            - Brand handling:
              * If brand_name is already Arabic, copy it EXACTLY as given (no changes).
              * If brand_name is in English, transliterate it ONCE to Arabic and reuse that consistently.
              * Do NOT merge brand and model into a new brand term.
            - Latin tokens to KEEP as-is (do NOT translate to Arabic): "2-Group", "3-Group", "Group", "SKU", voltage tokens like "110V", "220V", frequency/phase tokens like "1-Phase", "3-Phase".
            - Colors: if multiple are listed in English, render all in Arabic joined by "و".
            - Liters: always render as "لتر" in Arabic. Do not use "لترات" unless grammatically necessary.
            - Do NOT start sentences with filler like "تعتبر". Start directly.
            - Keep SKU exactly as provided (ASCII), preceded by Arabic label where needed.
            - If a feature name like "UltraVent" appears, transliterate appropriately while preserving meaning in context.
            HARD RULES (NEVER BREAK)
            - Do NOT mention product weight unless handheld and specified.
            - Do NOT include dimensions or weight in benefits or FAQ.
            - Dimensions, if provided, must be in WxDxH or LxDxH exactly; add "(عرض × عمق × ارتفاع)" once.
            - Do NOT invent details not in specs.
            - Do NOT reference packaging/shipping/case quantity or origin if "Made in China".
            - No em dashes — use commas or periods.
            - Do NOT use "fl oz"; use "oz".
            OUTPUT STRUCTURE — RETURN JSON ONLY (no markdown, no extra prose)
            {{
              "description": ["paragraph1", "paragraph2", "paragraph3", "paragraph4"],
              "benefits": [ // EXACTLY {k} items
                {{"benefit": "Benefit Title", "feature": "One crisp sentence linking to a listed spec."}}
              ],
              "faqs": [ // EXACTLY 5 items
                {{"question": "Question?", "answer": "Answer."}}
              ]
            }}
            GUIDANCE FOR CONTENT
            1) DESCRIPTION (4 paragraphs; ~300–350 chars each; end with periods)
               - P1: "The {{brand_name}} {{product_name}} and SKU {{sku}} is …" If brand or SKU missing, omit gracefully.
               - P2: Core technical specifications and control features; keep Latin tokens unmodified (e.g., 2-Group).
               - P3: If dimensions exist, include WxDxH or LxDxH; brief install/safety notes.
               - P4: Certifications, warranty, commercial suitability; accessories only if listed.
            2) BENEFITS — EXACTLY {k} pairs (no dimensions or weight); each supported by listed features.
            3) TECHNICAL FAQ — EXACTLY 5 Q&A (short, factual; no dimensions/weight unless handheld).
            IMPORTANT
            - Output must be VALID JSON only, with exactly the required list lengths.
            - End every sentence with a period.
            """).strip()
            # Build prompt & call model (pass API key explicitly)
            prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            msgs = prompt.format_messages(attribute_json=attribute_json, k=k)
            with st.spinner("Translating to Arabic JSON via OpenAI..."):
                llm = ChatOpenAI(
                    api_key=openai_key, # key from sidebar
                    model=model_name, # default gpt-5 per your request
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
                # Strict JSON parse with tiny repair for trailing commas
                try:
                    result_json = json.loads(raw)
                except json.JSONDecodeError:
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
