# app.py
# pip install streamlit pandas pymysql numpy SQLAlchemy langchain-core langchain-openai
import json
import re
import numpy as np
import pandas as pd
import pymysql
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.caches import BaseCache  # Add this line
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
# --- Fix Pydantic forward-ref issue for ChatOpenAI on some stacks ---
try:
    ChatOpenAI.model_rebuild()
except Exception:
    pass
# ... (rest of your code remains unchanged)

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
model_name = st.sidebar.selectbox("Model", ["gpt-4o", "gpt-4.1", "gpt-4o-mini"], index=0)
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
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):  return float(obj)
    if isinstance(obj, (np.bool_,)):     return bool(obj)
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

            # PROMPTS
            SYSTEM_PROMPT = f"""
You are a HORECA Arabic content specialist for B2B eCommerce. Produce Arabic output with these rules:

Style:
- Clear, factual Modern Standard Arabic for commercial buyers. Active voice, no fluff.
- Use ONLY the provided JSON attributes. Do not invent or assume anything.
- No em dashes; use commas or periods.
- Use the word "تجاري" for commercial use, never "مهني".
- Keep numbers/units exactly as given. Do not paraphrase values.

Brands & SKU (fixes applied):
- Keep SKUs in ASCII and do not translate them. If SKU exists, include it verbatim after the Arabic label.
- Do not output placeholders like N/A. Omit missing fields silently.
- Transliterate brand names (e.g., Nuova Simonelli → نوفا سيمونيلي, Sanremo → سانريمو, Rational → راشونال, La Cimbali → لا تشيمبالي, Slayer Espresso → سلاير إسبريسو).
- Do not merge brand and model into a new brand term.

Terminology:
- Do not start sentences with "تعتبر". Start directly and factually.
- If multiple colors exist in English, render them all in Arabic joined with "و".
- Prefer "لتر" over plural "لترات" when it reads naturally after numerals.

Dimensions & Electrical:
- If dimensions are present, present them as WxDxH or LxDxH exactly, and add "(عرض × عمق × ارتفاع)" once in Arabic.
- If both 110 and 220 are present, mention both as "110، 220 فولت".
- If 2-Group or 3-Group appears, reflect it clearly as a capability without altering SKU text.

Modules & Certifications:
- UltraVent for combi oven → "ألتراإفنت" with "الفرن متعدد الوظائف" where relevant.
- Certifications: mention once, e.g., "تحمل شهادات NSF وUL." Do not expand full names unless explicitly asked.

Output format:
- Return VALID JSON only. No markdown, no extra prose.
- The JSON must contain three keys: description, benefits, faqs.
- description: array of exactly 4 Arabic paragraphs, each about 300–350 characters, each ends with a period.
- benefits: array of EXACTLY {k} objects, each with keys "benefit" and "feature". No dimensions or weight here. Each sentence ends with a period.
- faqs: array of EXACTLY 5 objects with keys "question" and "answer". Technical, product-specific, short, and factual. No dimensions/weight unless handheld. Each ends with a period.
""".strip()

            HUMAN_PROMPT = """
Use ONLY the following product JSON as your source of truth:

PRODUCT_JSON_START
{attribute_json}
PRODUCT_JSON_END

Tasks:
1) Write 4 Arabic paragraphs as described in the system rules. If brand or SKU is missing in the JSON, omit gracefully without inventing. Do not include weights unless handheld.
2) Produce exactly {k} benefit-feature pairs. Each benefit must link directly to a listed spec present in the JSON.
3) Produce exactly 5 technical FAQs. Refer to the product type, not the SKU.

Return JSON only with keys: description, benefits, faqs.
Ensure it is strictly valid JSON and follows the counts exactly.
""".strip()

            # Build prompt & call model (pass API key explicitly)
            sys_tmpl = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)
            hum_tmpl = HumanMessagePromptTemplate.from_template(HUMAN_PROMPT)
            prompt = ChatPromptTemplate.from_messages([sys_tmpl, hum_tmpl])
            msgs = prompt.format_messages(attribute_json=attribute_json, k=k)

            with st.spinner("Translating to Arabic JSON via OpenAI..."):
                llm = ChatOpenAI(
                    api_key=openai_key,     # key from sidebar
                    model=model_name,       # e.g., gpt-4o
                    temperature=float(temperature),
                    timeout=120,
                )
                resp = llm.invoke(msgs)
                raw = resp.content.strip()

                # Strict JSON parse with tiny repair for trailing commas
                try:
                    result_json = json.loads(raw)
                except json.JSONDecodeError:
                    fixed = re.sub(r",\s*([}\]])", r"\1", raw)
                    result_json = json.loads(fixed)

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
