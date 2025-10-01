# app.py
# pip install streamlit pandas pymysql numpy SQLAlchemy openai
import json
import re
import textwrap
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import streamlit as st
from openai import OpenAI

# --- Page Config ---
st.set_page_config(page_title="HORECA Arabic Content Generator", layout="wide")
st.title("HORECA Arabic Content Generator (Single Product)")
st.caption("Connect to DB ➔ Find Product ID ➔ Generate Content ➔ Save to DB")

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("1. Database Connection")
    host = st.text_input("Host", value="horecadbdevelopment.c1c86oy8g663.me-south-1.rds.amazonaws.com")
    port = st.number_input("Port", value=3306, step=1)
    user = st.text_input("Username", value="horecaDbUAE")
    password = st.text_input("Password", value="Blackmango2025", type="password")
    db_name = st.text_input("Database", value="horecadbuae")

    st.header("2. OpenAI Configuration")
    openai_key = st.text_input("OPENAI_API_KEY", type="password", help="Paste your OpenAI API key here")
    
    st.divider()
    load_btn = st.button("Connect & Load Data", type="primary")

# -------------------------
# SQL Queries
# -------------------------
FETCH_SQL = textwrap.dedent("""
    SELECT p.id, p.sku, p.name AS product_name, b.name AS brand_name, a.name AS attribute_name, pa.attribute_value
    FROM ec_products AS p
    LEFT JOIN ec_brands AS b ON b.id = p.brand_id
    JOIN product_attributes AS pa ON pa.product_id = p.id
    JOIN attributes AS a ON a.id = pa.attribute_id
    WHERE p.is_published = 1
    ORDER BY p.id, a.name;
""")

CREATE_TABLE_SQL = textwrap.dedent("""
    CREATE TABLE IF NOT EXISTS product_arabic_content (
        product_id BIGINT UNSIGNED NOT NULL PRIMARY KEY,
        description_arabic JSON,
        benefits_arabic JSON,
        faqs_arabic JSON,
        error_message TEXT,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        CONSTRAINT fk_product_content_id FOREIGN KEY (product_id) REFERENCES ec_products(id) ON DELETE CASCADE
    );
""")

# -------------------------
# Data Loading and Helper Functions
# -------------------------
@st.cache_data(show_spinner="Connecting to database and loading products...", ttl=600)
def load_and_prepare_data(_h, _u, _p, _prt, _db):
    engine_str = f"mysql+pymysql://{_u}:{_p}@{_h}:{_prt}/{_db}"
    engine = create_engine(engine_str)
    with engine.connect() as connection:
        connection.execute(text(CREATE_TABLE_SQL))
        df = pd.read_sql(FETCH_SQL, connection)
    
    if df.empty: return pd.DataFrame()

    meta = ["id", "sku", "product_name", "brand_name"]
    wide = df.pivot_table(index=meta, columns="attribute_name", values="attribute_value", aggfunc="first").reset_index()
    attr_cols = [c for c in wide.columns if c not in meta]
    wide["attribute_combined"] = wide[attr_cols].apply(lambda r: {k: v for k, v in r.items() if pd.notna(v)}, axis=1)
    return wide[meta + ["attribute_combined"]]

def decide_benefit_count(num_attrs: int) -> int:
    if num_attrs <= 4: return 5
    if num_attrs == 5: return 6
    if num_attrs == 6: return 7
    if 7 <= num_attrs <= 8: return 8
    if 9 <= num_attrs <= 10: return 9
    return 10

def is_arabic(text):
    if not text: return False
    arabic_count = sum(1 for char in text if '\u0600' <= char <= '\u0FEFF')
    total_chars = len(text.replace(" ", ""))
    return (arabic_count / total_chars) > 0.7 if total_chars > 0 else False

def check_content_arabic(desc, benefits, faqs):
    texts = desc + [b.get('benefit', '') for b in benefits] + [b.get('feature', '') for b in benefits] + \
            [f.get('question', '') for f in faqs] + [f.get('answer', '') for f in faqs]
    return all(is_arabic(t) for t in texts if t.strip())

# --- Core OpenAI Generation Logic (from your script) ---
def generate_with_openai(row, client):
    """Generates content for a single product row with robust retries."""
    pid = str(row['id'])
    attributes = row.get('attribute_combined', {})
    k = decide_benefit_count(len(attributes))

    product_payload = {
        "brand_name": row.get('brand_name'),
        "product_name": row.get('product_name'),
        "sku": row.get('sku'),
        "attributes": attributes
    }
    attribute_json = json.dumps(product_payload, ensure_ascii=False, indent=2)

    prompt_template = textwrap.dedent("""
    Use ONLY this product JSON as your source of truth: {attribute_json}

    ROLE & AUDIENCE: Act as a HORECA content specialist for a B2B eCommerce platform. Audience: chefs, commercial kitchen operators.
    
    TONE & STYLE: Modern Standard Arabic (MSA), clear, factual, B2B. Active voice. Use ONLY listed attributes. All numbers in Arabic numerals (٠١٢٣٤٥٦٧٨٩). All content RTL.
    
    CRITICAL RULES: Transliterate English brand names ONCE and reuse. Do NOT merge brand, model, SKU. Translate "Group" in "1-Group" as "مجموعة".
    
    HARD RULES: NO inventing details. NO section titles. Translate units and values exactly.
    
    OUTPUT STRUCTURE — RETURN VALID JSON ONLY (no markdown):
    {{
      "description": ["paragraph1", "paragraph2", "paragraph3", "paragraph4"],
      "benefits": [ // EXACTLY {k} items: {{"benefit": "...", "feature": "..."}} ],
      "faqs": [ // EXACTLY 5 items: {{"question": "...", "answer": "..."}} ]
    }}
    """)
    prompt = prompt_template.format(attribute_json=attribute_json, k=k)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            retry_reason = "" if attempt == 0 else "Previous output was invalid. Strictly follow all rules, especially JSON format, Arabic language, and exact counts."
            resp = client.chat.completions.create(
                model="gpt-4o",  # Model is locked here as requested
                temperature=0.3,
                messages=[{"role": "user", "content": prompt + f"\n\n{retry_reason}"}]
            )
            message_content = resp.choices[0].message.content.strip()
            
            match = re.search(r"\{[\s\S]*\}", message_content)
            if not match:
                if attempt == max_retries - 1:
                    return {"pid": pid, "error": f"Failed after {max_retries} attempts: No valid JSON found."}
                continue

            data = json.loads(match.group(0))
            is_content_arabic = check_content_arabic(data.get("description", []), data.get("benefits", []), data.get("faqs", []))
            if len(data.get("benefits", [])) == k and len(data.get("faqs", [])) == 5 and is_content_arabic:
                return {"pid": pid, "data": data, "source_json": attribute_json} # Success
            else:
                if attempt == max_retries - 1:
                    return {"pid": pid, "error": f"Failed validation after {max_retries} attempts."}

        except Exception as e:
            if attempt == max_retries - 1:
                return {"pid": pid, "error": f"API call failed after {max_retries} attempts: {e}"}
    
    return {"pid": pid, "error": f"Failed after {max_retries} attempts for unknown reasons."}

# --- Database Saving Function for a Single Product ---
def save_single_result_to_db(result, _h, _u, _p, _prt, _db):
    engine_str = f"mysql+pymysql://{_u}:{_p}@{_h}:{_prt}/{_db}"
    engine = create_engine(engine_str)
    
    with engine.connect() as connection:
        with connection.begin() as transaction:
            upsert_sql = text("""
                INSERT INTO product_arabic_content (product_id, description_arabic, benefits_arabic, faqs_arabic, error_message)
                VALUES (:pid, :desc, :ben, :faq, :err)
                ON DUPLICATE KEY UPDATE
                description_arabic = VALUES(description_arabic),
                benefits_arabic = VALUES(benefits_arabic),
                faqs_arabic = VALUES(faqs_arabic),
                error_message = VALUES(error_message);
            """)
            
            params = {
                "pid": result.get('pid'),
                "desc": json.dumps(result['data'].get('description'), ensure_ascii=False) if result.get('data') else None,
                "ben": json.dumps(result['data'].get('benefits'), ensure_ascii=False) if result.get('data') else None,
                "faq": json.dumps(result['data'].get('faqs'), ensure_ascii=False) if result.get('data') else None,
                "err": result.get('error')
            }
            connection.execute(upsert_sql, params)
            transaction.commit()
    st.success(f"Successfully saved result for product ID {result.get('pid')} to the database!")

# -------------------------
# Main App UI
# -------------------------
if load_btn:
    df = load_and_prepare_data(host, user, password, port, db_name)
    st.session_state["product_df"] = df
    if df.empty:
        st.warning("Query returned no data. Check database connection or SQL query.")
    else:
        st.success(f"Successfully loaded and prepared {len(df)} products.")

tab1, tab2 = st.tabs(["1. View Loaded Products", "2. Generate Single Product"])

with tab1:
    if "product_df" in st.session_state:
        st.dataframe(st.session_state["product_df"][['id', 'sku', 'product_name', 'brand_name']])
    else:
        st.info("Click 'Connect & Load Data' in the sidebar to begin.")

with tab2:
    if "product_df" not in st.session_state:
        st.warning("Please load data first using the sidebar button.")
    else:
        product_id_input = st.text_input("Enter Product ID to generate content for:", placeholder="e.g., 8441")
        
        if st.button("Generate Content", type="primary"):
            if not product_id_input.strip():
                st.error("Please enter a Product ID.")
            elif not openai_key:
                st.error("Please enter your OpenAI API Key in the sidebar.")
            else:
                try:
                    product_id = int(product_id_input)
                    df = st.session_state["product_df"]
                    product_row = df[df['id'] == product_id]

                    if product_row.empty:
                        st.error(f"Product ID {product_id} not found in the loaded data.")
                    else:
                        with st.spinner("Generating content... This may take a moment. The process will retry up to 3 times on failure."):
                            client = OpenAI(api_key=openai_key)
                            result = generate_with_openai(product_row.iloc[0], client)
                            st.session_state['last_result'] = result
                except ValueError:
                    st.error("Product ID must be a number.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    if 'last_result' in st.session_state:
        st.divider()
        result = st.session_state['last_result']
        
        if result.get("error"):
            st.error(f"Error for Product ID {result.get('pid')}:")
            st.code(result.get("error"), language="text")
        elif result.get("data"):
            st.success(f"Successfully generated content for Product ID {result.get('pid')}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### English Source Data")
                st.code(result.get("source_json"), language="json")
            with col2:
                st.markdown("### Generated Arabic Content")
                st.json(result.get("data"))

            if st.button("💾 Save this Result to Database"):
                try:
                    save_single_result_to_db(result, host, user, password, port, db_name)
                except Exception as e:
                    st.error(f"Failed to save to database: {e}")
