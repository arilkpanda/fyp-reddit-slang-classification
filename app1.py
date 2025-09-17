import os import os 

import jsonimport json

import timeimport time

import joblibimport joblib

import stringimport string

import warningsimport warnings

import numpy as npimport numpy as np

import pandas as pdimport pandas as pd

import altair as altimport altair as alt

import streamlit as stimport streamlit as st

import seaborn as snsimport seaborn as sns

import matplotlib.pyplot as pltimport matplotlib.pyplot as plt



from collections import defaultdictfrom collections import defaultdict

from scipy.sparse import hstack, csr_matrixfrom scipy.sparse import hstack, csr_matrix

from sklearn.metrics import classification_report, confusion_matrixfrom sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.sequence import pad_sequencesfrom tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import load_modelfrom tensorflow.keras.models import load_model

from gensim.models import Word2Vec, KeyedVectorsfrom gensim.models import Word2Vec, KeyedVectors



warnings.filterwarnings("ignore")warnings.filterwarnings("ignore")

st.set_page_config(st.set_page_config(

    page_title="Reddit Slang Classifier",    page_title="Reddit Slang Classifier",

    layout="wide",    layout="wide",

    initial_sidebar_state="expanded"    initial_sidebar_state="expanded"

))



# Add the rest of app1.py content here...# Add Reddit Sans font
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Reddit+Sans:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Reddit Sans', sans-serif;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Reddit Sans', sans-serif;
            font-weight: 700;
        }
        
        .stButton button {
            font-family: 'Reddit Sans', sans-serif;
            font-weight: 500;
        }
        
        .stTextArea textarea {
            font-family: 'Reddit Sans', sans-serif;
        }
        
        .stSelectbox div {
            font-family: 'Reddit Sans', sans-serif;
        }
        
        .stTab {
            font-family: 'Reddit Sans', sans-serif;
        }
        
        div[data-testid="stExpander"] {
            font-family: 'Reddit Sans', sans-serif;
        }
        
        div[data-testid="stTable"] {
            font-family: 'Reddit Sans', sans-serif;
        }
        
        .stMarkdown {
            font-family: 'Reddit Sans', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Constants and utilities
# -----------------------
CLASS_LABELS = ['Abbreviation', 'Meme/Irony', 'Other', 'Profanity', 'Sentiment/Expression']
CLASS_COLORS = {
    'Abbreviation': '#4C78A8',
    'Meme/Irony': '#F58518',
    'Other': '#54A24B',
    'Profanity': '#E45756',
    'Sentiment/Expression': "#C0CA3F" 
}

MODELS_DIR = "."  # Load models from current directory
DATA_DIR = "data"

def color_for(label):
    return CLASS_COLORS.get(label, "#999999")

# -----------------------
# Ensemble wrapper 
# -----------------------
class EnsembleModel:
    def __init__(self, lgbm, svm, meta_clf, tfidf, tokenizer, mlb, w2v_model, idf_dict, bilstm_path=None):
        self.lgbm = lgbm
        self.svm = svm
        self.meta_clf = meta_clf
        self.tfidf = tfidf
        self.tokenizer = tokenizer
        self.mlb = mlb
        self.w2v_model = w2v_model
        self.idf_dict = idf_dict
        self.bilstm = None
        self.bilstm_path = bilstm_path
        if bilstm_path:
            self.load_bilstm(bilstm_path)

    def load_bilstm(self, path):
        self.bilstm = load_model(path)

    def get_tfidf_w2v_vector(self, text, dim=100):
        tokens = str(text).split()
        vecs, weights = [], []
        for token in tokens:
            if token in self.w2v_model.key_to_index and token in self.idf_dict:
                vecs.append(self.w2v_model[token])
                weights.append(self.idf_dict[token])
        if not vecs:
            return np.zeros(dim)
        vecs = np.array(vecs)
        weights = np.array(weights).reshape(-1, 1)
        return np.sum(vecs * weights, axis=0) / np.sum(weights)

    def predict_proba_all(self, texts):
        X_tfidf = self.tfidf.transform(texts)
        dim = self.w2v_model.vector_size if hasattr(self.w2v_model, "vector_size") else 100
        X_w2v = np.vstack([self.get_tfidf_w2v_vector(x, dim=dim) for x in texts])
        X_hybrid = hstack([X_tfidf, csr_matrix(X_w2v)])

        X_seq = pad_sequences(self.tokenizer.texts_to_sequences(texts), maxlen=100, padding="post")

        proba_lgbm = self.lgbm.predict_proba(X_hybrid)
        proba_svm = self.svm.predict_proba(X_hybrid)
        proba_bilstm = self.bilstm.predict(X_seq) if self.bilstm is not None else np.zeros((len(texts), len(self.mlb.classes_)))

        # Meta-level features are concatenated base probabilities
        meta_X = np.hstack([proba_lgbm, proba_svm, proba_bilstm])
        proba_ensemble = None
        if hasattr(self.meta_clf, "predict_proba"):
            proba_ensemble = self.meta_clf.predict_proba(meta_X)
        else:
            # Fallback: average voting if meta doesn't expose probabilities
            proba_ensemble = (proba_lgbm + proba_svm + proba_bilstm) / 3.0

        return {
            "lgbm": proba_lgbm,
            "svm": proba_svm,
            "bilstm": proba_bilstm,
            "ensemble": proba_ensemble
        }

    def predict(self, texts, which="ensemble"):
        probs = self.predict_proba_all(texts)[which]
        idx = np.argmax(probs, axis=1)
        labels = [self.mlb.classes_[i] if hasattr(self.mlb, "classes_") else CLASS_LABELS[i] for i in idx]
        return labels, probs

# -----------------------
# Cached loaders
# -----------------------
@st.cache_resource(show_spinner=False)
def load_pickle(path):
    with open(path, "rb") as f:
        return joblib.load(f)

@st.cache_resource(show_spinner=False)
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

@st.cache_resource(show_spinner=False)
def load_word2vec():
    import gensim.downloader as api
    try:
        return api.load("glove-wiki-gigaword-100")
    except Exception as e:
        print(f"Failed to load pretrained Word2Vec model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_keras(path):
    return load_model(path)

@st.cache_resource(show_spinner=False)
def lazy_import_shap():
    import shap
    return shap

# -----------------------
# Model/materialization
# -----------------------
def find(path):
    return path if os.path.exists(path) else None

def try_load_artifacts():
    artifacts = {}
    loading_status = {}
    
    # Load artifacts from current directory with status tracking
    required_files = {
        "tfidf": "tfidf_vectorizer.pkl",
        "tokenizer": "tokenizer.pkl",
        "mlb": "mlb.pkl",
        "idf_dict": "idf_dict.pkl",
        "svm": "svm_v2.pkl",
        "lgbm": "lgbm_v2.pkl",
        "ensemble": "ensemble_model.pkl",
        "bilstm": "bilstm_ensemble.keras"
    }
    
    # Load pretrained word2vec model
    try:
        artifacts["w2v"] = load_word2vec()
        loading_status["w2v"] = {"file_exists": True, "loaded": artifacts["w2v"] is not None}
    except Exception as e:
        artifacts["w2v"] = None
        loading_status["w2v"] = {"file_exists": False, "loaded": False, "error": str(e)}
    
    for key, filename in required_files.items():
        filepath = find(os.path.join(MODELS_DIR, filename))
        loading_status[key] = {"file_exists": filepath is not None, "loaded": False}
        
        try:
            if filepath:
                if key == "w2v":
                    artifacts[key] = load_word2vec(filepath)
                elif key == "bilstm":
                    artifacts[key] = load_keras(filepath)
                    artifacts["bilstm_path"] = filepath
                else:
                    artifacts[key] = load_pickle(filepath)
                loading_status[key]["loaded"] = artifacts[key] is not None
            else:
                if key == "idf_dict":
                    artifacts[key] = defaultdict(lambda: 1.0)
                    loading_status[key]["loaded"] = True
                else:
                    artifacts[key] = None
        except Exception as e:
            loading_status[key]["error"] = str(e)
            artifacts[key] = None
    
    # Print loading status
    print("\nModel Loading Status:")
    for key, status in loading_status.items():
        status_str = f"✓ Loaded" if status["loaded"] else "✗ Failed"
        if "error" in status:
            status_str += f" (Error: {status['error']})"
        print(f"{key}: {status_str}")
    
    return artifacts

ART = try_load_artifacts()

def build_ensemble_from_parts():
    if ART["lgbm"] and ART["svm"] and ART["mlb"] and ART["tfidf"] and ART["tokenizer"] and ART["w2v"]:
        ens = EnsembleModel(
            lgbm=ART["lgbm"],
            svm=ART["svm"],
            meta_clf=None,  
            tfidf=ART["tfidf"],
            tokenizer=ART["tokenizer"],
            mlb=ART["mlb"],
            w2v_model=ART["w2v"],
            idf_dict=ART["idf_dict"],
            bilstm_path=ART["bilstm_path"]
        )
        if ART["ensemble"]:
            # Attach BiLSTM if needed
            if (getattr(ART["ensemble"], "bilstm", None) is None) and ART["bilstm_path"]:
                try:
                    ART["ensemble"].load_bilstm(ART["bilstm_path"])
                except Exception:
                    pass
            return ART["ensemble"]
        return ens
    return None

ENSEMBLE = build_ensemble_from_parts()

# -----------------------
# Lightweight interpretability helpers
# -----------------------
def tokenize(text):
    # Minimal whitespace + punctuation split to align with TF-IDF tokenization
    return str(text).translate(str.maketrans("", "", string.punctuation)).lower().split()

def linear_token_contribs(text, clf, vectorizer, class_index):
    # Works for linear SVM/logistic with calibrated probabilities
    if not hasattr(clf, "coef_"):
        # Handle OneVsRestClassifier case
        if hasattr(clf, "estimators_"):
            base_clf = clf.estimators_[class_index]
            if hasattr(base_clf, "coef_"):
                tokens = tokenize(text)
                feature_names = np.array(vectorizer.get_feature_names_out())
                vec = vectorizer.transform([text])
                coef = base_clf.coef_.ravel() 
                nz_idx = vec.nonzero()[1]
                nz_vals = vec.data
                feats = feature_names[nz_idx]
                contribs = coef[nz_idx] * nz_vals
                token_contrib = defaultdict(float)
                for f, c in zip(feats, contribs):
                    token_contrib[f] += c
                pairs = [(t, token_contrib.get(t, 0.0)) for t in tokens]
                agg = defaultdict(float)
                for t, c in pairs:
                    agg[t] += c
                return sorted(agg.items(), key=lambda x: abs(x[1]), reverse=True)
        return []
        
    tokens = tokenize(text)
    feature_names = np.array(vectorizer.get_feature_names_out())
    vec = vectorizer.transform([text])
    coef = clf.coef_[class_index] if len(clf.coef_.shape) > 1 else clf.coef_.ravel()
    # Contribution ≈ coef * tfidf value for each feature present
    nz_idx = vec.nonzero()[1]
    nz_vals = vec.data
    feats = feature_names[nz_idx]
    contribs = coef[nz_idx] * nz_vals
    # Map contributions back to tokens by feature name match
    token_contrib = defaultdict(float)
    for f, c in zip(feats, contribs):
        token_contrib[f] += c
    # Return sorted per-token contributions
    pairs = [(t, token_contrib.get(t, 0.0)) for t in tokens]
    # Merge duplicates by token
    agg = defaultdict(float)
    for t, c in pairs:
        agg[t] += c
    return sorted(agg.items(), key=lambda x: abs(x[1]), reverse=True)

def format_token_highlights(contribs, top_k=10):
    """Format token contributions with both colored tokens and importance bars"""
    if not contribs:
        return "No token-level attributions available."
        
    contribs = contribs[:top_k]
    total_abs = sum(abs(c) for _, c in contribs)  # Sum of absolute values for percentage calculation
    max_abs = max(1e-9, max(abs(c) for _, c in contribs))  # For color intensity
    
    # First show the colored tokens at the top
    spans = []
    for tok, c in contribs:
        intensity = min(1.0, abs(c) / max_abs)
        if c >= 0:
            color = f"rgba(228,87,86,{0.2 + 0.6*intensity})"
        else:
            color = f"rgba(76,120,168,{0.2 + 0.6*intensity})"
        spans.append(f'<span style="background-color:{color}; padding:2px 4px; border-radius:4px; margin:2px; display:inline-block">{tok}</span>')
    
    tokens_html = " ".join(spans)
    
    # Then create importance bars for each token
    bars_html = '<div style="margin-top:20px;">'
    for tok, c in contribs:
        rel_importance = (abs(c) / total_abs * 100) if total_abs > 0 else 0  # Normalize to sum to 100%
        bar_width = rel_importance 
        direction = "🔺" if c >= 0 else "🔻"
        bar_color = "#FF4500" if c >= 0 else "#4C78A8" 
        bars_html += f'''
            <div style="margin:8px 0; display:flex; align-items:center; gap:10px;">
                <div style="width:100px; font-family:monospace;">{tok}</div>
                <div style="flex-grow:1; background-color:#f0f0f0; height:12px; border-radius:6px; overflow:hidden;">
                    <div style="width:{bar_width}%; height:100%; background-color:{bar_color};"></div>
                </div>
                <div style="width:60px; text-align:right; font-size:0.9em;">{rel_importance:.1f}%</div>
                <div style="width:40px; text-align:center; font-size:1.2em;">
                    {direction}
                </div>
            </div>'''
    bars_html += '</div>'
    
    return f'''
    <div style="margin-bottom:20px;">
        <div style="margin-bottom:15px;">{tokens_html}</div>
        {bars_html}
    </div>
    '''

# -----------------------
# UI Components 
# -----------------------
def header():
    st.markdown("""
        <h1 style="text-align: center;">
            Tracing Internet Vernacular with the<br>
            <span style="color: #FF4500;">Reddit</span> 
            Slang Classifier
        </h1>
    """, unsafe_allow_html=True)

def prob_chart(probs, labels):
    df = pd.DataFrame({"class": labels, "probability": probs})
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("class:N", sort="-x"),
        color=alt.Color("class:N", scale=alt.Scale(domain=labels, range=[color_for(c) for c in labels]), legend=None),
        tooltip=["class", alt.Tooltip("probability:Q", format=".3f")]
    ).properties(height=160)
    st.altair_chart(chart, use_container_width=True)

def performance_metrics():
    st.markdown("### Classification Reports")
    model = st.selectbox(
        "Select Model",
        ["Ensemble", "SVM", "LightGBM", "BiLSTM"],
        index=0
    )
    st.divider()
        
    detailed_metrics = {
            "Ensemble": {
                "Abbreviation": {"Precision": 0.92, "Recall": 0.91, "F1-Score": 0.92, "Support": 38847},
                "Meme/Irony": {"Precision": 0.91, "Recall": 0.85, "F1-Score": 0.88, "Support": 10465},
                "Other": {"Precision": 0.92, "Recall": 0.85, "F1-Score": 0.88, "Support": 8011},
                "Profanity": {"Precision": 0.96, "Recall": 0.96, "F1-Score": 0.96, "Support": 91984},
                "Sentiment/Expression": {"Precision": 0.92, "Recall": 0.90, "F1-Score": 0.91, "Support": 40575},
                "Overall Metrics": {
                    "Micro Avg": {"Precision": 0.94, "Recall": 0.93, "F1-Score": 0.93, "Support": 189882},
                    "Macro Avg": {"Precision": 0.93, "Recall": 0.89, "F1-Score": 0.91, "Support": 189882},
                    "Weighted Avg": {"Precision": 0.94, "Recall": 0.93, "F1-Score": 0.93, "Support": 189882},
                    "Accuracy": 0.9047
                }
            },
            "SVM": {
                "Abbreviation": {"Precision": 0.91, "Recall": 0.91, "F1-Score": 0.92, "Support": 38847},
                "Meme/Irony": {"Precision": 0.88, "Recall": 0.83, "F1-Score": 0.86, "Support": 10465},
                "Other": {"Precision": 0.89, "Recall": 0.80, "F1-Score": 0.85, "Support": 8011},
                "Profanity": {"Precision": 0.93, "Recall": 0.97, "F1-Score": 0.96, "Support": 91984},
                "Sentiment/Expression": {"Precision": 0.93, "Recall": 0.89, "F1-Score": 0.91, "Support": 40575},
                "Overall Metrics": {
                    "Micro Avg": {"Precision": 0.94, "Recall": 0.92, "F1-Score": 0.93, "Support": 189882},
                    "Macro Avg": {"Precision": 0.92, "Recall": 0.88, "F1-Score": 0.90, "Support": 189882},
                    "Weighted Avg": {"Precision": 0.94, "Recall": 0.92, "F1-Score": 0.93, "Support": 189882},
                    "Accuracy": 0.8928
                }
            },
            "LightGBM": {
                "Abbreviation": {"Precision": 0.90, "Recall": 0.93, "F1-Score": 0.92, "Support": 38847},
                "Meme/Irony": {"Precision": 0.90, "Recall": 0.87, "F1-Score": 0.88, "Support": 10465},
                "Other": {"Precision": 0.89, "Recall": 0.82, "F1-Score": 0.85, "Support": 8011},
                "Profanity": {"Precision": 0.95, "Recall": 0.97, "F1-Score": 0.96, "Support": 91984},
                "Sentiment/Expression": {"Precision": 0.91, "Recall": 0.92, "F1-Score": 0.91, "Support": 40575},
                "Overall Metrics": {
                    "Micro Avg": {"Precision": 0.92, "Recall": 0.94, "F1-Score": 0.93, "Support": 189882},
                    "Macro Avg": {"Precision": 0.91, "Recall": 0.90, "F1-Score": 0.90, "Support": 189882},
                    "Weighted Avg": {"Precision": 0.92, "Recall": 0.94, "F1-Score": 0.93, "Support": 189882},
                    "Accuracy": 0.8981
                }
            },
            "BiLSTM": {
                "Abbreviation": {"Precision": 0.94, "Recall": 0.87, "F1-Score": 0.90, "Support": 11850},
                "Meme/Irony": {"Precision": 0.88, "Recall": 0.80, "F1-Score": 0.84, "Support": 3267},
                "Other": {"Precision": 0.92, "Recall": 0.83, "F1-Score": 0.87, "Support": 2493},
                "Profanity": {"Precision": 0.93, "Recall": 0.96, "F1-Score": 0.94, "Support": 27303},
                "Sentiment/Expression": {"Precision": 0.93, "Recall": 0.84, "F1-Score": 0.88, "Support": 12267},
                "Overall Metrics": {
                    "Micro Avg": {"Precision": 0.93, "Recall": 0.90, "F1-Score": 0.91, "Support": 57180},
                    "Macro Avg": {"Precision": 0.92, "Recall": 0.86, "F1-Score": 0.89, "Support": 57180},
                    "Weighted Avg": {"Precision": 0.93, "Recall": 0.90, "F1-Score": 0.91, "Support": 57180},
                    "Accuracy": 0.8845
                }
            }
        }
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Per-Category Performance")
        categories_df = pd.DataFrame([
            {
                "Category": cat,
                "Precision": detailed_metrics[model][cat]["Precision"],
                "Recall": detailed_metrics[model][cat]["Recall"],
                "F1-Score": detailed_metrics[model][cat]["F1-Score"],
                "Support": detailed_metrics[model][cat]["Support"]
            }
            for cat in ["Abbreviation", "Meme/Irony", "Other", "Profanity", "Sentiment/Expression"]
        ])
        st.table(categories_df.set_index("Category").style.format({
            "Precision": "{:.2f}",
            "Recall": "{:.2f}",
            "F1-Score": "{:.2f}",
            "Support": "{:,.0f}"
        }))
    
    with col2:
        # Display overall metrics
        st.markdown("##### Overall Metrics")
        overall_metrics = detailed_metrics[model]["Overall Metrics"]
        overall_df = pd.DataFrame([
            {"Metric": k, **v} for k, v in overall_metrics.items()
            if k != "Accuracy" 
        ])
        st.table(overall_df.set_index("Metric").style.format({
            "Precision": "{:.2f}",
            "Recall": "{:.2f}",
            "F1-Score": "{:.2f}",
            "Support": "{:,.0f}"
        }))
    
    # Display accuracy separately with enhanced styling
    overall_metrics = detailed_metrics[model]["Overall Metrics"]
    st.markdown(
        f"""
        <div style="
            background-color: #474140;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #FFFFFF;
            margin: 1rem 0;
        ">
            <span style="font-size: 1.1em; font-weight: bold; color: #FFFFFF;">Subset Accuracy:</span>
            <span style="font-size: 1.2em; font-weight: bold; color: #FFFFFF; margin-left: 0.5rem;">
                {overall_metrics['Accuracy']:.4f}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

def research_insights():
    st.markdown("""
        <h4>
            Trends in <span style="color: #FF4500;">Reddit</span> Comment Categories Over the Years (2014-2024)
        </h4>
        """, unsafe_allow_html=True)

    category_trends = {
            'Abbreviation': [21566, 24984, 29258, 33289, 34723, 34396, 36275, 40989, 41781, 43302, 43987],
            'Meme/Irony': [14232, 15594, 15160, 15414, 15108, 15088, 14941, 15593, 16362, 16304, 17264],
            'Other': [4768, 6856, 6161, 5103, 6943, 7160, 7511, 7847, 8346, 8499, 6390],
            'Profanity': [93510, 93481, 86975, 84917, 84283, 80840, 76762, 74922, 73487, 70788, 61508],
            'Sentiment/Expression': [25565, 28426, 30909, 34079, 36483, 35348, 37461, 41802, 41701, 43724, 38363]
        }
        
    years = range(2014, 2025)
    data = []
    for cat in CLASS_LABELS:
        for year, value in zip(years, category_trends[cat]):
            data.append({
               "year": year,
                "category": cat,
                "number_of_comments": value
            })
        
    df = pd.DataFrame(data)
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("year:O", 
                axis=alt.Axis(title="Year", grid=True)),
        y=alt.Y("number_of_comments:Q",
                axis=alt.Axis(title="Number of Comments", grid=True),
                scale=alt.Scale(domain=[0, 90000])),
        color=alt.Color("category:N", 
                          scale=alt.Scale(domain=CLASS_LABELS, 
                                        range=[color_for(c) for c in CLASS_LABELS])),
        tooltip=[
            alt.Tooltip("year:O", title="Year"),
            alt.Tooltip("category:N", title="Category"),
            alt.Tooltip("number_of_comments:Q", title="Number of Comments")
        ]
    ).properties(
        height=400,
    )
    st.altair_chart(chart, use_container_width=True)
    
    st.markdown("#### Key Findings")
    st.markdown("""
    1. **Profanity dominates but declines steadily**  
    &nbsp;- Peaked above **90k** in 2014.  
    &nbsp;- Shows a consistent downward trend, dropping to ~**60k by 2024**.  
    &nbsp;- Suggests a cultural or moderation-driven shift away from overt profanity.

    2. **Abbreviation and Sentiment/Expression are rising categories**  
    &nbsp;- Both grew steadily from 2014 to 2021, peaking around **40k–45k**.  
    &nbsp;- Slight dip after 2023, but still much higher than 2014 levels.  
    &nbsp;- Indicates shorthand communication and expressive language are becoming more central.

    3. **Meme/Irony remains stable but modest**  
    &nbsp;- Hovering between **15k–20k** across the decade.  
    &nbsp;- A consistent but niche mode of expression.

    **Final Takeaway:** The decline in profanity and the rise of abbreviations and sentiment-driven language align 
    with broader digital linguistic trends, especially among younger users (like Gen Alpha), who 
    tend to use short, socially coded, and normalized forms of profanities over raw mainstream derogatories.
    """)
    st.divider()
    
    st.markdown("#### Relative Distribution of Categories Over Time")
    
    # Calculate percentages for each year
    yearly_totals = df.groupby('year')['number_of_comments'].sum().reset_index()
    df_with_totals = df.merge(yearly_totals, on='year', suffixes=('', '_total'))
    df_with_totals['percentage'] = (df_with_totals['number_of_comments'] / df_with_totals['number_of_comments_total'])

    # Create stacked area chart
    stacked_area = alt.Chart(df_with_totals).mark_area().encode(
        x=alt.X('year:O', axis=alt.Axis(title='Year')),
        y=alt.Y('percentage:Q', 
                axis=alt.Axis(title='Percentage of Comments', format='.0%'),
                stack="normalize"),
        color=alt.Color('category:N', 
                       scale=alt.Scale(domain=CLASS_LABELS, range=[color_for(c) for c in CLASS_LABELS])),
        tooltip=[
            alt.Tooltip('year:O', title='Year'),
            alt.Tooltip('category:N', title='Category'),
            alt.Tooltip('percentage:Q', title='Percentage', format='.1f'),
            alt.Tooltip('number_of_comments:Q', title='Total Comments', format=',')
        ]
    ).properties(
        height=300,
    )
    
    st.altair_chart(stacked_area, use_container_width=True)
    st.divider()
    

    st.markdown("#### Correlation Between Category Trends")  

    corr_data = df.pivot(index='year', columns='category', values='number_of_comments')
    correlations = corr_data.corr()
    
    # Create correlation heatmap
    corr_df = pd.DataFrame(
        correlations.values,
        index=correlations.index,
        columns=correlations.columns
    )
    
    corr_melted = corr_df.reset_index().melt(id_vars='category', var_name='category2', value_name='correlation')
    
    correlation_heatmap = alt.Chart(corr_melted).mark_rect().encode(
        x=alt.X('category:N', title=None),
        y=alt.Y('category2:N', title=None),
        color=alt.Color('correlation:Q',
                       scale=alt.Scale(domain=[-1, 1], scheme='blueorange'),
                       legend=alt.Legend(title='Correlation')),
        tooltip=[
            alt.Tooltip('category:N', title='Category 1'),
            alt.Tooltip('category2:N', title='Category 2'),
            alt.Tooltip('correlation:Q', title='Correlation', format='.2f')
        ]
    ).properties(
        width=400,
        height=550,
    )
    
    st.altair_chart(correlation_heatmap, use_container_width=True)
    
    st.markdown("#### Key Findings")
    st.markdown("""
    1. **Abbreviation vs. Profanity → Strong Negative Correlation**  
    &nbsp;- As the use of abbreviations rises, profanity usage tends to fall.  
    &nbsp;- This suggests a generational or stylistic shift: younger users may prefer shorthand and coded language over raw profanity.
 
    2. **Abbreviation vs. Meme/Irony → Strong Positive Correlation**  
    &nbsp;- These two categories often move together.  
    &nbsp;- Indicates that meme culture and abbreviation-heavy communication are intertwined.
                    
    3. **Profanity vs. Sentiment/Expression → Negative Correlation**  
    &nbsp;- When profanity decreases, expressive/emotional language increases.  
    &nbsp;- Suggests that users are shifting from aggressive or raw expression toward more nuanced emotional communication.

    **Final Takeaway:** Reddit’s linguistic landscape is evolving from shock-value profanity toward coded brevity, irony, 
    and emotional resonance. The correlations reinforce that these categories aren’t just independent trends instead they’re actively 
    replacing one another in shaping how people communicate online.
    """)
    st.divider()

# -----------------------
# Pages
# -----------------------
def page_home():
    header()
    st.divider()

    st.markdown("### Quick Classification")
    text = st.text_area(
        "Enter a short comment/text containing common slang phrases",
        height=100,
        placeholder="Example: ngl i'm lowkey nervous about this presentation"
    )

    model_type = st.selectbox(
        "Model",
        ["Ensemble", "SVM only", "LightGBM only", "BiLSTM only"],
        index=0
    )

    # Initialize session states
    if 'classification_done' not in st.session_state:
        st.session_state.classification_done = False
    if 'slang_results' not in st.session_state:
        st.session_state.slang_results = None
    if 'classification_results' not in st.session_state:
        st.session_state.classification_results = None
        
    st.markdown("""
        <style>
        div[data-testid="stButton"] button {
            background-color: #FF4500;
            color: white;
        }
        div[data-testid="stButton"] button:hover {
            background-color: #cc3700;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    if st.button("Classify", type="primary"):
        if not text.strip():
            st.warning("Please enter some text to classify.")
            return
            
        st.session_state.classification_done = True
        
        # Common internet slang dictionary
        slang_dict = {
            # Common Abbreviations
            "afk": "Away From Keyboard",
            "aka": "Also Known As",
            "asap": "As Soon As Possible",
            "diy": "Do It Yourself",
            "fomo": "Fear Of Missing Out",
            "fr": "For real - Used to emphasize truthfulness or agreement",
            "gtg": "Got To Go",
            "icl": "I Can't Lie",
            "idk": "I Don't Know",
            "idek": "I Don't Even Know",
            "ikr": "I Know Right",
            "ifkr": "I Fucking Know Right",
            "ig": "I Guess",
            "imo": "In My Opinion",
            "irl": "In Real Life",
            "iykyk": "If You Know You Know",
            "lol": "Laughing Out Loud",
            "lmao": "Laughing My Ass Off",
            "ngl": "Not Gonna lie",
            "nvm": "Never Mind",
            "ong": "On God - Expression of truth/agreement",
            "rn": "Right Now",
            "rofl": "Rolling On Floor Laughing",
            "roflmao": "Rolling On Floor Laughing My Ass Off",
            "smh": "Shaking My Head",
            "tbh": "To Be Honest",
            "tf": "The Fuck",
            "ts": "That Shit",
            "wtf": "What The Fuck",
            "wth": "What The Hell",

            # Internet Culture/Memes
            "amogus": "Misspelling of 'Among Us' - Used in memes",
            "ate": "Slang for did something perfectly",
            "bussin": "Extremely good",
            "cap": "Lie or fake",
            "chad": "Alpha male stereotype",
            "gigachad": "Exaggerated form of 'chad'",
            "cheugy": "Out of date or trying too hard",
            "cringe": "Embarrassing or awkward",
            "dank": "Cool or high quality (often used for memes)",
            "dead": "Extremely funny",
            "ded": "Dead (informal spelling)",
            "delulu": "Delusional",
            "delusionship": "Delusional relationship",
            "drip": "Fashionable or trendy appearance",
            "dogwater": "Extremely bad or terrible",
            "doomscrolling": "Continuously scrolling through negative news",
            "fanum tax": "Taking someone's food (from YouTuber Fanum)",
            "fax": "Facts",
            "fleek": "Perfect or flawless",
            "on fleek": "Perfect or on point",
            "fire af": "Extremely good",
            "goofy ahh": "Silly or ridiculous",
            "gyat": "Reaction to something impressive",
            "gyatt": "Variation of 'gyat'",
            "highkey": "Obviously or explicitly",
            "huzz": "Casual greeting",
            "ick": "Sudden turnoff",
            "incel": "Involuntary celibate",
            "Karen": "Entitled or demanding person stereotype",
            "L": "Loss or fail",
            "lebron": "Reference to basketball player, used in memes",
            "let him cook": "Let someone do their thing",
            "lit": "Exciting or excellent",
            "locked in": "Focused or committed",
            "lowkey": "Subtle or secretive; Kind of or slightly.",
            "main character": "Center of attention",
            "mid": "Mediocre or average",
            "noice": "Nice (meme pronunciation)",
            "OK boomer": "Dismissive response to older generations",
            "quandale dingle": "Meme name",
            "ratio": "More replies than likes",
            "rizz": "Charisma or game",
            "rizzler": "Someone with rizz",
            "savage": "Fierce or harsh in a good way",
            "sheesh": "Expression of impression",
            "sigma": "Lone wolf personality type",
            "simp": "Someone overly attentive to others",
            "skibidi": "Reference to viral content",
            "slay": "Expression of approval",
            "slayed": "Did something perfectly",
            "sus": "Suspicious",
            "sussy": "Suspicious (cutesy form)",
            "sussy baka": "Suspicious person (anime-influenced)",
            "thicc": "Curvaceous or thick",
            "vibe": "Mood or atmosphere",
            "vibe check": "Assessing someone's mood",
            "woke": "Socially aware",
            "Zoomer": "Gen Z member",

            # Modern Relationship Terms
            "aura": "Someone's energy or presence",
            "alpha": "Dominant personality",
            "bae": "Before Anyone Else - Term of endearment",
            "bestie": "Best friend",
            "situationship": "Undefined romantic relationship",
            "ghosted": "Suddenly cutting off communication",
            "ghosting": "Act of suddenly cutting off communication",

            # Profanity and Strong Language
            "bitch": "Derogatory term",
            "dick": "Vulgar term for male anatomy",
            "fuck": "Strong expletive",
            "fucking": "Intensifier expletive",
            "fucker": "Derogatory term",
            "nigga": "Racial term (offensive)",
            "nigger": "Racial slur (highly offensive)",
            "pussy": "Vulgar term for female anatomy",
            "ass": "Vulgar term for posterior",
            "asshole": "Vulgar insult",
            "bastard": "Derogatory term",
            "bitchass": "Compound insult",
            "bloody": "Mild expletive (British)",
            "bollocks": "Vulgar term (British)",
            "bullshit": "Expression of disbelief",
            "cock": "Vulgar term for male anatomy",
            "crap": "Mild expletive",
            "cunt": "Strong vulgar term",
            "damn": "Mild expletive",
            "dayum": "Emphatic form of 'damn'",
            "dickhead": "Vulgar insult",
            "douche": "Derogatory term",
            "douchebag": "Insulting term",
            "dumbass": "Insulting term",
            "effing": "Euphemistic form of 'fucking'",
            "goddamn": "Religious expletive",
            "gyatdamn": "Stylized 'goddamn'",
            "gyatdayum": "Stylized 'dayum'",
            "jackass": "Insulting term",
            "motherfucker": "Strong compound expletive",
            "mf": "Abbreviation of 'motherfucker'",
            "piss": "Mild expletive",
            "pissed": "Angry or upset",
            "pussyass": "Compound insult",
            "shit": "Common expletive",
            "shitty": "Poor quality",
            "shyt": "Alternative spelling of 'shit'",
            "son of a bitch": "Compound insult",
            "tits": "Vulgar term for breasts",
            "twat": "Vulgar insult",
            "whore": "Derogatory term",
            "omg": "Oh my god",
            "af": "As fuck - Used for emphasis",
            "rn": "Right now",
            "fam": "Family (used informally to refer to friends)",
            "sus": "Suspicious or suspect",
            "cap": "Lie or lying (no cap = no lie)",
            "based": "Agreeable or worthy of support; displaying independence in behavior or opinion",
            "goat": "Greatest Of All Time",
            "slay": "To do something exceptionally well",
            "bet": "Okay or agreement (similar to 'alright')",
            "bussin": "Really good, especially regarding food",
            "no cap": "No lie or being completely honest"
        }    
        
        # Split text into words and check for slang
        words = text.lower().split()
        found_slang = []
        
        # single-word slang
        for word in words:
            word = word.strip(".,!?")
            if word in slang_dict:
                found_slang.append((word, slang_dict[word]))
        
        # multi-word slang
        text_lower = text.lower()
        for slang in slang_dict:
            if " " in slang and slang in text_lower:
                found_slang.append((slang, slang_dict[slang]))
        
        st.session_state.slang_results = found_slang
        
        model_key = {
            "Ensemble": "ensemble",
            "SVM only": "svm",
            "LightGBM only": "lgbm",
            "BiLSTM only": "bilstm"
        }[model_type]

        labels, probs = ENSEMBLE.predict([text], which=model_key)
        label = labels[0]
        
        st.session_state.classification_results = {
            "text": text,
            "model_key": model_key,
            "label": label,
            "probs": probs,
            "probs_all": ENSEMBLE.predict_proba_all([text])
        }
        st.divider()

    if st.session_state.classification_done:
        if st.session_state.slang_results:
            st.markdown("### Identified Slang Terms")
            for term, definition in st.session_state.slang_results:
                st.markdown(f"- **{term}**: {definition}")
            st.divider()
        
        # Show classification results
        results = st.session_state.classification_results
        probs = results['probs'][0]
        sorted_indices = np.argsort(probs)[::-1] 
        top_class = ENSEMBLE.mlb.classes_[sorted_indices[0]]
        second_class = ENSEMBLE.mlb.classes_[sorted_indices[1]]
        top_prob = probs[sorted_indices[0]]
        second_prob = probs[sorted_indices[1]]
        
        if second_prob > 0.5:
            st.markdown(f"""#### Classification: <span style='color:{color_for(top_class)}'>{top_class}</span> and <span style='color:{color_for(second_class)}'>{second_class}</span>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""#### Classification: <span style='color:{color_for(top_class)}'>{top_class}</span>""", unsafe_allow_html=True)
            
        prob_chart(results['probs'][0], ENSEMBLE.mlb.classes_)

        st.markdown("---")
        st.markdown("""
            <style>
                div[data-testid="stExpander"] {
                    border-left-color: #FF4500 !important;
                }
                div[data-testid="stExpander"] div[class*="streamlit-expanderHeader"] {
                    color: #666;
                    transition: color 0.2s;
                }
                div[data-testid="stExpander"] div[class*="streamlit-expanderHeader"]:hover {
                    color: #FF4500;
                }
                div[data-testid="stExpander"][data-testid="stExpander"] > div[class*="streamlit-expanderContent"] {
                    border-left-color: #FF4500 !important;
                }
            </style>
        """, unsafe_allow_html=True)
        with st.expander("Classification Breakdown", expanded=False):
            tabs = st.tabs(["Token Importance", "Model Agreement"])
            
            with tabs[0]:
                results = st.session_state.classification_results
                model_key = results["model_key"]
                text = results["text"]
                probs = results["probs"]
                
                # Using SVM's token importance analysis for all models
                top_idx = np.argmax(probs[0])
                if ART["svm"] is not None and ART["tfidf"] is not None:
                    try:
                        contribs = linear_token_contribs(text, ART["svm"], ART["tfidf"], class_index=top_idx)
                        if contribs:
                            st.markdown(format_token_highlights(contribs), unsafe_allow_html=True)
                        else:
                            st.info("Could not calculate token importance for this prediction.")
                    except Exception as e:
                        st.error(f"Token analysis failed: {str(e)}")
                else:
                    st.warning("Required models (SVM and TF-IDF) not loaded.")
            
            with tabs[1]:
                results = st.session_state.classification_results
                probs_all = results["probs_all"]
                df = pd.DataFrame({
                    "Model": ["Ensemble", "SVM", "LightGBM", "BiLSTM"],
                    "Confidence": [
                        max(probs_all["ensemble"][0]),
                        max(probs_all["svm"][0]),
                        max(probs_all["lgbm"][0]),
                        max(probs_all["bilstm"][0])
                    ],
                    "Predicted": [
                        CLASS_LABELS[np.argmax(probs_all["ensemble"][0])],
                        CLASS_LABELS[np.argmax(probs_all["svm"][0])],
                        CLASS_LABELS[np.argmax(probs_all["lgbm"][0])],
                        CLASS_LABELS[np.argmax(probs_all["bilstm"][0])],
                    ]
                })
                st.table(df)
            
def page_performance():
    st.markdown("## Model Performance Analysis")
    performance_metrics()

def page_insights():
    st.markdown("## Research Insights & Category Evolution")
    research_insights()
       
def page_about():
    st.markdown("## About")
    
    st.markdown("### Project Overview")
    st.write("""
    This dashboard demonstrates an ensemble approach to classifying Reddit comments
    across five distinct categories and tracing the evolution of internet vernacular
    over the last decade (2014-2024).
    """)
    st.divider()

    st.markdown("### Methods & Data")
    st.write("""
    - **Data Collection:** 1M+ Reddit comments, stratified sampling
    - **Preprocessing:** TF-IDF vectorization, Word2Vec embeddings
    - **Model Architecture:** SVM + LightGBM + BiLSTM ensemble
    - **Dependencies:** TensorFlow 2.x, scikit-learn 1.0+ ...
    """)

def sidebar():
    with st.sidebar:
        st.markdown('## <span style="color: #FF4500;">󠁪󠁪 ⠀☰⠀ Navigation</span>', unsafe_allow_html=True)
        st.markdown("""
            <style>
                div[data-testid="stRadio"] > label {
                    display: none;
                }
                div[data-testid="stRadio"] > div {
                    gap: 0.5rem;
                }
                div[data-testid="stRadio"] > div > label {
                    padding: 0.5rem 1rem;
                    border-radius: 0.5rem;
                    color: #666;
                    width: 100%;
                    transition: all 0.2s;
                }
                div[data-testid="stRadio"] > div > label:hover {
                    background-color: #ff45001a;
                    color: #FF4500;
                }
                div[data-testid="stRadio"] > div > label[data-checked="true"] {
                    background-color: #ff45001a;
                    color: #FF4500;
                    font-weight: 600;
                    border-left: 3px solid #FF4500;
                }
            </style>
        """, unsafe_allow_html=True)
        page = st.radio(
            " ",
            ["Classifier", "Research Insights", "Model Performance", "About"],
            index=0,
            label_visibility="collapsed"
        )
   
    return page

# -----------------------
# Main
# -----------------------
def main():
    page = sidebar()
    
    if page == "Classifier":
        page_home()
    elif page == "Research Insights":
        page_insights()
    elif page == "Model Performance":
        page_performance()
    elif page == "About":
        page_about()

if __name__ == "__main__":
    main()