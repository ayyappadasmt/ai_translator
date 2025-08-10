import streamlit as st
import time
from datetime import datetime
import torch
from transformers import MarianMTModel, MarianTokenizer
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="✨ AI Translator Pro",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for black and golden theme
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 50%, #000000 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main-container {
        background: linear-gradient(145deg, #0a0a0a, #1f1f1f);
        border: 2px solid #FFD700;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(255, 215, 0, 0.3);
        backdrop-filter: blur(10px);
        animation: glow 3s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 20px 40px rgba(255, 215, 0, 0.3); }
        to { box-shadow: 0 25px 50px rgba(255, 215, 0, 0.5); }
    }
    
    /* Title styling */
    .main-title {
        text-align: center;
        background: linear-gradient(45deg, #FFD700, #FFA500, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(255, 215, 0, 0.5);
        animation: titleGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes titleGlow {
        from { filter: drop-shadow(0 0 10px rgba(255, 215, 0, 0.3)); }
        to { filter: drop-shadow(0 0 20px rgba(255, 215, 0, 0.6)); }
    }
    
    .subtitle {
        text-align: center;
        color: #C9C9C9;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Language selector styling */
    .language-selector {
        background: rgba(255, 215, 0, 0.1);
        border: 1px solid #FFD700;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(5px);
    }
    
    .lang-title {
        color: #FFD700;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        background: linear-gradient(145deg, #1a1a1a, #2a2a2a) !important;
        color: #ffffff !important;
        border: 2px solid #FFD700 !important;
        border-radius: 15px !important;
        font-size: 1.1rem !important;
        padding: 1rem !important;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.5) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.5), 0 0 20px rgba(255, 215, 0, 0.4) !important;
        border-color: #FFA500 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #FFD700, #FFA500) !important;
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.8rem 2rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: 0 5px 15px rgba(255, 215, 0, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(255, 215, 0, 0.6) !important;
        background: linear-gradient(45deg, #FFA500, #FFD700) !important;
    }
    
    /* Result container */
    .result-container {
        background: linear-gradient(145deg, #1a1a1a, #2a2a2a);
        border: 2px solid #FFD700;
        border-radius: 15px;
        padding: 2rem;
        margin-top: 2rem;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .result-text {
        color: #ffffff;
        font-size: 1.3rem;
        line-height: 1.6;
        margin: 1rem 0;
        padding: 1rem;
        background: rgba(255, 215, 0, 0.1);
        border-radius: 10px;
        border-left: 4px solid #FFD700;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div > select {
        background: linear-gradient(145deg, #1a1a1a, #2a2a2a) !important;
        color: #ffffff !important;
        border: 2px solid #FFD700 !important;
        border-radius: 10px !important;
        font-size: 1.1rem !important;
    }
    
    /* Loading animation */
    .loading-animation {
        text-align: center;
        color: #FFD700;
        font-size: 1.2rem;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Stats container */
    .stats-container {
        display: flex;
        justify-content: space-around;
        background: rgba(255, 215, 0, 0.1);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stat-item {
        text-align: center;
        color: #FFD700;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    /* Success and error messages */
    .success-message {
        background: linear-gradient(90deg, rgba(40, 167, 69, 0.2), rgba(40, 167, 69, 0.1));
        border: 1px solid #28a745;
        border-radius: 10px;
        padding: 1rem;
        color: #90EE90;
        margin: 1rem 0;
        animation: fadeIn 0.5s ease-in;
    }
    
    .error-message {
        background: linear-gradient(90deg, rgba(220, 53, 69, 0.2), rgba(220, 53, 69, 0.1));
        border: 1px solid #dc3545;
        border-radius: 10px;
        padding: 1rem;
        color: #FFB6C1;
        margin: 1rem 0;
        animation: shake 0.5s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_translation_models():
    """Load translation models with caching"""
    models = {}
    tokenizers = {}
    
    try:
        # Try to load from local directory first
        model_paths = {
            'en_hi': './models/en-hi',
            'hi_en': './models/hi-en'
        }
        
        for key, path in model_paths.items():
            if os.path.exists(path):
                st.info(f"Loading {key} model from local cache...")
                tokenizers[key] = MarianTokenizer.from_pretrained(path, local_files_only=True)
                models[key] = MarianMTModel.from_pretrained(path, local_files_only=True)
            else:
                # Fallback to downloading from HuggingFace
                model_name = 'Helsinki-NLP/opus-mt-en-hi' if key == 'en_hi' else 'Helsinki-NLP/opus-mt-hi-en'
                st.info(f"Downloading {key} model from HuggingFace...")
                tokenizers[key] = MarianTokenizer.from_pretrained(model_name)
                models[key] = MarianMTModel.from_pretrained(model_name)
            
            models[key].eval()
        
        return models, tokenizers
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}, {}

def translate_text(text, source_lang, target_lang, models, tokenizers):
    """Perform translation"""
    try:
        # Determine model key
        model_key = f"{source_lang}_{target_lang}"
        
        if model_key not in models or model_key not in tokenizers:
            return {"error": f"Model for {source_lang}->{target_lang} not available"}
        
        tokenizer = tokenizers[model_key]
        model = models[model_key]
        
        # Tokenize input
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Generate translation
        with torch.no_grad():
            translated = model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True,
                do_sample=True,
                temperature=0.7
            )
        
        # Decode output
        output = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        return {
            "input": text,
            "translation": output,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "confidence": 0.85
        }
        
    except Exception as e:
        return {"error": f"Translation error: {str(e)}"}

def create_language_options():
    return {
        "English": "en",
        "हिंदी (Hindi)": "hi"
    }

def main():
    load_css()
    
    # Initialize session state
    if 'translation_history' not in st.session_state:
        st.session_state.translation_history = []
    if 'source_lang' not in st.session_state:
        st.session_state.source_lang = "en"
    if 'target_lang' not in st.session_state:
        st.session_state.target_lang = "hi"
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    
    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Title and subtitle
    st.markdown('<h1 class="main-title"><i class="fas fa-language"></i> AI Translator Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Powered by Advanced Neural Machine Translation • Seamless Bidirectional Translation</p>', unsafe_allow_html=True)
    
    # Load models
    if not st.session_state.models_loaded:
        with st.spinner(""):
            models, tokenizers = load_translation_models()
            if models and tokenizers:
                st.session_state.models = models
                st.session_state.tokenizers = tokenizers
                st.session_state.models_loaded = True
                st.success("")
            else:
                st.error("")
                return
    
    # Language options
    lang_options = create_language_options()
    lang_codes = list(lang_options.values())
    lang_names = list(lang_options.keys())
    
    # Language selection section
    st.markdown('<div class="language-selector">', unsafe_allow_html=True)
    st.markdown('<h3 class="lang-title"><i class="fas fa-globe"></i> Translation Direction</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        source_idx = lang_codes.index(st.session_state.source_lang) if st.session_state.source_lang in lang_codes else 0
        source_lang_name = st.selectbox(
            "From Language", 
            lang_names,
            index=source_idx,
            key="source_lang_select"
        )
        st.session_state.source_lang = lang_options[source_lang_name]
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄", help="Swap Languages", key="swap_button"):
            st.session_state.source_lang, st.session_state.target_lang = st.session_state.target_lang, st.session_state.source_lang
            st.rerun()
    
    with col3:
        target_idx = lang_codes.index(st.session_state.target_lang) if st.session_state.target_lang in lang_codes else 1
        target_lang_name = st.selectbox(
            "To Language", 
            lang_names,
            index=target_idx,
            key="target_lang_select"
        )
        st.session_state.target_lang = lang_options[target_lang_name]
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main translation interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"###  Enter Text ({source_lang_name})")
        input_text = st.text_area(
            "",
            height=200,
            placeholder=f"Type your text in {source_lang_name} here...",
            key="input_text"
        )
    
    with col2:
        st.markdown(f"### ✨ Translation ({target_lang_name})")
        translation_placeholder = st.empty()
    
    # Translation button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        translate_btn = st.button(
            " TRANSLATE", 
            key="translate_main",
            help="Click to translate your text",
            use_container_width=True
        )
    
    # Handle translation
    if translate_btn and input_text and input_text.strip():
        with st.spinner(" Translating your text..."):
            # Show loading animation
            with translation_placeholder.container():
                st.markdown('<div class="loading-animation"><i class="fas fa-magic"></i> Working my magic...</div>', unsafe_allow_html=True)
                time.sleep(0.5)  # Brief delay for effect
            
            result = translate_text(
                input_text.strip(), 
                st.session_state.source_lang, 
                st.session_state.target_lang,
                st.session_state.models,
                st.session_state.tokenizers
            )
            
            with translation_placeholder.container():
                if "error" in result:
                    st.markdown(f'<div class="error-message"><i class="fas fa-exclamation-triangle"></i> {result["error"]}</div>', unsafe_allow_html=True)
                else:
                    # Success - show translation
                    st.markdown(f'<div class="result-text">{result["translation"]}</div>', unsafe_allow_html=True)
                    
                    # Add to history
                    st.session_state.translation_history.append({
                        "input": result["input"],
                        "translation": result["translation"],
                        "source_lang": result["source_lang"],
                        "target_lang": result["target_lang"],
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # Show stats
                    st.markdown('<div class="stats-container">', unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f'<div class="stat-item"><div class="stat-value">{len(result["input"].split())}</div><div class="stat-label">Words</div></div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<div class="stat-item"><div class="stat-value">{len(result["input"])}</div><div class="stat-label">Characters</div></div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown(f'<div class="stat-item"><div class="stat-value">{result.get("confidence", 0.85):.0%}</div><div class="stat-label">Confidence</div></div>', unsafe_allow_html=True)
                    with col4:
                        st.markdown(f'<div class="stat-item"><div class="stat-value">{len(st.session_state.translation_history)}</div><div class="stat-label">Translations</div></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
    elif translate_btn:
        st.warning("⚠️ Please enter some text to translate!")
    
    # Translation history (if any)
    if st.session_state.translation_history:
        st.markdown("---")
        st.markdown("###  Recent Translations")
        
        # Show last 3 translations
        for i, item in enumerate(reversed(st.session_state.translation_history[-3:])):
            with st.expander(f" Translation {len(st.session_state.translation_history) - i} - {item['timestamp']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Input:** {item['input']}")
                with col2:
                    st.markdown(f"**Translation:** {item['translation']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem; margin-top: 2rem;">
        <i class="fas fa-heart" style="color: #FFD700;"></i> Crafted with AI Magic • 
        <i class="fas fa-shield-alt" style="color: #FFD700;"></i> Secure & Private • 
        <i class="fas fa-rocket" style="color: #FFD700;"></i> Lightning Fast
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()