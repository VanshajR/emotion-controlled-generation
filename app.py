import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import pandas as pd

st.set_page_config(
    page_title="Emotion-Controlled Dialogue Generation",
    page_icon="🎭",
    layout="wide"
)

# Title
st.title("🎭 Emotion-Controlled Dialogue Generation")
st.markdown("""
**Research Achievement:** Lightweight prefix conditioning for controllable text generation  
**Best Model:** 38.2% emotion accuracy (+9.8pp improvement over baseline, 2.67x better than random)
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎯 Interactive Demo", "📊 Model Comparison", "🔍 Emotion Classifier"])

@st.cache_resource
def load_models():
    """Load all models"""
    with st.spinner("Loading models..."):
        # Emotion classifier
        classifier = pipeline(
            "text-classification",
            model="models/emotion_classifier_roberta",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Best model (Prefix-Small)
        prefix_model = AutoModelForCausalLM.from_pretrained("models/gpt2_prefix_v2")
        prefix_tokenizer = AutoTokenizer.from_pretrained("models/gpt2_prefix_v2")
        
        # Baseline model
        baseline_model = AutoModelForCausalLM.from_pretrained("models/gpt2_baseline")
        baseline_tokenizer = AutoTokenizer.from_pretrained("models/gpt2_baseline")
        
        if torch.cuda.is_available():
            prefix_model = prefix_model.to("cuda")
            baseline_model = baseline_model.to("cuda")
        
        return classifier, prefix_model, prefix_tokenizer, baseline_model, baseline_tokenizer

try:
    emotion_classifier, prefix_model, prefix_tokenizer, baseline_model, baseline_tokenizer = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    models_loaded = False

EMOTIONS = ["happy", "sad", "angry", "fear", "disgust", "surprise", "neutral"]
EMOTION_EMOJIS = {"happy": "😊", "sad": "😢", "angry": "😠", "fear": "😨", "disgust": "🤢", "surprise": "😲", "neutral": "😐"}

# ============================================
# TAB 1: Interactive Generation Demo
# ============================================
with tab1:
    st.header("🎯 Live Generation with Emotion Control")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⚙️ Configuration")
        
        selected_emotion = st.selectbox(
            "Target Emotion",
            EMOTIONS,
            format_func=lambda x: f"{EMOTION_EMOJIS[x]} {x.capitalize()}"
        )
        
        context = st.text_area(
            "Dialogue Context",
            value="How was your day?",
            height=100,
            help="Enter a dialogue utterance"
        )
        
        temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
        max_length = st.slider("Max Length", 20, 100, 50, 10)
        
        generate_btn = st.button("🎯 Generate Response", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📝 Generated Output")
        
        if models_loaded and generate_btn and context:
            with st.spinner("Generating..."):
                # Create prompt
                prompt = f"Respond with {selected_emotion} emotion: {context}"
                
                # Generate
                inputs = prefix_tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                outputs = prefix_model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    pad_token_id=prefix_tokenizer.eos_token_id
                )
                
                full_text = prefix_tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = full_text.replace(prompt, "").strip()
                
                # Take first sentence if too long
                if '.' in response:
                    response = response.split('.')[0] + '.'
                
                st.success(response)
                
                # Classify generated response
                result = emotion_classifier(response)[0]
                detected = result['label']
                confidence = result['score']
                
                st.divider()
                st.subheader("🔍 Emotion Verification")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Target", f"{EMOTION_EMOJIS[selected_emotion]} {selected_emotion}")
                with col_b:
                    st.metric("Detected", f"{EMOTION_EMOJIS.get(detected, '❓')} {detected}")
                with col_c:
                    match = selected_emotion == detected
                    st.metric("Result", "✅ Match" if match else "❌ Miss")
                
                st.progress(confidence, text=f"Confidence: {confidence:.1%}")

# ============================================
# TAB 2: Model Comparison
# ============================================
with tab2:
    st.header("📊 Baseline vs Prefix Conditioning Comparison")
    
    st.markdown("""
    **Demonstrate the 9.8 percentage point improvement** our prefix conditioning method achieved.  
    Generate responses with both models and see the difference in emotion control.
    """)
    
    comp_emotion = st.selectbox(
        "Select Target Emotion",
        EMOTIONS,
        format_func=lambda x: f"{EMOTION_EMOJIS[x]} {x.capitalize()}",
        key="comp_emotion"
    )
    
    comp_context = st.text_input(
        "Dialogue Context",
        value="How was your day?",
        key="comp_context"
    )
    
    if st.button("🔬 Compare Models", type="primary", use_container_width=True):
        if models_loaded and comp_context:
            col_base, col_prefix = st.columns(2)
            
            with col_base:
                st.subheader("🔵 Baseline Model (No Conditioning)")
                with st.spinner("Generating baseline..."):
                    # Baseline generates without prefix
                    inputs = baseline_tokenizer(comp_context, return_tensors="pt")
                    if torch.cuda.is_available():
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    
                    outputs = baseline_model.generate(
                        **inputs,
                        max_length=50,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=baseline_tokenizer.eos_token_id
                    )
                    
                    baseline_text = baseline_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    baseline_response = baseline_text.replace(comp_context, "").strip()
                    if '.' in baseline_response:
                        baseline_response = baseline_response.split('.')[0] + '.'
                    
                    st.info(baseline_response)
                    
                    # Classify
                    base_result = emotion_classifier(baseline_response)[0]
                    st.write(f"**Detected:** {EMOTION_EMOJIS.get(base_result['label'], '❓')} {base_result['label']} ({base_result['score']:.1%})")
                    if base_result['label'] == comp_emotion:
                        st.success("✅ Correct emotion!")
                    else:
                        st.error(f"❌ Wrong emotion (wanted {comp_emotion})")
            
            with col_prefix:
                st.subheader("🟢 Prefix Model (Our Method)")
                with st.spinner("Generating with prefix..."):
                    # Prefix model with emotion conditioning
                    prompt = f"Respond with {comp_emotion} emotion: {comp_context}"
                    inputs = prefix_tokenizer(prompt, return_tensors="pt")
                    if torch.cuda.is_available():
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    
                    outputs = prefix_model.generate(
                        **inputs,
                        max_length=50,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=prefix_tokenizer.eos_token_id
                    )
                    
                    prefix_text = prefix_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    prefix_response = prefix_text.replace(prompt, "").strip()
                    if '.' in prefix_response:
                        prefix_response = prefix_response.split('.')[0] + '.'
                    
                    st.info(prefix_response)
                    
                    # Classify
                    prefix_result = emotion_classifier(prefix_response)[0]
                    st.write(f"**Detected:** {EMOTION_EMOJIS.get(prefix_result['label'], '❓')} {prefix_result['label']} ({prefix_result['score']:.1%})")
                    if prefix_result['label'] == comp_emotion:
                        st.success("✅ Correct emotion!")
                    else:
                        st.error(f"❌ Wrong emotion (wanted {comp_emotion})")
            
            # Summary
            st.divider()
            st.subheader("📈 Comparison Summary")
            
            summary_cols = st.columns(3)
            with summary_cols[0]:
                st.metric("Baseline Accuracy", "28.3%", help="Accuracy on full test set")
            with summary_cols[1]:
                st.metric("Prefix Accuracy", "38.2%", delta="+9.8pp", help="Our model's accuracy")
            with summary_cols[2]:
                improvement = ((38.2 - 28.3) / 28.3) * 100
                st.metric("Relative Improvement", f"{improvement:.1f}%", help="34.6% better")

# ============================================
# TAB 3: Emotion Classifier Demo
# ============================================
with tab3:
    st.header("🔍 Emotion Classifier (RoBERTa)")
    st.markdown("""
    **Fine-tuned RoBERTa classifier** trained on GoEmotions dataset.  
    **Accuracy:** 57.8% on 7-class emotion classification
    """)
    
    test_text = st.text_area(
        "Enter text to classify:",
        value="I'm so excited about this project! It's going to be amazing!",
        height=100
    )
    
    if st.button("🔍 Classify Emotion", type="primary"):
        if models_loaded and test_text:
            with st.spinner("Classifying..."):
                result = emotion_classifier(test_text)[0]
                emotion = result['label']
                score = result['score']
                
                st.success(f"## {EMOTION_EMOJIS.get(emotion, '❓')} **{emotion.upper()}**")
                st.progress(score, text=f"Confidence: {score:.1%}")
                
                # Show all predictions
                all_results = emotion_classifier(test_text, top_k=7)
                
                st.divider()
                st.subheader("📊 All Emotion Probabilities")
                
                df = pd.DataFrame([
                    {"Emotion": f"{EMOTION_EMOJIS.get(r['label'], '❓')} {r['label']}", "Probability": f"{r['score']:.1%}"}
                    for r in all_results
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)

# ============================================
# Footer with metrics
# ============================================
st.divider()
st.subheader("📊 Project Metrics")

metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("Best Model", "Prefix-Small")
with metric_cols[1]:
    st.metric("Emotion Accuracy", "38.2%")
with metric_cols[2]:
    st.metric("Improvement", "+9.8pp")
with metric_cols[3]:
    st.metric("Test Samples", "6,740")

st.markdown("""
---
**GitHub:** [emotion-controlled-generation](https://github.com/your-username/emotion-controlled-generation)  
**Models:** Available on HuggingFace Hub  
**Report:** See PROJECT_REPORT.md for technical details
""")
