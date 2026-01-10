import streamlit as st
import json
from predictor import NERPredictor

try:
    from st_ner_annotate import st_ner_annotate
except ImportError:
    st_ner_annotate = None

st.set_page_config(page_title="Medical NER Transformer", layout="wide")

st.title("🏥 Medical Named Entity Recognition")
st.markdown("""
This application uses a synthetic-trained Transformer model to identify **Diseases** and **Medications** in medical text.
The model uses subtoken-level prediction with word-level max-pooling of logits.
""")

@st.cache_resource
def load_predictor():
    return NERPredictor()

predictor = load_predictor()

# Sidebar for example sentences
st.sidebar.title("Examples")
examples = [
    "Patient presents with symptoms of Diabetes and was prescribed Metformin 500mg.",
    "History of Hypertension and Asthma. Currently on Albuterol as needed.",
    "The patient was diagnosed with Pneumonia and started on Amoxicillin.",
    "No history of Chronic Kidney Disease or Heart Failure.",
]

selected_example = st.sidebar.selectbox("Choose an example", [""] + examples)

input_text = st.text_area("Enter medical text for NER analysis:", 
                         value=selected_example if selected_example else "",
                         height=150)

if st.button("Analyze Entities"):
    if input_text:
        with st.spinner("Analyzing..."):
            predictions = predictor.predict(input_text)
            
            # Format for st_ner_annotate or custom display
            # st_ner_annotate usually expects a list of dictionaries with text and label
            
            if st_ner_annotate:
                st.subheader("Annotated Text")
                # Prepare data for st_ner_annotate
                # Note: st_ner_annotate might expect a specific format. 
                # If it's the one from the repo, it often takes (text, labels)
                # But let's try a simple display first or use its default if available.
                
                # Based on the repo name st_ner_annotate, it's often used for manual annotation
                # but can be pre-filled.
                
                # If we can't reliably use st_ner_annotate for display only, 
                # we'll use a custom colored display.
                
                # For now, let's provide a custom visualization that mimics it
                def get_color(label):
                    colors = {
                        "DISEASE": "#ff4b4b",  # Red
                        "MEDICATION": "#1f77b4", # Blue
                        "O": "transparent"
                    }
                    return colors.get(label, "transparent")

                st.markdown("### Visualization")
                html_elements = []
                for p in predictions:
                    text = p['text'].replace('\n', '<br>')
                    if p['label'] != 'O':
                        color = get_color(p['label'])
                        html_elements.append(
                            f'<span style="background-color: {color}; color: white; padding: 2px 6px; border-radius: 4px; margin: 0 2px; font-weight: bold;">'
                            f'{text} <small style="opacity: 0.8; font-size: 0.7em;">{p["label"]}</small></span>'
                        )
                    else:
                        html_elements.append(text)
                
                st.markdown(f'<div style="line-height: 2.5; font-size: 1.1em; border: 1px solid #ddd; padding: 20px; border-radius: 10px;">'
                            f'{"".join(html_elements)}</div>', unsafe_allow_html=True)
                
            else:
                st.warning("st_ner_annotate not found. Using fallback visualization.")
                # Fallback implementation (already done above)

            # Display a table of entities
            st.subheader("Entities Found")
            entities = [p for p in predictions if p['label'] != 'O']
            if entities:
                st.table(entities)
            else:
                st.info("No entities detected.")
                
    else:
        st.warning("Please enter some text first.")

st.sidebar.markdown("---")
st.sidebar.info("Model: NERTransformer (v1.0)\nArchitecture: Encoder-only Transformer\nLabels: DISEASE, MEDICATION")
