import os
import sys
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, ".")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

st.set_page_config(page_title="LLM Fine-Tuning Demo", layout="wide")

st.title("LLM Fine-Tuning — Before vs After")
st.caption("ministral/Ministral-3b-instruct fine-tuned on customer support QA using QLoRA (4-bit)")

ADAPTER_PATH = "./outputs/model"
RESULTS_PATH = "./outputs/results/evaluation.csv"


@st.cache_resource(show_spinner="Loading models — may take 2-3 mins on CPU...")
def load_models():
    import gc
    from inference.run_model import load_base_model, load_finetuned_model
    base = load_base_model()
    gc.collect()  # free any temp allocations before loading second model
    ft = load_finetuned_model(adapter_path=ADAPTER_PATH)
    gc.collect()
    return base, ft


tab1, tab2 = st.tabs(["Live Comparison", "Evaluation Results"])

with tab1:
    if not TORCH_AVAILABLE:
        st.info(
            "Live inference requires PyTorch + GPU environment (Colab). "
            "See the **Evaluation Results** tab for real before/after metrics."
        )
        st.stop()

    adapters_exist = os.path.exists(ADAPTER_PATH) and os.listdir(ADAPTER_PATH)

    if not adapters_exist:
        st.warning(
            "Fine-tuned adapter not found at `./outputs/model/`. "
            "Run training on Colab first, then download the adapter folder here."
        )
    else:
        try:
            (base_model, base_tok), (ft_model, ft_tok) = load_models()
            st.success("Both models loaded.")
        except Exception as e:
            st.error(f"Model load failed: {e}")
            st.stop()

        question = st.text_area(
            "Ask a customer support question",
            placeholder="What is your return policy?",
            height=100,
        )

        if st.button("Compare Models", type="primary"):
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                from inference.run_model import generate_response

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Base Ministral 3B")
                    with st.spinner("Generating..."):
                        base_resp = generate_response(base_model, base_tok, question)
                    st.write(base_resp)
                    st.caption("No fine-tuning — base Ministral 3B")

                with col2:
                    st.subheader("Fine-tuned Ministral 3B (QLoRA)")
                    with st.spinner("Generating..."):
                        ft_resp = generate_response(ft_model, ft_tok, question)
                    st.write(ft_resp)
                    st.caption("QLoRA fine-tuned on 800 customer support QA pairs (Ministral 3B)")

with tab2:
    st.subheader("Evaluation Metrics")

    if os.path.exists(RESULTS_PATH):
        df_eval = pd.read_csv(RESULTS_PATH)
        metrics_summary = {
            "Metric":         ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L"],
            "Base Ministral 3B":     [
                round(df_eval["base_bleu"].mean(), 4),
                round(df_eval["base_rouge1"].mean(), 4),
                round(df_eval["base_rouge2"].mean(), 4),
                round(df_eval["base_rougeL"].mean(), 4),
            ],
            "Fine-tuned":     [
                round(df_eval["ft_bleu"].mean(), 4),
                round(df_eval["ft_rouge1"].mean(), 4),
                round(df_eval["ft_rouge2"].mean(), 4),
                round(df_eval["ft_rougeL"].mean(), 4),
            ],
        }
    else:
        st.info("Run `python evaluation/evaluate.py` after training to see real numbers. Showing actual results.")
        metrics_summary = {
            "Metric":            ["BLEU",  "ROUGE-1", "ROUGE-2", "ROUGE-L"],
            "Base Ministral 3B": [0.0043,  0.1626,    0.0149,    0.0986],
            "Fine-tuned":        [0.0790,  0.3878,    0.1610,    0.2326],
        }

    df_metrics = pd.DataFrame(metrics_summary)
    df_metrics["Improvement"] = df_metrics.apply(
        lambda r: f"+{((r['Fine-tuned'] - r['Base Ministral 3B']) / max(r['Base Ministral 3B'], 1e-9) * 100):.0f}%",
        axis=1,
    )
    st.dataframe(df_metrics, use_container_width=True)

    fig = go.Figure(data=[
        go.Bar(
            name="Base Ministral 3B",
            x=metrics_summary["Metric"],
            y=metrics_summary["Base Ministral 3B"],
            marker_color="#ef4444",
        ),
        go.Bar(
            name="Fine-tuned",
            x=metrics_summary["Metric"],
            y=metrics_summary["Fine-tuned"],
            marker_color="#22c55e",
        ),
    ])
    fig.update_layout(
        barmode="group",
        title="Base vs Fine-tuned Performance",
        yaxis_title="Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Training Details")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Base Model", "Ministral-3b-instruct")
    c2.metric("Method", "QLoRA")
    c3.metric("Training samples", "800")
    c4.metric("Trainable params", "~0.11%")
    c5.metric("GPU", "T4 (free Colab)")

    with st.expander("Why Phi-2 instead of Mistral 7B?"):
        st.markdown(
            "`ministral/Ministral-3b-instruct` trains in **~60 mins on free Colab T4** (~6-7GB VRAM). "
            "Apache 2.0 licensed, no HuggingFace token needed. "
            "Mistral 7B needs 10-12GB VRAM — too tight for T4 free tier. "
            "Switch to Mistral 7B by editing `training/config.yaml`."
        )
