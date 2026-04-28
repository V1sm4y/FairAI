import io
import json
import tempfile
import zipfile
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from bias_analysis import (
    _json_default,
    build_risk_summary,
    evaluate_fairness,
    load_model_bundle,
    run_analysis_workflow,
    save_model_bundle,
)
from bias_insights import generate_bias_insights, get_bias_reduction_suggestions

st.set_page_config(
    page_title="FairAI Studio",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_DATA_PATH = Path(__file__).resolve().parent / "sample_data.csv"


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --fairai-ink: #163130;
            --fairai-ink-soft: #486665;
            --fairai-paper: #f7f9fb;
            --fairai-card: rgba(255, 255, 255, 0.94);
            --fairai-line: rgba(22, 49, 48, 0.10);
            --fairai-accent: #c1663a;
            --fairai-accent-soft: #f3d9c8;
            --fairai-support: #2d8f7c;
            --fairai-shadow: 0 14px 34px rgba(22, 49, 48, 0.08);
        }

        .stApp {
            background: linear-gradient(180deg, #f7f9fb 0%, #eef4f2 100%);
            color: var(--fairai-ink);
        }

        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: var(--fairai-ink);
            font-family: Georgia, Cambria, "Trebuchet MS", serif;
            letter-spacing: 0;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #143534 0%, #1c4745 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stMarkdown *,
        [data-testid="stSidebar"] .stRadio,
        [data-testid="stSidebar"] .stRadio *,
        [data-testid="stSidebar"] .stCheckbox,
        [data-testid="stSidebar"] .stCheckbox *,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSelectbox label *,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stSlider label *,
        [data-testid="stSidebar"] .stNumberInput label,
        [data-testid="stSidebar"] .stNumberInput label *,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
            color: #f7f4ee !important;
            opacity: 1 !important;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            opacity: 1 !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="select"] *,
        [data-testid="stSidebar"] [data-baseweb="select"] input {
            color: #163130 !important;
        }

        [data-testid="stSidebar"] .stNumberInput input {
            color: #163130 !important;
            background: #fffaf2 !important;
        }

        [data-testid="stSidebar"] code {
            color: #143534 !important;
            background: rgba(255, 250, 242, 0.72) !important;
        }

        div[data-baseweb="popover"] [role="listbox"],
        div[data-baseweb="popover"] [role="option"] {
            background: #fffaf2 !important;
            color: #163130 !important;
        }

        div[data-baseweb="popover"] [role="option"]:hover {
            background: #f3e3d4 !important;
        }

        .hero-card {
            padding: 1.35rem 1.5rem;
            border: 1px solid var(--fairai-line);
            border-radius: 8px;
            background:
                linear-gradient(120deg, rgba(255, 255, 255, 0.98), rgba(243, 249, 247, 0.92)),
                linear-gradient(135deg, rgba(193, 102, 58, 0.05), rgba(45, 143, 124, 0.06));
            box-shadow: var(--fairai-shadow);
            margin-bottom: 1.2rem;
        }

        .hero-kicker {
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: var(--fairai-support);
            font-weight: 700;
            margin-bottom: 0.45rem;
        }

        .hero-copy {
            color: var(--fairai-ink-soft);
            font-size: 1rem;
            line-height: 1.55;
            max-width: 64rem;
        }

        .metric-card {
            border-radius: 8px;
            border: 1px solid var(--fairai-line);
            background: var(--fairai-card);
            box-shadow: var(--fairai-shadow);
            padding: 1rem 1rem 0.95rem;
            min-height: 7.4rem;
        }

        .metric-label {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.73rem;
            color: var(--fairai-ink-soft);
            margin-bottom: 0.35rem;
        }

        .metric-value {
            font-size: 2rem;
            line-height: 1;
            font-weight: 700;
            color: var(--fairai-ink);
            margin-bottom: 0.45rem;
        }

        .metric-note {
            color: var(--fairai-ink-soft);
            font-size: 0.92rem;
            line-height: 1.45;
        }

        .section-shell {
            border-radius: 8px;
            border: 1px solid var(--fairai-line);
            background: rgba(255, 255, 255, 0.82);
            box-shadow: var(--fairai-shadow);
            padding: 1rem 1.15rem 0.75rem;
            margin-bottom: 1rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.84);
            border-radius: 8px;
            border: 1px solid rgba(22, 49, 48, 0.08);
            padding: 0.55rem 1rem;
        }

        .report-note {
            border: 1px solid rgba(193, 102, 58, 0.18);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            background: #fff8f4;
            color: var(--fairai-ink-soft);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_sample_dataframe():
    return pd.read_csv(SAMPLE_DATA_PATH)


def read_uploaded_dataframe(uploaded_file):
    return pd.read_csv(io.BytesIO(uploaded_file.getvalue()))


def render_metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title, body):
    st.markdown(
        f"""
        <div class="section-shell">
            <h3 style="margin-bottom:0.25rem;">{title}</h3>
            <div style="color:#486665; line-height:1.5;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_metric(value):
    return 0.0 if value is None else float(value)


def metric_text(value):
    return "N/A" if value is None else f"{float(value):.3f}"


def percent_text(value):
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def improvement_label(improvement_summary):
    headline = improvement_summary.get("headline") or {}
    before_value = float(headline.get("before") or 0.0)
    after_value = float(headline.get("after") or 0.0)
    relative = float(headline.get("relative_reduction") or 0.0)
    if before_value <= 0:
        return "Stable"
    if relative > 0:
        return f"{relative * 100:.0f}%"
    if after_value > before_value:
        return "Worse"
    return "0%"


def improvement_note(improvement_summary):
    headline = improvement_summary.get("headline") or {}
    label = headline.get("label", "largest fairness gap")
    before_value = float(headline.get("before") or 0.0)
    after_value = float(headline.get("after") or 0.0)
    if after_value <= before_value:
        return f"{label} moved from {before_value:.3f} to {after_value:.3f}."
    return f"{label} increased from {before_value:.3f} to {after_value:.3f}."


def improvement_table(improvement_summary):
    rows = []
    for item in improvement_summary.get("metrics", []):
        rows.append(
            {
                "fairness_metric": item["label"],
                "before_gap": item["before"],
                "after_gap": item["after"],
                "absolute_reduction": item["absolute_reduction"],
                "relative_reduction": percent_text(item["relative_reduction"]),
            }
        )
    return pd.DataFrame(rows)


def threshold_table(threshold_sweep):
    rows = threshold_sweep.get("rows", [])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).rename(
        columns={
            "threshold": "threshold",
            "accuracy": "accuracy",
            "demographic_parity_gap": "predicted_positive_gap",
            "tpr_gap": "tpr_gap",
            "fpr_gap": "fpr_gap",
            "max_gap": "largest_gap",
        }
    )


def guess_positive_label(values):
    hints = [
        "1",
        "true",
        "yes",
        "approved",
        "accept",
        "selected",
        "hire",
        "pass",
        "positive",
        "good",
    ]
    values = list(values)
    if not values:
        return None
    for hint in hints:
        for value in values:
            if hint in str(value).strip().lower():
                return value
    return values[-1]


def format_bullet_list(items):
    return "\n".join([f"- {item}" for item in items])


def risk_badge(level):
    palette = {
        "Low": ("#2d8f7c", "#e8f6f2"),
        "Moderate": ("#b06a13", "#fff0d8"),
        "High": ("#c1663a", "#fde7db"),
        "Severe": ("#a63f3f", "#fde2e2"),
    }
    ink, background = palette.get(level, ("#486665", "#edf3f0"))
    return (
        f"<span style='display:inline-block;padding:0.28rem 0.7rem;border-radius:999px;"
        f"background:{background};color:{ink};font-weight:700;font-size:0.86rem;'>{level} Risk</span>"
    )


def ensure_widget_value(key, options, default_value):
    if not options:
        return
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = default_value if default_value in options else options[0]


def pick_default_column(columns, preferred_names, fallback_index):
    lowered = {column.lower(): column for column in columns}
    for preferred in preferred_names:
        if preferred.lower() in lowered:
            return lowered[preferred.lower()]
    return columns[min(fallback_index, len(columns) - 1)]


def metrics_table(results):
    rows = []
    for group, metrics in results["group_metrics"].items():
        rows.append(
            {
                "group": group,
                "group_size": metrics["group_size"],
                "predicted_positive_rate": metrics["demographic_parity"],
                "tpr": metrics["true_positive_rate"],
                "fpr": metrics["false_positive_rate"],
            }
        )
    return pd.DataFrame(rows)


def confusion_table(results, label):
    rows = []
    for group, metrics in results["group_metrics"].items():
        matrix = metrics["confusion_matrix"]
        rows.append(
            {
                "stage": label,
                "group": group,
                "true_negative": matrix[0][0],
                "false_positive": matrix[0][1],
                "false_negative": matrix[1][0],
                "true_positive": matrix[1][1],
            }
        )
    return pd.DataFrame(rows)


def plot_group_distribution(dataset_summary):
    data = pd.DataFrame(dataset_summary["group_distribution"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(data["group"], data["count"], color=["#c1663a", "#2d8f7c", "#466e91", "#8f6f2d"][: len(data)])
    ax.set_title("Sensitive Group Distribution")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", labelrotation=25)
    ax.set_facecolor("#fbf8f1")
    fig.patch.set_facecolor("#fbf8f1")
    fig.tight_layout()
    return fig


def plot_target_distribution(dataset_summary):
    data = pd.DataFrame(dataset_summary["target_distribution"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(data["label"].astype(str), data["count"], color="#486665")
    ax.set_title("Target Distribution")
    ax.set_ylabel("Count")
    ax.set_facecolor("#fbf8f1")
    fig.patch.set_facecolor("#fbf8f1")
    fig.tight_layout()
    return fig


def plot_gap_comparison(before, after):
    labels = ["Predicted + Gap", "TPR Gap", "FPR Gap"]
    before_values = [
        safe_metric(before["disparities"]["demographic_parity"]),
        safe_metric(before["fairness_gap"]["tpr_difference"]),
        safe_metric(before["fairness_gap"]["fpr_difference"]),
    ]
    after_values = [
        safe_metric(after["disparities"]["demographic_parity"]),
        safe_metric(after["fairness_gap"]["tpr_difference"]),
        safe_metric(after["fairness_gap"]["fpr_difference"]),
    ]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - width / 2, before_values, width, label="Before", color="#c1663a")
    ax.bar(x + width / 2, after_values, width, label="After", color="#2d8f7c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(before_values + after_values + [0.12]) * 1.2)
    ax.set_ylabel("Gap size")
    ax.set_title("Fairness Gap Comparison")
    ax.legend(frameon=False)
    ax.set_facecolor("#fbf8f1")
    fig.patch.set_facecolor("#fbf8f1")
    fig.tight_layout()
    return fig


def plot_threshold_tradeoff(threshold_sweep):
    data = pd.DataFrame(threshold_sweep.get("rows", []))
    fig, ax_accuracy = plt.subplots(figsize=(8.5, 4.5))
    if data.empty:
        ax_accuracy.text(0.5, 0.5, "No threshold sweep available", ha="center", va="center")
        return fig

    ax_accuracy.plot(data["threshold"], data["accuracy"], marker="o", color="#335c81", label="Accuracy")
    ax_accuracy.set_xlabel("Decision threshold")
    ax_accuracy.set_ylabel("Accuracy", color="#335c81")
    ax_accuracy.tick_params(axis="y", labelcolor="#335c81")
    ax_accuracy.set_ylim(0, 1)

    ax_gap = ax_accuracy.twinx()
    ax_gap.plot(data["threshold"], data["max_gap"], marker="s", color="#c1663a", label="Largest fairness gap")
    ax_gap.set_ylabel("Largest fairness gap", color="#c1663a")
    ax_gap.tick_params(axis="y", labelcolor="#c1663a")
    ax_gap.set_ylim(0, max(float(data["max_gap"].max()) * 1.25, 0.10))

    ax_accuracy.set_title("Fairness vs Accuracy by Threshold")
    ax_accuracy.grid(axis="y", alpha=0.18)
    ax_accuracy.set_facecolor("#fbf8f1")
    fig.patch.set_facecolor("#fbf8f1")
    fig.tight_layout()
    return fig


def plot_group_metric_comparison(before, after):
    groups = sorted(set(before["group_metrics"]) | set(after["group_metrics"]))
    before_tpr = [before["group_metrics"].get(group, {}).get("true_positive_rate") or 0 for group in groups]
    after_tpr = [after["group_metrics"].get(group, {}).get("true_positive_rate") or 0 for group in groups]
    before_fpr = [before["group_metrics"].get(group, {}).get("false_positive_rate") or 0 for group in groups]
    after_fpr = [after["group_metrics"].get(group, {}).get("false_positive_rate") or 0 for group in groups]

    x = np.arange(len(groups))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - 1.5 * width, before_tpr, width, label="Before TPR", color="#335c81")
    ax.bar(x - 0.5 * width, after_tpr, width, label="After TPR", color="#68a0cf")
    ax.bar(x + 0.5 * width, before_fpr, width, label="Before FPR", color="#a84e4e")
    ax.bar(x + 1.5 * width, after_fpr, width, label="After FPR", color="#e08383")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.tick_params(axis="x", labelrotation=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Group Metrics Before vs After")
    ax.legend(frameon=False, ncols=2)
    ax.set_facecolor("#fbf8f1")
    fig.patch.set_facecolor("#fbf8f1")
    fig.tight_layout()
    return fig


def plot_feature_importance(feature_importance):
    data = pd.DataFrame(feature_importance["top_raw_features"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if data.empty:
        ax.text(0.5, 0.5, "No feature importance available", ha="center", va="center")
        return fig
    data = data.sort_values("importance", ascending=True)
    ax.barh(data["raw_feature"], data["importance"], color="#2d8f7c")
    ax.set_title("Top Feature Importance")
    ax.set_xlabel("Absolute coefficient contribution")
    ax.set_facecolor("#fbf8f1")
    fig.patch.set_facecolor("#fbf8f1")
    fig.tight_layout()
    return fig


def build_bundle_zip(model, metadata):
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path, metadata_path = save_model_bundle(model, metadata, Path(temp_dir))
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.write(model_path, arcname=Path(model_path).name)
            zip_file.write(metadata_path, arcname=Path(metadata_path).name)
        buffer.seek(0)
        return buffer.getvalue()


def load_bundle_from_upload(uploaded_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        bundle_path = Path(temp_dir) / "fairai_bundle.zip"
        bundle_path.write_bytes(uploaded_file.getvalue())
        with zipfile.ZipFile(bundle_path, "r") as zip_file:
            zip_file.extractall(temp_dir)
        return load_model_bundle(temp_dir)


def build_executive_report_html(analysis, tuned_results=None, tuned_threshold=None):
    risk_summary = analysis["risk_summary"]
    improvement_summary = analysis["improvement_summary"]
    target_configuration = analysis["target_configuration"]
    metadata = analysis["metadata"]
    before = analysis["results_before"]
    after = tuned_results or analysis["results_after"]
    threshold_label = f"{tuned_threshold:.2f}" if tuned_threshold is not None else "0.50"

    metric_rows = []
    for item in improvement_summary.get("metrics", []):
        metric_rows.append(
            "<tr>"
            f"<td>{escape(item['label'])}</td>"
            f"<td>{item['before']:.3f}</td>"
            f"<td>{item['after']:.3f}</td>"
            f"<td>{item['absolute_reduction']:.3f}</td>"
            f"<td>{percent_text(item['relative_reduction'])}</td>"
            "</tr>"
        )

    group_rows = []
    for group, metrics in after["group_metrics"].items():
        group_rows.append(
            "<tr>"
            f"<td>{escape(str(group))}</td>"
            f"<td>{metrics['group_size']}</td>"
            f"<td>{metrics['demographic_parity']:.3f}</td>"
            f"<td>{metric_text(metrics['true_positive_rate'])}</td>"
            f"<td>{metric_text(metrics['false_positive_rate'])}</td>"
            "</tr>"
        )

    insights = "".join([f"<li>{escape(item)}</li>" for item in analysis["insights"]])
    recommendations = "".join([f"<li>{escape(item)}</li>" for item in analysis["recommendations"]])

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>FairAI Executive Report</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; color: #163130; margin: 40px; line-height: 1.5; }}
    h1, h2 {{ margin-bottom: 0.25rem; }}
    .kicker {{ color: #2d8f7c; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700; font-size: 0.78rem; }}
    .summary {{ border: 1px solid #d9e5e2; border-left: 5px solid #2d8f7c; padding: 16px; border-radius: 8px; background: #f5fbf9; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 18px 0; }}
    .metric {{ border: 1px solid #d9e5e2; border-radius: 8px; padding: 12px; }}
    .label {{ color: #486665; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }}
    .value {{ font-size: 1.6rem; font-weight: 700; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 22px; }}
    th, td {{ border-bottom: 1px solid #d9e5e2; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eef4f2; }}
    .note {{ color: #486665; font-size: 0.92rem; }}
  </style>
</head>
<body>
  <div class="kicker">FairAI Studio</div>
  <h1>Executive Fairness Report</h1>
  <p class="note">Audit target: {escape(target_configuration['display_name'])}. Sensitive view: {escape(metadata['sensitive_column'])}.</p>

  <div class="summary">
    <strong>{escape(risk_summary['level'])} bias risk.</strong>
    {escape(risk_summary['summary'])}
  </div>

  <div class="grid">
    <div class="metric"><div class="label">Risk Score</div><div class="value">{risk_summary['score']}/100</div></div>
    <div class="metric"><div class="label">Gap Reduction</div><div class="value">{improvement_label(improvement_summary)}</div></div>
    <div class="metric"><div class="label">Accuracy Before</div><div class="value">{before['accuracy']:.3f}</div></div>
    <div class="metric"><div class="label">Accuracy After</div><div class="value">{after['accuracy']:.3f}</div></div>
  </div>

  <h2>Fairness Movement</h2>
  <table>
    <thead><tr><th>Metric</th><th>Before</th><th>After</th><th>Reduction</th><th>Relative</th></tr></thead>
    <tbody>{''.join(metric_rows)}</tbody>
  </table>

  <h2>Group Outcomes at Threshold {threshold_label}</h2>
  <table>
    <thead><tr><th>Group</th><th>Rows</th><th>Predicted Positive Rate</th><th>TPR</th><th>FPR</th></tr></thead>
    <tbody>{''.join(group_rows)}</tbody>
  </table>

  <h2>Key Insights</h2>
  <ul>{insights}</ul>

  <h2>Recommended Next Steps</h2>
  <ul>{recommendations}</ul>

  <p class="note">Responsible use: FairAI supports auditing and monitoring. It should not be used as the sole decision-maker for high-stakes outcomes.</p>
</body>
</html>
""".strip()


def build_report_bundle_zip(analysis, tuned_results=None, tuned_threshold=None):
    report_json = json.dumps(analysis["report"], indent=2, default=_json_default)
    report_html = build_executive_report_html(analysis, tuned_results=tuned_results, tuned_threshold=tuned_threshold)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("fairai_analysis_report.json", report_json)
        zip_file.writestr("fairai_executive_report.html", report_html)
    buffer.seek(0)
    return buffer.getvalue()


def render_prediction_results(model, metadata, prediction_frame, result_key):
    feature_frame = prediction_frame[metadata["feature_columns"]]
    predictions = model.predict(feature_frame)
    probabilities = model.predict_proba(feature_frame)[:, 1] if hasattr(model, "predict_proba") else [None] * len(predictions)
    inverse_mapping = {int(value): str(key) for key, value in metadata["target_mapping"].items()}

    output = prediction_frame.copy()
    output["predicted_binary"] = predictions.astype(int)
    output["predicted_label"] = [inverse_mapping.get(int(prediction), str(prediction)) for prediction in predictions]
    output["positive_probability"] = [
        float(probability) if probability is not None else None for probability in probabilities
    ]

    st.dataframe(output, use_container_width=True)
    st.download_button(
        "Download predictions as CSV",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name="fairai_predictions.csv",
        mime="text/csv",
        key=f"download_{result_key}",
        use_container_width=True,
    )


def render_manual_prediction_form(model, metadata, form_key):
    st.markdown("#### Single prediction")
    st.caption("Try a candidate profile and inspect the model output using the current fairness-aware model.")
    profiles = metadata.get("feature_profiles", {})
    feature_values = {}
    columns = st.columns(2)

    with st.form(form_key):
        for index, feature in enumerate(metadata["feature_columns"]):
            profile = profiles.get(feature, {})
            bucket = columns[index % 2]
            if profile.get("kind") == "numeric":
                feature_values[feature] = bucket.number_input(
                    feature,
                    value=float(profile.get("default", 0.0)),
                    help=f"Observed range: {profile.get('min', 0.0):.2f} to {profile.get('max', 0.0):.2f}",
                )
            else:
                options = profile.get("options", [])
                if options:
                    feature_values[feature] = bucket.selectbox(feature, options=options, index=0, key=f"{form_key}_{feature}")
                else:
                    feature_values[feature] = bucket.text_input(feature, value=str(profile.get("default", "")))

        submitted = st.form_submit_button("Predict outcome", use_container_width=True)

    if submitted:
        single_row = pd.DataFrame([feature_values])[metadata["feature_columns"]]
        render_prediction_results(model, metadata, single_row, result_key=f"{form_key}_single")


def render_batch_prediction_form(model, metadata, form_key):
    st.markdown("#### Batch prediction")
    batch_file = st.file_uploader(
        "Upload a CSV with the same feature columns used during training.",
        type="csv",
        key=f"{form_key}_batch_upload",
    )
    if batch_file is None:
        return

    batch_df = pd.read_csv(io.BytesIO(batch_file.getvalue()))
    missing = [column for column in metadata["feature_columns"] if column not in batch_df.columns]
    if missing:
        st.error(f"Prediction file is missing required columns: {missing}")
        return

    render_prediction_results(model, metadata, batch_df, result_key=f"{form_key}_batch")


def render_loaded_model_panel():
    st.markdown("#### Load a saved model bundle")
    st.caption("Upload a bundle downloaded from FairAI Studio to reuse the trained model later.")
    bundle_file = st.file_uploader("Model bundle (.zip)", type="zip", key="model_bundle_upload")
    if bundle_file is None:
        return None, None

    try:
        model, metadata = load_bundle_from_upload(bundle_file)
        st.success("Saved model bundle loaded successfully.")
        return model, metadata
    except Exception as exc:
        st.error(f"Could not load the bundle: {exc}")
        return None, None


def main():
    inject_styles()

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Fairness Audit Workspace</div>
            <h1 style="margin:0 0 0.35rem 0;">FairAI Studio</h1>
            <div class="hero-copy">
                Audit binary decision models for bias, compare fairness before and after mitigation,
                and turn the results into a decision-ready workflow with visuals, warnings, and reusable model bundles.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "dataset_source" not in st.session_state:
        st.session_state["dataset_source"] = "Use sample dataset"

    with st.sidebar:
        st.header("Audit Controls")
        source_choice = st.radio(
            "Dataset source",
            ["Use sample dataset", "Upload CSV"],
            index=0,
            key="dataset_source",
        )
        uploaded_file = None
        if source_choice == "Upload CSV":
            uploaded_file = st.file_uploader("Upload a CSV dataset", type="csv")

    try:
        if source_choice == "Use sample dataset":
            source_df = load_sample_dataframe()
            source_label = "sample_data.csv"
        elif uploaded_file is not None:
            source_df = read_uploaded_dataframe(uploaded_file)
            source_label = uploaded_file.name
        else:
            source_df = None
            source_label = "No dataset loaded"
    except Exception as exc:
        st.error(f"Could not read the dataset: {exc}")
        source_df = None
        source_label = "Dataset error"

    if source_df is None:
        st.info("Upload a CSV or switch to the sample dataset to start the fairness audit.")
        return

    columns = source_df.columns.tolist()
    default_target = pick_default_column(columns, ["hired", "loan_approved", "approved", "target", "label"], len(columns) - 1)
    default_sensitive = pick_default_column(columns, ["gender", "sex", "race", "ethnicity", "age_group"], 0)

    with st.sidebar:
        ensure_widget_value("target_column_selector", columns, default_target)
        target_column = st.selectbox("Target column", options=columns, key="target_column_selector")
        target_values = sorted(source_df[target_column].dropna().unique().tolist(), key=lambda value: str(value))
        if len(target_values) < 2:
            st.error("The selected target column needs at least 2 distinct values.")
            return
        guessed_positive = guess_positive_label(target_values)
        ensure_widget_value("positive_label_selector", target_values, guessed_positive)
        positive_label = st.selectbox(
            "Positive outcome",
            options=target_values,
            key="positive_label_selector",
            format_func=str,
        )
        remaining_values = [value for value in target_values if str(value) != str(positive_label)]
        if len(target_values) > 2:
            negative_label = st.text_input("Grouped negative label", value="Other")
            st.caption(
                f"FairAI will convert `{target_column}` into `{positive_label}` vs `{negative_label or 'Other'}` "
                f"for the audit."
            )
        else:
            negative_label = str(remaining_values[0]) if remaining_values else "Other"
            st.caption(f"FairAI will audit `{positive_label}` vs `{negative_label}`.")

        sensitive_options = [column for column in columns if column != target_column]
        if not sensitive_options:
            st.error("The dataset needs at least one non-target column to use as a sensitive attribute.")
            return
        chosen_sensitive = default_sensitive if default_sensitive != target_column else sensitive_options[0]
        ensure_widget_value("sensitive_column_selector", sensitive_options, chosen_sensitive)
        sensitive_column = st.selectbox(
            "Sensitive attribute",
            options=sensitive_options,
            key="sensitive_column_selector",
        )
        if "intersectional_enabled" not in st.session_state:
            st.session_state["intersectional_enabled"] = False
        intersectional_enabled = st.checkbox("Intersectional audit", key="intersectional_enabled")
        intersectional_columns = None
        if intersectional_enabled:
            current_intersectional = [
                column for column in st.session_state.get("intersectional_columns_selector", []) if column in sensitive_options
            ]
            if not current_intersectional:
                st.session_state["intersectional_columns_selector"] = [sensitive_column]
            else:
                st.session_state["intersectional_columns_selector"] = current_intersectional
            intersectional_columns = st.multiselect(
                "Protected attributes",
                options=sensitive_options,
                key="intersectional_columns_selector",
            )
            if not intersectional_columns:
                intersectional_columns = [sensitive_column]
        mitigation_method = st.selectbox("Mitigation", options=["reweighting", "resampling"])
        top_features = st.slider("Top features to show", min_value=3, max_value=12, value=8)
        test_size = st.slider("Test split", min_value=0.10, max_value=0.40, value=0.30, step=0.05)
        random_state = st.number_input("Random seed", min_value=1, max_value=9999, value=42)
        run_clicked = st.button("Run fairness audit", type="primary", use_container_width=True)

    active_sensitive_view = " + ".join(intersectional_columns) if intersectional_columns else sensitive_column
    if intersectional_columns:
        sensitive_group_count = source_df[intersectional_columns].fillna("Unknown").astype(str).agg(" | ".join, axis=1).nunique()
    else:
        sensitive_group_count = source_df[sensitive_column].nunique(dropna=True)

    render_section_title(
        "Dataset Preview",
        f"Working dataset: <strong>{source_label}</strong>. Select the decision target and the sensitive attribute in the sidebar, then run the audit.",
    )
    left, right = st.columns([1.3, 1])
    with left:
        st.dataframe(source_df.head(12), use_container_width=True)
    with right:
        st.markdown("#### Quick profile")
        st.write(f"Rows: {len(source_df)}")
        st.write(f"Columns: {len(source_df.columns)}")
        st.write(f"Potential features: {max(len(source_df.columns) - 1, 0)}")
        st.write(f"Target unique values: {source_df[target_column].nunique(dropna=True)}")
        st.write(f"Sensitive view: {active_sensitive_view}")
        st.write(f"Audited groups: {sensitive_group_count}")
        st.write(f"Audit outcome: {positive_label} vs {negative_label}")

    if run_clicked:
        try:
            with st.spinner("Training the model, auditing fairness, and building the comparison report..."):
                st.session_state["analysis"] = run_analysis_workflow(
                    df=source_df.copy(),
                    target_column=target_column,
                    sensitive_column=sensitive_column,
                    intersectional_columns=intersectional_columns,
                    positive_label=positive_label,
                    negative_label=negative_label,
                    mitigation=mitigation_method,
                    top_features=top_features,
                    test_size=test_size,
                    random_state=int(random_state),
                )
                st.session_state["analysis_source_label"] = source_label
        except Exception as exc:
            st.error(f"Audit failed: {exc}")

    analysis = st.session_state.get("analysis")
    if analysis is None:
        st.info("Run the fairness audit to unlock comparison charts, warnings, and prediction tools.")
        return

    before = analysis["results_before"]
    after = analysis["results_after"]
    dataset_summary = analysis["dataset_summary"]
    target_configuration = analysis["target_configuration"]
    risk_summary = analysis["risk_summary"]
    improvement_summary = analysis["improvement_summary"]

    card_a, card_b, card_c, card_d, card_e, card_f = st.columns(6)
    with card_a:
        render_metric_card("Bias Risk", risk_summary["level"], f"{risk_summary['score']}/100 - {risk_summary['trend']} trend")
    with card_b:
        render_metric_card("Gap Reduced", improvement_label(improvement_summary), improvement_note(improvement_summary))
    with card_c:
        render_metric_card("Accuracy Before", f"{before['accuracy']:.3f}", "Baseline performance on the test split.")
    with card_d:
        render_metric_card("Accuracy After", f"{after['accuracy']:.3f}", "Performance after mitigation is applied.")
    with card_e:
        render_metric_card("TPR Gap", metric_text(after["fairness_gap"]["tpr_difference"]), "Smaller is fairer. Measures opportunity difference across groups.")
    with card_f:
        render_metric_card("FPR Gap", metric_text(after["fairness_gap"]["fpr_difference"]), "Smaller is fairer. Measures harmful false approvals across groups.")

    overview_tab, diagnostics_tab, threshold_tab, prediction_tab, download_tab = st.tabs(
        ["Overview", "Diagnostics", "Threshold Lab", "Prediction Lab", "Downloads"]
    )
    tuned_threshold = 0.50
    tuned_results = after
    tuned_risk_summary = risk_summary

    with overview_tab:
        render_section_title(
            "Executive Summary",
            (
                f"Audit target: <strong>{target_configuration['display_name']}</strong>. "
                f"Sensitive attribute: <strong>{analysis['metadata']['sensitive_column']}</strong>. "
                f"FairAI currently rates this model as {risk_badge(risk_summary['level'])} "
                f"with a score of <strong>{risk_summary['score']}/100</strong>."
            ),
        )

        if risk_summary["level"] in {"High", "Severe"}:
            st.error(risk_summary["summary"])
        elif risk_summary["level"] == "Moderate":
            st.warning(risk_summary["summary"])
        else:
            st.success(risk_summary["summary"])

        insight_col, recommendation_col = st.columns(2)
        with insight_col:
            st.markdown("#### Key insights")
            st.markdown(format_bullet_list(analysis["insights"]))
        with recommendation_col:
            st.markdown("#### Recommended next steps")
            st.markdown(format_bullet_list(analysis["recommendations"]))

        # AI-powered insights using Gemini
        st.markdown("---")
        st.markdown("#### 🤖 AI Insights for Bias Reduction")
        
        with st.spinner("Generating AI-powered recommendations..."):
            try:
                # Build fairness report from analysis
                fairness_report = {
                    "metrics": {
                        "accuracy_before": before.get("accuracy", 0),
                        "accuracy_after": after.get("accuracy", 0),
                        "disparity_before": before.get("disparity", 0),
                        "disparity_after": after.get("disparity", 0),
                    },
                    "disparity_analysis": {
                        "risk_level": risk_summary.get("level", "Unknown"),
                        "risk_score": risk_summary.get("score", 0),
                    },
                    "risk_factors": analysis.get("warnings", []),
                }
                
                ai_insights = generate_bias_insights(fairness_report)
                st.markdown(ai_insights)
                
                # Also provide specific suggestions for the sensitive column
                if sensitive_column:
                    disparity = before.get("disparity", 0)
                    with st.expander(f"💡 Specific recommendations for '{sensitive_column}'"):
                        suggestions = get_bias_reduction_suggestions(sensitive_column, disparity)
                        st.markdown(suggestions)
                        
            except Exception as e:
                st.info(f"AI insights currently unavailable. Run a fairness audit to enable this feature.")

        render_section_title(
            "Bias Warnings",
            "These are the headline alerts and tradeoffs FairAI surfaced while comparing the baseline model with the mitigated model.",
        )
        for warning in analysis["warnings"]:
            if "worsened" in warning or "reduced accuracy" in warning:
                st.warning(warning)
            elif "improved" in warning:
                st.success(warning)
            else:
                st.info(warning)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Sensitive groups")
            st.dataframe(pd.DataFrame(dataset_summary["group_distribution"]), use_container_width=True)
            st.pyplot(plot_group_distribution(dataset_summary), clear_figure=True)
        with col2:
            st.markdown("#### Target balance")
            st.dataframe(pd.DataFrame(dataset_summary["target_distribution"]), use_container_width=True)
            st.pyplot(plot_target_distribution(dataset_summary), clear_figure=True)

        st.markdown("#### Fairness comparison")
        st.pyplot(plot_gap_comparison(before, after), clear_figure=True)
        st.markdown("#### Fairness movement")
        st.dataframe(improvement_table(improvement_summary), use_container_width=True)

    with diagnostics_tab:
        st.markdown("#### Risk score breakdown")
        risk_components = pd.DataFrame(
            [{"component": key, "points": value} for key, value in risk_summary["components"].items()]
        )
        st.dataframe(risk_components, use_container_width=True)

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.markdown("#### Before mitigation")
            st.dataframe(metrics_table(before), use_container_width=True)
        with metric_col2:
            st.markdown("#### After mitigation")
            st.dataframe(metrics_table(after), use_container_width=True)

        st.markdown("#### Group rate changes")
        st.pyplot(plot_group_metric_comparison(before, after), clear_figure=True)

        st.markdown("#### Feature importance")
        st.pyplot(plot_feature_importance(analysis["feature_importance"]), clear_figure=True)

        conf_before, conf_after = st.columns(2)
        with conf_before:
            st.markdown("#### Confusion matrices before")
            st.dataframe(confusion_table(before, "before"), use_container_width=True)
        with conf_after:
            st.markdown("#### Confusion matrices after")
            st.dataframe(confusion_table(after, "after"), use_container_width=True)

    with threshold_tab:
        render_section_title(
            "Decision Threshold Tuning",
            "Adjust the positive-decision threshold and compare the fairness and accuracy tradeoff before choosing an operating point.",
        )
        threshold_sweep = analysis["threshold_sweep"]
        best_threshold = threshold_sweep.get("best_fairness_threshold") or {"threshold": 0.50}
        st.caption(
            f"Fairness-focused threshold from the sweep: {float(best_threshold['threshold']):.2f}. "
            "The slider starts at the standard 0.50 operating point."
        )
        tuned_threshold = st.slider(
            "Decision threshold",
            min_value=0.20,
            max_value=0.80,
            value=0.50,
            step=0.05,
            help="Higher thresholds approve fewer positive outcomes; lower thresholds approve more.",
        )
        tuned_results = evaluate_fairness(
            analysis["model_after"],
            analysis["test_features"],
            analysis["test_labels"],
            analysis["test_sensitive"],
            threshold=tuned_threshold,
        )
        tuned_risk_summary = build_risk_summary(dataset_summary, before, tuned_results)

        tuned_a, tuned_b, tuned_c, tuned_d = st.columns(4)
        with tuned_a:
            render_metric_card("Tuned Accuracy", f"{tuned_results['accuracy']:.3f}", "Accuracy at the selected threshold.")
        with tuned_b:
            render_metric_card("Tuned Risk", tuned_risk_summary["level"], f"{tuned_risk_summary['score']}/100 risk score.")
        with tuned_c:
            render_metric_card("Tuned TPR Gap", metric_text(tuned_results["fairness_gap"]["tpr_difference"]), "Opportunity gap at this threshold.")
        with tuned_d:
            render_metric_card("Tuned FPR Gap", metric_text(tuned_results["fairness_gap"]["fpr_difference"]), "False-positive gap at this threshold.")

        st.pyplot(plot_threshold_tradeoff(threshold_sweep), clear_figure=True)
        st.markdown("#### Threshold sweep")
        st.dataframe(threshold_table(threshold_sweep), use_container_width=True)

        tuned_left, tuned_right = st.columns(2)
        with tuned_left:
            st.markdown("#### Group metrics at selected threshold")
            st.dataframe(metrics_table(tuned_results), use_container_width=True)
        with tuned_right:
            st.markdown("#### Tuned risk components")
            st.dataframe(
                pd.DataFrame(
                    [{"component": key, "points": value} for key, value in tuned_risk_summary["components"].items()]
                ),
                use_container_width=True,
            )

    with prediction_tab:
        st.markdown("### Predict with the current session model")
        st.caption(
            "This uses the mitigated model from the latest run, so users can inspect a fairness-aware prediction workflow."
        )
        render_manual_prediction_form(analysis["model_after"], analysis["metadata"], form_key="session_prediction")
        render_batch_prediction_form(analysis["model_after"], analysis["metadata"], form_key="session_prediction")

        st.markdown("---")
        loaded_model, loaded_metadata = render_loaded_model_panel()
        if loaded_model is not None and loaded_metadata is not None:
            render_manual_prediction_form(loaded_model, loaded_metadata, form_key="loaded_prediction")
            render_batch_prediction_form(loaded_model, loaded_metadata, form_key="loaded_prediction")

    with download_tab:
        st.markdown("### Analysis report")
        report_json = json.dumps(analysis["report"], indent=2, default=_json_default)
        st.download_button(
            "Download JSON report",
            data=report_json.encode("utf-8"),
            file_name="fairai_analysis_report.json",
            mime="application/json",
            use_container_width=True,
        )
        executive_report = build_executive_report_html(
            analysis,
            tuned_results=tuned_results,
            tuned_threshold=tuned_threshold,
        )
        st.download_button(
            "Download executive HTML report",
            data=executive_report.encode("utf-8"),
            file_name="fairai_executive_report.html",
            mime="text/html",
            use_container_width=True,
        )
        st.download_button(
            "Download complete report bundle",
            data=build_report_bundle_zip(analysis, tuned_results=tuned_results, tuned_threshold=tuned_threshold),
            file_name="fairai_report_bundle.zip",
            mime="application/zip",
            use_container_width=True,
        )

        st.markdown("### Model bundle")
        st.caption("Save the mitigated model now, then load the same bundle later in the Prediction Lab.")
        bundle_bytes = build_bundle_zip(analysis["model_after"], analysis["metadata"])
        st.download_button(
            "Download model bundle",
            data=bundle_bytes,
            file_name="fairai_model_bundle.zip",
            mime="application/zip",
            use_container_width=True,
        )

        st.markdown("### Responsible use")
        st.markdown(
            """
            <div class="report-note">
                FairAI supports auditing, monitoring, and documentation. It should not be used as the sole
                decision-maker for hiring, lending, healthcare, education, or other high-stakes outcomes.
            </div>
            """
            ,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
