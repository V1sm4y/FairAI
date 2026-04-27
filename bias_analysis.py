import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_dataframe(csv_path, target_column, sensitive_column):
    df = pd.read_csv(csv_path)
    required_columns = {target_column, sensitive_column}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df = df.dropna(subset=[target_column, sensitive_column]).reset_index(drop=True)
    if df.empty:
        raise ValueError("Dataset is empty after dropping rows with missing target/sensitive values.")
    return df


def prepare_model_inputs(df, target_column, sensitive_column, mapping=None, excluded_feature_columns=None):
    y_raw = df[target_column].copy()
    sensitive = df[sensitive_column].copy()
    X = df.drop(columns=[target_column])
    protected_columns = set(excluded_feature_columns or [])
    protected_columns.add(sensitive_column)
    protected_columns = [column for column in protected_columns if column in X.columns]
    if protected_columns:
        X = X.drop(columns=protected_columns)

    if mapping is None:
        y, mapping = _encode_binary_target(y_raw)
    else:
        y = _map_target_series(y_raw, mapping)
        if y.isna().any():
            raise ValueError("Loaded target mapping does not cover all prediction labels.")

    return X.reset_index(drop=True), y.reset_index(drop=True), sensitive.reset_index(drop=True), mapping


def _build_preprocessor(X):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["string", "object", "category", "bool"]).columns.tolist()

    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No features available for model training after removing target and sensitive columns.")

    return ColumnTransformer(transformers, remainder="drop")


def train_model(X, y, sample_weight=None):
    if len(np.unique(y)) != 2:
        raise ValueError("Target variable must be binary for logistic regression.")

    preprocessor = _build_preprocessor(X)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(solver="liblinear", max_iter=1000)),
        ]
    )
    fit_params = {"classifier__sample_weight": sample_weight} if sample_weight is not None else {}
    model.fit(X, y, **fit_params)
    return model


def predict_with_threshold(model, X, threshold=0.50):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
        return (probabilities >= threshold).astype(int)
    return model.predict(X)


def evaluate_fairness(model, X, y, sensitive, threshold=0.50):
    y = pd.Series(y).reset_index(drop=True)
    sensitive = pd.Series(sensitive).fillna("Unknown").astype(str).reset_index(drop=True)
    y_pred = predict_with_threshold(model, X, threshold=threshold)
    groups = sorted(sensitive.unique(), key=str)
    metrics = {}

    for group in groups:
        mask = sensitive == group
        y_group = y[mask]
        yhat_group = y_pred[mask.to_numpy()]
        if len(y_group) == 0:
            continue

        predicted_positive_rate = np.mean(yhat_group == 1)
        positive_mask = y_group == 1
        negative_mask = y_group == 0
        tpr = np.mean(yhat_group[positive_mask] == 1) if positive_mask.any() else np.nan
        fpr = np.mean(yhat_group[negative_mask] == 1) if negative_mask.any() else np.nan
        cm = confusion_matrix(y_group, yhat_group, labels=[0, 1])

        metrics[str(group)] = {
            "group_size": int(mask.sum()),
            "demographic_parity": float(predicted_positive_rate),
            "true_positive_rate": float(tpr) if not np.isnan(tpr) else None,
            "equal_opportunity": float(tpr) if not np.isnan(tpr) else None,
            "false_positive_rate": float(fpr) if not np.isnan(fpr) else None,
            "confusion_matrix": cm.tolist(),
        }

    accuracy = float(accuracy_score(y, y_pred))
    disparities = _compute_disparities(metrics)
    return {
        "accuracy": accuracy,
        "threshold": float(threshold),
        "group_metrics": metrics,
        "disparities": disparities,
        "fairness_gap": {
            "tpr_difference": disparities["true_positive_rate"],
            "fpr_difference": disparities["false_positive_rate"],
        },
    }


def _compute_disparities(group_metrics):
    disparity = {}
    for key in ["demographic_parity", "true_positive_rate", "false_positive_rate"]:
        values = [m[key] for m in group_metrics.values() if m[key] is not None]
        disparity[key] = float(max(values) - min(values)) if values else None
    return disparity


def build_intersectional_group(df, columns):
    if not columns:
        raise ValueError("Choose at least one column for an intersectional audit.")

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Intersectional audit columns are missing from the dataset: {missing}")

    protected_frame = df[columns].fillna("Unknown").astype(str)
    return protected_frame.apply(
        lambda row: " | ".join([f"{column}={row[column]}" for column in columns]),
        axis=1,
    )


def summarize_fairness_change(results_before, results_after):
    metric_labels = {
        "demographic_parity": "predicted positive rate gap",
        "true_positive_rate": "true positive rate gap",
        "false_positive_rate": "false positive rate gap",
    }
    metrics = []
    for key, label in metric_labels.items():
        before_value = float(results_before["disparities"].get(key) or 0.0)
        after_value = float(results_after["disparities"].get(key) or 0.0)
        absolute_reduction = before_value - after_value
        relative_reduction = absolute_reduction / before_value if before_value > 0 else 0.0
        metrics.append(
            {
                "metric": key,
                "label": label,
                "before": before_value,
                "after": after_value,
                "absolute_reduction": absolute_reduction,
                "relative_reduction": relative_reduction,
            }
        )

    headline = max(metrics, key=lambda item: item["before"]) if metrics else None
    return {
        "headline": headline,
        "metrics": metrics,
        "accuracy_before": float(results_before["accuracy"]),
        "accuracy_after": float(results_after["accuracy"]),
        "accuracy_change": float(results_after["accuracy"] - results_before["accuracy"]),
    }


def sweep_thresholds(model, X, y, sensitive, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.20, 0.81, 0.05), 2)

    rows = []
    for threshold in thresholds:
        results = evaluate_fairness(model, X, y, sensitive, threshold=float(threshold))
        gaps = [
            float(results["disparities"].get("demographic_parity") or 0.0),
            float(results["fairness_gap"].get("tpr_difference") or 0.0),
            float(results["fairness_gap"].get("fpr_difference") or 0.0),
        ]
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(results["accuracy"]),
                "demographic_parity_gap": gaps[0],
                "tpr_gap": gaps[1],
                "fpr_gap": gaps[2],
                "max_gap": float(max(gaps)),
            }
        )

    if not rows:
        return {"rows": [], "best_fairness_threshold": None}

    best = min(rows, key=lambda row: (row["max_gap"], -row["accuracy"]))
    return {"rows": rows, "best_fairness_threshold": best}


def analyze_dataset(df, target_column, sensitive_column, target_mapping):
    sensitive_values = df[sensitive_column].fillna("Unknown").astype(str)
    group_counts = sensitive_values.value_counts(dropna=False).sort_index()
    target_counts = df[target_column].value_counts(dropna=False).sort_index()
    y_binary = _map_target_series(df[target_column], target_mapping)

    group_distribution = [
        {
            "group": str(group),
            "count": int(count),
            "percentage": float(count / len(df)),
            "positive_rate": float(y_binary[sensitive_values == group].mean()),
        }
        for group, count in group_counts.items()
    ]

    target_distribution = [
        {
            "label": str(label),
            "count": int(count),
            "percentage": float(count / len(df)),
        }
        for label, count in target_counts.items()
    ]

    group_ratio = float(group_counts.min() / group_counts.max()) if not group_counts.empty and group_counts.max() else 1.0
    target_ratio = float(target_counts.min() / target_counts.max()) if not target_counts.empty and target_counts.max() else 1.0

    imbalance_detection = {
        "group_ratio": group_ratio,
        "target_ratio": target_ratio,
        "group_imbalance_detected": bool(group_ratio < 0.60 or (group_counts.min() / len(df)) < 0.20),
        "target_imbalance_detected": bool(target_ratio < 0.50 or (target_counts.min() / len(df)) < 0.30),
    }

    return {
        "row_count": int(len(df)),
        "feature_count": int(len(df.columns) - 1),
        "group_distribution": group_distribution,
        "target_distribution": target_distribution,
        "imbalance_detection": imbalance_detection,
    }


def prepare_target_for_audit(df, target_column, positive_label=None, negative_label=None):
    df = df.copy()
    unique_values = pd.Series(df[target_column].dropna().unique()).tolist()
    if not unique_values:
        raise ValueError("Target column has no usable values.")

    if positive_label is None:
        if len(unique_values) != 2:
            raise ValueError(
                "Target column must have exactly 2 distinct values, or supply a positive class so FairAI can binarize it."
            )
        sorted_values = sorted(unique_values, key=lambda value: str(value))
        return df, {
            "original_target_column": target_column,
            "audit_target_column": target_column,
            "mode": "native_binary",
            "created_binary_target": False,
            "positive_label": str(sorted_values[1]),
            "negative_label": str(sorted_values[0]),
            "grouped_negative_values": [str(sorted_values[0])],
            "original_unique_values": [str(value) for value in unique_values],
            "display_name": f"{target_column}: {sorted_values[1]} vs {sorted_values[0]}",
        }

    positive_value = _resolve_label_value(unique_values, positive_label)
    other_values = [value for value in unique_values if not _values_equal(value, positive_value)]
    if not other_values:
        raise ValueError("FairAI needs at least one non-positive target value to build a binary outcome.")

    if negative_label is None or not str(negative_label).strip():
        negative_label = str(other_values[0]) if len(other_values) == 1 else "Other"

    negative_label = str(negative_label).strip()
    if negative_label == str(positive_value):
        negative_label = f"Not {positive_value}"

    positive_mask = df[target_column].apply(lambda value: _values_equal(value, positive_value))
    df[target_column] = np.where(positive_mask, str(positive_value), negative_label)

    return df, {
        "original_target_column": target_column,
        "audit_target_column": target_column,
        "mode": "binarized" if len(unique_values) > 2 else "custom_binary",
        "created_binary_target": bool(len(unique_values) > 2),
        "positive_label": str(positive_value),
        "negative_label": negative_label,
        "grouped_negative_values": [str(value) for value in other_values],
        "original_unique_values": [str(value) for value in unique_values],
        "display_name": f"{target_column}: {positive_value} vs {negative_label}",
    }


def summarize_feature_profiles(X, max_categories=12):
    profiles = {}
    for column in X.columns:
        series = X[column].dropna()
        if pd.api.types.is_numeric_dtype(X[column]):
            profiles[str(column)] = {
                "kind": "numeric",
                "dtype": str(X[column].dtype),
                "default": float(series.median()) if not series.empty else 0.0,
                "min": float(series.min()) if not series.empty else 0.0,
                "max": float(series.max()) if not series.empty else 1.0,
            }
        else:
            values = [str(value) for value in series.astype(str).value_counts().head(max_categories).index.tolist()]
            profiles[str(column)] = {
                "kind": "categorical",
                "dtype": str(X[column].dtype),
                "default": values[0] if values else "",
                "options": values,
            }
    return profiles


def build_risk_summary(dataset_summary, results_before, results_after):
    metric_labels = {
        "demographic_parity": "predicted positive rate gap",
        "true_positive_rate": "true positive rate gap",
        "false_positive_rate": "false positive rate gap",
    }
    before_gaps = {key: float(results_before["disparities"].get(key) or 0.0) for key in metric_labels}
    after_gaps = {key: float(results_after["disparities"].get(key) or 0.0) for key in metric_labels}
    largest_metric = max(after_gaps, key=after_gaps.get)
    largest_value = after_gaps[largest_metric]
    average_gap = float(np.mean(list(after_gaps.values()))) if after_gaps else 0.0
    accuracy_drop = max(0.0, float(results_before["accuracy"] - results_after["accuracy"]))

    residual_component = min(45.0, (largest_value / 0.15) * 45.0)
    average_component = min(20.0, (average_gap / 0.10) * 20.0)
    group_component = 10.0 if dataset_summary["imbalance_detection"]["group_imbalance_detected"] else 0.0
    target_component = 8.0 if dataset_summary["imbalance_detection"]["target_imbalance_detected"] else 0.0
    accuracy_component = min(12.0, (accuracy_drop / 0.10) * 12.0)
    improvement_credit = 5.0 if max(before_gaps.values()) - largest_value > 0.03 else 0.0

    score = int(round(np.clip(
        residual_component + average_component + group_component + target_component + accuracy_component - improvement_credit,
        0,
        100,
    )))

    if score >= 65:
        level = "Severe"
        color = "red"
    elif score >= 45:
        level = "High"
        color = "orange"
    elif score >= 25:
        level = "Moderate"
        color = "yellow"
    else:
        level = "Low"
        color = "green"

    before_largest = max(before_gaps.values()) if before_gaps else 0.0
    if largest_value < before_largest - 0.02:
        trend = "improving"
    elif largest_value > before_largest + 0.02:
        trend = "worsening"
    else:
        trend = "mixed"

    summary = (
        f"{level} bias risk. The largest remaining issue is the {metric_labels[largest_metric]} "
        f"at {largest_value:.3f}; mitigation trend is {trend}."
    )
    return {
        "score": score,
        "level": level,
        "color": color,
        "trend": trend,
        "summary": summary,
        "largest_remaining_gap_metric": largest_metric,
        "largest_remaining_gap_label": metric_labels[largest_metric],
        "largest_remaining_gap_value": largest_value,
        "average_gap_value": average_gap,
        "accuracy_drop": accuracy_drop,
        "components": {
            "residual_gap_points": round(residual_component, 2),
            "average_gap_points": round(average_component, 2),
            "group_imbalance_points": round(group_component, 2),
            "target_imbalance_points": round(target_component, 2),
            "accuracy_tradeoff_points": round(accuracy_component, 2),
            "improvement_credit": round(improvement_credit, 2),
        },
    }


def generate_model_insights(dataset_summary, results_before, results_after, risk_summary, target_configuration):
    insights = []
    if target_configuration.get("created_binary_target"):
        grouped = ", ".join(target_configuration["grouped_negative_values"][:4])
        if len(target_configuration["grouped_negative_values"]) > 4:
            grouped += ", ..."
        insights.append(
            f"FairAI audited `{target_configuration['positive_label']}` as the positive outcome and grouped "
            f"{grouped} into `{target_configuration['negative_label']}`."
        )

    positive_rate_story = _describe_group_extremes(results_after, "demographic_parity", "predicted positive rate")
    if positive_rate_story:
        insights.append(positive_rate_story)

    largest_metric = risk_summary["largest_remaining_gap_metric"]
    group_gap_story = _describe_group_extremes(results_after, largest_metric, risk_summary["largest_remaining_gap_label"])
    if group_gap_story and group_gap_story != positive_rate_story:
        insights.append(group_gap_story)

    accuracy_drop = risk_summary["accuracy_drop"]
    if accuracy_drop > 0.0:
        insights.append(
            f"Mitigation changed test accuracy from {results_before['accuracy']:.3f} to {results_after['accuracy']:.3f}, "
            f"a {accuracy_drop:.3f} drop that should be weighed against the fairness gains."
        )

    if dataset_summary["imbalance_detection"]["target_imbalance_detected"]:
        insights.append(
            "The target distribution is imbalanced, so headline accuracy should be interpreted alongside per-group error rates."
        )

    return insights[:4]


def generate_recommendations(dataset_summary, results_before, results_after, feature_importance, risk_summary, target_configuration):
    recommendations = []
    if target_configuration.get("created_binary_target") and len(target_configuration.get("grouped_negative_values", [])) > 1:
        recommendations.append(
            "Validate that collapsing multiple raw outcomes into one negative class matches the policy question you want to audit."
        )

    if dataset_summary["imbalance_detection"]["group_imbalance_detected"]:
        recommendations.append(
            "Collect or upweight more examples from underrepresented sensitive groups before trusting this model in production."
        )

    if dataset_summary["imbalance_detection"]["target_imbalance_detected"]:
        recommendations.append(
            "Track recall and false-positive behavior alongside accuracy because the outcome classes are imbalanced."
        )

    if risk_summary["largest_remaining_gap_value"] > 0.08:
        recommendations.append(
            "Keep a human review step for borderline decisions until the largest fairness gap drops below a safer threshold."
        )

    if results_before["accuracy"] - results_after["accuracy"] > 0.05:
        recommendations.append(
            "Compare reweighting against resampling and tune the decision threshold to recover some accuracy after mitigation."
        )

    if feature_importance["top_raw_features"]:
        top_feature = feature_importance["top_raw_features"][0]["raw_feature"]
        recommendations.append(
            f"Review `{top_feature}` as a possible proxy feature and document why it is appropriate for the decision policy."
        )

    if not recommendations:
        recommendations.append("Current fairness gaps are relatively small; keep monitoring with fresh data before deployment.")

    deduped = []
    for recommendation in recommendations:
        if recommendation not in deduped:
            deduped.append(recommendation)
    return deduped[:5]


def _describe_group_extremes(results, metric_key, metric_label):
    metric_label = metric_label.replace(" gap", "")
    scored_groups = []
    for group, metrics in results["group_metrics"].items():
        value = metrics.get(metric_key)
        if value is not None:
            scored_groups.append((group, value))

    if len(scored_groups) < 2:
        return None

    highest_group, highest_value = max(scored_groups, key=lambda item: item[1])
    lowest_group, lowest_value = min(scored_groups, key=lambda item: item[1])
    if highest_group == lowest_group:
        return None

    return (
        f"After mitigation, `{highest_group}` still has the highest {metric_label} ({highest_value:.3f}) "
        f"while `{lowest_group}` has the lowest ({lowest_value:.3f})."
    )


def extract_feature_importance(model, top_n=10):
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    coefficients = classifier.coef_[0]

    transformed_names, raw_feature_names = _extract_transformed_feature_names(preprocessor)
    if len(transformed_names) != len(coefficients):
        transformed_names = [f"feature_{idx}" for idx in range(len(coefficients))]
        raw_feature_names = transformed_names[:]

    transformed_importance = pd.DataFrame(
        {
            "transformed_feature": transformed_names,
            "raw_feature": raw_feature_names,
            "coefficient": coefficients,
            "importance": np.abs(coefficients),
        }
    ).sort_values("importance", ascending=False)

    raw_importance = (
        transformed_importance.groupby("raw_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )

    return {
        "top_raw_features": raw_importance.head(top_n).to_dict(orient="records"),
        "top_transformed_features": transformed_importance.head(top_n).to_dict(orient="records"),
    }


def _extract_transformed_feature_names(preprocessor):
    transformed_names = []
    raw_feature_names = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue

        if isinstance(transformer, Pipeline) and "onehot" in transformer.named_steps:
            encoder = transformer.named_steps["onehot"]
            encoded_names = encoder.get_feature_names_out(columns)
            index = 0
            for column, categories in zip(columns, encoder.categories_):
                width = len(categories)
                for feature_name in encoded_names[index:index + width]:
                    transformed_names.append(str(feature_name))
                    raw_feature_names.append(str(column))
                index += width
        else:
            for column in columns:
                transformed_names.append(str(column))
                raw_feature_names.append(str(column))

    return transformed_names, raw_feature_names


def generate_bias_warnings(dataset_summary, results_before, results_after=None):
    warnings = []
    imbalance = dataset_summary["imbalance_detection"]

    if imbalance["group_imbalance_detected"]:
        warnings.append(
            f"Sensitive groups are imbalanced (smallest/largest group ratio={imbalance['group_ratio']:.2f}), which can amplify unfair outcomes."
        )

    if imbalance["target_imbalance_detected"]:
        warnings.append(
            f"Target classes are imbalanced (minority/majority ratio={imbalance['target_ratio']:.2f}); accuracy may look strong even when minority outcomes are weak."
        )

    threshold_labels = {
        "demographic_parity": "predicted positive rate gap",
        "true_positive_rate": "true positive rate gap",
        "false_positive_rate": "false positive rate gap",
    }
    for metric, label in threshold_labels.items():
        value = results_before["disparities"].get(metric)
        if value is not None and value > 0.10:
            warnings.append(f"High {label} detected before mitigation ({value:.3f}).")

    if results_after is not None:
        for metric, label in threshold_labels.items():
            before_value = results_before["disparities"].get(metric)
            after_value = results_after["disparities"].get(metric)
            if before_value is None or after_value is None:
                continue
            if after_value < before_value - 1e-6:
                warnings.append(f"Mitigation improved the {label} from {before_value:.3f} to {after_value:.3f}.")
            elif after_value > before_value + 1e-6:
                warnings.append(f"Mitigation worsened the {label} from {before_value:.3f} to {after_value:.3f}.")

        if results_after["accuracy"] < results_before["accuracy"] - 0.05:
            warnings.append(
                f"Mitigation reduced accuracy by more than 5 percentage points ({results_before['accuracy']:.3f} -> {results_after['accuracy']:.3f})."
            )

    if not warnings:
        warnings.append("No strong bias warning triggered with the current thresholds.")

    return warnings


def apply_mitigation(df, target_column, sensitive_column, target_mapping, method="reweighting"):
    if method not in {"reweighting", "resampling"}:
        raise ValueError("Mitigation method must be 'reweighting' or 'resampling'.")

    df = df.copy()
    target_binary = _map_target_series(df[target_column], target_mapping)

    if method == "reweighting":
        weighted_df = df.copy()
        weighted_df["_target_binary"] = target_binary
        counts = weighted_df.groupby([sensitive_column, "_target_binary"]).size().rename("count").reset_index()
        counts["weight"] = 1 / counts["count"]
        counts["weight"] = counts["weight"] / counts["weight"].sum() * len(weighted_df)
        weight_map = counts.set_index([sensitive_column, "_target_binary"])["weight"].to_dict()
        sample_weight = weighted_df.apply(lambda row: weight_map[(row[sensitive_column], row["_target_binary"])], axis=1).to_numpy()
        weighted_df = weighted_df.drop(columns=["_target_binary"])
        return weighted_df, sample_weight

    resampling_df = df.copy()
    resampling_df["_target_binary"] = target_binary
    grouped = resampling_df.groupby([sensitive_column, "_target_binary"], group_keys=False)
    max_size = grouped.size().max()
    resampled = (
        grouped.apply(lambda group: resample(group, replace=True, n_samples=max_size, random_state=42))
        .reset_index(drop=True)
        .drop(columns=["_target_binary"])
    )
    return resampled, None


def compare_results(before, after):
    print("\n=== Model Comparison ===")
    print(f"Accuracy before mitigation: {before['accuracy']:.4f}")
    print(f"Accuracy after mitigation:  {after['accuracy']:.4f}")
    print("\nDisparity differences:")
    for metric, before_value in before["disparities"].items():
        after_value = after["disparities"].get(metric)
        if before_value is None or after_value is None:
            continue
        print(f"  {metric}: before={before_value:.4f}, after={after_value:.4f}, change={after_value - before_value:+.4f}")

    print("\nGroup metrics before vs after:")
    groups = sorted(set(before["group_metrics"]) | set(after["group_metrics"]))
    for group in groups:
        before_metrics = before["group_metrics"].get(group)
        after_metrics = after["group_metrics"].get(group)
        print(f"\nGroup: {group}")
        if before_metrics:
            print(
                "  before "
                f"TPR={_fmt_metric(before_metrics['true_positive_rate'])}, "
                f"FPR={_fmt_metric(before_metrics['false_positive_rate'])}, "
                f"Predicted+={before_metrics['demographic_parity']:.4f}"
            )
        if after_metrics:
            print(
                "  after  "
                f"TPR={_fmt_metric(after_metrics['true_positive_rate'])}, "
                f"FPR={_fmt_metric(after_metrics['false_positive_rate'])}, "
                f"Predicted+={after_metrics['demographic_parity']:.4f}"
            )


def print_dataset_summary(dataset_summary):
    print("\n=== Dataset Summary ===")
    print(f"Rows: {dataset_summary['row_count']}")
    print("\nSensitive group distribution:")
    for entry in dataset_summary["group_distribution"]:
        print(
            f"  {entry['group']}: count={entry['count']}, share={entry['percentage']:.2%}, positive_rate={entry['positive_rate']:.4f}"
        )

    print("\nTarget distribution:")
    for entry in dataset_summary["target_distribution"]:
        print(f"  {entry['label']}: count={entry['count']}, share={entry['percentage']:.2%}")

    imbalance = dataset_summary["imbalance_detection"]
    print(
        "\nImbalance detection:"
        f" group_ratio={imbalance['group_ratio']:.3f},"
        f" target_ratio={imbalance['target_ratio']:.3f},"
        f" group_imbalance={imbalance['group_imbalance_detected']},"
        f" target_imbalance={imbalance['target_imbalance_detected']}"
    )


def print_results(title, results):
    print(f"\n=== {title} ===")
    print(f"Accuracy: {results['accuracy']:.4f}")
    for group, metrics in results["group_metrics"].items():
        print(
            f"Group {group}: "
            f"TPR={_fmt_metric(metrics['true_positive_rate'])}, "
            f"FPR={_fmt_metric(metrics['false_positive_rate'])}, "
            f"Predicted+={metrics['demographic_parity']:.4f}, "
            f"size={metrics['group_size']}, "
            f"confusion_matrix={metrics['confusion_matrix']}"
        )
    print(
        "Fairness gaps:"
        f" TPR difference={_fmt_metric(results['fairness_gap']['tpr_difference'])},"
        f" FPR difference={_fmt_metric(results['fairness_gap']['fpr_difference'])}"
    )
    print(f"All disparities: {results['disparities']}")


def print_feature_importance(feature_importance):
    print("\n=== Feature Importance (Basic) ===")
    for item in feature_importance["top_raw_features"]:
        print(f"  {item['raw_feature']}: {item['importance']:.4f}")


def print_warnings(warnings):
    print("\n=== Bias Warnings ===")
    for warning in warnings:
        print(f"  - {warning}")


def print_risk_summary(risk_summary):
    print("\n=== Bias Risk Summary ===")
    print(f"Risk level: {risk_summary['level']} ({risk_summary['score']}/100)")
    print(f"Trend: {risk_summary['trend']}")
    print(f"Largest remaining gap: {risk_summary['largest_remaining_gap_label']} = {risk_summary['largest_remaining_gap_value']:.4f}")
    print(f"Summary: {risk_summary['summary']}")


def print_list_section(title, items):
    print(f"\n=== {title} ===")
    for item in items:
        print(f"  - {item}")


def save_report(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    return report_path


def plot_group_outcomes(results, output_dir, filename):
    groups = list(results["group_metrics"].keys())
    values = [results["group_metrics"][group]["demographic_parity"] for group in groups]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(groups, values, color="#2A7FFF")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted positive rate")
    ax.set_title("Group Outcomes")
    fig.tight_layout()

    chart_path = output_dir / filename
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)
    return chart_path


def plot_fairness_comparison(before, after, output_dir, filename):
    labels = ["Demographic Parity", "TPR Gap", "FPR Gap"]
    before_values = [
        before["disparities"]["demographic_parity"],
        before["fairness_gap"]["tpr_difference"],
        before["fairness_gap"]["fpr_difference"],
    ]
    after_values = [
        after["disparities"]["demographic_parity"],
        after["fairness_gap"]["tpr_difference"],
        after["fairness_gap"]["fpr_difference"],
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, before_values, width, label="Before", color="#D96B2B")
    ax.bar(x + width / 2, after_values, width, label="After", color="#2D9D78")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Gap size")
    ax.set_title("Fairness Comparison")
    ax.legend()
    fig.tight_layout()

    chart_path = output_dir / filename
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)
    return chart_path


def plot_before_after_metrics(before, after, output_dir, filename):
    groups = sorted(set(before["group_metrics"]) | set(after["group_metrics"]))
    before_tpr = [before["group_metrics"].get(group, {}).get("true_positive_rate") or 0 for group in groups]
    after_tpr = [after["group_metrics"].get(group, {}).get("true_positive_rate") or 0 for group in groups]
    before_fpr = [before["group_metrics"].get(group, {}).get("false_positive_rate") or 0 for group in groups]
    after_fpr = [after["group_metrics"].get(group, {}).get("false_positive_rate") or 0 for group in groups]

    x = np.arange(len(groups))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * width, before_tpr, width, label="Before TPR", color="#1B6CA8")
    ax.bar(x - 0.5 * width, after_tpr, width, label="After TPR", color="#58A6FF")
    ax.bar(x + 0.5 * width, before_fpr, width, label="Before FPR", color="#A64545")
    ax.bar(x + 1.5 * width, after_fpr, width, label="After FPR", color="#E57373")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Before vs After by Group")
    ax.legend()
    fig.tight_layout()

    chart_path = output_dir / filename
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)
    return chart_path


def save_model_bundle(model, metadata, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "fairai_model.joblib"
    metadata_path = output_dir / "fairai_model_metadata.json"
    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")
    return model_path, metadata_path


def load_model_bundle(model_dir):
    model_dir = Path(model_dir)
    model_path = model_dir / "fairai_model.joblib"
    metadata_path = model_dir / "fairai_model_metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Expected fairai_model.joblib and fairai_model_metadata.json in the model directory.")

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def predict_with_loaded_model(model, metadata, predict_json=None, predict_csv=None):
    if bool(predict_json) == bool(predict_csv):
        raise ValueError("Provide exactly one of --predict-json or --predict-csv when using --load-model-dir.")

    if predict_json:
        raw_payload = json.loads(predict_json)
        records = raw_payload if isinstance(raw_payload, list) else [raw_payload]
        prediction_frame = pd.DataFrame(records)
    else:
        prediction_frame = pd.read_csv(predict_csv)

    expected_features = metadata["feature_columns"]
    missing = [column for column in expected_features if column not in prediction_frame.columns]
    if missing:
        raise ValueError(f"Prediction input is missing required feature columns: {missing}")

    prediction_frame = prediction_frame[expected_features]
    predictions = model.predict(prediction_frame)
    probabilities = model.predict_proba(prediction_frame)[:, 1] if hasattr(model, "predict_proba") else [None] * len(predictions)
    inverse_mapping = {int(value): key for key, value in metadata["target_mapping"].items()}

    outputs = []
    for index, (prediction, probability) in enumerate(zip(predictions, probabilities)):
        outputs.append(
            {
                "row_index": index,
                "predicted_label": inverse_mapping.get(int(prediction), int(prediction)),
                "predicted_binary": int(prediction),
                "positive_probability": float(probability) if probability is not None else None,
            }
        )
    return outputs


def generate_sample_dataset(csv_path):
    np.random.seed(42)
    n = 800
    genders = np.random.choice(["Female", "Male"], size=n, p=[0.55, 0.45])
    ages = np.random.choice(["<30", "30-50", ">50"], size=n, p=[0.35, 0.45, 0.20])
    income = np.random.normal(loc=70_000, scale=20_000, size=n).astype(int)
    education = np.random.choice(["High School", "Bachelor", "Master"], size=n, p=[0.25, 0.50, 0.25])
    base_score = (income / 10000) + np.where(education == "Master", 3, np.where(education == "Bachelor", 2, 1))
    bias_term = np.where(genders == "Male", 1.5, 0.0) + np.where(ages == "<30", -0.5, 0.0)
    probabilities = 1 / (1 + np.exp(-(base_score + bias_term - 6)))
    hired = (np.random.rand(n) < probabilities).astype(int)

    pd.DataFrame(
        {
            "gender": genders,
            "age_group": ages,
            "income": income,
            "education_level": education,
            "hired": hired,
        }
    ).to_csv(csv_path, index=False)
    print(f"Sample CSV data generated at: {csv_path}")


def run_analysis_workflow(
    df,
    target_column,
    sensitive_column,
    intersectional_columns=None,
    positive_label=None,
    negative_label=None,
    mitigation="reweighting",
    top_features=10,
    test_size=0.30,
    random_state=42,
    output_dir=None,
    save_model_dir=None,
):
    audit_df, target_configuration = prepare_target_for_audit(
        df,
        target_column,
        positive_label=positive_label,
        negative_label=negative_label,
    )
    protected_columns = []
    if intersectional_columns:
        protected_columns = []
        for column in intersectional_columns:
            if column != target_column and column not in protected_columns:
                protected_columns.append(column)
        if sensitive_column not in protected_columns and sensitive_column != target_column:
            protected_columns.insert(0, sensitive_column)
        sensitive_audit_column = "__fairai_intersectional_group"
        audit_df[sensitive_audit_column] = build_intersectional_group(audit_df, protected_columns)
        sensitive_display_name = " + ".join(protected_columns)
    else:
        protected_columns = [sensitive_column]
        sensitive_audit_column = sensitive_column
        sensitive_display_name = sensitive_column

    _, _, _, target_mapping = prepare_model_inputs(
        audit_df,
        target_column,
        sensitive_audit_column,
        excluded_feature_columns=protected_columns,
    )
    dataset_summary = analyze_dataset(audit_df, target_column, sensitive_audit_column, target_mapping)
    dataset_summary["sensitive_column"] = sensitive_display_name
    dataset_summary["sensitive_columns"] = protected_columns
    dataset_summary["audited_group_column"] = sensitive_audit_column

    y_binary = _map_target_series(audit_df[target_column], target_mapping)
    df_train, df_test = train_test_split(
        audit_df,
        test_size=test_size,
        random_state=random_state,
        stratify=y_binary,
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    X_train, y_train, _, _ = prepare_model_inputs(
        df_train,
        target_column,
        sensitive_audit_column,
        mapping=target_mapping,
        excluded_feature_columns=protected_columns,
    )
    X_test, y_test, sensitive_test, _ = prepare_model_inputs(
        df_test,
        target_column,
        sensitive_audit_column,
        mapping=target_mapping,
        excluded_feature_columns=protected_columns,
    )
    dataset_summary["feature_count"] = int(len(X_train.columns))

    model_before = train_model(X_train, y_train)
    results_before = evaluate_fairness(model_before, X_test, y_test, sensitive_test)

    mitigated_train_df, sample_weight = apply_mitigation(
        df_train,
        target_column,
        sensitive_audit_column,
        target_mapping=target_mapping,
        method=mitigation,
    )

    if sample_weight is None:
        X_mitigated_train, y_mitigated_train, _, _ = prepare_model_inputs(
            mitigated_train_df,
            target_column,
            sensitive_audit_column,
            mapping=target_mapping,
            excluded_feature_columns=protected_columns,
        )
        model_after = train_model(X_mitigated_train, y_mitigated_train)
    else:
        model_after = train_model(X_train, y_train, sample_weight=sample_weight)

    results_after = evaluate_fairness(model_after, X_test, y_test, sensitive_test)
    improvement_summary = summarize_fairness_change(results_before, results_after)
    threshold_sweep = sweep_thresholds(model_after, X_test, y_test, sensitive_test)
    feature_importance = extract_feature_importance(model_before, top_n=top_features)
    warnings = generate_bias_warnings(dataset_summary, results_before, results_after)
    risk_summary = build_risk_summary(dataset_summary, results_before, results_after)
    insights = generate_model_insights(dataset_summary, results_before, results_after, risk_summary, target_configuration)
    recommendations = generate_recommendations(
        dataset_summary,
        results_before,
        results_after,
        feature_importance,
        risk_summary,
        target_configuration,
    )
    chart_paths = {}
    report_path = None
    model_path = None
    metadata_path = None

    report = {
        "dataset_summary": dataset_summary,
        "target_mapping": target_mapping,
        "target_configuration": target_configuration,
        "split": {
            "train_rows": int(len(df_train)),
            "test_rows": int(len(df_test)),
            "test_size": float(test_size),
            "random_state": int(random_state),
        },
        "sensitive_columns": protected_columns,
        "audited_group_column": sensitive_audit_column,
        "mitigation_method": mitigation,
        "improvement_summary": improvement_summary,
        "threshold_sweep": threshold_sweep,
        "risk_summary": risk_summary,
        "insights": insights,
        "recommendations": recommendations,
        "before_mitigation": results_before,
        "after_mitigation": results_after,
        "feature_importance": feature_importance,
        "warnings": warnings,
        "charts": chart_paths,
    }

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        chart_paths = {
            "group_outcomes_before": str(plot_group_outcomes(results_before, output_path, "group_outcomes_before.png")),
            "group_outcomes_after": str(plot_group_outcomes(results_after, output_path, "group_outcomes_after.png")),
            "fairness_comparison": str(plot_fairness_comparison(results_before, results_after, output_path, "fairness_comparison.png")),
            "before_after_metrics": str(plot_before_after_metrics(results_before, results_after, output_path, "before_after_metrics.png")),
        }
        report["charts"] = chart_paths
        report_path = save_report(report, output_path)

    metadata = {
        "target_column": target_column,
        "sensitive_column": sensitive_display_name,
        "sensitive_columns": protected_columns,
        "audited_group_column": sensitive_audit_column,
        "feature_columns": X_train.columns.tolist(),
        "feature_profiles": summarize_feature_profiles(X_train),
        "target_mapping": target_mapping,
        "target_configuration": target_configuration,
        "mitigation_method": mitigation,
    }
    if save_model_dir is not None:
        model_path, metadata_path = save_model_bundle(model_after, metadata, Path(save_model_dir))

    return {
        "dataset_summary": dataset_summary,
        "target_mapping": target_mapping,
        "target_configuration": target_configuration,
        "report": report,
        "report_path": str(report_path) if report_path is not None else None,
        "chart_paths": chart_paths,
        "feature_importance": feature_importance,
        "warnings": warnings,
        "improvement_summary": improvement_summary,
        "threshold_sweep": threshold_sweep,
        "risk_summary": risk_summary,
        "insights": insights,
        "recommendations": recommendations,
        "results_before": results_before,
        "results_after": results_after,
        "model_before": model_before,
        "model_after": model_after,
        "metadata": metadata,
        "model_path": str(model_path) if model_path is not None else None,
        "metadata_path": str(metadata_path) if metadata_path is not None else None,
        "audit_df": audit_df,
        "train_df": df_train,
        "test_df": df_test,
        "train_features": X_train,
        "test_features": X_test,
        "test_labels": y_test,
        "test_sensitive": sensitive_test,
    }


def run_analysis(args):
    df = load_dataframe(args.csv, args.target, args.sensitive)
    intersectional_columns = None
    if args.intersectional:
        intersectional_columns = [column.strip() for column in args.intersectional.split(",") if column.strip()]
    workflow = run_analysis_workflow(
        df=df,
        target_column=args.target,
        sensitive_column=args.sensitive,
        intersectional_columns=intersectional_columns,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
        mitigation=args.mitigation,
        top_features=args.top_features,
        test_size=args.test_size,
        random_state=args.random_state,
        output_dir=args.output_dir,
        save_model_dir=args.save_model_dir,
    )
    print_dataset_summary(workflow["dataset_summary"])
    print_results("Before Mitigation", workflow["results_before"])
    print_results("After Mitigation", workflow["results_after"])
    compare_results(workflow["results_before"], workflow["results_after"])
    print_feature_importance(workflow["feature_importance"])
    print_warnings(workflow["warnings"])
    print_risk_summary(workflow["risk_summary"])
    print_list_section("Key Insights", workflow["insights"])
    print_list_section("Recommendations", workflow["recommendations"])

    if workflow["report_path"] is not None:
        print(f"\nSaved analysis report to: {workflow['report_path']}")
        print("Saved charts:")
        for label, chart_path in workflow["chart_paths"].items():
            print(f"  {label}: {chart_path}")

    if workflow["model_path"] is not None and workflow["metadata_path"] is not None:
        print(f"\nSaved mitigated model to: {workflow['model_path']}")
        print(f"Saved model metadata to: {workflow['metadata_path']}")


def _values_equal(left, right):
    if pd.isna(left) or pd.isna(right):
        return False
    return left == right or str(left) == str(right)


def _resolve_label_value(options, selected_label):
    for option in options:
        if _values_equal(option, selected_label):
            return option
    raise ValueError(f"Could not find target value '{selected_label}' in the dataset.")


def _encode_binary_target(y):
    unique_values = pd.Series(y.unique()).dropna().sort_values().tolist()
    if len(unique_values) != 2:
        raise ValueError("Target column must have exactly 2 distinct values.")
    mapping = {unique_values[0]: 0, unique_values[1]: 1}
    encoded = y.map(mapping)
    return encoded.astype(int), mapping


def _map_target_series(series, mapping):
    mapped = series.map(mapping)
    if mapped.isna().any():
        normalized_mapping = {str(key): value for key, value in mapping.items()}
        mapped = series.astype(str).map(normalized_mapping)
    return mapped.astype(int)


def _fmt_metric(value):
    return "N/A" if value is None else f"{value:.4f}"


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Bias detection, mitigation, visualization, and model persistence for binary classification.")
    parser.add_argument("--csv", type=str, help="Path to the input CSV dataset.")
    parser.add_argument("--target", type=str, help="Name of the target column.")
    parser.add_argument("--sensitive", type=str, help="Name of the sensitive attribute column.")
    parser.add_argument("--intersectional", type=str, help="Comma-separated protected columns to audit as combined groups.")
    parser.add_argument("--positive-label", type=str, help="Which target value should be treated as the positive outcome.")
    parser.add_argument("--negative-label", type=str, help="Optional name for the grouped negative outcome when binarizing a non-binary target.")
    parser.add_argument("--mitigation", choices=["reweighting", "resampling"], default="reweighting", help="Bias mitigation method to apply.")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Directory where reports and plots are saved.")
    parser.add_argument("--save-model-dir", type=str, help="Directory where the trained model bundle is saved.")
    parser.add_argument("--load-model-dir", type=str, help="Load a previously saved model bundle for prediction.")
    parser.add_argument("--predict-json", type=str, help="JSON object or JSON array containing prediction inputs.")
    parser.add_argument("--predict-csv", type=str, help="CSV file containing prediction inputs.")
    parser.add_argument("--top-features", type=int, default=10, help="How many features to show in the feature-importance summary.")
    parser.add_argument("--test-size", type=float, default=0.30, help="Test split size used for evaluation.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for train/test split and resampling.")
    parser.add_argument("--generate-sample", action="store_true", help="Generate a sample dataset file and exit.")
    parser.add_argument("--sample-output", type=str, default="sample_data.csv", help="Path where the sample dataset is saved.")
    args = parser.parse_args()

    if args.generate_sample:
        generate_sample_dataset(args.sample_output)
        return

    if args.load_model_dir:
        model, metadata = load_model_bundle(args.load_model_dir)
        predictions = predict_with_loaded_model(model, metadata, predict_json=args.predict_json, predict_csv=args.predict_csv)
        print(json.dumps({"predictions": predictions}, indent=2))
        return

    if not args.csv or not args.target or not args.sensitive:
        raise ValueError(
            "Provide --csv, --target, and --sensitive for analysis, use --generate-sample to create sample data, "
            "or use --load-model-dir with --predict-json/--predict-csv for prediction."
        )

    run_analysis(args)


if __name__ == "__main__":
    main()
