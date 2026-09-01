from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def resolve_report_paths(
    report: dict[str, dict],
    db_path: str | Path | None = None,
    table_name: str = "cv_photos2",
    image_dir: str | Path | None = None,
    dataset_df: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """
    Ensure all observation entries in report dictionary have 'paths' populated.
    Queries dataset_df or DuckDB table if paths are missing.
    """
    missing_obs_ids = set()
    for class_data in report.values():
        for category in ("fp", "fn"):
            for item in class_data.get(category, []):
                if not item.get("paths"):
                    missing_obs_ids.add(item["obs_id"])

    if not missing_obs_ids:
        return report

    path_map: dict[int, list[str]] = {}

    # Try resolving from dataset_df first if supplied
    if dataset_df is not None and not dataset_df.empty:
        idx_col = (
            "observation_id"
            if "observation_id" in dataset_df.columns
            else dataset_df.columns[0]
        )
        path_col = "path" if "path" in dataset_df.columns else None
        if path_col:
            df_filtered = dataset_df[dataset_df[idx_col].isin(missing_obs_ids)]
            for _, row in df_filtered.iterrows():
                obs_id = int(row[idx_col])
                paths_val = row[path_col]
                if isinstance(paths_val, str):
                    path_map[obs_id] = [paths_val]
                elif isinstance(paths_val, list):
                    path_map[obs_id] = paths_val

    # Query DuckDB for any remaining missing observation IDs
    remaining_obs = missing_obs_ids - set(path_map.keys())
    if remaining_obs and db_path and Path(db_path).exists():
        import duckdb

        with duckdb.connect(str(db_path)) as con:
            obs_tuple = tuple(remaining_obs)
            where_clause = (
                f"WHERE observation_id IN {obs_tuple}"
                if len(remaining_obs) > 1
                else f"WHERE observation_id = {list(remaining_obs)[0]}"
            )
            query = f"SELECT * FROM {table_name} {where_clause}"
            df_db = con.execute(query).fetch_df()

            if not df_db.empty:
                obs_col = (
                    "observation_id"
                    if "observation_id" in df_db.columns
                    else df_db.columns[0]
                )
                photo_col = (
                    "photo_id"
                    if "photo_id" in df_db.columns
                    else ("path" if "path" in df_db.columns else df_db.columns[1])
                )
                for obs_id, group in df_db.groupby(obs_col):
                    obs_id = int(obs_id)
                    photos = group[photo_col].tolist()
                    if image_dir and photo_col != "path":
                        photos = [
                            (
                                f"{str(image_dir).rstrip('/')}/{p}.jpg"
                                if not str(p).endswith(".jpg")
                                else f"{str(image_dir).rstrip('/')}/{p}"
                            )
                            for p in photos
                        ]
                    path_map[obs_id] = photos

    # Populate paths back into report
    for class_data in report.values():
        for category in ("fp", "fn"):
            for item in class_data.get(category, []):
                obs_id = item["obs_id"]
                if not item.get("paths") and obs_id in path_map:
                    item["paths"] = path_map[obs_id]

    return report


def plot_misclassified_observation(
    obs_entry: dict,
    label_name: str = "",
    error_type: str = "",
    max_cols: int = 4,
    save_path: str | Path | None = None,
):
    """
    Renders matplotlib figure showing observation photos
      with individual attention weights
    and a bar chart of the attention distribution across photos in the bag.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    obs_id = obs_entry.get("obs_id", "N/A")
    weights = obs_entry.get("weights", [])
    paths = obs_entry.get("paths", [])
    prob = obs_entry.get("prob", None)
    target = obs_entry.get("target", None)
    thresh = obs_entry.get("threshold", None)

    num_photos = max(len(weights), len(paths))
    if num_photos == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(
            0.5,
            0.5,
            f"No photos or weights for obs_id {obs_id}",
            ha="center",
            va="center",
        )
        ax.axis("off")
        return fig, ax

    cols = min(num_photos, max_cols)
    rows = (num_photos + cols - 1) // cols + 1  # extra row for attention bar plot

    fig = plt.figure(figsize=(cols * 3.5, rows * 3.5))

    title_str = (
        f"Obs ID: {obs_id} | Label: {label_name.upper()} | Type: {error_type.upper()}"
    )
    if prob is not None:
        title_str += f" | Pred Prob: {prob:.4f}"
    if thresh is not None:
        title_str += f" (Thresh: {thresh:.4f})"
    if target is not None:
        title_str += f" | Target: {target}"
    fig.suptitle(title_str, fontsize=12, fontweight="bold")

    # Plot photos
    for idx in range(num_photos):
        ax = fig.add_subplot(rows, cols, idx + 1)
        w = weights[idx] if idx < len(weights) else 0.0
        img_path = paths[idx] if idx < len(paths) else None

        if img_path and Path(img_path).exists():
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "Image Load Error", ha="center", va="center")
        else:
            ax.text(
                0.5,
                0.5,
                f"Photo {idx + 1}\n(No file path)",
                ha="center",
                va="center",
            )

        ax.set_title(f"Photo {idx + 1}\nWeight: {w:.3f} ({w * 100:.1f}%)", fontsize=10)
        ax.axis("off")

    # Plot attention weights bar chart spanning bottom row
    ax_bar = fig.add_subplot(rows, 1, rows)
    bars = ax_bar.bar([f"P{i + 1}" for i in range(len(weights))], weights, color="teal")
    ax_bar.set_ylabel("Attention Weight")
    ax_bar.set_ylim(0, max(1.0, max(weights, default=1.0) * 1.15))
    ax_bar.set_title("Bag Attention Weight Distribution", fontsize=10)

    for bar, w in zip(bars, weights):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{w:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, fig.axes


def review_misclassifications(
    report: dict | str | Path,
    db_path: str | Path | None = None,
    dataset_df: pd.DataFrame | None = None,
    image_dir: str | Path | None = None,
):
    """
    Launches an interactive ipywidgets GUI for
    Jupyter Notebooks to explore misclassifications.
    Falls back to returning resolved report if
    ipywidgets or display is unavailable.
    """
    if isinstance(report, (str, Path)):
        with open(report) as f:
            report_dict = json.load(f)
    else:
        report_dict = report

    # Resolve photo paths
    report_dict = resolve_report_paths(
        report_dict, db_path=db_path, image_dir=image_dir, dataset_df=dataset_df
    )

    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except ImportError:
        print(
            "ipywidgets or IPython display not available. "
            "Use plot_misclassified_observation(obs_entry, ...)"
        )
        return report_dict

    labels = list(report_dict.keys())
    if not labels:
        print("Report dictionary is empty.")
        return report_dict

    label_dropdown = widgets.Dropdown(
        options=labels, description="Class:", value=labels[0]
    )
    error_dropdown = widgets.Dropdown(
        options=[("False Positives (FP)", "fp"), ("False Negatives (FN)", "fn")],
        description="Error Type:",
        value="fp",
    )
    obs_slider = widgets.IntSlider(min=0, max=0, step=1, description="Obs Index:")
    out = widgets.Output()

    def update_obs_list(*args):
        lbl = label_dropdown.value
        err = error_dropdown.value
        items = report_dict.get(lbl, {}).get(err, [])
        if items:
            obs_slider.max = len(items) - 1
            obs_slider.value = 0
            obs_slider.disabled = False
        else:
            obs_slider.max = 0
            obs_slider.value = 0
            obs_slider.disabled = True
        render_figure()

    def render_figure(*args):
        with out:
            clear_output(wait=True)
            lbl = label_dropdown.value
            err = error_dropdown.value
            items = report_dict.get(lbl, {}).get(err, [])
            if not items:
                print(f"No {err.upper()} observations found for class '{lbl}'.")
                return
            idx = obs_slider.value
            if 0 <= idx < len(items):
                item = items[idx]
                import matplotlib.pyplot as plt

                fig, _ = plot_misclassified_observation(
                    item, label_name=lbl, error_type=err
                )
                display(fig)
                plt.close(fig)

    label_dropdown.observe(update_obs_list, names="value")
    error_dropdown.observe(update_obs_list, names="value")
    obs_slider.observe(render_figure, names="value")

    update_obs_list()

    controls = widgets.VBox(
        [widgets.HBox([label_dropdown, error_dropdown]), obs_slider, out]
    )
    display(controls)
    return report_dict
