
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import glob
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import json

from pandastable import Table  # pip install pandastable


# =========================
# Globals
# =========================
metadata_full = pd.DataFrame()
metadata_filtered = pd.DataFrame()

selected_file = None
selected_metadata = None

current_data = None
current_energy = None
current_angles = None
current_ei = None
current_errors = None

q_preview_df = pd.DataFrame()


# =========================
# Helpers: file scanning
# =========================
def build_metadata(folder_path: str) -> pd.DataFrame:
    records = []
    for file in glob.glob(os.path.join(folder_path, "*.nxspe")):
        name = os.path.basename(file).replace(".nxspe", "")
        if name.startswith("empty"):
            continue

        parts = name.split("_")
        if len(parts) < 4:
            continue

        try:
            sample = parts[0]
            temperature = float(parts[1])
            Ei = float(parts[2])
            scattering = parts[3]

            records.append(
                {
                    "filepath": file,
                    "sample": sample,
                    "temperature": temperature,
                    "Ei": Ei,
                    "scattering": scattering,
                }
            )
        except ValueError:
            continue

    return pd.DataFrame(records)


def validate_files(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    valid, missing = [], []
    if df.empty or "filepath" not in df.columns:
        return valid, missing

    for fp in df["filepath"].tolist():
        if os.path.exists(fp):
            valid.append(fp)
        else:
            missing.append(fp)
    return valid, missing


# =========================
# NXSPE load + plot
# =========================
def load_nxspe_data(file_path: str):
    with h5py.File(file_path, "r") as f:
        root_key = list(f.keys())[0]

        data = f[f"{root_key}/data/data"][()]
        energy = f[f"{root_key}/data/energy"][()]
        angles = f[f"{root_key}/data/polar"][()]
        ei = f[f"{root_key}/NXSPE_info/fixed_energy"][()]
        errors = f[f"{root_key}/data/error"][()]

    return data, energy, angles, ei, errors


def infer_q_axis_label() -> str:
    return "Q / Å⁻¹"


def plot_q_index(q_index: int):
    global current_data, current_energy, current_angles

    if current_data is None or current_energy is None or current_angles is None:
        messagebox.showerror("No dataset", "Select a file first.")
        return

    if q_index < 0 or q_index >= len(current_angles):
        messagebox.showerror("Invalid Q index", f"Q index {q_index} is out of range.")
        return

    q_value = current_angles[q_index]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(current_energy[:-1], current_data[q_index, :])
    ax.set_xlabel("ω / meV")
    ax.set_ylabel("S(ω, q)")
    ax.set_title(f"Q index {q_index} | Q = {q_value:.4f}")
    plt.show()


def plot_selected_preview_q():
    if q_preview_df.empty:
        messagebox.showerror("No Q preview", "No Q preview rows available yet.")
        return

    try:
        row = q_pt.getSelectedRow()
    except Exception:
        row = None

    if row is None or row < 0 or row >= len(q_preview_df):
        messagebox.showerror("No selection", "Select a row in the Q preview table first.")
        return

    q_index = int(q_preview_df.iloc[row]["q_index"])
    plot_q_index(q_index)


# =========================
# Path selection
# =========================
def set_path(entry_field):
    path = filedialog.askdirectory()
    if path:
        entry_field.delete(0, tk.END)
        entry_field.insert(0, path)


# =========================
# Filtering
# =========================
def refresh_filter_controls():
    global metadata_full

    if metadata_full.empty:
        return

    tmin, tmax = float(metadata_full["temperature"].min()), float(metadata_full["temperature"].max())
    emin, emax = float(metadata_full["Ei"].min()), float(metadata_full["Ei"].max())

    temp_min_scale.configure(from_=tmin, to=tmax)
    temp_max_scale.configure(from_=tmin, to=tmax)
    ei_min_scale.configure(from_=emin, to=emax)
    ei_max_scale.configure(from_=emin, to=emax)

    temp_min_var.set(tmin)
    temp_max_var.set(tmax)
    ei_min_var.set(emin)
    ei_max_var.set(emax)

    types = sorted(metadata_full["scattering"].unique().tolist())
    type_combo["values"] = ["All"] + types
    type_var.set("All")

    update_filter_labels()
    apply_filter()


def update_filter_labels():
    temp_label_var.set(f"Temp: {temp_min_var.get():.3g} → {temp_max_var.get():.3g}")
    ei_label_var.set(f"Ei: {ei_min_var.get():.3g} → {ei_max_var.get():.3g}")


def apply_filter(*_):
    global metadata_full, metadata_filtered, pt

    if metadata_full.empty:
        metadata_filtered = pd.DataFrame()
        pt.model.df = metadata_filtered
        pt.redraw()
        count_var.set("Matching files: 0")
        return

    t_lo, t_hi = sorted([float(temp_min_var.get()), float(temp_max_var.get())])
    e_lo, e_hi = sorted([float(ei_min_var.get()), float(ei_max_var.get())])
    scatt = type_var.get()

    filtered = metadata_full[
        metadata_full["temperature"].between(t_lo, t_hi)
        & metadata_full["Ei"].between(e_lo, e_hi)
    ].copy()

    if scatt != "All":
        filtered = filtered[filtered["scattering"] == scatt].copy()

    metadata_filtered = filtered

    pt.model.df = metadata_filtered
    pt.redraw()
    count_var.set(f"Matching files: {len(metadata_filtered)}")

    if not metadata_filtered.empty:
        _, missing = validate_files(metadata_filtered)
        if missing:
            print(f"Missing files: {len(missing)}")


def load_folder():
    global metadata_full

    folder_path = txt_path.get().strip()
    if not folder_path or not os.path.exists(folder_path):
        messagebox.showerror("Path error", "Folder path does not exist.")
        return

    metadata_full = build_metadata(folder_path)

    if metadata_full.empty:
        pt.model.df = pd.DataFrame()
        pt.redraw()
        count_var.set("Matching files: 0")
        messagebox.showinfo("No data", "No valid .nxspe files found.")
        return

    refresh_filter_controls()


# =========================
# File selection / dataset load
# =========================
def handle_row_click(event):
    global selected_file, selected_metadata
    global current_data, current_energy, current_angles, current_ei, current_errors

    try:
        row_clicked = pt.get_row_clicked(event)
        if row_clicked < 0:
            return

        if pt.model.df.empty:
            return

        selected_row = pt.model.df.iloc[row_clicked]
        file_path = selected_row["filepath"]

        if not os.path.exists(file_path):
            messagebox.showerror("Missing file", f"File not found:\n{file_path}")
            return

        data, energy, angles, ei, errors = load_nxspe_data(file_path)

        selected_file = file_path
        selected_metadata = selected_row.to_dict()

        current_data = data
        current_energy = energy
        current_angles = angles
        current_ei = ei
        current_errors = errors

        q_min = float(np.min(current_angles))
        q_max = float(np.max(current_angles))
        n_q = int(len(current_angles))

        dataset_label_var.set(
            f"Selected: {os.path.basename(selected_file)} | "
            f"Q range: {q_min:.4f} → {q_max:.4f} | "
            f"Q values: {n_q}"
        )

        qmin_var.set(q_min)
        qmax_var.set(q_max)
        qbins_var.set(min(10, n_q))
        q_index_var.set(max(0, n_q - 1))

        update_q_preview()

    except Exception as e:
        messagebox.showerror("Load failed", str(e))


# =========================
# Q preview / grouping
# =========================
def update_q_preview(*_):
    global q_preview_df

    if current_angles is None or len(current_angles) == 0:
        q_preview_df = pd.DataFrame()
        q_pt.model.df = q_preview_df
        q_pt.redraw()
        return

    q_lo, q_hi = sorted([float(qmin_var.get()), float(qmax_var.get())])
    n_bins = max(1, int(qbins_var.get()))

    q_values = np.asarray(current_angles)
    q_indices = np.arange(len(q_values))

    mask = (q_values >= q_lo) & (q_values <= q_hi)

    if not np.any(mask):
        q_preview_df = pd.DataFrame(columns=["q_index", "q_value", "included", "q_bin", "bin_lower", "bin_upper"])
        q_pt.model.df = q_preview_df
        q_pt.redraw()
        q_summary_var.set("No Q values in selected range.")
        return

    included_q = q_values[mask]
    included_idx = q_indices[mask]

    edges = np.linspace(float(included_q.min()), float(included_q.max()), n_bins + 1)
    bin_ids = np.digitize(included_q, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    rows = []
    for idx, qv, b in zip(included_idx, included_q, bin_ids):
        rows.append(
            {
                "q_index": int(idx),
                "q_value": float(qv),
                "included": True,
                "q_bin": int(b),
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
            }
        )

    excluded_q = q_values[~mask]
    excluded_idx = q_indices[~mask]
    for idx, qv in zip(excluded_idx, excluded_q):
        rows.append(
            {
                "q_index": int(idx),
                "q_value": float(qv),
                "included": False,
                "q_bin": None,
                "bin_lower": None,
                "bin_upper": None,
            }
        )

    q_preview_df = pd.DataFrame(rows).sort_values("q_index").reset_index(drop=True)
    q_pt.model.df = q_preview_df
    q_pt.redraw()

    q_summary_var.set(
        f"Included Q values: {len(included_q)} | Bins: {n_bins} | "
        f"Range: {included_q.min():.4f} → {included_q.max():.4f}"
    )


def plot_q_from_entry():
    try:
        q_index = int(q_index_var.get())
    except Exception:
        messagebox.showerror("Invalid input", "Q index must be an integer.")
        return
    plot_q_index(q_index)


# =========================
# Export
# =========================
def build_job_spec():
    if selected_file is None or selected_metadata is None:
        raise ValueError("No file selected.")

    if q_preview_df.empty:
        raise ValueError("No Q preview available.")

    included = q_preview_df[q_preview_df["included"] == True].copy()

    job = {
        "dataset": {
            "filepath": selected_file,
            "metadata": selected_metadata,
        },
        "q_selection": {
            "q_min": float(min(qmin_var.get(), qmax_var.get())),
            "q_max": float(max(qmin_var.get(), qmax_var.get())),
            "n_q_bins": int(qbins_var.get()),
            "resolved_assignments": included.to_dict(orient="records"),
        },
        "model": {
            "name": model_var.get(),
        },
    }
    return job


def export_job():
    try:
        job = build_job_spec()
    except Exception as e:
        messagebox.showerror("Export error", str(e))
        return

    save_path = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json")],
        title="Save analysis job spec",
    )

    if not save_path:
        return

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2)

    messagebox.showinfo("Saved", f"Job spec exported:\n{save_path}")


# =========================
# Scrollable layout
# =========================
root = tk.Tk()
root.title("QUENS GUI")
root.geometry("1100x850")

canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

def _on_mousewheel(event):
    try:
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    except Exception:
        pass

canvas.bind_all("<MouseWheel>", _on_mousewheel)


# =========================
# GUI
# =========================
lbl_title = tk.Label(scrollable_frame, text="QUENS data analysis system", font=("Arial", 14))
lbl_title.pack(pady=10)

# Path row
path_frame = tk.Frame(scrollable_frame)
path_frame.pack(fill="x", padx=10, pady=5)

txt_path = tk.Entry(path_frame, width=120)
txt_path.pack(side="left", padx=(0, 8))

btn_get_path = tk.Button(path_frame, text="Select folder", command=lambda: set_path(txt_path))
btn_get_path.pack(side="left", padx=(0, 8))

btn_run = tk.Button(path_frame, text="Load", command=load_folder, bg="white")
btn_run.pack(side="left")

# Filter controls
filter_frame = tk.LabelFrame(scrollable_frame, text="File filters", padx=10, pady=10)
filter_frame.pack(fill="x", padx=10, pady=5)

temp_min_var = tk.DoubleVar(value=0.0)
temp_max_var = tk.DoubleVar(value=0.0)
ei_min_var = tk.DoubleVar(value=0.0)
ei_max_var = tk.DoubleVar(value=0.0)

temp_label_var = tk.StringVar(value="Temp: -")
ei_label_var = tk.StringVar(value="Ei: -")

tk.Label(filter_frame, textvariable=temp_label_var).grid(row=0, column=0, sticky="w")
temp_min_scale = ttk.Scale(filter_frame, orient="horizontal", variable=temp_min_var, command=lambda _: (update_filter_labels(), apply_filter()))
temp_max_scale = ttk.Scale(filter_frame, orient="horizontal", variable=temp_max_var, command=lambda _: (update_filter_labels(), apply_filter()))
temp_min_scale.grid(row=1, column=0, sticky="ew", padx=(0, 10))
temp_max_scale.grid(row=2, column=0, sticky="ew", padx=(0, 10))

tk.Label(filter_frame, textvariable=ei_label_var).grid(row=0, column=1, sticky="w")
ei_min_scale = ttk.Scale(filter_frame, orient="horizontal", variable=ei_min_var, command=lambda _: (update_filter_labels(), apply_filter()))
ei_max_scale = ttk.Scale(filter_frame, orient="horizontal", variable=ei_max_var, command=lambda _: (update_filter_labels(), apply_filter()))
ei_min_scale.grid(row=1, column=1, sticky="ew")
ei_max_scale.grid(row=2, column=1, sticky="ew")

type_var = tk.StringVar(value="All")
tk.Label(filter_frame, text="Type:").grid(row=3, column=0, sticky="w", pady=(8, 0))
type_combo = ttk.Combobox(filter_frame, textvariable=type_var, state="readonly", width=20)
type_combo.grid(row=3, column=0, sticky="w", pady=(8, 0), padx=(50, 0))
type_combo.bind("<<ComboboxSelected>>", apply_filter)

count_var = tk.StringVar(value="Matching files: 0")
tk.Label(filter_frame, textvariable=count_var).grid(row=3, column=1, sticky="e", pady=(8, 0))

filter_frame.columnconfigure(0, weight=1)
filter_frame.columnconfigure(1, weight=1)

# File selection table
frame_table = tk.Frame(scrollable_frame, height=220)
frame_table.pack(fill="x", padx=10, pady=10)
frame_table.pack_propagate(False)

pt = Table(frame_table, dataframe=pd.DataFrame(), showtoolbar=True, showstatusbar=True)
pt.show()
pt.bind("<ButtonRelease-1>", handle_row_click)

# Dataset summary
dataset_label_var = tk.StringVar(value="No dataset selected")
tk.Label(scrollable_frame, textvariable=dataset_label_var, anchor="w").pack(fill="x", padx=10, pady=5)

# Q inspection controls
q_frame = tk.LabelFrame(scrollable_frame, text="Q inspection and binning", padx=10, pady=10)
q_frame.pack(fill="x", padx=10, pady=5)

qmin_var = tk.DoubleVar(value=0.0)
qmax_var = tk.DoubleVar(value=0.0)
qbins_var = tk.IntVar(value=10)
q_index_var = tk.IntVar(value=0)

tk.Label(q_frame, text="Q min").grid(row=0, column=0, sticky="w")
tk.Entry(q_frame, textvariable=qmin_var, width=10).grid(row=0, column=1, sticky="w", padx=(5, 15))

tk.Label(q_frame, text="Q max").grid(row=0, column=2, sticky="w")
tk.Entry(q_frame, textvariable=qmax_var, width=10).grid(row=0, column=3, sticky="w", padx=(5, 15))

tk.Label(q_frame, text="Number of Q bins").grid(row=0, column=4, sticky="w")
tk.Entry(q_frame, textvariable=qbins_var, width=8).grid(row=0, column=5, sticky="w", padx=(5, 15))

tk.Button(q_frame, text="Update Q preview", command=update_q_preview, bg="white").grid(row=0, column=6, sticky="w", padx=(10, 0))

tk.Label(q_frame, text="Plot raw Q index").grid(row=1, column=0, sticky="w", pady=(10, 0))
tk.Entry(q_frame, textvariable=q_index_var, width=10).grid(row=1, column=1, sticky="w", pady=(10, 0), padx=(5, 15))
tk.Button(q_frame, text="Plot selected Q", command=plot_q_from_entry, bg="white").grid(row=1, column=2, sticky="w", pady=(10, 0))

q_summary_var = tk.StringVar(value="No Q preview yet.")
tk.Label(q_frame, textvariable=q_summary_var).grid(row=2, column=0, columnspan=7, sticky="w", pady=(10, 0))

# Q preview table
q_preview_frame = tk.LabelFrame(scrollable_frame, text="Q preview table", padx=5, pady=5)
q_preview_frame.pack(fill="both", expand=True, padx=10, pady=10)

q_table_holder = tk.Frame(q_preview_frame, height=260)
q_table_holder.pack(fill="x")
q_table_holder.pack_propagate(False)

q_pt = Table(q_table_holder, dataframe=pd.DataFrame(), showtoolbar=True, showstatusbar=True)
q_pt.show()

q_buttons = tk.Frame(q_preview_frame)
q_buttons.pack(fill="x", pady=(8, 0))
tk.Button(q_buttons, text="Plot highlighted Q row", command=plot_selected_preview_q, bg="white").pack(side="left")

# Model selection
model_frame = tk.LabelFrame(scrollable_frame, text="Model selection", padx=10, pady=10)
model_frame.pack(fill="x", padx=10, pady=5)

model_var = tk.StringVar(value="Lorentzian")
ttk.Combobox(
    model_frame,
    textvariable=model_var,
    state="readonly",
    values=["Lorentzian", "Gaussian", "DHO"]
).pack(anchor="w")

# Export
export_frame = tk.Frame(scrollable_frame)
export_frame.pack(fill="x", padx=10, pady=20)

tk.Button(export_frame, text="Export analysis job", command=export_job, bg="white").pack(anchor="w")

root.mainloop()
