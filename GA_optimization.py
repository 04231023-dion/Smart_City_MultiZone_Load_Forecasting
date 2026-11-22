import json
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ==========================================================
# KONFIGURASI PATH FILE ARTEFAK MINGGU 13
# Sesuaikan DATA_DIR jika file .csv/.json disimpan di subfolder
# (misal: Path("artifacts/W13") atau Path("data/w13"))
# ==========================================================
DATA_DIR = Path(".")


@st.cache_data
def load_ga_artifacts(base_dir: Path):
    log_df = pd.read_csv(base_dir / "ga_log_konvergensi.csv")
    feat_df = pd.read_csv(base_dir / "ga_fitur_sebelum_sesudah.csv")
    comp_df = pd.read_csv(base_dir / "ga_perbandingan_model.csv")
    with open(base_dir / "ga_selected_features.json", "r", encoding="utf-8") as f:
        selected_json = json.load(f)
    img_path = base_dir / "ga_convergence_curve.png"
    return log_df, feat_df, comp_df, selected_json, img_path


def render_ga_page():
    st.title("🧬 Genetic Algorithm (GA) Optimization")
    st.caption(
        "Minggu 13 — Seleksi fitur otomatis menggunakan Genetic Algorithm "
        "untuk optimasi model RNN pada dataset Tetouan City Power Consumption."
    )

    try:
        log_df, feat_df, comp_df, selected_json, img_path = load_ga_artifacts(DATA_DIR)
    except FileNotFoundError as e:
        st.error(
            f"File artefak GA tidak ditemukan: `{e.filename}`. "
            "Pastikan file CSV dan JSON hasil GA berada di folder yang sesuai dengan `DATA_DIR`."
        )
        return

    # ----------------------------------------------------------
    # 1. RINGKASAN METRIK UTAMA
    # ----------------------------------------------------------
    best_seed = selected_json.get("best_seed", "-")
    selected_features = selected_json.get("selected_features", [])
    chromosome = selected_json.get("chromosome", [])
    total_features = len(chromosome) if chromosome else len(feat_df)

    # Ambil nilai perbandingan dari tabel
    baseline_row = comp_df.iloc[0]
    ga_row = comp_df.iloc[1]
    r2_delta = ga_row["Val R2"] - baseline_row["Val R2"]
    rmse_delta = ga_row["Val RMSE (kW)"] - baseline_row["Val RMSE (kW)"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Seed", str(best_seed))
    col2.metric("Fitur Terpilih", f"{len(selected_features)} / {total_features}")
    col3.metric("Val R² (GA+RNN)", f"{ga_row['Val R2']:.4f}", f"{r2_delta:+.4f}")
    col4.metric(
        "Val RMSE (GA+RNN)",
        f"{ga_row['Val RMSE (kW)']:.2f} kW",
        f"{rmse_delta:+.2f} kW",
        delta_color="inverse",
    )

    st.divider()

    # ----------------------------------------------------------
    # 2. FORMULASI FITNESS & REPRESENTASI KROMOSOM
    # ----------------------------------------------------------
    st.subheader("1. Fungsi Fitness & Kromosom Terbaik")
    st.markdown(
        """
        Setiap individu direpresentasikan sebagai **kromosom biner** sepanjang 18 bit
        (`1` = fitur dipilih, `0` = fitur dieliminasi). Fungsi *fitness* memaksimalkan
        performa validasi (meminimalkan *Validation MSE*) sekaligus memberi penalti
        terhadap jumlah fitur agar subset tetap ringkas tanpa menyentuh *test set*.
        """
    )
    st.latex(r"\text{Fitness} = \frac{1}{\text{Val MSE} + \alpha \cdot \frac{k}{N}}")
    st.caption(
        "Keterangan: $k$ = jumlah fitur terpilih, $N$ = total fitur kandidat (18), "
        "$\\alpha$ = bobot penalti kompleksitas fitur."
    )

    chrom_str = "".join(str(bit) for bit in chromosome)
    st.markdown(f"**Kromosom Terbaik (Seed {best_seed}):**")
    st.code(f"Biner : {chrom_str}\nArray : {chromosome}", language="text")

    st.divider()

    # ----------------------------------------------------------
    # 3. FITUR SEBELUM VS SESUDAH SELEKSI
    # ----------------------------------------------------------
    st.subheader("2. Fitur Sebelum vs Sesudah Seleksi GA")
    tab_selected, tab_all = st.tabs(["Daftar Fitur Terpilih (10)", "Tabel Lengkap (18 Fitur)"])

    with tab_selected:
        st.write(f"Sebanyak **{len(selected_features)} fitur** terpilih dari {total_features} kandidat:")
        cols = st.columns(2)
        for idx, feat in enumerate(selected_features):
            cols[idx % 2].markdown(f"- ✅ `{feat}`")

    with tab_all:
        status_filter = st.radio(
            "Filter Status Fitur:",
            options=["Semua", "Terpilih", "Dieliminasi"],
            horizontal=True,
        )
        filtered_feat = (
            feat_df if status_filter == "Semua" else feat_df[feat_df["Status"] == status_filter]
        )
        st.dataframe(filtered_feat, use_container_width=True, hide_index=True)

    st.divider()

    # ----------------------------------------------------------
    # 4. CONVERGENCE CURVE PER SEED
    # ----------------------------------------------------------
    st.subheader("3. Kurva Konvergensi Multi-Seed")
    seeds = sorted(log_df["Seed"].unique().tolist())
    selected_seeds = st.multiselect(
        "Pilih Seed untuk ditampilkan:",
        options=seeds,
        default=seeds,
    )

    if selected_seeds:
        fig, ax = plt.subplots(figsize=(9, 4))
        for s in selected_seeds:
            sub = log_df[log_df["Seed"] == s]
            ax.plot(
                sub["Generasi"],
                sub["Best_Fitness"],
                marker="o",
                linewidth=2,
                label=f"Seed {s} (Best)",
            )
            ax.plot(
                sub["Generasi"],
                sub["Avg_Fitness"],
                linestyle="--",
                alpha=0.5,
                label=f"Seed {s} (Avg)",
            )
        ax.set_title("GA Convergence Curve per Seed")
        ax.set_xlabel("Generasi")
        ax.set_ylabel("Fitness")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Pilih minimal satu seed untuk menampilkan grafik konvergensi.")

    with st.expander("Lihat Gambar Kurva Konvergensi Asli & Log Lengkap"):
        if img_path.exists():
            st.image(str(img_path), caption="ga_convergence_curve.png", use_container_width=True)
        st.dataframe(log_df, use_container_width=True, hide_index=True)

    st.divider()

    # ----------------------------------------------------------
    # 5. PERBANDINGAN BASELINE RNN VS GA + RNN
    # ----------------------------------------------------------
    st.subheader("4. Perbandingan Baseline RNN vs GA + RNN")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.info(
        "**Catatan Analisis:** Model Baseline RNN menggunakan 9 fitur awal, sedangkan "
        "GA + RNN memilih 10 fitur terbaik dari 18 fitur kandidat (termasuk fitur *lag* "
        "dan *cyclical time*). Hasilnya, Val RMSE turun signifikan dari **2990.07 kW** "
        "menjadi **1510.03 kW** dan Val $R^2$ meningkat dari **0.5422** menjadi **0.9189** "
        "dengan waktu komputasi yang sedikit lebih cepat (**73.58 detik** vs **76.78 detik**)."
    )


if __name__ == "__main__":
    st.set_page_config(page_title="GA Optimization - Kelompok 7", page_icon="🧬", layout="wide")
    render_ga_page()
Displaying ga_optimization.py.
