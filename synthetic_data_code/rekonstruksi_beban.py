"""
Rekonstruksi profil beban tahunan - Pulau Gili Ketapang.
Implementasi Tahap 1-5 sesuai metodologi (metodologi_rekonstruksi_beban.tex).
"""
import calendar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

try:
    from prophet import Prophet
except ImportError:  # pragma: no cover - dependency opsional
    Prophet = None

try:
    if not hasattr(np, "NaN"):
        np.NaN = np.nan  # shim: neuralprophet 0.8.0 masih memakai alias np.NaN
        # yang dihapus di NumPy>=2.0; lebih aman menambal alias di sini
        # daripada menurunkan versi NumPy untuk seluruh proyek.
    from neuralprophet import NeuralProphet
except ImportError:  # pragma: no cover - dependency opsional
    NeuralProphet = None

try:
    from statsmodels.tsa.statespace.structural import UnobservedComponents
except ImportError:  # pragma: no cover - dependency opsional
    UnobservedComponents = None

WBP_HOURS = list(range(17, 22))  # 17:00-21:59, definisi tarif PLN
LWBP_HOURS = [h for h in range(24) if h not in WBP_HOURS]

# Target kalibrasi bulanan (kW), diadaptasi dari Falfi (2024), Gambar IV.3
MONTHLY_TARGETS = {
    1: {"WBP": 358, "LWBP": 289}, 2: {"WBP": 351, "LWBP": 277},
    3: {"WBP": 345, "LWBP": 301}, 4: {"WBP": 369, "LWBP": 330},
    5: {"WBP": 389, "LWBP": 319}, 6: {"WBP": 378, "LWBP": 304},
    7: {"WBP": 357, "LWBP": 292}, 8: {"WBP": 364, "LWBP": 283},
    9: {"WBP": 371, "LWBP": 284}, 10: {"WBP": 397, "LWBP": 315},
    11: {"WBP": 394, "LWBP": 316}, 12: {"WBP": 402, "LWBP": 316},
}

# Beban puncak tahunan (kW), diadaptasi dari Falfi (2024), Gambar IV.2
ANNUAL_PEAK = {
    2014: 350, 2015: 381, 2016: 330, 2017: 319, 2018: 374,
    2019: 413, 2020: 454, 2021: 474, 2022: 446, 2023: 441,
}


# --------------------------------------------------------------------
# Pemuatan data beban riil Gili Ketapang (log genset + rekap harian PLN)
# --------------------------------------------------------------------
def _flag_outage_rows(df: pd.DataFrame) -> pd.Series:
    """Baris yang menyatakan genset padam: teks ('padam'/'off', kapitalisasi/
    spasi bervariasi) pada beban_total_kw, ATAU kolom mesin='OFF' (numerik 0
    yang sudah ada di tempat lain pada data ini). Dipakai bersama oleh
    load_genset_log() dan estimate_unmet_demand() supaya definisi "outage"
    konsisten di keduanya."""
    is_padam_text = df["beban_total_kw"].astype(str).str.strip().str.lower().isin(
        ["padam", "off"]
    )
    is_off_mesin = df["mesin"].astype(str).str.strip().str.upper() == "OFF"
    return is_padam_text | is_off_mesin


def load_genset_log(path: str, treat_outage_as: str = "zero") -> pd.DataFrame:
    """
    Membaca log genset mentah (kolom tanggal, mesin, daya_kw, beban_total_kw)
    dan mengembalikannya dalam struktur df_obs standar (index DatetimeIndex
    per jam, kolom 'load_kw') yang dikonsumsi extract_daily_shape(),
    reconstruct_annual_profile(), dan validate_mape().

    treat_outage_as:
    - "zero" (default, perilaku lama): jam padam (teks 'padam'/'off' pada
      beban_total_kw, ATAU mesin='OFF') dipetakan ke 0 kW -- diperlakukan
      sebagai permintaan nyata nol.
    - "missing": jam padam DIBUANG (seperti baris NaN), bukan dipetakan ke 0.
      **Catatan metodologis penting** (temuan review eksternal): memetakan
      padam ke 0 kW lalu memasukkannya ke perhitungan p(h,tau) (Tahap 1)
      berisiko bias bentuk kurva turun tepat di jam yang justru butuh
      kapasitas terbesar, jika padam terkonsentrasi di jam beban tinggi --
      sistem baru bisa under-sized dari data yang sudah "dijinakkan" oleh
      keterbatasan sistem lama. "missing" menghindari bias ini dengan
      mengecualikan jam-jam tsb dari p(h,tau) sama sekali; gunakan bersama
      estimate_unmet_demand() untuk mengukur energi yang hilang akibat
      pengecualian ini, alih-alih menyembunyikannya begitu saja.

    Empat pembersihan diterapkan pada data mentah:
    1. beban_total_kw dipaksa numerik; baris outage (lihat treat_outage_as)
       ditangani sesuai mode yang dipilih.
    2. Baris yang beban_total_kw-nya tetap tidak bisa diparse (benar-benar
       kosong/NaN) dibuang -- tidak bisa direkayasa, bukan bagian dari
       kontrak df_obs.
    3. Timestamp dibulatkan ke jam TERDEKAT (mis. '04:59:59' -> '05:00:00')
       karena logger mencatat dengan jitter -1 detik di sekitar jam bulat.
       Sebelum 2026-09-11 timestamp di-floor, sehingga bacaan '10:59:59' (milik
       jam 11:00) jatuh ke 10:00 dan dirata-rata dengan bacaan 10:00 yang asli:
       pada log riil 881 jam tercampur (selisih sampai 203 kW) dan hanya 2.624
       dari 3.528 jam yang tersisa. Dengan round tidak ada tabrakan, dan hasilnya
       identik dengan load_clean_2026.py pada studi DataScarceMicrogrid.
    4. Saat 2 mesin berjalan bersamaan, beban_total_kw dicatat identik pada
       kedua baris (timestamp terduplikasi) -- di-dedup dengan mean() per
       jam bulat (bukan first(), supaya bila ternyata nilainya berbeda,
       hasilnya tetap representatif alih-alih diam-diam membuang informasi).
    """
    if treat_outage_as not in ("zero", "missing"):
        raise ValueError(f"treat_outage_as harus 'zero' atau 'missing', dapat {treat_outage_as!r}")

    df = pd.read_csv(path, parse_dates=["tanggal"])
    is_outage = _flag_outage_rows(df)
    if treat_outage_as == "zero":
        df.loc[is_outage, "beban_total_kw"] = 0.0
    else:
        df.loc[is_outage, "beban_total_kw"] = np.nan
    df["beban_total_kw"] = pd.to_numeric(df["beban_total_kw"], errors="coerce")
    df = df.dropna(subset=["beban_total_kw"])
    df["jam_bulat"] = df["tanggal"].dt.round("h")
    load_kw = df.groupby("jam_bulat")["beban_total_kw"].mean()
    load_kw.index.name = None
    return load_kw.to_frame("load_kw")


def estimate_unmet_demand(path: str, shape: pd.DataFrame, targets: dict) -> pd.DataFrame:
    """
    Mengestimasi energi tak terlayani (unserved energy, kWh) pada jam-jam
    padam yang dikecualikan oleh load_genset_log(path, treat_outage_as="missing").
    Untuk tiap jam padam, dihitung beban yang SEHARUSNYA terjadi seandainya
    genset tidak padam -- via formula shape-times-target yang sama dengan
    Tahap 3 (Persamaan reconstruction), shape[h,tau] * target[window] /
    rata-rata shape pada window itu -- lalu selisihnya terhadap 0 (realisasi
    riil saat padam) adalah estimasi energi tak terlayani jam itu (1 jam
    interval, jadi kW == kWh).

    shape: hasil extract_daily_shape() (idealnya dari df_obs yang SUDAH
    dibersihkan dengan treat_outage_as="missing", supaya shape itu sendiri
    tidak ikut bias oleh jam padam).
    targets: dict bulanan {'WBP':..,'LWBP':..} (mis. dari
    build_monthly_targets_from_daily()).

    Return: DataFrame indeks=bulan, kolom 'unmet_energy_kwh', 'n_outage_hours'.
    """
    df = pd.read_csv(path, parse_dates=["tanggal"])
    is_outage = _flag_outage_rows(df)
    outage = df[is_outage].copy()
    outage["jam_bulat"] = outage["tanggal"].dt.round("h")  # sama dengan load_genset_log()
    outage = outage.drop_duplicates(subset="jam_bulat")

    p_bar = {
        "WBP": shape.loc[WBP_HOURS].mean(),
        "LWBP": shape.loc[LWBP_HOURS].mean(),
    }

    records = []
    for ts in outage["jam_bulat"]:
        tau = day_type(ts)
        if tau not in shape.columns:
            tau = "weekend" if ts.dayofweek >= 5 else "weekday"
        window = "WBP" if ts.hour in WBP_HOURS else "LWBP"
        target = targets[ts.month][window]
        predicted_kw = shape.loc[ts.hour, tau] * target / p_bar[window][tau]
        records.append({"month": ts.month, "predicted_kw": predicted_kw})

    if not records:
        return pd.DataFrame(columns=["unmet_energy_kwh", "n_outage_hours"])

    rec_df = pd.DataFrame(records)
    summary = rec_df.groupby("month").agg(
        unmet_energy_kwh=("predicted_kw", "sum"),
        n_outage_hours=("predicted_kw", "size"),
    )
    return summary


def build_monthly_targets_from_daily(path: str) -> dict:
    """
    Membaca rekap beban harian (kolom tanggal, lwbp_siang_kw, wbp_malam_kw)
    dan mengembalikan dict target bulanan berformat identik dengan
    MONTHLY_TARGETS ({bulan: {'WBP':.., 'LWBP':..}}), sehingga bisa langsung
    dipakai sebagai argumen `targets` pada reconstruct_annual_profile()
    atau validate_mape().

    Catatan jujur:
    - lwbp_siang_kw/wbp_malam_kw diperlakukan sebagai PROKSI rata-rata
      WBP/LWBP harian -- bukan rata-rata per-jam presisi seperti definisi
      WBP_HOURS/LWBP_HOURS yang dipakai di tempat lain pada modul ini.
    - Seluruh baris dipakai apa adanya untuk rata-rata bulanan, walau
      sebagian bulan awal cakupan data menunjukkan blok nilai yang
      berulang identik (indikasi kemungkinan placeholder) -- atas
      keputusan eksplisit bahwa data tidak difilter/di-drop.
    """
    df = pd.read_csv(path, parse_dates=["tanggal"])
    monthly = df.groupby(df["tanggal"].dt.month)[["wbp_malam_kw", "lwbp_siang_kw"]].mean()
    return {
        int(month): {"WBP": float(row["wbp_malam_kw"]), "LWBP": float(row["lwbp_siang_kw"])}
        for month, row in monthly.iterrows()
    }


def day_type(ts: pd.Timestamp, holidays: set | None = None) -> str:
    """
    tau(d) in {'weekday', 'weekend', 'holiday'}. 'holiday' takes precedence
    over weekend/weekday whenever ts's calendar date is in `holidays`
    (national public holidays + joint-leave/cuti bersama days -- see
    fetch_public_holidays() in akuisisi_data_terbuka.py). If holidays is
    None (default), behaves exactly as the original 2-category split.
    """
    if holidays and ts.date() in holidays:
        return "holiday"
    return "weekend" if ts.dayofweek >= 5 else "weekday"


# --------------------------------------------------------------------
# Tahap 1 - Ekstraksi bentuk kurva harian empiris (Pers. 1-2)
# --------------------------------------------------------------------
def extract_daily_shape(df_obs: pd.DataFrame, holidays: set | None = None) -> pd.DataFrame:
    """
    df_obs: DataFrame indeks datetime per jam, kolom 'load_kw', mencakup
    periode observasi M_obs (4 bulan data riil).
    holidays: opsional, set tanggal (datetime.date) hari libur nasional/cuti
    bersama -- lihat fetch_public_holidays(). Bila diberikan dan minimal satu
    hari libur jatuh di dalam M_obs, tau mendapat kategori ketiga 'holiday'.
    Bila tidak ada hari libur dalam M_obs (atau holidays=None), hasil identik
    dengan versi 2-kategori (weekday/weekend) -- lihat catatan fallback pada
    reconstruct_annual_profile().
    Return: DataFrame indeks=jam(0-23), kolom subset dari
    ['weekday','weekend','holiday'] berisi p(h,tau).
    """
    df = df_obs.copy()
    df["hour"] = df.index.hour
    df["date"] = df.index.date
    df["tau"] = df.index.map(lambda ts: day_type(ts, holidays))

    daily_mean = df.groupby("date")["load_kw"].transform("mean")
    df["ratio"] = df["load_kw"] / daily_mean  # L(h,d)/Lbar(d), Pers.(1)

    shape = (
        df.groupby(["hour", "tau"])["ratio"]
        .mean()  # p(h,tau), Pers.(2)
        .unstack("tau")
    )
    return shape


# --------------------------------------------------------------------
# Tahap 3 - Penskalaan konsisten WBP/LWBP (Pers. 3-4)
# --------------------------------------------------------------------
def reconstruct_annual_profile(
    df_obs: pd.DataFrame,
    shape: pd.DataFrame,
    targets: dict,
    year: int,
    holidays: set | None = None,
) -> pd.Series:
    """
    Membangun profil jam-an tahunan; bulan dalam M_obs diisi data riil,
    bulan lainnya direkonstruksi via Pers.(4).

    holidays: sama seperti pada extract_daily_shape(). Fallback penting:
    bila tanggal yang direkonstruksi adalah hari libur tetapi shape TIDAK
    memiliki kolom 'holiday' (karena tidak ada hari libur yang jatuh di
    dalam M_obs saat p(h,tau) diekstraksi), tau diturunkan ke 'weekend'
    sebagai pendekatan konservatif (bukan KeyError) -- didokumentasikan
    sebagai keterbatasan eksplisit pada metodologi (lihat subbagian
    "Kalender Hari Libur" pada dokumen).
    """
    idx = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    out = pd.Series(index=idx, dtype=float)
    obs_months = set(df_obs.index.to_period("M"))
    has_holiday_shape = "holiday" in shape.columns

    p_bar = {
        "WBP": shape.loc[WBP_HOURS].mean(),  # p_bar^WBP(tau), Pers.(3)
        "LWBP": shape.loc[LWBP_HOURS].mean(),  # p_bar^LWBP(tau), Pers.(3)
    }

    for ts in idx:
        if ts.to_period("M") in obs_months and ts in df_obs.index:
            out[ts] = df_obs.loc[ts, "load_kw"]
            continue

        tau = day_type(ts, holidays)
        if tau == "holiday" and not has_holiday_shape:
            tau = "weekend"  # fallback: lihat docstring
        window = "WBP" if ts.hour in WBP_HOURS else "LWBP"
        target = targets[ts.month][window]
        out[ts] = shape.loc[ts.hour, tau] * target / p_bar[window][tau]  # Pers.(4)

    return out


# --------------------------------------------------------------------
# Tahap 4 (pra) - Cek load factor: apakah g dari data PUNCAK representatif
# untuk memproyeksikan profil berbasis RATA-RATA WBP/LWBP? (temuan review
# eksternal: keduanya diam-diam diasumsikan tumbuh dengan laju sama)
# --------------------------------------------------------------------
def compute_load_factor_from_monthly_targets(
    monthly_targets: dict, peak: float, reference_year: int
) -> dict:
    """
    Menghitung load factor (puncak/rata-rata) dan rata-rata per-window
    (WBP/LWBP) dari MONTHLY_TARGETS-style dict (rata-rata bulanan konstan
    per jam dalam window, ditimbang jumlah hari tiap bulan pada
    reference_year -- untuk menangani tahun kabisat secara konsisten).

    peak: beban puncak tahun itu (mis. dari ANNUAL_PEAK[reference_year]).

    Return: {'peak', 'avg_overall', 'avg_wbp', 'avg_lwbp', 'load_factor'}.
    """
    total_wbp_wh = total_wbp_hrs = total_lwbp_wh = total_lwbp_hrs = 0.0
    for m, t in monthly_targets.items():
        ndays = calendar.monthrange(reference_year, m)[1]
        total_wbp_wh += t["WBP"] * len(WBP_HOURS) * ndays
        total_wbp_hrs += len(WBP_HOURS) * ndays
        total_lwbp_wh += t["LWBP"] * len(LWBP_HOURS) * ndays
        total_lwbp_hrs += len(LWBP_HOURS) * ndays

    avg_wbp = total_wbp_wh / total_wbp_hrs
    avg_lwbp = total_lwbp_wh / total_lwbp_hrs
    avg_overall = (total_wbp_wh + total_lwbp_wh) / (total_wbp_hrs + total_lwbp_hrs)
    return {
        "peak": float(peak),
        "avg_overall": avg_overall,
        "avg_wbp": avg_wbp,
        "avg_lwbp": avg_lwbp,
        "load_factor": float(peak) / avg_overall,
    }


def compute_load_factor_from_profile(annual: pd.Series) -> dict:
    """
    Sama seperti compute_load_factor_from_monthly_targets(), tapi dari
    profil jam-an yang sudah direkonstruksi penuh (mis. hasil
    reconstruct_annual_profile()) -- dipakai untuk periode yang datanya
    berupa profil, bukan MONTHLY_TARGETS konstan.
    """
    avg_wbp = float(annual[annual.index.hour.isin(WBP_HOURS)].mean())
    avg_lwbp = float(annual[annual.index.hour.isin(LWBP_HOURS)].mean())
    avg_overall = float(annual.mean())
    peak = float(annual.max())
    return {
        "peak": peak,
        "avg_overall": avg_overall,
        "avg_wbp": avg_wbp,
        "avg_lwbp": avg_lwbp,
        "load_factor": peak / avg_overall,
    }


def compare_load_factor_two_periods(period_old: dict, period_new: dict, n_years: float) -> dict:
    """
    Membandingkan 2 titik load factor (masing-masing hasil
    compute_load_factor_from_monthly_targets()/compute_load_factor_from_profile())
    untuk menguji apakah laju pertumbuhan beban PUNCAK (g dari
    compute_growth_rate(), Persamaan growth-rate) representatif untuk
    memproyeksikan level RATA-RATA WBP/LWBP yang dipakai Persamaan
    projection -- alih-alih diam-diam diasumsikan sama.

    Catatan jujur wajib: ini HANYA 2 titik waktu (bukan deret historis
    panjang), karena data rata-rata WBP/LWBP multi-tahun tidak tersedia --
    hasilnya indikatif, bukan estimasi tren yang solid secara statistik.
    Sebagian perbedaan yang teramati juga bisa berasal dari perbedaan
    metodologi pengukuran antar-sumber (tesis Falfi 2024 vs rekonstruksi
    2026 dari data genset+harian), bukan murni perubahan pola demand.

    Return: {'g_wbp', 'g_lwbp', 'load_factor_change_pct'} dari period_old
    ke period_new selama n_years tahun.
    """
    g_wbp = (period_new["avg_wbp"] / period_old["avg_wbp"]) ** (1 / n_years) - 1
    g_lwbp = (period_new["avg_lwbp"] / period_old["avg_lwbp"]) ** (1 / n_years) - 1
    lf_change_pct = (
        (period_new["load_factor"] - period_old["load_factor"]) / period_old["load_factor"]
    )
    return {"g_wbp": g_wbp, "g_lwbp": g_lwbp, "load_factor_change_pct": lf_change_pct}


# --------------------------------------------------------------------
# Tahap 4 - Proyeksi multi-tahun (Pers. 5-6)
# --------------------------------------------------------------------
def compute_growth_rate(peak_series: dict) -> float:
    """
    Konvensi n = selisih tahun (CAGR standar). Falfi (2024) melaporkan
    g=2,34%/tahun; nilai ini kemungkinan dihitung dengan n=10 (jumlah
    titik data) alih-alih n=9 (selisih tahun), sehingga hasil fungsi ini
    (~2,60%) berbeda tipis. Konfirmasi konvensi ke sumber asli sebelum
    dipakai pada laporan akhir.
    """
    years = sorted(peak_series)
    n = years[-1] - years[0]
    return (peak_series[years[-1]] / peak_series[years[0]]) ** (1 / n) - 1  # Pers.(5)


def project_growth(profile: pd.Series, growth_rate: float, n_years: int) -> pd.Series:
    return profile * (1 + growth_rate) ** n_years  # Pers.(6)


def bootstrap_confidence_interval(
    estimator_fn, n: int, n_boot: int = 2000, seed: int | None = None, ci: float = 0.95
) -> dict:
    """
    Bootstrap confidence interval generik (nonparametrik, resampling
    persentil -- bukan asumsi Normal) untuk parameter yang diestimasi dari
    n<=10 titik (temuan review eksternal: g, alpha/beta VIIRS, dst
    dilaporkan 2 desimal tanpa CI, memberi kesan presisi yang tidak
    didukung ukuran sampel sekecil itu).

    estimator_fn(idx: np.ndarray) -> float: dipanggil dengan array indeks
    hasil resampling DENGAN PENGEMBALIAN dari range(n) (ukuran n). Pemanggil
    menutup (closure) atas data aslinya sendiri dan menentukan cara
    resampling yang sesuai secara statistik untuk kasusnya -- mis. resample
    indeks TAHUN untuk laju pertumbuhan year-over-year (compute_yoy_growth_rates()),
    atau resample RESIDUAL (bukan titik data mentah) untuk bootstrap
    regresi OLS (residual bootstrap standar: y_boot = fitted + residual[idx]),
    supaya struktur x tetap dan cuma ketidakpastian noise yang di-resample.
    estimator_fn(np.arange(n)) (tanpa resampling) dipakai sebagai titik
    estimasi asli.

    Return: {'point_estimate', 'ci_low', 'ci_high', 'boot_std'}. CI persentil
    (ci=0.95 -> persentil ke-2.5 dan ke-97.5 dari n_boot estimasi bootstrap),
    bukan CI Normal-asimtotik -- lebih sesuai untuk n sekecil ini.
    """
    rng = np.random.default_rng(seed)
    point_estimate = estimator_fn(np.arange(n))

    boot_estimates = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_estimates[b] = estimator_fn(idx)

    alpha = (1 - ci) / 2
    ci_low, ci_high = np.percentile(boot_estimates, [alpha * 100, (1 - alpha) * 100])
    return {
        "point_estimate": float(point_estimate),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "boot_std": float(boot_estimates.std(ddof=1)),
    }


# --------------------------------------------------------------------
# Tahap 4b - Proyeksi probabilistik (skenario pertumbuhan)
# Mengadaptasi prinsip pohon-skenario Fioriti et al. (2021) untuk kasus
# satu situs (single-site): skenario dikalibrasi dari distribusi laju
# pertumbuhan year-over-year historis milik lokasi itu sendiri, bukan
# dari basis data multi-situs eksternal (yang tidak tersedia untuk
# Gili Ketapang). Lihat metodologi_rekonstruksi_beban.tex, subbagian
# "Proyeksi Probabilistik" pada Tahap 4.
# --------------------------------------------------------------------
def compute_yoy_growth_rates(peak_series: dict) -> np.ndarray:
    """Laju pertumbuhan year-over-year historis, g_y = P_y/P_{y-1} - 1."""
    years = sorted(peak_series)
    values = np.array([peak_series[y] for y in years], dtype=float)
    return values[1:] / values[:-1] - 1


def compute_growth_scenarios(peak_series: dict) -> dict:
    """
    Diskritisasi 3-titik (Low/Mid/High) yang mempertahankan rata-rata
    (mean-preserving) dari distribusi Normal yang di-fit pada laju
    pertumbuhan year-over-year historis. Setiap skenario punya
    probabilitas TEPAT 1/3 secara konstruksi (batas tersil distribusi
    Normal-standar), dengan nilai representatif = rata-rata bersyarat
    (conditional mean) Normal terpotong (truncated) pada tersil
    tersebut -- rumus baku teori peluang, bukan pilihan sembarang:

        E[Z | a < Z < b] = (phi(a) - phi(b)) / (Phi(b) - Phi(a))

    dengan phi = pdf Normal-standar, Phi = cdf Normal-standar.

    Return: {'low': {'g':..., 'prob': 1/3}, 'mid': {...}, 'high': {...}}
    Sifat: rata-rata tertimbang probabilitas dari g_low,g_mid,g_high
    sama persis dengan mu (rata-rata historis) -- lihat unit test.
    """
    g_yoy = compute_yoy_growth_rates(peak_series)
    mu, sigma = g_yoy.mean(), g_yoy.std(ddof=1)

    z1 = stats.norm.ppf(1 / 3)
    z2 = stats.norm.ppf(2 / 3)

    def truncated_mean(a, b):
        Fa = stats.norm.cdf(a) if a is not None else 0.0
        Fb = stats.norm.cdf(b) if b is not None else 1.0
        pa = stats.norm.pdf(a) if a is not None else 0.0
        pb = stats.norm.pdf(b) if b is not None else 0.0
        return (pa - pb) / (Fb - Fa)

    e_low = truncated_mean(None, z1)
    e_mid = truncated_mean(z1, z2)
    e_high = truncated_mean(z2, None)

    return {
        "low": {"g": mu + sigma * e_low, "prob": 1 / 3},
        "mid": {"g": mu + sigma * e_mid, "prob": 1 / 3},
        "high": {"g": mu + sigma * e_high, "prob": 1 / 3},
    }


def project_growth_scenarios(
    profile: pd.Series, scenarios: dict, n_years: int
) -> dict:
    """Terapkan project_growth() untuk setiap skenario. Return dict {nama: profil}."""
    return {
        name: project_growth(profile, s["g"], n_years)
        for name, s in scenarios.items()
    }


# --------------------------------------------------------------------
# Tahap 4c - Perbandingan dengan metode proyeksi alternatif dari literatur
# (metode diusulkan <=15 tahun terakhir; lihat subbagian "Comparison with
# Alternative Projection Methods from the Literature" pada dokumen). Semua
# fungsi di bawah dipakai HANYA sebagai perbandingan/validasi silang terhadap
# CAGR/skenario probabilistik di atas -- tidak menggantikan Persamaan (5)-(9).
# --------------------------------------------------------------------
def fit_linear_growth(peak_series: dict) -> dict:
    """
    Regresi least-squares P_y = m*(y - y0) + c pada data beban puncak tahunan.
    Pembanding paling sederhana terhadap CAGR (compound/exponential) -- matematika
    identik dengan fit_linear_trend() di akuisisi_data_terbuka.py (dipakai untuk
    tren radiance VIIRS), diterapkan ulang di sini untuk beban puncak.
    Return: {'m':..., 'c':..., 'r2':..., 'first_year':...}
    """
    years = sorted(peak_series)
    y0 = years[0]
    x = np.array([y - y0 for y in years], dtype=float)
    values = np.array([peak_series[y] for y in years], dtype=float)

    n = len(x)
    mean_x, mean_v = x.mean(), values.mean()
    cov = np.sum((x - mean_x) * (values - mean_v))
    var_x = np.sum((x - mean_x) ** 2)
    m = cov / var_x
    c = mean_v - m * mean_x

    predicted = m * x + c
    ss_res = np.sum((values - predicted) ** 2)
    ss_tot = np.sum((values - mean_v) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {"m": m, "c": c, "r2": r2, "first_year": y0}


def project_linear_growth(fit: dict, target_year: int) -> float:
    """Proyeksi P(target_year) dari hasil fit_linear_growth()."""
    return fit["m"] * (target_year - fit["first_year"]) + fit["c"]


def estimate_saturation_capacity(
    building_count: int = 2356, per_household_kw_range: tuple = (0.3, 0.6)
) -> tuple:
    """
    Estimasi kasar batas atas (carrying capacity K) beban puncak Gili Ketapang,
    dari jumlah bangunan OpenStreetMap (Bagian OSM dokumen, 2.356 bangunan)
    dikalikan asumsi beban puncak per rumah tangga -- ASUMSI EKSPLISIT, bukan
    hasil kalibrasi, dipakai sebagai input opsional 'cap' untuk
    fit_prophet_growth(growth='logistic').
    Return: (K_low, K_high) dalam kW.
    """
    low, high = per_household_kw_range
    return (building_count * low, building_count * high)


def fit_prophet_growth(
    peak_series: dict, growth: str = "linear", cap: float | None = None
) -> dict:
    """
    Prophet (Taylor & Letham, 2018) diterapkan pada beban puncak tahunan sebagai
    pembanding CAGR: mode growth='linear' (tren piecewise + changepoint otomatis)
    atau growth='logistic' (saturasi ke kapasitas 'cap', lihat
    estimate_saturation_capacity()). Seasonality mingguan/harian/tahunan
    dimatikan -- data kita tahunan, bukan observasi sub-tahunan, sehingga
    seasonality bawaan Prophet tidak relevan; yang dipakai murni komponen tren.
    Return: {'model': Prophet, 'df': DataFrame pelatihan, 'growth': str}
    """
    if Prophet is None:
        raise ImportError("Fungsi ini membutuhkan prophet. Jalankan: pip install prophet")
    if growth == "logistic" and cap is None:
        raise ValueError("growth='logistic' membutuhkan argumen cap (lihat estimate_saturation_capacity()).")

    years = sorted(peak_series)
    df = pd.DataFrame({
        "ds": pd.to_datetime([f"{y}-01-01" for y in years]),
        "y": [peak_series[y] for y in years],
    })
    if growth == "logistic":
        df["cap"] = cap

    model = Prophet(
        growth=growth,
        weekly_seasonality=False,
        daily_seasonality=False,
        yearly_seasonality=False,
    )
    model.fit(df)
    return {"model": model, "df": df, "growth": growth, "cap": cap}


def project_prophet_growth(fit: dict, target_year: int) -> float:
    """Proyeksi P(target_year) dari hasil fit_prophet_growth()."""
    future = pd.DataFrame({"ds": [pd.Timestamp(f"{target_year}-01-01")]})
    if fit["growth"] == "logistic":
        future["cap"] = fit["cap"]
    forecast = fit["model"].predict(future)
    return float(forecast["yhat"].iloc[0])


def fit_neuralprophet_growth(peak_series: dict) -> dict:
    """
    NeuralProphet (Triebe et al., 2021 -- preprint arXiv:2111.15397, belum
    peer-reviewed pada saat dokumen ini ditulis) diterapkan dengan n_lags=0:
    modul auto-regresi/covariate berbasis neural network (AR-Net) DIMATIKAN
    karena 10 titik data tahunan jauh di bawah kebutuhan minimal untuk komponen
    tersebut belajar bermakna -- yang dipakai murni komponen tren piecewise-nya,
    secara konsep setara Prophet-linear tapi dioptimasi via PyTorch/gradient
    descent, bukan Stan.
    Return: {'model': NeuralProphet, 'df': DataFrame pelatihan}
    """
    if NeuralProphet is None:
        raise ImportError("Fungsi ini membutuhkan neuralprophet. Jalankan: pip install neuralprophet")

    years = sorted(peak_series)
    first_year = years[0]
    # Titik waktu dispasikan tepat 365 hari (bukan tanggal kalender 1 Januari
    # asli) agar NeuralProphet dapat menyimpulkan frekuensi tetap ('365D');
    # tahun kalender asli (365/366 hari, kabisat) membuat inferensi frekuensi
    # internal paket ini gagal (ValueError "Invalid frequency: NaT" -- bug
    # library saat menulis kode ini). Pergeseran akibat hari kabisat (<=2 hari
    # per dekade) diabaikan -- tidak berpengaruh pada tren pertumbuhan tahunan.
    epoch = pd.Timestamp(f"{first_year}-01-01")
    df = pd.DataFrame({
        "ds": [epoch + pd.Timedelta(days=365 * (y - first_year)) for y in years],
        "y": [peak_series[y] for y in years],
    })
    model = NeuralProphet(
        n_lags=0,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        learning_rate=0.1,  # nilai eksplisit -- lewati auto learning-rate-finder
        # (fitur tersebut tidak kompatibel dengan versi PyTorch/checkpoint terbaru)
    )
    model.fit(df, freq="365D")
    return {"model": model, "df": df, "first_year": first_year, "epoch": epoch}


def project_neuralprophet_growth(fit: dict, target_year: int) -> float:
    """
    Proyeksi P(target_year) dari hasil fit_neuralprophet_growth(), via
    make_future_dataframe() bawaan NeuralProphet (bekerja andal di sini karena
    frekuensi pelatihan sudah memakai jarak tetap 365 hari, lihat catatan pada
    fit_neuralprophet_growth()).
    """
    periods = target_year - (fit["first_year"] + len(fit["df"]) - 1)
    future = fit["model"].make_future_dataframe(
        fit["df"], periods=periods, n_historic_predictions=False
    )
    forecast = fit["model"].predict(future)
    return float(forecast["yhat1"].iloc[-1])


def fit_bsts_local_trend(peak_series: dict, exog: pd.Series | None = None) -> dict:
    """
    Model local-linear-trend state-space (Kalman filter, estimasi maximum-
    likelihood via statsmodels.UnobservedComponents), dipakai sebagai pembanding
    dalam semangat Bayesian Structural Time Series (Scott & Varian, 2014).
    CATATAN JUJUR: kontribusi asli Scott & Varian memakai MCMC penuh + prior
    spike-and-slab untuk seleksi regressor; implementasi ini adalah
    penyederhanaan yang wajar (MLE/Kalman filter, bukan MCMC) -- didokumentasikan
    eksplisit, bukan diklaim sebagai replikasi penuh.
    exog: opsional, mis. radiance VIIRS R_sum(y) (Tabel viirs dokumen) sebagai
    regressor eksogen -- fitur inti BSTS asli (regresi pada predictor eksternal).
    Return: {'results': ..., 'years': [...], 'exog_used': bool}
    """
    if UnobservedComponents is None:
        raise ImportError("Fungsi ini membutuhkan statsmodels. Jalankan: pip install statsmodels")

    years = sorted(peak_series)
    endog = pd.Series(
        [peak_series[y] for y in years],
        index=pd.PeriodIndex([str(y) for y in years], freq="Y"),
    )
    exog_aligned = None
    if exog is not None:
        exog_aligned = exog.reindex(years).to_numpy().reshape(-1, 1)

    model = UnobservedComponents(endog, level="local linear trend", exog=exog_aligned)
    results = model.fit(disp=False)
    return {
        "results": results,
        "years": years,
        "exog_used": exog is not None,
        "exog_last": None if exog is None else exog.reindex(years).to_numpy()[-1],
    }


def project_bsts_local_trend(fit: dict, n_years: int) -> float:
    """
    Proyeksi n_years ke depan dari titik data terakhir. Bila model dilatih
    dengan exog, nilai exog periode proyeksi diasumsikan sama dengan nilai
    terakhir yang teramati (naive hold-last-value) -- disebutkan eksplisit
    sebagai keterbatasan, bukan diasumsikan tanpa catatan.
    """
    exog_future = None
    if fit["exog_used"]:
        exog_future = np.full((n_years, 1), fit["exog_last"])
    forecast = fit["results"].get_forecast(steps=n_years, exog=exog_future)
    return float(forecast.predicted_mean.iloc[-1])


def fit_gm11(peak_series: dict) -> dict:
    """
    Grey Model GM(1,1) (Deng, 1982), formula tertutup persis Hu (2017)
    Eq.~1-7 -- DIHITUNG sebagai pembanding empiris di tabel metode-proyeksi
    (bukan cuma dikutip naratif), meski usianya >15 tahun (Section
    "Scope"), karena review eksternal menunjukkan justru metode inilah yang
    paling relevan untuk n=10 (dirancang khusus untuk seri sesingkat 4 titik).

    Langkah: AGO (Accumulated Generating Operation) x1_k = sum_{j<=k} x0_j,
    lalu selesaikan a,b dari persamaan beda hijau x0_k + a*z1_k = b (z1_k =
    rata-rata bergerak x1_k dan x1_{k-1}) via least-squares, lalu formula
    tertutup x1_hat_k = (x0_1 - b/a)*exp(-a*(k-1)) + b/a.

    Return: {'a', 'b', 'x0_1', 'first_year', 'r2'} (r2 dihitung pada nilai
    asli x0 hasil IAGO/first-difference dari x1_hat, bukan pada x1 kumulatif).
    """
    years = sorted(peak_series)
    first_year = years[0]
    x0 = np.array([peak_series[y] for y in years], dtype=float)
    n = len(x0)

    x1 = np.cumsum(x0)
    z1 = 0.5 * (x1[1:] + x1[:-1])  # background value, k=2..n

    # Grey difference equation: x0_k = -a*z1_k + b, k=2..n -> least squares [-z1, 1] @ [a,b] = x0
    B = np.column_stack([-z1, np.ones(n - 1)])
    Y = x0[1:]
    a, b = np.linalg.lstsq(B, Y, rcond=None)[0]

    x1_hat = (x0[0] - b / a) * np.exp(-a * np.arange(n)) + b / a
    x0_hat = np.empty(n)
    x0_hat[0] = x0[0]
    x0_hat[1:] = x1_hat[1:] - x1_hat[:-1]
    r2 = _r_squared_local(x0, x0_hat)

    return {"a": float(a), "b": float(b), "x0_1": float(x0[0]), "first_year": first_year, "r2": r2}


def project_gm11(fit: dict, target_year: int) -> float:
    """Ekstrapolasi GM(1,1) ke target_year via formula tertutup Hu (2017) Eq.~7."""
    k = target_year - fit["first_year"]  # x0_1 berindeks k=1 (tahun pertama)
    a, b, x0_1 = fit["a"], fit["b"], fit["x0_1"]
    x1_hat_k = (x0_1 - b / a) * np.exp(-a * k) + b / a
    x1_hat_km1 = (x0_1 - b / a) * np.exp(-a * (k - 1)) + b / a
    return float(x1_hat_k - x1_hat_km1)


def _r_squared_local(observed: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - observed.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def fit_nngm11(
    peak_series: dict,
    eta: float = 0.01,
    epochs: int = 5000,
    seed: int | None = None,
    max_step_a: float = 1e-4,
    max_step_b: float = 0.05,
) -> dict:
    """
    NNGM(1,1) (Hu et al., 2001; direplikasi dari Hu (2017) Eq.~8-14). Single-
    layer perceptron 1-neuron dengan bobot a,b (sama peran dengan GM(1,1)),
    dilatih via gradient descent pada cost function E(a,b) =
    0.5*sum(x0_k - x0_hat_k)^2 alih-alih least-squares pada background
    value z1_k -- menghindari ketergantungan pada z1_k yang menurut Hu et
    al. (2001) "tidak mudah ditentukan" secara prinsip. Untuk tiap epoch,
    pilih SATU sampel (k,1,1) k=2..n secara acak, hitung x0_hat_k via Eq.~7,
    lalu update a,b via Eq.~9-14 (delta_a, delta_b, dengan turunan V_ak, V_bk).

    Dua stabilisasi numerik DITAMBAHKAN secara eksplisit (bukan di paper
    asli, TIDAK disembunyikan sebagai replikasi murni):
    1. Inisialisasi a,b dari solusi GM(1,1) (fit_gm11()) +- 20% acak,
       bukan rentang lebar sembarang. Formula Eq.~7 melibatkan b/a dan
       1/a^2 -- inisialisasi acak lebar (mis. a~U(0.01,0.5) independen
       dari skala b) membuat b/a meledak dan gradient descent divergen
       jadi NaN dalam <20 epoch pada data ini (diverifikasi langsung).
       Solusi GM(1,1) menjamin b/a berada di skala yang wajar.
    2. Gradient clipping (max_step_a, max_step_b) pada tiap update --
       tanpanya, turunan V_ak/V_bk (yang memuat suku 1/a^2) sesekali
       menghasilkan langkah ekstrem meski inisialisasi sudah wajar.
    Berdasarkan data ANNUAL_PEAK, kombinasi ini stabil dan menghasilkan
    R^2=0.621 (vs GM(1,1) R^2=0.674) -- tanpa stabilisasi ini, pelatihan
    meledak jadi NaN pada data yang sama.

    Return: {'a', 'b', 'x0_1', 'first_year', 'r2'} -- format sama dengan
    fit_gm11() supaya project_nngm11() dan project_gm11() identik strukturnya.
    """
    years = sorted(peak_series)
    first_year = years[0]
    x0 = np.array([peak_series[y] for y in years], dtype=float)
    n = len(x0)

    gm_init = fit_gm11(peak_series)
    rng = np.random.default_rng(seed)
    a = gm_init["a"] * (1 + rng.uniform(-0.2, 0.2))
    b = gm_init["b"] * (1 + rng.uniform(-0.2, 0.2))

    def predict(k, a, b):
        # x0_hat_k = (1-e^a)(x0_1 - b/a) e^{-a(k-1)}, k>=2 (Hu 2017 Eq. 7)
        return (1 - np.exp(a)) * (x0[0] - b / a) * np.exp(-a * (k - 1))

    for _ in range(epochs):
        k = rng.integers(2, n + 1)  # 2..n inklusif
        x0_hat_k = predict(k, a, b)
        error = x0[k - 1] - x0_hat_k

        exp_neg_a_km1 = np.exp(-a * (k - 1))
        exp_a = np.exp(a)
        V_ak = (
            (-exp_a) * (x0[0] - b / a) * exp_neg_a_km1
            + (1 - exp_a) * (b / a**2) * exp_neg_a_km1
            + (1 - exp_a) * (x0[0] - b / a) * (-(k - 1)) * exp_neg_a_km1
        )
        V_bk = (1 - exp_a) * (-1 / a) * exp_neg_a_km1

        a = a + np.clip(eta * error * V_ak, -max_step_a, max_step_a)
        b = b + np.clip(eta * error * V_bk, -max_step_b, max_step_b)

    x0_hat = np.empty(n)
    x0_hat[0] = x0[0]
    for k in range(2, n + 1):
        x0_hat[k - 1] = predict(k, a, b)
    r2 = _r_squared_local(x0, x0_hat)

    return {"a": float(a), "b": float(b), "x0_1": float(x0[0]), "first_year": first_year, "r2": r2}


def project_nngm11(fit: dict, target_year: int) -> float:
    """Ekstrapolasi NNGM(1,1) ke target_year via Eq.~7 (Hu 2017), bobot a,b hasil fit_nngm11()."""
    k = target_year - fit["first_year"] + 1
    a, b, x0_1 = fit["a"], fit["b"], fit["x0_1"]
    return float((1 - np.exp(a)) * (x0_1 - b / a) * np.exp(-a * (k - 1)))


# --------------------------------------------------------------------
# Tahap 5 - Validasi (Pers. 7)
# --------------------------------------------------------------------
def check_wbp_window_against_real_data(df_obs: pd.DataFrame) -> pd.Series:
    """
    Poin sekunder dari review eksternal: WBP_HOURS=17:00-21:59 (definisi
    tarif PLN, dipakai di seluruh modul ini sejak awal) diambil begitu saja
    dari dokumen Falfi (2024) tanpa pernah dikonfirmasi ulang terhadap pola
    beban aktual pada data pengukuran real (df_obs, mis. hasil
    load_genset_log()). Fungsi ini menyediakan pengecekan langsung: rata-rata
    load_kw per jam-dalam-hari (0-23), dikumpulkan dari SELURUH baris df_obs
    (lintas bulan/tipe hari), supaya bisa dibandingkan dengan WBP_HOURS untuk
    melihat apakah jendela tarif itu memang berimpit dengan jam-jam beban
    tertinggi secara empiris, atau menyimpang.

    Catatan jujur: ini pengecekan KEWAJARAN (apakah jam WBP_HOURS ada di
    antara jam-jam beban tertinggi), bukan estimasi ulang batas WBP/LWBP itu
    sendiri -- df_obs tidak berlabel WBP/LWBP secara independen (cuma
    beban_puncak_harian.csv yang membedakan siang/malam per hari, tanpa jam
    pasti), sehingga tidak ada "ground truth" jam-per-jam untuk direstimasi.

    Return: Series index=jam (0..23), nilai=rata-rata load_kw.
    """
    return df_obs.groupby(df_obs.index.hour)["load_kw"].mean()


def validate_mape(df_obs: pd.DataFrame, targets: dict) -> float:
    df = df_obs.copy()
    df["window"] = np.where(df.index.hour.isin(WBP_HOURS), "WBP", "LWBP")
    df["month"] = df.index.month

    monthly_obs = df.groupby(["month", "window"])["load_kw"].mean()

    errors = []
    for (month, window), obs_val in monthly_obs.items():
        target_val = targets[month][window]
        errors.append(abs(obs_val - target_val) / target_val)

    return float(np.mean(errors) * 100)


def validate_temperature_correlation(
    annual: pd.Series, temperature: pd.Series
) -> dict:
    """
    Tahap 5 - proksi validasi eksternal opsional: korelasi rata-rata bulanan
    antara profil beban hasil rekonstruksi dan suhu udara ambien (mis. hasil
    fetch_temperature_openmeteo() + temperature_records_to_series() pada
    akuisisi_data_terbuka.py). Dipakai sebagai pemeriksaan kewajaran arah
    tren musiman (bukan validasi kuantitatif langsung seperti Pers.(7)),
    mengikuti semangat item "validasi eksternal via proksi musiman lain"
    pada dokumen.

    annual: Series beban jam-an (index datetime), hasil reconstruct_annual_profile().
    temperature: Series suhu (index datetime, resolusi jam/harian), sumber eksternal.

    Return: {'r': korelasi Pearson rata-rata bulanan, 'monthly': DataFrame
    index=bulan, kolom=['load_kw','temp_c']}.
    """
    load_m = annual.resample("MS").mean()
    temp_m = temperature.resample("MS").mean()
    common = load_m.index.intersection(temp_m.index)
    df = pd.DataFrame({"load_kw": load_m.loc[common], "temp_c": temp_m.loc[common]})
    r = float(df["load_kw"].corr(df["temp_c"])) if len(df) >= 2 else float("nan")
    return {"r": r, "monthly": df}


# --------------------------------------------------------------------
# Integrasi ke MicroGridsPy-SESAM
# --------------------------------------------------------------------
def build_microgridspy_demand_csv(
    annual: pd.Series,
    growth_rate: float,
    n_years: int,
    output_path: str = "Demand_gili_ketapang.csv",
) -> pd.DataFrame:
    """
    Membangun Demand.csv sesuai format MicroGridsPy-SESAM, diverifikasi dari
    template asli di Code/Inputs/Demand.csv (repo pengguna):
    - delimiter ';', desimal ',';
    - baris = Periods (8760, hari kabisat 29 Feb dipotong bila ada);
    - kolom = 1..n_years (tahun proyek), header kolom indeks dikosongkan;
    - tahun ke-1 memakai profil dasar (tanpa pertumbuhan); tahun ke-y
      dikalikan (1+growth_rate)^(y-1), Pers.(6).
    """
    values = annual.to_numpy()
    if len(values) == 8784:  # tahun kabisat -> buang 24 jam terakhir (31 Des)
        values = values[:8760]
    elif len(values) != 8760:
        raise ValueError(
            f"Panjang profil tahunan harus 8760 atau 8784 jam, dapat {len(values)}"
        )

    data = {
        str(y): values * (1 + growth_rate) ** (y - 1) for y in range(1, n_years + 1)
    }
    df = pd.DataFrame(data, index=range(8760))
    df.to_csv(output_path, sep=";", decimal=",", index=True, index_label="")
    return df


# --------------------------------------------------------------------
# Visualisasi
# --------------------------------------------------------------------
def plot_annual_profile(
    annual: pd.Series, df_obs: pd.DataFrame, save_path: str = "profil_tahunan.png"
) -> None:
    """Profil tahunan hasil rekonstruksi, menyorot periode data riil."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(
        annual.index, annual.values, color="steelblue", lw=0.6,
        label="Profil terkalibrasi (rekonstruksi)",
    )
    obs_mask = annual.index.isin(df_obs.index)
    ax.plot(
        annual.index[obs_mask], annual.values[obs_mask], color="crimson", lw=0.8,
        label="Data riil (observasi)",
    )
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Beban (kW)")
    ax.set_title("Profil Beban Tahunan Gili Ketapang: Rekonstruksi vs Data Riil")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_monthly_comparison(
    annual: pd.Series, targets: dict, save_path: str = "perbandingan_bulanan.png"
) -> None:
    """Perbandingan rata-rata bulanan WBP/LWBP: target vs hasil rekonstruksi."""
    df = annual.to_frame("load_kw")
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["window"] = np.where(df["hour"].isin(WBP_HOURS), "WBP", "LWBP")
    monthly = df.groupby(["month", "window"])["load_kw"].mean().unstack("window")

    months = list(range(1, 13))
    labels = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
              "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    target_wbp = [targets[m]["WBP"] for m in months]
    target_lwbp = [targets[m]["LWBP"] for m in months]

    x = np.arange(len(months))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 1.5 * width, target_wbp, width, label="Target WBP (Falfi, 2024)", color="firebrick")
    ax.bar(x - 0.5 * width, monthly.loc[months, "WBP"], width, label="Rekonstruksi WBP", color="lightcoral")
    ax.bar(x + 0.5 * width, target_lwbp, width, label="Target LWBP (Falfi, 2024)", color="navy")
    ax.bar(x + 1.5 * width, monthly.loc[months, "LWBP"], width, label="Rekonstruksi LWBP", color="lightsteelblue")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Beban rata-rata (kW)")
    ax.set_title("Validasi: Target Kalibrasi vs Hasil Rekonstruksi per Bulan")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------
# Eksekusi utama
# --------------------------------------------------------------------
if __name__ == "__main__":
    df_obs = pd.read_csv(
        "beban_giliketapang_4bulan.csv", index_col=0, parse_dates=True
    )

    shape = extract_daily_shape(df_obs)  # Tahap 1
    annual = reconstruct_annual_profile(
        df_obs, shape, MONTHLY_TARGETS, year=2024
    )  # Tahap 2-3
    g = compute_growth_rate(ANNUAL_PEAK)  # Tahap 4
    annual_proj = project_growth(annual, g, n_years=1)
    mape = validate_mape(df_obs, MONTHLY_TARGETS)  # Tahap 5

    print(f"Growth rate: {g:.4%}")
    print(f"MAPE validasi (overlap bulan observasi): {mape:.2f}%")

    plot_annual_profile(annual, df_obs)
    plot_monthly_comparison(annual, MONTHLY_TARGETS)

    annual_proj.to_csv("profil_beban_tahunan_terkalibrasi.csv")

    # Integrasi ke MicroGridsPy-SESAM (sesuaikan n_years dengan parameter
    # 'Years' pada Model_Configuration.csv proyek MicroGridsPy)
    build_microgridspy_demand_csv(
        annual, g, n_years=20, output_path="Demand_gili_ketapang_rekonstruksi.csv"
    )
