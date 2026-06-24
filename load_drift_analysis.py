"""
配电网负荷时序漂移分析脚本  v2.0
=====================================
核心改进：同期对齐策略
  - 完整年（12月全覆盖）之间对比：使用全年数据
  - 涉及不完整年（如 2026 仅 1-3 月）的对比：
    所有年份只取同一月份区间，确保公平比较
  - 每张图标题/标签自动标注所用的数据范围

依赖：pandas, numpy, matplotlib, scipy
使用：
    python load_drift_analysis.py --file data.csv --high 420 --low 40
"""

import argparse, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# ── 自动注册中文字体 ──────────────────────────────────────
import matplotlib.font_manager as _fm
_CJK_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
]
_FONT_NAME = "DejaVu Sans"
for _fp in _CJK_CANDIDATES:
    if os.path.exists(_fp):
        try:
            _fm.fontManager.addfont(_fp)
            _FONT_NAME = _fm.FontProperties(fname=_fp).get_name()
            print(f"  [字体] 已加载: {_FONT_NAME}")
            break
        except Exception:
            continue

plt.rcParams.update({
    "font.family":        _FONT_NAME,
    "axes.unicode_minus": False,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
})

YEAR_COLORS  = {2024: "#2196F3", 2025: "#FF5722", 2026: "#4CAF50"}
YEAR_MARKERS = {2024: "o",        2025: "s",        2026: "^"}
MONTH_NAMES  = ["1月","2月","3月","4月","5月","6月",
                "7月","8月","9月","10月","11月","12月"]


# ═══════════════════════════════════════════════════════
#  1. 数据加载
# ═══════════════════════════════════════════════════════
def load_data(filepath):
    raw = pd.read_csv(filepath, header=None, names=["datetime", "load"])
    raw["datetime"] = pd.to_datetime(raw["datetime"], errors="coerce")
    raw["load"]     = pd.to_numeric(raw["load"], errors="coerce")
    raw = raw.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    raw = raw.resample("5min").mean()
    raw["load"] = raw["load"].interpolate(method="time", limit=3)
    raw["year"]    = raw.index.year
    raw["month"]   = raw.index.month
    raw["hour"]    = raw.index.hour
    raw["weekday"] = raw.index.weekday
    raw["hm"]      = raw["hour"] * 60 + raw.index.minute
    return raw


# ═══════════════════════════════════════════════════════
#  2. 同期对齐工具
# ═══════════════════════════════════════════════════════
def detect_period_info(raw):
    """
    检测每年的月份覆盖情况，确定公共对比月份区间。
    """
    year_months = {}
    for yr, g in raw.groupby("year"):
        year_months[yr] = set(int(m) for m in g["month"].unique())

    all_months = set(range(1, 13))
    full_years    = sorted(y for y, ms in year_months.items() if ms == all_months)
    partial_years = sorted(y for y, ms in year_months.items() if ms != all_months)

    # 公共月份 = 所有年份均有数据的月份交集
    common_months = set(range(1, 13))
    for ms in year_months.values():
        common_months &= ms
    common_months = sorted(common_months)

    return {
        "year_months":   year_months,
        "full_years":    full_years,
        "partial_years": partial_years,
        "common_months": common_months,
    }


def period_label(months):
    """[1,2,3] -> '1-3月'，[1..12] -> '全年'"""
    if not months:
        return "无数据"
    if sorted(months) == list(range(1, 13)):
        return "全年"
    ms = sorted(months)
    return f"{ms[0]}-{ms[-1]}月"


def get_plot_data(raw, pinfo, years_to_plot):
    """
    决定是否需要月份对齐，返回 (data_dict, used_months, note)。
    - 若涉及不完整年：所有年只取公共月份
    - 若全为完整年：使用全年数据
    """
    has_partial = any(y in pinfo["partial_years"] for y in years_to_plot)
    used_months = pinfo["common_months"] if has_partial else list(range(1, 13))
    note = f"（同期对比：{period_label(used_months)}）" if has_partial else "（全年）"

    data_dict = {}
    for yr in years_to_plot:
        sub = raw[raw["year"] == yr]
        sub = sub[sub["month"].isin(used_months)]
        data_dict[yr] = sub
    return data_dict, used_months, note


# ═══════════════════════════════════════════════════════
#  3. 统计指标
# ═══════════════════════════════════════════════════════
def _adf_test(x, maxlag=None):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if maxlag is None:
        maxlag = int(np.floor(12*(n/100)**0.25))
    maxlag = min(maxlag, n//4)
    dx = np.diff(x)
    lags = min(maxlag, len(dx)-2)
    y = dx[lags:]
    X = np.column_stack([x[lags:-1]] +
                        [dx[lags-i-1:-i-1] for i in range(lags)] +
                        [np.ones(len(y)), np.arange(len(y))])
    try:
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        e  = y - X @ coef
        s2 = np.dot(e, e) / max(len(y)-X.shape[1], 1)
        se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)*s2))
        t  = coef[0] / (se[0]+1e-15)
        p  = float(np.clip(np.interp(-t,[3.13,3.41,3.96],[0.10,0.05,0.01]),0.001,0.999))
        return float(t), p
    except Exception:
        return np.nan, np.nan


def _kpss_test(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    e = x - x.mean()
    S = np.cumsum(e)
    eta = np.sum(S**2)/n**2
    s2  = np.sum(e**2)/n
    L = int(np.floor(4*(n/100)**0.25))
    for j in range(1, L+1):
        s2 += 2*(1-j/(L+1))*np.dot(e[j:],e[:-j])/n
    stat = eta/max(s2,1e-15)
    p = float(np.clip(np.interp(stat,[0.347,0.463,0.739][::-1],[0.10,0.05,0.01][::-1]),0.01,0.15))
    return float(stat), p


def _stl_decompose(series_h, period=24):
    s = series_h.values.astype(float)
    n = len(s)
    half = period//2
    trend = np.full(n, np.nan)
    for i in range(half, n-half):
        trend[i] = np.nanmean(s[i-half:i+half+1])
    trend[:half] = trend[half]
    trend[n-half:] = trend[n-half-1]
    seasonal = np.zeros(n)
    for p in range(period):
        idx = np.arange(p, n, period)
        seasonal[idx] = np.nanmean(s[idx]-trend[idx])
    return trend, seasonal, s-trend-seasonal


def compute_stats(raw, pinfo, high_thr, low_thr):
    """
    每年计算两套指标：
      'full'   - 该年实际拥有的全部数据
      'common' - 仅公共月份数据（跨年公平对比用）
    """
    results = {}
    cm = pinfo["common_months"]
    for yr, grp in raw.groupby("year"):
        entry = {"year": yr}
        for key, sub in [("full", grp), ("common", grp[grp["month"].isin(cm)])]:
            s = sub["load"].dropna()
            if len(s) == 0:
                continue
            mu, sigma = s.mean(), s.std()
            med = s.median()
            cv  = sigma/mu if mu else np.nan
            p5,p25,p75,p95 = np.percentile(s,[5,25,75,95])
            n_hi = int((s>high_thr).sum())
            n_lo = int((s<low_thr).sum())
            dm_max = s.resample("D").max()
            dm_min = s.resample("D").min()
            def max_run(mask):
                mx=cur=0
                for v in mask:
                    cur=cur+1 if v else 0; mx=max(mx,cur)
                return mx
            try:
                s_h = s.resample("h").mean().dropna()
                adf_t, adf_p   = _adf_test(s_h.values)
                kpss_t, kpss_p = _kpss_test(s_h.values)
            except Exception:
                adf_t=adf_p=kpss_t=kpss_p=np.nan
            n = len(s)
            entry[key] = dict(
                n=n, mean=round(mu,3), std=round(sigma,3),
                median=round(med,3), cv=round(cv,4),
                skew=round(float(s.skew()),4), kurt=round(float(s.kurt()),4),
                peak=round(s.max(),3), valley=round(s.min(),3),
                pk2pk=round(s.max()-s.min(),3),
                p5=round(p5,3), p25=round(p25,3), p75=round(p75,3), p95=round(p95,3),
                n_high=n_hi, pct_high=round(n_hi/n*100,2),
                n_low=n_lo,  pct_low=round(n_lo/n*100,2),
                days_high=int((dm_min>high_thr).sum()),
                days_low =int((dm_max<low_thr).sum()),
                max_run_high_h=round(max_run(s>high_thr)*5/60,1),
                max_run_low_h =round(max_run(s<low_thr)*5/60,1),
                adf_p =round(float(adf_p),4)  if not np.isnan(adf_p)  else "N/A",
                is_adf="✓平稳" if (not np.isnan(adf_p) and adf_p<0.05) else "✗非平稳",
                kpss_p=round(float(kpss_p),4) if not np.isnan(kpss_p) else "N/A",
                is_kpss="✓平稳" if (not np.isnan(kpss_p) and kpss_p>0.05) else "✗非平稳",
            )
        results[yr] = entry
    return results


def print_stats_table(stats, pinfo, high_thr, low_thr):
    yrs = sorted(stats.keys())
    cm  = pinfo["common_months"]
    fields = [
        ("数据点数","n"),("均值(MW)","mean"),("标准差","std"),
        ("中位数","median"),("变异系数","cv"),("峰值","peak"),
        ("谷值","valley"),("峰谷差","pk2pk"),
        ("P5","p5"),("P95","p95"),
        ("超高点数","n_high"),("超高占比%","pct_high"),
        ("超低点数","n_low"),("超低占比%","pct_low"),
        ("最长连超高h","max_run_high_h"),("最长连超低h","max_run_low_h"),
        ("ADF p值","adf_p"),("ADF平稳","is_adf"),
        ("KPSS p值","kpss_p"),("KPSS平稳","is_kpss"),
    ]
    print("\n"+"═"*95)
    print("  配电网负荷时序漂移分析报告  v2.0（同期对齐）")
    print("═"*95)
    for mode, label in [("full","各年实际全量数据"), ("common",f"公共月份 {period_label(cm)}")]:
        print(f"\n  ▶ [{label}]")
        print(f"  {'指标':<18}"+"".join(f"{y:>14}" for y in yrs))
        print("  "+"-"*(20+14*len(yrs)))
        for fname, fkey in fields:
            row = f"  {fname:<18}"
            for yr in yrs:
                v = stats[yr].get(mode,{}).get(fkey,"—")
                row += f"{str(v):>14}"
            print(row)
    print("\n"+"─"*95)
    print(f"  同期漂移小结（公共月份 {period_label(cm)}）")
    print("─"*95)
    for i in range(1, len(yrs)):
        a = stats[yrs[i-1]].get("common",{})
        b = stats[yrs[i]].get("common",{})
        if a and b:
            dm=b["mean"]-a["mean"]; ds=b["std"]-a["std"]; dp=b["peak"]-a["peak"]
            print(f"  {yrs[i-1]}→{yrs[i]}: 均值漂移={dm:+.2f} MW  "
                  f"标准差漂移={ds:+.2f}  峰值漂移={dp:+.2f}")
    print("═"*95+"\n")


# ═══════════════════════════════════════════════════════
#  4. 绘图函数
# ═══════════════════════════════════════════════════════

def fig_overview(raw, stats, pinfo, save_path):
    yrs = sorted(raw["year"].unique())
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # (A) 日均值走势 — 各年完整时段，无需截断
    ax_a = fig.add_subplot(gs[0, :])
    daily = raw["load"].resample("D").mean()
    for yr in yrs:
        m = daily.index.year == yr
        ax_a.plot(daily.index[m], daily[m],
                  color=YEAR_COLORS.get(yr,"gray"), lw=0.8, alpha=0.7, label=str(yr))
    roll7 = daily.rolling(7, center=True).mean()
    ax_a.plot(roll7.index, roll7, color="black", lw=1.5, ls="--", label="7日滚动均值")
    ax_a.set_title("日均值负荷走势（各年完整时段，含不完整年已有数据）")
    ax_a.set_ylabel("有功功率 (MW)")
    ax_a.legend(loc="upper right")
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax_a.xaxis.get_majorticklabels(), rotation=30)

    # (B) 滚动30日波动性
    ax_b = fig.add_subplot(gs[1, 0])
    roll_std = raw["load"].resample("D").std().rolling(30, center=True).mean()
    for yr in yrs:
        m = roll_std.index.year == yr
        ax_b.plot(roll_std.index[m], roll_std[m],
                  color=YEAR_COLORS.get(yr,"gray"), lw=1.2, label=str(yr))
    ax_b.set_title("滚动30日波动性（日内标准差）")
    ax_b.set_ylabel("标准差 (MW)")
    ax_b.legend()
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))

    # (C) 年度箱线图 — 同期对齐
    ax_c = fig.add_subplot(gs[1, 1])
    data_dict, _, note = get_plot_data(raw, pinfo, yrs)
    bp_data = [data_dict[y]["load"].dropna().values for y in yrs]
    bp = ax_c.boxplot(bp_data, patch_artist=True, widths=0.5,
                      medianprops=dict(color="black", lw=2))
    for patch, yr in zip(bp["boxes"], yrs):
        patch.set_facecolor(YEAR_COLORS.get(yr,"gray")); patch.set_alpha(0.7)
    ax_c.set_xticks(range(1, len(yrs)+1))
    ax_c.set_xticklabels([str(y) for y in yrs])
    ax_c.set_title(f"年度负荷箱线图 {note}")
    ax_c.set_ylabel("有功功率 (MW)")

    # (D) 均值 ± 标准差（同期）
    ax_d = fig.add_subplot(gs[2, 0])
    x = np.arange(len(yrs))
    means  = [data_dict[y]["load"].mean() for y in yrs]
    stds   = [data_dict[y]["load"].std()  for y in yrs]
    colors = [YEAR_COLORS.get(y,"gray") for y in yrs]
    bars = ax_d.bar(x, means, yerr=stds, capsize=5, color=colors,
                    alpha=0.75, ecolor="black", error_kw=dict(lw=1.5))
    ax_d.set_xticks(x); ax_d.set_xticklabels([str(y) for y in yrs])
    ax_d.set_title(f"年均值 ± 标准差 {note}")
    ax_d.set_ylabel("有功功率 (MW)")
    for bar, m in zip(bars, means):
        ax_d.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                  f"{m:.1f}", ha="center", va="bottom", fontsize=8)

    # (E) 峰谷差 & CV（同期）
    ax_e = fig.add_subplot(gs[2, 1])
    pk2pk = [data_dict[y]["load"].max()-data_dict[y]["load"].min() for y in yrs]
    cvs   = [data_dict[y]["load"].std()/data_dict[y]["load"].mean()*100 for y in yrs]
    ax_e2 = ax_e.twinx()
    ax_e.bar(x-0.2, pk2pk, 0.35, color=colors, alpha=0.7,  label="峰谷差 (MW)")
    ax_e2.bar(x+0.2, cvs,  0.35, color=colors, alpha=0.4, hatch="///", label="CV (%)")
    ax_e.set_xticks(x); ax_e.set_xticklabels([str(y) for y in yrs])
    ax_e.set_title(f"年度峰谷差 & 变异系数 {note}")
    ax_e.set_ylabel("峰谷差 (MW)"); ax_e2.set_ylabel("变异系数 (%)")
    h1,l1=ax_e.get_legend_handles_labels(); h2,l2=ax_e2.get_legend_handles_labels()
    ax_e.legend(h1+h2, l1+l2, loc="upper right")

    fig.suptitle("配电网负荷时序漂移分析 — 总览", fontsize=15, fontweight="bold", y=0.98)
    fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    print(f"  [✓] 图1 已保存: {save_path}")


def fig_anomaly(raw, pinfo, high_thr, low_thr, save_path):
    yrs = sorted(raw["year"].unique())
    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    fig.subplots_adjust(hspace=0.5)

    # (A) 全时序
    ax = axes[0]
    for yr in yrs:
        m = raw["year"] == yr
        ax.plot(raw.index[m], raw.loc[m,"load"],
                color=YEAR_COLORS.get(yr,"gray"), lw=0.4, alpha=0.6, label=str(yr))
    ax.axhline(high_thr, color="red",   ls="--", lw=1.5, label=f"高阈值 {high_thr}")
    ax.axhline(low_thr,  color="orange",ls="--", lw=1.5, label=f"低阈值 {low_thr}")
    ax.scatter(raw.index[raw["load"]>high_thr], raw.loc[raw["load"]>high_thr,"load"],
               s=2, color="red",    zorder=5, alpha=0.5)
    ax.scatter(raw.index[raw["load"]<low_thr],  raw.loc[raw["load"]<low_thr,"load"],
               s=2, color="orange", zorder=5, alpha=0.5)
    ax.set_title("全时序负荷 + 异常点（各年完整时段）")
    ax.set_ylabel("有功功率 (MW)")
    ax.legend(ncol=4); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # (B) 月度异常计数（同期）
    ax2 = axes[1]
    data_dict, used_months, note = get_plot_data(raw, pinfo, yrs)
    x = np.arange(len(used_months))
    width = 0.8/(len(yrs)*2)
    patches = []
    for i, yr in enumerate(yrs):
        sub = data_dict[yr].copy()
        sub["hi"] = (sub["load"]>high_thr).astype(int)
        sub["lo"] = (sub["load"]<low_thr).astype(int)
        hi = [sub.loc[sub["month"]==m,"hi"].sum() for m in used_months]
        lo = [sub.loc[sub["month"]==m,"lo"].sum() for m in used_months]
        off = (i-len(yrs)/2+0.5)*width*2
        c = YEAR_COLORS.get(yr,"gray")
        ax2.bar(x+off-width/2, hi, width, color=c, alpha=0.8)
        ax2.bar(x+off+width/2, lo, width, color=c, alpha=0.4, hatch="//")
        patches += [Patch(color=c,alpha=0.8,label=f"{yr}超高"),
                    Patch(color=c,alpha=0.4,hatch="//",label=f"{yr}超低")]
    ax2.legend(handles=patches, ncol=len(yrs)*2, fontsize=7)
    ax2.set_xticks(x); ax2.set_xticklabels([MONTH_NAMES[m-1] for m in used_months])
    ax2.set_title(f"月度异常点数对比 {note}"); ax2.set_ylabel("异常点数")

    # (C) 日均值超阈值
    ax3 = axes[2]
    dm = raw["load"].resample("D").mean()
    ax3.plot(dm.index, dm, color="steelblue", lw=0.8, label="日均值")
    ax3.fill_between(dm.index, high_thr, dm, where=dm>high_thr,
                     alpha=0.35, color="red", label="超高区")
    ax3.fill_between(dm.index, dm, low_thr,  where=dm<low_thr,
                     alpha=0.35, color="orange", label="超低区")
    ax3.axhline(high_thr,color="red",ls="--",lw=1,alpha=0.5)
    ax3.axhline(low_thr, color="orange",ls="--",lw=1,alpha=0.5)
    ax3.set_title("日均值 — 超阈值区域"); ax3.set_ylabel("有功功率 (MW)")
    ax3.legend(); ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.suptitle("负荷异常分析", fontsize=15, fontweight="bold")
    fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    print(f"  [✓] 图2 已保存: {save_path}")


def fig_distribution(raw, pinfo, save_path):
    yrs = sorted(raw["year"].unique())
    data_dict, _, note = get_plot_data(raw, pinfo, yrs)

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # (A) KDE
    ax_a = fig.add_subplot(gs[0, 0])
    for yr in yrs:
        s = data_dict[yr]["load"].dropna()
        if len(s)<10: continue
        kde = gaussian_kde(s, bw_method="silverman")
        xs  = np.linspace(s.min(), s.max(), 400)
        ax_a.plot(xs, kde(xs), lw=2, color=YEAR_COLORS.get(yr,"gray"), label=str(yr))
        ax_a.axvline(s.mean(), color=YEAR_COLORS.get(yr,"gray"), ls=":", lw=1)
    ax_a.set_title(f"负荷概率密度分布（KDE）{note}")
    ax_a.set_xlabel("有功功率 (MW)"); ax_a.set_ylabel("密度"); ax_a.legend()

    # (B) 分位数漂移
    ax_b = fig.add_subplot(gs[0, 1])
    pcts = [5, 25, 50, 75, 95]
    x = np.arange(len(pcts))
    for yr in yrs:
        s = data_dict[yr]["load"].dropna()
        if len(s)<10: continue
        ax_b.plot(x, [np.percentile(s,p) for p in pcts],
                  marker=YEAR_MARKERS.get(yr,"o"), lw=2,
                  color=YEAR_COLORS.get(yr,"gray"), label=str(yr))
    ax_b.set_xticks(x); ax_b.set_xticklabels([f"P{p}" for p in pcts])
    ax_b.set_title(f"分位数漂移对比 {note}")
    ax_b.set_ylabel("有功功率 (MW)"); ax_b.legend()

    # (C) 逐时均值热力图
    ax_c = fig.add_subplot(gs[1, :])
    hour_arr = np.full((len(yrs), 24), np.nan)
    for i, yr in enumerate(yrs):
        for h in range(24):
            v = data_dict[yr].loc[data_dict[yr]["hour"]==h, "load"].mean()
            hour_arr[i, h] = v
    im = ax_c.imshow(hour_arr, aspect="auto", cmap="RdYlGn_r",
                     extent=[-0.5,23.5,len(yrs)-0.5,-0.5])
    ax_c.set_yticks(range(len(yrs))); ax_c.set_yticklabels([str(y) for y in yrs])
    ax_c.set_xticks(range(24))
    ax_c.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)
    ax_c.set_title(f"逐时均值热力图 {note}")
    plt.colorbar(im, ax=ax_c, label="MW", shrink=0.8)
    for i in range(len(yrs)):
        for h in range(24):
            v = hour_arr[i,h]
            if not np.isnan(v):
                ax_c.text(h, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5)

    fig.suptitle("负荷分布漂移分析", fontsize=15, fontweight="bold")
    fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    print(f"  [✓] 图3 已保存: {save_path}")


def fig_same_period(raw, pinfo, save_path):
    yrs = sorted(raw["year"].unique())
    cm  = pinfo["common_months"]
    note = f"（同期对比：{period_label(cm)}）"

    # 场景只从公共月份中选取
    all_scenarios = [
        ("夏季高峰（7月工作日）", 7,  "wd_max", 4),
        ("冬季低谷（1月周末）",   1,  "wd_min", 5),
        ("春季过渡（4月工作日）", 4,  "wd_max", 4),
        ("秋季稳定（10月工作日）",10, "wd_max", 4),
        ("1月工作日",             1,  "wd_max", 4),
        ("2月工作日（含春节）",   2,  "wd_max", 4),
        ("3月工作日",             3,  "wd_max", 4),
        ("3月周末",               3,  "wd_min", 5),
    ]
    scenarios = [(t,m,k,v) for t,m,k,v in all_scenarios if m in cm][:4]
    if not scenarios:
        scenarios = all_scenarios[:4]

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(18, 4.5*len(scenarios)))
    if len(scenarios)==1: axes=[axes]
    fig.subplots_adjust(hspace=0.55)

    for ax, (title, month, wd_key, wd_val) in zip(axes, scenarios):
        for yr in yrs:
            sub = raw[(raw["year"]==yr) & (raw["month"]==month)]
            if wd_key=="wd_max": sub = sub[sub["weekday"]<=wd_val]
            else:                sub = sub[sub["weekday"]>=wd_val]
            if sub.empty: continue
            hm_m = sub.groupby("hm")["load"].mean()
            hm_s = sub.groupby("hm")["load"].std()
            xs = hm_m.index/60
            ax.plot(xs, hm_m, color=YEAR_COLORS.get(yr,"gray"),
                    lw=2, marker=YEAR_MARKERS.get(yr,"o"), markevery=24, ms=4, label=str(yr))
            ax.fill_between(xs, hm_m-hm_s, hm_m+hm_s,
                            alpha=0.15, color=YEAR_COLORS.get(yr,"gray"))
        ax.set_title(f"同期对比：{title}（阴影=±1σ）{note}")
        ax.set_xlabel("时刻 (h)"); ax.set_ylabel("有功功率 (MW)")
        ax.set_xticks(range(0,25,2))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0,25,2)])
        ax.legend(loc="upper right")

    fig.suptitle(f"同期典型场景跨年对比 {note}", fontsize=15, fontweight="bold")
    fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    print(f"  [✓] 图4 已保存: {save_path}")


def fig_monthly_drift(raw, pinfo, save_path):
    yrs = sorted(raw["year"].unique())
    cm  = pinfo["common_months"]
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))
    fig.subplots_adjust(hspace=0.45)

    # (A) 月度均值热力图（全月份，金框标注公共区间）
    ax = axes[0]
    arr = np.full((len(yrs), 12), np.nan)
    for i, yr in enumerate(yrs):
        sub = raw[raw["year"]==yr]
        for j, m in enumerate(range(1,13)):
            vals = sub.loc[sub["month"]==m,"load"]
            arr[i,j] = vals.mean() if len(vals)>0 else np.nan
    im = ax.imshow(arr, aspect="auto", cmap="coolwarm",
                   extent=[-0.5,11.5,len(yrs)-0.5,-0.5])
    ax.set_yticks(range(len(yrs))); ax.set_yticklabels([str(y) for y in yrs])
    ax.set_xticks(range(12)); ax.set_xticklabels(MONTH_NAMES)
    plt.colorbar(im, ax=ax, label="月均值 (MW)", shrink=0.8)
    for i in range(len(yrs)):
        for j in range(12):
            v = arr[i,j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=9, fontweight="bold")
    # 金色边框标注公共区间
    if cm and sorted(cm) != list(range(1,13)):
        x0, x1 = cm[0]-1-0.48, cm[-1]-1+0.48
        ax.add_patch(plt.Rectangle((x0,-0.48), x1-x0, len(yrs)-0.04,
                                   lw=2.5, edgecolor="gold",
                                   facecolor="none", zorder=5))
        ax.set_title(f"月度均值热力图  ▶ 金框 = 公共对比区间（{period_label(cm)}）")
    else:
        ax.set_title("月度均值热力图（全年）")

    # (B) 月度峰/均/谷折线 + 公共区间背景
    ax2 = axes[1]
    x = np.arange(12)
    for yr in yrs:
        sub = raw[raw["year"]==yr]
        ym = set(int(m) for m in sub["month"].unique())
        maxv  = [sub.loc[sub["month"]==m,"load"].max()  if m in ym else np.nan for m in range(1,13)]
        meanv = [sub.loc[sub["month"]==m,"load"].mean() if m in ym else np.nan for m in range(1,13)]
        minv  = [sub.loc[sub["month"]==m,"load"].min()  if m in ym else np.nan for m in range(1,13)]
        c = YEAR_COLORS.get(yr,"gray")
        ax2.plot(x, maxv,  color=c, lw=1.5, ls="-",  marker="^", ms=5, label=f"{yr} 峰")
        ax2.plot(x, meanv, color=c, lw=1.5, ls="--", marker="o", ms=4, label=f"{yr} 均")
        ax2.plot(x, minv,  color=c, lw=1.5, ls=":",  marker="v", ms=5, label=f"{yr} 谷")
    if cm and sorted(cm) != list(range(1,13)):
        ax2.axvspan(cm[0]-1-0.4, cm[-1]-1+0.4, alpha=0.08, color="gold",
                    label=f"公共区间（{period_label(cm)}）")
    ax2.set_xticks(x); ax2.set_xticklabels(MONTH_NAMES)
    ax2.set_title("月度峰/均/谷值跨年对比（金色区域=共有月份）")
    ax2.set_ylabel("有功功率 (MW)")
    ax2.legend(loc="upper right", ncol=len(yrs)+1, fontsize=8)

    fig.suptitle("月度负荷漂移分析", fontsize=15, fontweight="bold")
    fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    print(f"  [✓] 图5 已保存: {save_path}")


def fig_stl_decompose(raw, pinfo, save_path):
    yrs = sorted(raw["year"].unique())
    n = len(yrs)
    fig, axes = plt.subplots(n, 4, figsize=(20, 4.5*n))
    if n==1: axes = axes.reshape(1,-1)
    fig.subplots_adjust(hspace=0.55, wspace=0.35)

    for row, yr in enumerate(yrs):
        is_partial = yr in pinfo["partial_years"]
        sfx = f"（{period_label(sorted(pinfo['year_months'][yr]))}）" if is_partial else "（全年）"
        s   = raw.loc[raw["year"]==yr,"load"].dropna()
        s_h = s.resample("h").mean().dropna()
        if len(s_h)<48: continue
        try:
            trend, seasonal, resid = _stl_decompose(s_h, period=24)
        except Exception as e:
            print(f"  [!] STL {yr} 失败:{e}"); continue
        for col, (data, dlab) in enumerate([
            (s_h.values,"原始序列"),(trend,"趋势项"),(seasonal,"季节项"),(resid,"残差")
        ]):
            ax = axes[row, col]
            ax.plot(s_h.index, data, lw=0.6,
                    color=YEAR_COLORS.get(yr,"steelblue"), alpha=0.85)
            ax.set_title(f"{yr}{sfx} — {dlab}")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=7)
            if col==0: ax.set_ylabel("MW")

    fig.suptitle("STL 季节性分解（小时粒度）", fontsize=15, fontweight="bold")
    fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    print(f"  [✓] 图6 已保存: {save_path}")


def fig_weekday_pattern(raw, pinfo, save_path):
    yrs = sorted(raw["year"].unique())
    data_dict, _, note = get_plot_data(raw, pinfo, yrs)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.35)
    for ax, (wk, label) in zip(axes, [("work","工作日（周一~五）"),("rest","周末（周六~日）")]):
        for yr in yrs:
            sub = data_dict[yr]
            sub = sub[sub["weekday"]<=4] if wk=="work" else sub[sub["weekday"]>=5]
            if sub.empty: continue
            hm_m = sub.groupby("hm")["load"].mean()
            hm_s = sub.groupby("hm")["load"].std()
            xs = hm_m.index/60
            ax.plot(xs, hm_m, color=YEAR_COLORS.get(yr,"gray"), lw=2, label=str(yr))
            ax.fill_between(xs, hm_m-hm_s, hm_m+hm_s,
                            alpha=0.15, color=YEAR_COLORS.get(yr,"gray"))
        ax.set_title(f"均值日曲线：{label} {note}")
        ax.set_xlabel("时刻 (h)"); ax.set_ylabel("有功功率 (MW)")
        ax.set_xticks(range(0,25,2))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0,25,2)])
        ax.legend()

    fig.suptitle(f"工作日 vs 周末负荷模式跨年对比 {note}", fontsize=15, fontweight="bold")
    fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    print(f"  [✓] 图7 已保存: {save_path}")


def fig_metrics_summary(raw, stats, pinfo, save_path):
    yrs = sorted(stats.keys())
    if len(yrs)<2: return
    _, _, note = get_plot_data(raw, pinfo, yrs)

    fig = plt.figure(figsize=(18, 8))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.45)

    # (A) 雷达图（common 统计）
    ax_a = fig.add_subplot(gs[0, 0], polar=True)
    mlabs = ["均值","标准差","峰值","CV×100","P95","P5"]
    mkeys = ["mean","std","peak","cv","p95","p5"]
    scales = {}
    for k in mkeys:
        vals = [float(stats[y].get("common",{}).get(k,0) or 0) for y in yrs]
        scales[k] = (min(vals), max(vals))
    N = len(mlabs)
    angles = np.linspace(0,2*np.pi,N,endpoint=False).tolist()+[0]
    for yr in yrs:
        d = stats[yr].get("common",{})
        vs = []
        for k in mkeys:
            v = float(d.get(k,0) or 0)*(100 if k=="cv" else 1)
            mn,mx = scales[k]
            vs.append((v-mn)/(mx-mn+1e-9))
        vs += vs[:1]
        ax_a.plot(angles, vs, lw=2, color=YEAR_COLORS.get(yr,"gray"), label=str(yr))
        ax_a.fill(angles, vs, alpha=0.1, color=YEAR_COLORS.get(yr,"gray"))
    ax_a.set_xticks(angles[:-1]); ax_a.set_xticklabels(mlabs)
    ax_a.set_title(f"时序指标雷达图（归一化）{note}", pad=15)
    ax_a.legend(loc="upper right", bbox_to_anchor=(1.35,1.1))

    # (B) 漂移量瀑布图（common 统计）
    ax_b = fig.add_subplot(gs[0, 1])
    dkeys  = ["mean","std","peak","valley","p95","p5"]
    dlabs  = ["均值","标准差","峰值","谷值","P95","P5"]
    x = np.arange(len(dkeys))
    width = 0.8/(len(yrs)-1)
    for i in range(1, len(yrs)):
        pa = stats[yrs[i-1]].get("common",{})
        pb = stats[yrs[i]].get("common",{})
        drifts = [float(pb.get(k,0) or 0)-float(pa.get(k,0) or 0) for k in dkeys]
        offset = (i-1-(len(yrs)-2)/2)*width
        for j,(d,c) in enumerate(zip(drifts,["red" if d>0 else "green" for d in drifts])):
            ax_b.bar(x[j]+offset, d, width*0.85, color=c, alpha=0.7,
                     label=f"{yrs[i-1]}→{yrs[i]}" if j==0 else "")
            ax_b.text(x[j]+offset, d+np.sign(d)*0.5, f"{d:+.1f}",
                      ha="center", va="bottom" if d>0 else "top", fontsize=7)
    ax_b.axhline(0, color="black", lw=1)
    ax_b.set_xticks(x); ax_b.set_xticklabels(dlabs)
    ax_b.set_title(f"年际漂移量（红=增/绿=减）{note}")
    ax_b.set_ylabel("漂移量 (MW)"); ax_b.legend()

    fig.suptitle(f"时序漂移指标综合摘要 {note}", fontsize=15, fontweight="bold")
    fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    print(f"  [✓] 图8 已保存: {save_path}")


# ═══════════════════════════════════════════════════════
#  5. 主函数
# ═══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="配电网负荷时序漂移分析 v2.0")
    parser.add_argument("--file",   default="C:/Users/admin/Desktop/最近/1.28/IEEE-Transactions-LaTeX2e-templates-and-instructions/dataset/丽水全社会负荷.csv")
    parser.add_argument("--high",   type=float, default=420.0)
    parser.add_argument("--low",    type=float, default=40.0)
    parser.add_argument("--outdir", default="/mnt/user-data/outputs")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  配电网负荷时序漂移分析 v2.0 — 同期对齐")
    print(f"  文件: {args.file}")
    print(f"  高阈值={args.high} MW  低阈值={args.low} MW")
    print(f"{'='*65}\n")

    raw   = load_data(args.file)
    pinfo = detect_period_info(raw)

    print(f"  时间范围: {raw.index.min()} → {raw.index.max()}")
    print(f"  完整年  : {pinfo['full_years']}")
    partial_detail = " | ".join(
        f"{y}:{sorted(pinfo['year_months'][y])}" for y in pinfo['partial_years'])
    print(f"  不完整年: {pinfo['partial_years']}  （各月: {partial_detail}）")
    print(f"  公共月份: {period_label(pinfo['common_months'])}  {pinfo['common_months']}")
    print()

    stats = compute_stats(raw, pinfo, args.high, args.low)
    print_stats_table(stats, pinfo, args.high, args.low)

    print("▶ 开始绘图（共8张）...")
    out = args.outdir
    fig_overview       (raw, stats, pinfo, f"{out}/01_overview.png")
    fig_anomaly        (raw, pinfo, args.high, args.low, f"{out}/02_anomaly.png")
    fig_distribution   (raw, pinfo, f"{out}/03_distribution.png")
    fig_same_period    (raw, pinfo, f"{out}/04_same_period.png")
    fig_monthly_drift  (raw, pinfo, f"{out}/05_monthly_drift.png")
    fig_stl_decompose  (raw, pinfo, f"{out}/06_stl_decompose.png")
    fig_weekday_pattern(raw, pinfo, f"{out}/07_weekday_pattern.png")
    fig_metrics_summary(raw, stats, pinfo, f"{out}/08_metrics_summary.png")

    print(f"\n✅  全部完成！图片已保存至 {out}/")


if __name__ == "__main__":
    main()
