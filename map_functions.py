# =============================================================================
# map_functions.py
#
# Visualization functions for the Housing Coverage Optimization Model.
# Produces county- and state-level choropleth maps and a housing-type
# stacked bar chart.  Alaska and Hawaii are shown as inset boxes.
#
# Usage:
#   from src.map_functions import load_and_merge_data, generate_all_maps_at_level
#
#   gdf = load_and_merge_data(
#       results_csv  = "outputs/coverage_results.csv",
#       geocodes_xlsx= "data/all-geocodes-v2020.xlsx",
#       county_shp   = "data/cb_2024_us_county_5m/cb_2024_us_county_5m.shp",
#       output_dir   = "outputs/maps",
#       scenario_tag = "Base",
#   )
#   for level in ["county", "state"]:
#       generate_all_maps_at_level(gdf, level=level)
# =============================================================================

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# STATE ABBREVIATION MAPPING (FIPS → 2-letter code)
# =============================================================================
STATE_ABBREV = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY", "60": "AS", "66": "GU", "69": "MP", "72": "PR",
    "78": "VI",
}

# Inset box positions in figure-relative coords [left, bottom, width, height]
_AK_INSET = [0.01, 0.01, 0.20, 0.22]
_HI_INSET = [0.21, 0.01, 0.11, 0.13]
_AK_XLIM  = (-180, -130);  _AK_YLIM = (50, 72)
_HI_XLIM  = (-162, -154);  _HI_YLIM = (18, 24)

# These are set once by load_and_merge_data() and used by all map functions
OUTPUT_DIR   = "outputs/maps"
SCENARIO_TAG = "Base"


# =============================================================================
# DATA LOADING & MERGING
# =============================================================================

def load_and_merge_data(
    results_csv:   str = "outputs/coverage_results.csv",
    geocodes_xlsx: str = "data/all-geocodes-v2020.xlsx",
    county_shp:    str = "data/cb_2024_us_county_5m/cb_2024_us_county_5m.shp",
    output_dir:    str = "outputs/maps",
    scenario_tag:  str = "Base",
) -> gpd.GeoDataFrame:
    """
    Load optimisation results, merge with census geocodes and county shapefile.

    Parameters
    ----------
    results_csv   : path to coverage_results.csv produced by housing_model.py
    geocodes_xlsx : path to all-geocodes-v2020.xlsx (Census Bureau)
    county_shp    : path to cb_2024_us_county_5m shapefile
    output_dir    : directory for map outputs (created if absent)
    scenario_tag  : label used for output filenames

    Returns
    -------
    GeoDataFrame ready for all mapping functions
    """
    global OUTPUT_DIR, SCENARIO_TAG
    OUTPUT_DIR   = output_dir
    SCENARIO_TAG = scenario_tag
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    cov     = pd.read_csv(results_csv)
    geo_raw = pd.read_excel(geocodes_xlsx, dtype=str)

    # State name lookup
    geo_states = geo_raw[geo_raw["Summary Level"] == "040"].copy()
    state_map  = {
        str(r["State Code (FIPS)"]).strip().zfill(2):
        str(r["Area Name (including legal/statistical area description)"]).strip().lower()
        for _, r in geo_states.iterrows()
    }

    # County GEOID lookup
    geo_county = geo_raw[geo_raw["Summary Level"] == "050"].copy()
    geo_county["full_name"] = (geo_county["Area Name (including legal/statistical area description)"]
                               .fillna("").astype(str).str.strip())
    name_parts = geo_county["full_name"].str.split(",", n=1, expand=True)
    geo_county["county_part"] = (name_parts[0].fillna("").astype(str).str.strip()
                                 if name_parts.shape[1] >= 1 else geo_county["full_name"])

    def _clean(s):
        if pd.isna(s) or s == "":
            return ""
        s = str(s).lower().strip()
        for sfx in [" county", " parish", " borough", " census area",
                    " city", " municipality", " city and borough"]:
            if s.endswith(sfx):
                s = s[:-len(sfx)].strip()
        return s

    geo_county["county_clean"] = geo_county["county_part"].apply(_clean)
    geo_county["state_code"]   = (geo_county["State Code (FIPS)"]
                                  .fillna("").astype(str).str.strip().str.zfill(2))
    geo_county["state_clean"]  = geo_county["state_code"].map(state_map).fillna("")
    geo_county["GEOID"]        = (geo_county["state_code"] +
                                  geo_county["County Code (FIPS)"]
                                  .fillna("").astype(str).str.strip().str.zfill(3))
    geo_county = geo_county[
        (geo_county["GEOID"].str.len() == 5) &
        (geo_county["state_clean"] != "") &
        (geo_county["county_clean"] != "")
    ].drop_duplicates(subset=["state_clean", "county_clean"], keep="first")

    cov["state_clean"]  = cov["state"].fillna("").astype(str).str.lower().str.strip()
    cov["county_clean"] = cov["county"].fillna("").astype(str).apply(_clean)
    cov_geo = cov.merge(geo_county[["state_clean", "county_clean", "GEOID"]],
                        on=["state_clean", "county_clean"], how="inner")
    print(f"  {len(cov_geo):,} counties matched to geocodes")

    # Shapefile
    gdf = gpd.read_file(county_shp)
    gdf["GEOID"]  = gdf["GEOID"].astype(str).str.zfill(5)
    cov_geo["GEOID"] = cov_geo["GEOID"].astype(str).str.zfill(5)

    # Filter: keep only US states (FIPS ≤ 56), drop CT planning regions
    gdf = gdf[gdf["GEOID"].str[:2].astype(int) <= 56].copy()
    gdf = gdf[~((gdf["GEOID"].str[:2] == "09") &
                (gdf["GEOID"].str[2:].astype(int) > 9))].copy()
    gdf = gdf[(gdf["GEOID"].str[2:].astype(int) >= 1) &
              (gdf["GEOID"].str[2:].astype(int) <= 999)].copy()

    gdf_merged = gdf.merge(cov_geo, on="GEOID", how="left")
    print(f"✓ {len(gdf_merged):,} counties for mapping "
          f"({gdf_merged['ELI_coverage_pct'].notna().sum():,} with model results)")

    out_path = os.path.join(OUTPUT_DIR, f"{SCENARIO_TAG}_coverage_by_county.csv")
    gdf_merged.drop(columns="geometry").to_csv(out_path, index=False)
    return gdf_merged


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def get_geo_data(gdf: gpd.GeoDataFrame,
                 level: str = "county") -> tuple[gpd.GeoDataFrame, str]:
    """Dissolve to state level if requested; otherwise return county GDF."""
    if level.lower() == "state":
        if "STATEFP" not in gdf.columns and "GEOID" in gdf.columns:
            gdf = gdf.copy()
            gdf["STATEFP"] = gdf["GEOID"].str[:2]
        if "STATEFP" not in gdf.columns:
            return gdf, "county"
        state_gdf = gdf.dissolve(by="STATEFP", aggfunc="sum", numeric_only=True)
        for tier in ["ELI", "VLI", "LI"]:
            sc, dc = f"{tier}_served", f"{tier}_deficit"
            if sc in state_gdf.columns and dc in state_gdf.columns:
                state_gdf[f"{tier}_coverage_pct"] = (
                    state_gdf[sc] / state_gdf[dc].replace(0, np.nan) * 100
                ).fillna(0)
        return state_gdf, "state"
    return gdf, "county"


def _split_ak_hi(gdf: gpd.GeoDataFrame):
    """Split GDF into (continental, Alaska, Hawaii)."""
    if gdf.index.name == "STATEFP":
        fips = gdf.index.astype(str).str.zfill(2)
    else:
        fips = gdf["GEOID"].astype(str).str[:2]
    return (gdf[~fips.isin(["02", "15"])],
            gdf[fips == "02"],
            gdf[fips == "15"])


def add_state_annotations(ax, plot_gdf: gpd.GeoDataFrame,
                           value_col: str = None, label_type=None):
    """Annotate state centroids with 2-letter code and optional numeric value."""
    if "geometry" not in plot_gdf.columns:
        return
    try:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    except Exception:
        xlim, ylim = (-180, -65), (18, 72)

    for idx, row in plot_gdf.iterrows():
        if row["geometry"] is None:
            continue
        try:
            cx, cy     = row["geometry"].centroid.x, row["geometry"].centroid.y
            if not (xlim[0] <= cx <= xlim[1] and ylim[0] <= cy <= ylim[1]):
                continue
            fips       = str(idx) if isinstance(idx, str) else ""
            code       = STATE_ABBREV.get(fips, fips)
            if value_col and value_col in row.index:
                val  = row.get(value_col, 0)
                ann  = f"{code}\n{val:.0f}" if val > 0 else code
            else:
                ann  = code
            fs = 3 if code in ("RI", "DE", "CT", "NJ", "MD", "MA") else 6
            ax.text(cx, cy, ann, fontsize=fs, ha="center", va="center",
                    fontweight="normal", color="black",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              alpha=0.3, edgecolor="none"))
        except Exception:
            continue


def _plot_inset(fig, bounds, gdf_sub, xlim, ylim, *,
                column=None, cmap=None, vmin=None, vmax=None,
                edge_color="black", line_width=0.5,
                special_masks=None, special_colors=None,
                value_col=None, label=None):
    """Render a small inset axes (AK or HI) onto fig."""
    if gdf_sub is None or gdf_sub.empty:
        return None
    ax_in = fig.add_axes(bounds)
    ax_in.set_aspect("auto")
    if column and column in gdf_sub.columns:
        gdf_sub.plot(column=column, ax=ax_in, cmap=cmap,
                     vmin=vmin, vmax=vmax,
                     edgecolor=edge_color, linewidth=line_width,
                     legend=False,
                     missing_kwds={"color": "lightgrey",
                                   "edgecolor": edge_color,
                                   "linewidth": line_width})
    else:
        gdf_sub.plot(ax=ax_in, color="lightgrey",
                     edgecolor=edge_color, linewidth=line_width)
    if special_masks and special_colors:
        for key, mask in special_masks.items():
            loc = mask.reindex(gdf_sub.index, fill_value=False)
            if loc.any():
                gdf_sub[loc].plot(ax=ax_in, color=special_colors[key],
                                  edgecolor=edge_color, linewidth=line_width)
    if value_col:
        add_state_annotations(ax_in, gdf_sub, value_col=value_col)
    ax_in.set_xlim(xlim); ax_in.set_ylim(ylim); ax_in.set_axis_off()
    for sp in ax_in.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color("gray")
    if label:
        ax_in.set_title(label, fontsize=7, pad=2, loc="center")
    return ax_in


# =============================================================================
# MAP: COVERAGE RATE
# =============================================================================

def create_coverage_map_any_level(gdf: gpd.GeoDataFrame,
                                   income_tier: str = "ELI",
                                   level: str = "county"):
    """Choropleth of housing deficit coverage rate for one income tier."""
    plot_gdf, actual_level = get_geo_data(gdf, level)
    print(f"\nCreating {actual_level.upper()}-LEVEL coverage map for {income_tier}...")

    cov_col = f"{income_tier}_coverage_pct"
    def_col = f"{income_tier}_deficit"
    srv_col = f"{income_tier}_served"

    if cov_col not in plot_gdf.columns:
        if srv_col in plot_gdf.columns and def_col in plot_gdf.columns:
            plot_gdf[cov_col] = (
                plot_gdf[srv_col] / plot_gdf[def_col].replace(0, np.nan) * 100
            ).fillna(0)
        else:
            return None

    deficit_data = plot_gdf[def_col].fillna(0) if def_col in plot_gdf.columns else pd.Series([0])
    served_data  = plot_gdf[srv_col].fillna(0) if srv_col in plot_gdf.columns else pd.Series([0])
    units_name   = "States" if actual_level == "state" else "Counties"
    n_deficit    = (plot_gdf[def_col] > 0).sum() if def_col in plot_gdf.columns else 0
    tot_deficit  = deficit_data.sum()
    tot_served   = served_data.sum()
    remaining    = tot_deficit - tot_served
    overall_cov  = (tot_served / tot_deficit * 100) if tot_deficit > 0 else 0

    cont, ak, hi = _split_ak_hi(plot_gdf)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    ec, lw  = ("white", 0.2) if actual_level == "county" else ("black", 0.5)

    cont.plot(column=cov_col, ax=ax, legend=True, cmap="RdYlGn",
              vmin=0, vmax=100, edgecolor=ec, linewidth=lw,
              missing_kwds={"color": "lightgrey", "edgecolor": ec, "linewidth": lw},
              legend_kwds={"label": "Coverage Rate (%)", "orientation": "vertical",
                           "shrink": 0.4, "aspect": 15, "pad": 0.02})
    cbar = ax.get_figure().axes[-1]
    cbar.tick_params(labelsize=10)
    cbar.set_ylabel("Coverage Rate (%)", fontsize=10)

    if actual_level == "state":
        add_state_annotations(ax, cont, value_col=cov_col)

    ax.set_aspect("auto"); ax.set_xlim(-130, -65); ax.set_ylim(24, 50)
    ax.set_axis_off()

    tier_names = {"ELI": "Extremely Low Income",
                  "VLI": "Very Low Income", "LI": "Low Income"}
    ax.set_title(
        f'{"State" if actual_level=="state" else "County"}-Level '
        f'{tier_names.get(income_tier, income_tier)} ({income_tier}) '
        f'Housing Deficit Coverage Rate',
        fontsize=12, fontweight="normal", pad=8)

    plt.tight_layout()
    _plot_inset(fig, _AK_INSET, ak, _AK_XLIM, _AK_YLIM,
                column=cov_col, cmap="RdYlGn", vmin=0, vmax=100,
                edge_color=ec, line_width=lw,
                value_col=cov_col if actual_level == "state" else None,
                label="AK")
    _plot_inset(fig, _HI_INSET, hi, _HI_XLIM, _HI_YLIM,
                column=cov_col, cmap="RdYlGn", vmin=0, vmax=100,
                edge_color=ec, line_width=lw,
                value_col=cov_col if actual_level == "state" else None,
                label="HI")

    out = os.path.join(OUTPUT_DIR, f"{actual_level}_coverage_rate_{income_tier}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show(); plt.close()

    print(f"✓ Saved: {out}")
    print(f"  {units_name} w/ Deficit: {n_deficit:,}   Total Deficit: {tot_deficit:,.0f}   "
          f"Households Served: {tot_served:,.0f}   Remaining: {remaining:,.0f}   "
          f"Overall Coverage: {overall_cov:.1f}%")

    return {"level": actual_level, "tier": income_tier,
            "units_with_deficit": n_deficit, "total_deficit": tot_deficit,
            "total_served": tot_served, "overall_coverage": overall_cov}


# =============================================================================
# MAP: SUBSIDY EFFICIENCY
# =============================================================================

def create_subsidy_efficiency_map_any_level(gdf: gpd.GeoDataFrame,
                                             income_tier: str = None,
                                             level: str = "county"):
    """Choropleth of subsidy efficiency (households served per $1M)."""
    plot_gdf, actual_level = get_geo_data(gdf, level)
    label = income_tier if income_tier else "Overall"
    print(f"\nCreating {actual_level.upper()}-LEVEL {label} efficiency map...")

    if "total_served" not in plot_gdf.columns:
        tiers = ["ELI", "VLI", "LI"]
        sc = [f"{t}_served" for t in tiers if f"{t}_served" in plot_gdf.columns]
        plot_gdf["total_served"] = plot_gdf[sc].sum(axis=1) if sc else 0
    if "total_deficit" not in plot_gdf.columns:
        tiers = ["ELI", "VLI", "LI"]
        dc = [f"{t}_deficit" for t in tiers if f"{t}_deficit" in plot_gdf.columns]
        plot_gdf["total_deficit"] = plot_gdf[dc].sum(axis=1) if dc else 0

    def _eff(row):
        ts  = row.get("total_served",      0)
        sub = row.get("total_subsidy_paid", 0)
        if income_tier:
            s  = row.get(f"{income_tier}_served",  0)
            d  = row.get(f"{income_tier}_deficit", 0)
            if d == 0:            return np.nan
            if s == 0:            return 0
            if sub > 0 and ts > 0:
                sub = sub * (s / ts)
            else:
                return np.nan
            return s / (sub / 1_000_000) if sub > 0 else np.nan
        else:
            td = row.get("total_deficit", 0)
            if td == 0:           return np.nan
            if ts == 0:           return 0
            if sub == 0:          return np.nan
            return ts / (sub / 1_000_000)

    eff_col = f"{income_tier}_efficiency" if income_tier else "subsidy_efficiency"
    plot_gdf[eff_col] = plot_gdf.apply(_eff, axis=1)
    nm = plot_gdf[eff_col].notna() & (plot_gdf[eff_col] != 0)

    max_eff = median_eff = 0
    if nm.any():
        ne         = plot_gdf.loc[nm, eff_col]
        max_eff    = ne.max()
        median_eff = ne.median()
        q95        = ne.quantile(0.95)
        print(f"   Max: {max_eff:.1f}   Median: {median_eff:.1f}   "
              f"95th: {q95:.1f}   Top {(plot_gdf[eff_col] >= q95).sum()} {actual_level}s")

    SPEC = {"no_demand": "#D3D3D3", "infeasible": "#FF6B6B",
            "market_viable": "#90EE90"}
    if income_tier:
        dc, sc = f"{income_tier}_deficit", f"{income_tier}_served"
        no_dem = plot_gdf[dc] == 0 if dc in plot_gdf.columns else pd.Series(False, index=plot_gdf.index)
        infeas = ((plot_gdf[dc] > 0) & (plot_gdf[sc] == 0)
                  if dc in plot_gdf.columns and sc in plot_gdf.columns
                  else pd.Series(False, index=plot_gdf.index))
        mktv   = (plot_gdf[eff_col].isna() & (plot_gdf[sc] > 0)
                  if sc in plot_gdf.columns
                  else pd.Series(False, index=plot_gdf.index))
    else:
        no_dem = plot_gdf["total_deficit"] == 0
        infeas = (plot_gdf["total_deficit"] > 0) & (plot_gdf["total_subsidy_paid"] == 0)
        mktv   = (plot_gdf["total_subsidy_paid"] == 0) & (plot_gdf["total_served"] > 0)

    spec_masks = {"no_demand": no_dem, "infeasible": infeas, "market_viable": mktv}
    cont, ak, hi = _split_ak_hi(plot_gdf)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    ec, lw  = ("white", 0.2) if actual_level == "county" else ("black", 0.5)
    cont_nm = nm.reindex(cont.index, fill_value=False)

    if cont_nm.any():
        cont[cont_nm].plot(column=eff_col, ax=ax, cmap="viridis",
                           vmin=0, vmax=max_eff, edgecolor=ec, linewidth=lw,
                           legend=True,
                           legend_kwds={"label": "Households Served per $1M",
                                        "orientation": "vertical",
                                        "shrink": 0.4, "aspect": 15, "pad": 0.02})
    for key, mask in spec_masks.items():
        lm = mask.reindex(cont.index, fill_value=False)
        if lm.any():
            cont[lm].plot(ax=ax, color=SPEC[key], edgecolor=ec, linewidth=lw)

    if actual_level == "state":
        add_state_annotations(ax, cont, value_col=eff_col)

    ax.set_aspect("auto"); ax.set_xlim(-130, -65); ax.set_ylim(24, 50)
    ax.set_axis_off()
    tier_names = {"ELI": "Extremely Low Income",
                  "VLI": "Very Low Income", "LI": "Low Income"}
    t = (f'{tier_names.get(income_tier, income_tier)} ({income_tier}) '
         if income_tier else "Overall ")
    ax.set_title(
        f'{"State" if actual_level=="state" else "County"}-Level '
        f'{t}Efficiency: Households Served per $1M',
        fontsize=12, fontweight="bold", pad=8)

    legend_els = [mpatches.Patch(facecolor=SPEC[k], edgecolor=ec, label=v)
                  for k, v in [("no_demand", "No Demand"),
                                ("infeasible", "Infeasible"),
                                ("market_viable", "Market Viable")]
                  if spec_masks[k].any()]
    if legend_els:
        ax.legend(handles=legend_els, loc="lower left", fontsize=9)

    plt.tight_layout()
    _plot_inset(fig, _AK_INSET, ak, _AK_XLIM, _AK_YLIM,
                column=eff_col, cmap="viridis", vmin=0, vmax=max_eff,
                edge_color=ec, line_width=lw,
                special_masks=spec_masks, special_colors=SPEC,
                value_col=eff_col if actual_level == "state" else None,
                label="AK")
    _plot_inset(fig, _HI_INSET, hi, _HI_XLIM, _HI_YLIM,
                column=eff_col, cmap="viridis", vmin=0, vmax=max_eff,
                edge_color=ec, line_width=lw,
                special_masks=spec_masks, special_colors=SPEC,
                value_col=eff_col if actual_level == "state" else None,
                label="HI")

    suf = f"{income_tier}_efficiency" if income_tier else "subsidy_efficiency"
    out = os.path.join(OUTPUT_DIR, f"{actual_level}_{suf}_map.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show(); plt.close()

    ms  = f"{max_eff:.1f}" if nm.any() else "N/A"
    mds = f"{median_eff:.1f}" if nm.any() else "N/A"
    print(f"✓ Saved: {out}")
    print(f"  {label} Efficiency (HH/$1M) — Max: {ms}   Median: {mds}   "
          f"No Demand: {no_dem.sum()}   Infeasible: {infeas.sum()}   "
          f"Market Viable: {mktv.sum()}")


# =============================================================================
# CHART: HOUSING-TYPE MIX
# =============================================================================

def create_unit_type_mix_stacked_bar(gdf: gpd.GeoDataFrame,
                                      level: str = "county"):
    """Stacked bar chart of unit-type distribution by income group."""
    plot_gdf, actual_level = get_geo_data(gdf, level)
    unit_cols    = [c for c in plot_gdf.columns
                    if c.startswith("Type") and "units_built" in c]
    housing_types = sorted(set(c.split("_")[0] for c in unit_cols))
    if not housing_types:
        print(f"  No unit type data found for {level}-level analysis")
        return None

    print(f"\nCreating {actual_level.upper()}-LEVEL unit type mix chart...")
    income_tiers = ["ELI", "VLI", "LI"]
    tier_labels  = {"ELI": "ELI\n(Extremely Low)",
                    "VLI": "VLI\n(Very Low)", "LI": "LI\n(Low)"}

    mat = np.zeros((len(income_tiers), len(housing_types)))
    for _, row in plot_gdf.iterrows():
        for hi, ht in enumerate(housing_types):
            uc = f"{ht}_units_built"
            if uc in row and row[uc] > 0:
                units = row[uc]
                tot   = sum(row.get(f"{t}_served", 0) for t in income_tiers)
                if tot > 0:
                    for ti, t in enumerate(income_tiers):
                        mat[ti, hi] += units * row.get(f"{t}_served", 0) / tot
                else:
                    for ti in range(len(income_tiers)):
                        mat[ti, hi] += units / len(income_tiers)

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.2))
    colors  = plt.cm.Set2(np.linspace(0, 1, len(housing_types)))
    xpos    = np.arange(len(income_tiers))
    bottom  = np.zeros(len(income_tiers))

    for hi, ht in enumerate(housing_types):
        ax.bar(xpos, mat[:, hi], 0.7, bottom=bottom,
               color=colors[hi], label=ht, edgecolor="white")
        bottom += mat[:, hi]

    ax.set_xticks(xpos)
    ax.set_xticklabels([tier_labels[t] for t in income_tiers])
    ax.set_xlabel("Income Group", fontsize=11)
    ax.set_ylabel("Number of Units", fontsize=11)
    ax.set_ylim(0, mat.sum(axis=1).max() * 1.15)

    for i in range(len(income_tiers)):
        tot = mat[i].sum(); cum = 0
        if tot > 0:
            for hi in range(len(housing_types)):
                v = mat[i, hi]
                if v > 0:
                    pct = v / tot * 100
                    if pct > 5:
                        ax.text(xpos[i], cum + v / 2, f"{pct:.0f}%",
                                ha="center", va="center", fontsize=9,
                                color="white", fontweight="normal")
                    cum += v
            ax.text(xpos[i], tot * 1.02, f"{tot:,.0f}",
                    ha="center", va="bottom", fontsize=10)

    total = sum(plot_gdf[f"{ht}_units_built"].sum() for ht in housing_types
                if f"{ht}_units_built" in plot_gdf.columns)
    ax.set_title(
        f"{actual_level.capitalize()}-Level Unit Type Distribution by Income Group\n"
        f"Total Units: {total:,.0f}",
        fontsize=12, fontweight="bold", pad=8)
    ax.legend(title="Unit Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, f"{actual_level}_unit_type_mix_stacked.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show(); plt.close()

    tier_str = "   ".join(f"{t}: {mat[ti].sum():,.0f} units"
                          for ti, t in enumerate(income_tiers))
    print(f"✓ Saved: {out}")
    print(f"  {tier_str}   |   Total: {total:,.0f}")


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def generate_all_maps_at_level(gdf: gpd.GeoDataFrame, level: str = "county"):
    """Generate all maps and charts at the specified level."""
    print(f"\n{'='*70}")
    print(f"GENERATING ALL {level.upper()}-LEVEL OUTPUTS")
    print("=" * 70)

    stats_list = []
    for tier in ["ELI", "VLI", "LI"]:
        s = create_coverage_map_any_level(gdf, tier, level=level)
        if s:
            stats_list.append(s)

    create_subsidy_efficiency_map_any_level(gdf, level=level)
    create_unit_type_mix_stacked_bar(gdf, level=level)

    for tier in ["ELI", "VLI", "LI"]:
        create_subsidy_efficiency_map_any_level(gdf, income_tier=tier, level=level)

    if stats_list:
        out = os.path.join(OUTPUT_DIR, f"{level}_coverage_summary.csv")
        pd.DataFrame(stats_list).to_csv(out, index=False)
        print(f"\n✓ Saved coverage summary → {out}")
