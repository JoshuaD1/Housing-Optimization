# =============================================================================
# housing_model.py
#
# Subsidized Housing Coverage Optimization Model
#
# Solves a county-level integer linear program (ILP) to maximize the number
# of low-income households served subject to a per-county subsidy budget.
# Four housing types (room, studio, 1BR, 2BR) are available across four
# income tiers (ELI, VLI, LI, NLI).  Construction costs are scaled by a
# state-level RS Means cost index.
#
# Usage (local):
#   python src/housing_model.py
#
# Usage (Google Colab):
#   See notebooks/housing_model_colab.ipynb
# =============================================================================

import os
import re
import time
import argparse

import pandas as pd
import numpy as np
from pulp import (
    LpProblem, LpMaximize, LpVariable, LpStatus,
    lpSum, value, PULP_CBC_CMD
)

# =============================================================================
# PATHS  (override via CLI flags or environment variables)
# =============================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "..", "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "..", "outputs")

INPUT_CSV   = os.path.join(DATA_DIR,   "County_CHAS_with_FMR_RSMeans.csv")
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "coverage_results.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# CONFIGURATION
# =============================================================================
SCENARIO_TAG       = "Base"
BUDGET_PER_COUNTY  = 5_000_000   # $ per county

INCOME_GROUPS = ["ELI", "VLI", "LI", "NLI"]
HOUSING_TYPES = [1, 2, 3, 4]
RENT_BURDEN_PCT = 0.30   # θ — max rent as share of income
AVG_INCOME_PCT  = 0.60   # α — income-mix constraint

INCOME_PARAMS = {
    "ELI": {"rho": 0.05, "hamfi_pct": 0.30, "default_income": 20_000},
    "VLI": {"rho": 0.05, "hamfi_pct": 0.50, "default_income": 35_000},
    "LI":  {"rho": 0.05, "hamfi_pct": 0.80, "default_income": 50_000},
    "NLI": {"rho": 0.05, "hamfi_pct": 1.00, "default_income": 70_000},
}

# National Fair Market Rents (monthly, $)
NATIONAL_FMR = {
    "FMR_Room":   875,
    "FMR_Studio": 1_000,
    "FMR_1BR":    1_200,
    "FMR_2BR":    1_450,
}

HOUSING_PARAMS = {
    1: {"Cap": 180_000, "LTV": 0.7, "r": 0.05, "T": 30, "Op": 2_800,
        "Type": "FMR_Room",   "unit_weight": 1.0,
        "annual_rent": 12 * NATIONAL_FMR["FMR_Room"]},
    2: {"Cap": 220_000, "LTV": 0.7, "r": 0.05, "T": 30, "Op": 3_200,
        "Type": "FMR_Studio", "unit_weight": 1.0,
        "annual_rent": 12 * NATIONAL_FMR["FMR_Studio"]},
    3: {"Cap": 280_000, "LTV": 0.7, "r": 0.05, "T": 30, "Op": 3_800,
        "Type": "FMR_1BR",    "unit_weight": 1.0,
        "annual_rent": 12 * NATIONAL_FMR["FMR_1BR"]},
    4: {"Cap": 350_000, "LTV": 0.7, "r": 0.05, "T": 30, "Op": 4_500,
        "Type": "FMR_2BR",    "unit_weight": 1.0,
        "annual_rent": 12 * NATIONAL_FMR["FMR_2BR"]},
}

# =============================================================================
# HELPERS
# =============================================================================

def clean_county_name(county_name: str) -> str:
    """Sanitise county name for use as a PuLP problem identifier."""
    if pd.isna(county_name):
        return "Unknown_County"
    cleaned = re.sub(r"[^\w]", "_", str(county_name))
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "County"


def compute_financial_metrics(state_multiplier: float = 1.0) -> dict:
    """
    Return per-housing-type financial metrics scaled by the state RS Means index.

    Parameters
    ----------
    state_multiplier : float
        RS Means cost index for the state (1.0 = national average).

    Returns
    -------
    dict  {housing_type: {Loan, Equity, PMT, Op, Cap}}
    """
    fin = {}
    for h, p in HOUSING_PARAMS.items():
        cap   = state_multiplier * p["Cap"]
        loan  = p["LTV"] * cap
        eq    = (1 - p["LTV"]) * cap
        r, T  = p["r"], p["T"]
        pmt   = loan * (r * (1 + r) ** T) / ((1 + r) ** T - 1)
        fin[h] = {"Loan": loan, "Equity": eq, "PMT": pmt,
                  "Op": p["Op"], "Cap": cap}
    return fin


def prepare_income_data(county_row: pd.Series) -> dict:
    """
    Extract demand (D), affordable-and-available units (AA), and HAMFI
    from a county row.  Incremental AA calculation matches CHAS methodology.
    """
    def _get(col, default=0):
        v = county_row.get(col, default)
        return float(v) if pd.notna(v) else float(default)

    eli_aa = _get("Affordable and Available Units At or Below 30% HAMFI (ELI)")
    vli_aa = _get("Affordable and Available Units At or Below 50% HAMFI (VLI)")
    li_aa  = _get("Units Affordable and Available to LI Renter Households")

    return {
        "ELI": {
            "D":     _get("ELI Renter Households"),
            "AA":    max(0, eli_aa),
            "HAMFI": _get("ELI HAMFI", INCOME_PARAMS["ELI"]["default_income"]),
        },
        "VLI": {
            "D":     _get("VLI Renter Households"),
            "AA":    max(0, vli_aa - eli_aa),
            "HAMFI": _get("VLI HAMFI", INCOME_PARAMS["VLI"]["default_income"]),
        },
        "LI": {
            "D":     _get("LI Renter Households"),
            "AA":    max(0, li_aa - vli_aa),
            "HAMFI": _get("LI HAMFI", INCOME_PARAMS["LI"]["default_income"]),
        },
        "NLI": {
            "D":     _get("NLI Renter Households"),
            "AA":    float("inf"),
            "HAMFI": _get("NLI HAMFI", INCOME_PARAMS["NLI"]["default_income"]),
        },
    }


# =============================================================================
# CORE ILP MODEL
# =============================================================================

def solve_coverage_model(county_row: pd.Series,
                         county_name: str,
                         budget: float,
                         state_multiplier: float = 1.0) -> dict:
    """
    Solve the housing coverage ILP for a single county.

    Decision variables
    ------------------
    x[i][h]       : integer — households of income group i placed in type h
    y[i][h]       : binary  — indicator (x[i][h] > 0)
    s[h]          : integer — total units of type h built
    subcost[i][h] : continuous — subsidy cost for group i in type h

    Objective (Eq. 1)
    -----------------
    Maximise  Σ_i Σ_h ω_h · x[i][h]

    Key constraints
    ---------------
    Eq. 2  Income-mix:  Σ β_i · x[i][h]  ≤  α · Σ x[i][h]
    Eq. 3  Demand cap:  Σ_h x[i][h]      ≤  D_i − AA_i
    Eq. 4  Unit count:  Σ_i x[i][h]      =  s[h]
    Eq. 7  Subsidy lb:  subcost[i][h]    ≥  (Op + PMT + ρ·Equity − θ·HAMFI) · x[i][h]
    Eq. 8  Indicator:   x[i][h]          ≤  D_i · y[i][h]
    Eq. 12 Budget:      Σ_i Σ_h subcost[i][h] ≤ B

    Returns
    -------
    dict with optimisation results and household/unit breakdowns.
    """
    income_data    = prepare_income_data(county_row)
    deficits       = {i: max(0, income_data[i]["D"] - income_data[i]["AA"])
                      for i in INCOME_GROUPS}
    rent_max       = {i: RENT_BURDEN_PCT * income_data[i]["HAMFI"]
                      for i in INCOME_GROUPS}
    FIN            = compute_financial_metrics(state_multiplier)
    fmr_annual     = {h: HOUSING_PARAMS[h]["annual_rent"] for h in HOUSING_TYPES}

    # Short-circuit: no demand
    if sum(deficits.values()) == 0:
        return _no_demand_result(county_name, county_row, deficits,
                                 state_multiplier)

    prob = LpProblem(f"Coverage_{clean_county_name(county_name)}", LpMaximize)

    x       = LpVariable.dicts("x",       (INCOME_GROUPS, HOUSING_TYPES),
                                lowBound=0, cat="Integer")
    y       = LpVariable.dicts("y",       (INCOME_GROUPS, HOUSING_TYPES),
                                cat="Binary")
    s       = LpVariable.dicts("s",       HOUSING_TYPES,
                                lowBound=0, cat="Integer")
    subcost = LpVariable.dicts("subcost", (INCOME_GROUPS, HOUSING_TYPES),
                                lowBound=0, cat="Continuous")

    # Objective
    prob += lpSum(HOUSING_PARAMS[h]["unit_weight"] * x[i][h]
                  for i in INCOME_GROUPS for h in HOUSING_TYPES)

    # Eq. 2 — income-mix constraint
    prob += (lpSum(INCOME_PARAMS[i]["hamfi_pct"] * x[i][h]
                   for i in INCOME_GROUPS for h in HOUSING_TYPES)
             <= AVG_INCOME_PCT *
             lpSum(x[i][h] for i in INCOME_GROUPS for h in HOUSING_TYPES))

    # Eqs. 3, 4, 7, 8, 12
    for i in INCOME_GROUPS:
        prob += lpSum(x[i][h] for h in HOUSING_TYPES) <= deficits[i]   # Eq. 3
        rho_i = INCOME_PARAMS[i]["rho"]
        D_i   = income_data[i]["D"]
        for h in HOUSING_TYPES:
            fm       = FIN[h]
            required = fm["Op"] + fm["PMT"] + rho_i * fm["Equity"]
            prob += subcost[i][h] >= max(0, required - rent_max[i]) * x[i][h]  # Eq. 7
            prob += x[i][h] <= D_i * y[i][h]                                   # Eq. 8

    for h in HOUSING_TYPES:
        prob += lpSum(x[i][h] for i in INCOME_GROUPS) == s[h]          # Eq. 4

    prob += lpSum(subcost[i][h]
                  for i in INCOME_GROUPS
                  for h in HOUSING_TYPES) <= budget                     # Eq. 12

    # Solve
    try:
        prob.solve(PULP_CBC_CMD(msg=False, timeLimit=10))
    except Exception:
        try:
            prob.solve(PULP_CBC_CMD(msg=False))
        except Exception:
            return _error_result(county_name, county_row, deficits,
                                 state_multiplier)

    result = {
        "county":            county_name,
        "state":             county_row.get("state", ""),
        "status":            LpStatus.get(prob.status, "Unknown"),
        "state_multiplier":  state_multiplier,
        "objective":         value(prob.objective) if prob.status == 1 else 0,
        "total_served":      0,
        "total_units_built": 0,
        "total_subsidy_paid": 0,
        "total_deficit":     sum(deficits.values()),
    }

    if prob.status == 1:
        result["total_units_built"]  = sum(value(s[h]) or 0
                                           for h in HOUSING_TYPES)
        result["total_subsidy_paid"] = sum(value(subcost[i][h]) or 0
                                           for i in INCOME_GROUPS
                                           for h in HOUSING_TYPES)
        total_served = 0
        for i in INCOME_GROUPS:
            served = sum(value(x[i][h]) or 0 for h in HOUSING_TYPES)
            result[f"{i}_served"]       = served
            result[f"{i}_deficit"]      = deficits[i]
            result[f"{i}_coverage_pct"] = (served / deficits[i] * 100
                                           if deficits[i] > 0 else 0)
            total_served += served
        result["total_served"] = total_served

        for h in HOUSING_TYPES:
            units_h      = value(s[h]) or 0
            total_rent_h = 0
            total_sub_h  = 0
            for i in INCOME_GROUPS:
                x_val = value(x[i][h]) or 0
                if x_val > 0:
                    total_rent_h += RENT_BURDEN_PCT * income_data[i]["HAMFI"] * x_val
                    total_sub_h  += value(subcost[i][h]) or 0
            result[f"Type{h}_units_built"]    = units_h
            result[f"Type{h}_rent_annual"]    = (total_rent_h / units_h
                                                 if units_h > 0 else 0)
            result[f"Type{h}_subsidy_annual"] = (total_sub_h / units_h
                                                 if units_h > 0 else 0)

        if result["total_units_built"] > 0:
            result["subsidy_per_unit"] = (result["total_subsidy_paid"] /
                                          result["total_units_built"])
        if result["total_subsidy_paid"] > 0:
            result["subsidy_efficiency"] = (result["total_served"] /
                                            (result["total_subsidy_paid"] / 1_000_000))
            result["private_leverage_ratio"] = (
                sum(FIN[h]["Equity"] * (value(s[h]) or 0) for h in HOUSING_TYPES)
                / result["total_subsidy_paid"])
    else:
        for i in INCOME_GROUPS:
            result[f"{i}_served"]       = 0
            result[f"{i}_deficit"]      = deficits[i]
            result[f"{i}_coverage_pct"] = 0
        for h in HOUSING_TYPES:
            result[f"Type{h}_units_built"]    = 0
            result[f"Type{h}_rent_annual"]    = 0
            result[f"Type{h}_subsidy_annual"] = 0

    return result


def _no_demand_result(county_name, county_row, deficits, state_multiplier):
    r = {"county": county_name, "state": county_row.get("state", ""),
         "status": "No_Demand", "state_multiplier": state_multiplier,
         "objective": 0, "total_served": 0, "total_units_built": 0,
         "total_subsidy_paid": 0, "total_deficit": sum(deficits.values())}
    for i in INCOME_GROUPS:
        r.update({f"{i}_served": 0, f"{i}_deficit": deficits[i],
                  f"{i}_coverage_pct": 0})
    for h in HOUSING_TYPES:
        r.update({f"Type{h}_units_built": 0, f"Type{h}_rent_annual": 0,
                  f"Type{h}_subsidy_annual": 0})
    return r


def _error_result(county_name, county_row, deficits, state_multiplier):
    r = _no_demand_result(county_name, county_row, deficits, state_multiplier)
    r["status"] = "Solver_Error"
    return r


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run(input_csv: str = INPUT_CSV,
        output_csv: str = OUTPUT_CSV,
        budget: float = BUDGET_PER_COUNTY) -> pd.DataFrame:
    """
    Run the model for every county in input_csv and save results to output_csv.

    Parameters
    ----------
    input_csv  : path to County_CHAS_with_FMR_RSMeans.csv
    output_csv : path for results output
    budget     : per-county subsidy budget ($)
    """
    print("=" * 70)
    print("HOUSING COVERAGE OPTIMIZATION MODEL")
    print("=" * 70)
    print(f"  Input  : {input_csv}")
    print(f"  Output : {output_csv}")
    print(f"  Budget : ${budget:,.0f} per county")
    print()

    df = pd.read_csv(input_csv)
    print(f"✓ Loaded {len(df):,} counties")

    results = []
    for idx, row in df.iterrows():
        county       = row.get("county", f"County_{idx}")
        state_mult   = float(row.get("2020 Weighted RS Means Index", 1.0) or 1.0)
        try:
            res = solve_coverage_model(row, county, budget, state_mult)
            results.append(res)
            if res["status"] not in ("Optimal", "No_Demand"):
                print(f"  {county}: {res['status']}")
        except Exception as e:
            print(f"  ✗ {county}: {str(e)[:60]}")
            results.append({"county": county, "status": "Error",
                             "state_multiplier": state_mult})

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved → {output_csv}")

    # Summary
    optimal    = results_df[results_df["status"] == "Optimal"]
    infeasible = results_df[results_df["status"] == "Infeasible"]
    no_demand  = results_df[results_df["status"] == "No_Demand"]
    print(f"\nMODEL STATUS:")
    print(f"  Optimal    : {len(optimal):,}  ({len(optimal)/len(results_df)*100:.1f}%)")
    print(f"  Infeasible : {len(infeasible):,}  ({len(infeasible)/len(results_df)*100:.1f}%)")
    print(f"  No Demand  : {len(no_demand):,}  ({len(no_demand)/len(results_df)*100:.1f}%)")

    if len(optimal) > 0:
        ts  = optimal["total_served"].sum()
        td  = optimal["total_deficit"].sum()
        sub = optimal["total_subsidy_paid"].sum()
        print(f"\nPERFORMANCE (optimal counties):")
        print(f"  Households served : {ts:,.0f}")
        print(f"  Coverage rate     : {ts/td*100:.1f}%" if td > 0 else "  Coverage rate: N/A")
        print(f"  Total subsidy     : ${sub:,.0f}")
        print(f"\nCOVERAGE BY INCOME GROUP:")
        for i in INCOME_GROUPS:
            if f"{i}_served" in optimal.columns:
                srv = optimal[f"{i}_served"].sum()
                dft = optimal[f"{i}_deficit"].sum()
                print(f"  {i}: {srv:,.0f}/{dft:,.0f} ({srv/dft*100:.1f}%)" if dft > 0
                      else f"  {i}: N/A")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Housing Coverage Optimization Model"
    )
    parser.add_argument("--input",  default=INPUT_CSV,
                        help="Path to input CSV (default: data/County_CHAS_with_FMR_RSMeans.csv)")
    parser.add_argument("--output", default=OUTPUT_CSV,
                        help="Path for results CSV (default: outputs/coverage_results.csv)")
    parser.add_argument("--budget", type=float, default=BUDGET_PER_COUNTY,
                        help=f"Per-county budget in $ (default: {BUDGET_PER_COUNTY:,})")
    args = parser.parse_args()
    run(args.input, args.output, args.budget)
