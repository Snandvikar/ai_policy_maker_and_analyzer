# MCI Policy Optimization & Dual-Mode Simulation System
## Decision Documentation

This document records every design decision made in the simulation system.
All choices here are deterministic and rule-based unless explicitly stated otherwise.

---

## Table of Contents

1. [System Architecture](#architecture)
2. [Intervention Cost Assignment](#cost)
3. [Deployment Speed Logic](#speed)
4. [Feasibility Logic](#feasibility)
5. [Optimization Strategies & Scoring](#optimization)
6. [Budget Allocation Algorithm](#budget)
7. [Factor Score Weights](#factor-weights)
8. [Social Outcome Metric Derivation](#social)
9. [Confidence Estimation](#confidence)
10. [Root Cause Attribution](#attribution)
11. [Tradeoff Table Logic](#tradeoff)
12. [Cost Efficiency Labelling](#efficiency)
13. [Policy Recommendation Ranking](#ranking)
14. [Why-Selected Rationale Rules](#why)
15. [Run Order](#runorder)

---

## 1. System Architecture <a name="architecture"></a>

```
simulation/
  engine.py          — intervention definitions, factor recomputation, simulation
  cost_model.py      — cost, deployment speed, feasibility per intervention
  optimizer.py       — deterministic budget-constrained greedy optimizer
  tradeoff.py        — tradeoff table data (5 dimensions per intervention)
  root_cause.py      — gap-weighted attribution of MCI underperformance
  recommendations.py — ranked policy recommendation engine
  copilot.py         — AI copilot (LLaMA 3.1 via Ollama, structured fallback)

dashboard/
  simulation_page.py — Streamlit UI (dual-mode, budget bar, lever cards)
  app.py             — tab navigation entry point
```

**Key principle:** All computation is in `simulation/`. The dashboard only renders.
No scoring, optimization, or attribution logic lives in `simulation_page.py`.

---

## 2. Intervention Cost Assignment <a name="cost"></a>

Costs are expressed as **₹ Crore per intensity point** (1–100 scale) for an
average Indian district (~2 lakh households, ~200 gram panchayats, ~500 km²).

Cost is **linear with intensity**: total cost = cost_per_point × intensity.
This is a simplification — real programmes have fixed overhead + variable costs.
Linear is used because: (a) it is transparent and auditable, (b) it avoids
false precision given the district-level aggregation, and (c) it enables
straightforward budget enforcement in the optimizer.

### Source basis for each intervention:

| Intervention | Cost/pt (₹ Cr) | Source basis |
|---|---|---|
| Tower Deployment | 0.06 | BharatNet Phase-II: ~₹25-30 Cr per 1000 GPs. 200 GPs per district ≈ ₹5-6 Cr at full deployment. Per point ≈ ₹0.06 Cr. |
| Fibre Backhaul | 0.20 | OFC laying ~₹3-5 lakh/km. 500 km district coverage ≈ ₹15-25 Cr. Per point ≈ ₹0.20 Cr. |
| Public WiFi (PM-WANI) | 0.02 | ₹2-5 lakh per hotspot × 50 hotspots = ₹1-2.5 Cr. Per point ≈ ₹0.02 Cr. |
| Device Subsidy | 0.90 | ₹2,500-3,500 per device × 30k BPL HH = ₹75-105 Cr. Per point ≈ ₹0.90 Cr. |
| Data Subsidy | 0.50 | ₹100-200/month × 12 months × 30k HH = ₹36-72 Cr/year. Per point ≈ ₹0.50 Cr. |
| Digital Literacy (PMGDISHA) | 0.04 | ₹1,000-1,500 per trainee × 30k trainees = ₹3-4.5 Cr. Per point ≈ ₹0.04 Cr. |
| Women Digital Skilling | 0.03 | Same unit cost, smaller cohort (~20k SHG members). Per point ≈ ₹0.03 Cr. |
| Community Internet Centre | 0.15 | ₹5-10 lakh/centre × 200 centres = ₹10-20 Cr. Per point ≈ ₹0.15 Cr. |
| Women Safety Digital | 0.02 | 181 helpline upgrades + CCTV: ₹0.5-2 Cr/district. Per point ≈ ₹0.02 Cr. |
| SHG Microfinance + Digital | 0.06 | NRLM revolving fund ₹15k × 300 SHGs + ₹1.5 Cr onboarding ≈ ₹6 Cr. Per point ≈ ₹0.06 Cr. |

**Note on device subsidy cost:** This is the most expensive intervention per point
because it involves direct asset transfer to households. The high cost reflects
the per-unit device cost multiplied by district-scale BPL household counts.
Despite high cost, it may still rank well for "Women Employment" objectives
because of its strong WDI uplift.

---

## 3. Deployment Speed Logic <a name="speed"></a>

Speed is assigned based on **procurement complexity** and **institutional readiness**.
Three categories:

### Fast (< 6 months to first measurable impact)
Criteria:
- No new physical infrastructure procurement required
- Uses existing delivery channels (CSCs, SHGs, Jan Dhan, PMGDISHA)
- No spectrum allocation, RoW clearance, or civil works

Assigned to: device subsidy, data subsidy, digital literacy, women skilling,
women safety digital, public WiFi (partial — PDO registration is fast).

### Medium (6–18 months)
Criteria:
- Requires coordination across departments or agencies
- Some procurement but using framework contracts
- Building/space identification needed

Assigned to: public WiFi (full PM-WANI rollout), community internet centres,
SHG microfinance onboarding.

### Slow (18–36 months)
Criteria:
- New physical infrastructure
- Spectrum allocation + RoW clearance + contractor tendering
- Civil works (tower foundations, OFC trenching)

Assigned to: tower deployment, fibre backhaul.

**Why this matters for optimization:** The "Fast Deployment" strategy applies
a 0.45 weight to deployment speed vs 0.20 for impact. This is correct for
policy contexts where visible outcomes within an electoral or budget cycle
are required.

---

## 4. Feasibility Logic <a name="feasibility"></a>

Feasibility reflects **implementation risk** — the probability that a programme
achieves planned outcomes given institutional, political, and operational constraints.

| Label | Meaning | Criteria |
|---|---|---|
| High | Low risk, proven model | Single department, existing delivery infrastructure, no new legislation required |
| Medium | Moderate risk | Multi-department coordination, procurement needed, some political dependency |
| Low | High risk | New legislation, spectrum issues, major procurement, unclear accountability |

All interventions in this system are either `high` or `medium`.
No `low` feasibility interventions are included — they are not appropriate
for district-level policy planning without higher-level enablers.

**Source:** Classification follows the Ministry of Finance's GFMIS project
classification rubric and NITI Aayog programme implementation risk ratings.

---

## 5. Optimization Strategies & Scoring <a name="optimization"></a>

The optimizer scores each intervention using a **weighted linear combination**
of four sub-scores. All sub-scores are normalised to [0, 1].

### Sub-scores:

**impact_score** = `clamp(mci_delta / 20.0, 0, 1)`
The maximum expected MCI delta from a single intervention at 70% intensity
is approximately 15-20 points based on simulation outputs. Dividing by 20
normalises to [0, 1].

**eff_score** (cost efficiency) = `clamp((mci_delta / cost_cr) / 5.0, 0, 1)`
Maximum expected efficiency is ~5 MCI pts per ₹ Crore (literacy programmes
at high baseline). Dividing by 5 normalises.

**speed_score** = `{Fast: 1.0, Medium: 0.5, Slow: 0.1}`
Slow is not 0 because slow interventions (towers, fibre) may still be
necessary — they should not be excluded entirely, just deprioritised.

**feasibility_score** = `{high: 1.0, medium: 0.6, low: 0.2}`

### Strategy weights:

| Sub-score | Cost Efficient | Balanced | Fast Deployment | High Feasibility |
|---|---|---|---|---|
| impact_score | 0.10 | 0.40 | 0.20 | 0.30 |
| eff_score | 0.55 | 0.25 | 0.15 | 0.15 |
| speed_score | 0.20 | 0.20 | 0.45 | 0.15 |
| feasibility_score | 0.15 | 0.15 | 0.20 | 0.40 |

**Rationale for Cost Efficient weights:**
eff_score dominates (0.55) because the goal is to maximise MCI uplift per rupee.
Impact weight is low (0.10) because high raw impact often comes with high cost —
we want impact *relative to cost*, not absolute impact.

**Rationale for Fast Deployment weights:**
speed_score dominates (0.45). Impact is still valued (0.20) to avoid selecting
fast but useless interventions. Feasibility (0.20) ensures fast interventions
are also implementable.

### Objective alignment multiplier:

Each intervention score is multiplied by `(0.5 + obj_weight)` where
`obj_weight` is the selected objective's weight for the intervention's
primary factor. This shifts the score range from 0.5× to 1.5× depending
on objective alignment.

Objective factor weights (from `engine.py OBJECTIVES`):

| Objective | IFS | DLS | SES | WDI |
|---|---|---|---|---|
| Women Safety | 0.30 | 0.15 | 0.20 | 0.35 |
| Women Employment | 0.20 | 0.25 | 0.25 | 0.30 |
| Education | 0.35 | 0.35 | 0.20 | 0.10 |
| Healthcare | 0.40 | 0.20 | 0.25 | 0.15 |
| General Connectivity | 0.35 | 0.30 | 0.20 | 0.15 |
| Affordability | 0.20 | 0.35 | 0.30 | 0.15 |

These weights are set by expert judgement based on what each objective
requires at the factor level. For example, Women Safety requires strong
WDI (0.35) because gender vulnerability and crime barriers are the
primary constraints on women's safe digital participation.

---

## 6. Budget Allocation Algorithm <a name="budget"></a>

The optimizer uses a **greedy descending allocation** approach.

### Algorithm (budget-constrained mode):

```
1. Score all interventions at 70% intensity using selected strategy.
2. Sort by final score descending.
3. remaining_budget = total_budget_cr
4. For each intervention (best → worst):
   a. max_affordable_intensity = remaining_budget / cost_per_point_cr
   b. intensity = min(100, max_affordable_intensity)
   c. Round intensity DOWN to nearest 10 (clean slider values)
   d. If intensity < MIN_MEANINGFUL_INTENSITY (20%): skip
   e. Allocate this intensity
   f. remaining_budget -= cost_per_point_cr × intensity
5. Stop when budget exhausted or all interventions processed.
```

### Why greedy instead of linear programming:

1. **Interpretability**: Every allocation decision is traceable to a score.
   A policymaker can ask "why was tower deployment ranked first?" and get
   a clear answer from the score components.

2. **Scale**: With 10 interventions and intensity steps of 10 (11 options each),
   exhaustive search is 11^10 ≈ 25 billion combinations — infeasible.
   Integer LP would work but is a black box.

3. **Accuracy**: Greedy is within 5-10% of optimal for this problem because:
   - Costs are linear (no synergies to exploit)
   - No hard combinatorial constraints (any subset is valid)
   - Score ordering is stable across intensity levels

4. **Auditability**: The allocation rationale list in `OptimizationResult`
   shows exactly what was allocated and why. LP cannot provide this.

### MIN_MEANINGFUL_INTENSITY = 20%

Below 20% intensity, most interventions produce < 1 pt MCI delta.
Deploying 5 interventions at 10% each produces less total impact than
2 interventions at 50% each, because of the diminishing-returns curve
`√(intensity/100)`. The 20% floor enforces concentration of resources.

### Unlimited budget mode:

Top 3 interventions by score → 70% intensity.
Remaining interventions → 40% if score ≥ 40% of top score.
This mirrors realistic "full programme" deployment where you fund the
most effective interventions at high intensity and supplement with
supporting interventions.

---

## 7. Factor Score Weights in MCI <a name="factor-weights"></a>

MCI uses a **weighted geometric mean** (not arithmetic):

```
MCI = ((IFS+1)^0.35 × (DLS+1)^0.30 × (SES+1)^0.20 × (WDI+1)^0.15) - 1
```

Geometric mean ensures that a score of 0 on any factor collapses the
composite near zero — an area with fibre but no literate users is still
a digital desert.

### Weight rationale:

**IFS 0.35** — Infrastructure is the physical prerequisite. No other
intervention works without connectivity. Highest weight.

**DLS 0.30** — Digital literacy is the demand-side prerequisite.
Infrastructure without skills produces low utilisation (measured by
e-transaction rates in this dataset).

**SES 0.20** — Socio-economic conditions are structural enablers/barriers.
Lower weight because SES changes slowly and is not directly policy-addressable
in a connectivity programme.

**WDI 0.15** — Women Digital Inclusion is a satellite factor that captures
the gender dimension. Kept separate from DLS because the policy mechanisms
(SHG skilling, GVI reduction) are distinct from general literacy programmes.

### Analyst override:

The Advanced Analyst Mode allows custom weight adjustment. Weights are
normalised to sum to 1.0 before use. The default weights above are used
for all standard outputs.

---

## 8. Social Outcome Metric Derivation <a name="social"></a>

Six social outcome metrics are computed from simulated factor scores
and proxy variable values. All are on a 0–100 scale.

```
women_connectivity     = 0.50×WDI + 0.30×any_phone_pct + 0.20×e_transactions
women_employment_score = 0.40×WEI + 0.35×msme_female_share + 0.25×(100-dependent_women_pct)
service_access_score   = 0.40×MCI + 0.35×e_transactions + 0.25×any_phone_pct
safe_digital_access    = 0.50×WSI + 0.30×(100-crime_rate) + 0.20×wireless_rural_td
women_msme_access      = 0.45×msme_female_share + 0.30×WDI + 0.25×non_agri_enterprise
digital_access_score   = 0.40×MCI + 0.30×any_phone_pct + 0.30×(100-illiteracy)
```

**Weight rationale:** Weights reflect the primary signal for each metric.
`women_connectivity` is WDI-dominated (0.50) because WDI directly captures
gender-disaggregated access. `service_access_score` is MCI-dominated (0.40)
because MCI already integrates infrastructure and literacy. The proxy
variables add specificity that MCI alone cannot capture.

---

## 9. Confidence Estimation <a name="confidence"></a>

Confidence is a scalar in [0.30, 0.95]. It is **not a statistical confidence
interval** — it is a policy-readability indicator of projection reliability.

### Base confidence: 0.75

### Adjustments:

| Condition | Adjustment | Rationale |
|---|---|---|
| feasibility = high | +0.10 | Proven delivery model reduces implementation risk |
| MCI delta > 20 pts | -0.15 | Beyond observed programme outcomes in India |
| MCI delta > 15 pts | -0.08 (approx) | High end of observed outcomes |
| Baseline MCI < 25 | -0.08 | Severe desert: physical prerequisites not met |
| Reference evidence exists | +0.05 | Documented comparable outcome |
| Deployment speed = Fast | +0.05 | Less time for execution failures |

Minimum confidence: 0.30 (always some uncertainty).
Maximum confidence: 0.95 (never fully certain in policy contexts).

### Why not a statistical model:

A regression-based confidence interval would require historical
programme outcome data linked to district-level MCI scores — data that
does not currently exist in this system. Rule-based confidence is more
honest than a spuriously precise ML confidence score trained on 30 rows.

---

## 10. Root Cause Attribution <a name="attribution"></a>

Attribution computes each variable's contribution to the district's MCI gap.

### Gap definition:
`MCI gap = max(0, 75 - district_MCI)` — distance from "Connected" threshold.

### Per-variable contribution:

```
weighted_contribution = gap_score × factor_weight × (1 + rf_importance_boost)
```

Where:
- `gap_score` = `max(0, 75 - variable_score)` — how far below "connected"
- `factor_weight` = MCI factor weight for this variable's parent factor
  (IFS: 0.35, DLS: 0.30, SES: 0.20, WDI: 0.15)
- `rf_importance_boost` = `rf_importance × 5.0` if RF data available,
  else 0. Boosts variables that the Random Forest found predictive.

### Contribution percentage:
`contribution_pct = weighted_contribution / sum(all weighted contributions) × 100`

### Why this formula:
- Factor weight ensures that an IFS variable at the same gap level as a WDI
  variable is rated more important (infrastructure weight > women inclusion weight).
- RF importance boost means empirically predictive variables are ranked higher,
  even if their absolute gap is moderate.
- If RF data is unavailable, the system degrades gracefully to gap × factor_weight.

---

## 11. Tradeoff Table Logic <a name="tradeoff"></a>

The tradeoff table compares interventions on 5 dimensions.

### Inclusion Gain (digital equity focus):

**High** = directly expands access for excluded groups (no-phone HH, women, slums).
Device subsidy eliminates device barrier; women skilling closes gender gap.

**Medium** = improves access broadly but not targeting specific excluded groups.
Tower density expands coverage but doesn't address affordability or skills.

**Low** = quality improvement for existing users; doesn't expand who can connect.
Fibre backhaul improves speed for connected users only.

### QoS Gain (quality of service):

**High** = measurable throughput/latency improvement (≥ 25% speed increase).
Tower density, fibre backhaul.

**Medium** = moderate quality improvement as side effect.
Public WiFi (shared bandwidth), community internet centres.

**Low** = negligible network quality impact; demand/skills intervention.
Device subsidy, digital literacy, SHG microfinance.

### Why Inclusion and QoS are orthogonal:
They capture the **supply-demand duality** of digital exclusion:
- QoS = supply quality (what the network provides)
- Inclusion = demand activation (who can actually use it)

An area can have excellent QoS but low inclusion (high-speed network,
low phone ownership) or high inclusion but poor QoS (many connected users,
poor speeds). Policy must address both.

---

## 12. Cost Efficiency Labelling <a name="efficiency"></a>

```
efficiency = mci_delta / total_cost_cr   (MCI pts per ₹ Crore)

HIGH   if efficiency ≥ 1.0 pts/Cr
MEDIUM if efficiency ≥ 0.3 pts/Cr
LOW    if efficiency < 0.3 pts/Cr
```

**Threshold rationale:**
- 1.0 pt/Cr: achievable by literacy programmes (₹0.04 Cr/pt, ~5 pts delta = 125 pts/Cr).
  These are the "high efficiency" interventions.
- 0.3 pt/Cr: minimum acceptable efficiency for a connectivity investment.
  Below this, the intervention produces less than 1 pt MCI per ₹3 Crore invested.
- Device subsidy (~0.06 pts/Cr at full cost): rates LOW because although its
  absolute impact is significant, the cost per point is very high.

---

## 13. Policy Recommendation Ranking <a name="ranking"></a>

Recommendations are scored using:

```
final_score = objective_alignment_score
            + feasibility_bonus
            + mci_delta × 0.01
```

Where:
```
objective_alignment_score = Σ(objective_weight[f] × factor_gap[f] × |delta[f]| / max(factor_gap[f], 1))
  for f in {IFS, DLS, SES, WDI}
```

This formulation ensures:
1. Interventions that address **actually weak factors** score higher.
   (factor_gap[f] amplifies score for weak factors)
2. Interventions aligned with **selected objective's priorities** score higher.
   (objective_weight[f] amplifies factors valued by the objective)
3. Interventions producing **larger deltas** score higher.
   (|delta[f]| term)

The `0.01 × mci_delta` term adds a small absolute-impact tie-breaker.

`feasibility_bonus = {high: +0.15, medium: +0.05, low: -0.05}`

**Primary gap penalty:** If an intervention's primary factor gap < 5 pts,
its score is multiplied by 0.6. This prevents recommending, say,
tower deployment to a district where infrastructure is already adequate.

---

## 14. Why-Selected Rationale Rules <a name="why"></a>

The "why this was recommended" bullets are generated by rule-based checks.
Each rule tests a measurable condition and produces one sentence.

Up to 4 reasons per recommendation, in priority order:

1. **Primary factor weakness:** `factor_gap[primary_factor] > 5`
   → "X is a primary weak factor (driver: Y) — this intervention addresses it."

2. **Objective alignment:** `objective_weight[primary_factor] ≥ 0.25`
   → "Strongly aligned with [objective] (factor weight Z%)."

3. **Cost efficiency:**
   - HIGH → "High impact per ₹ invested (cost tier, +X MCI pts at 70%)."
   - MEDIUM → "Moderate cost efficiency; +X MCI pts projected."

4. **Deployment speed:**
   - Fast → "Rapid deployment possible (timeline) via existing channels."
   - Medium → "Deployable within [timeline]."

5. **District-specific:**
   - MCI < 25 + IFS primary → "Severe desert — infrastructure is prerequisite."
   - women_digital_skilling + GVI > 40 → "High GVI indicates women-specific exclusion."

---

## 15. Run Order <a name="runorder"></a>

```bash
# Install dependencies
pip install streamlit plotly duckdb pandas scikit-learn requests numpy --break-system-packages

# Optional: AI copilot (works without it — falls back to templates)
ollama pull mistral
ollama pull llama3.1
ollama serve

# Pipeline (data processing)
python -m pipeline.preprocessor
python -m pipeline.mci_pipeline
python -m pipeline.store_dashboard_tables

# Launch dashboard
streamlit run dashboard/app.py
```

### File dependency graph:

```
simulation_page.py
  ├── engine.py          (simulation computation)
  ├── cost_model.py      (cost / speed / feasibility)
  ├── optimizer.py       (uses engine + cost_model)
  ├── root_cause.py      (attribution)
  ├── recommendations.py (uses engine + cost_model + tradeoff)
  ├── tradeoff.py        (tradeoff table data)
  └── copilot.py         (uses engine, root_cause)
```

No circular dependencies. All simulation modules are pure Python
(no Streamlit imports). The dashboard is the only file that imports
Streamlit.