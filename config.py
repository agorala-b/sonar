import os
from pathlib import Path
from datetime import datetime
from reportlab.lib import colors

# ── Global constants ──────────────────────────────────────────
RANDOM_STATE    = 42
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "your_api_key_here")
GROQ_MODEL      = "llama-3.3-70b-versatile"
OUTPUT_DIR      = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_PATH       = "MERGED_FINAL_DATA_FOR_ML (1) (1).csv"
SCORE_THRESHOLD = 3.00
REPORT_DATE     = datetime.now().strftime("%Y-%m-%d")
DATA_CUTOFF     = "2025-12-31"

# ── FFO Signal schema ─────────────────────────────────────────
SIGNAL_COLS  = ['f_accrual', 'f_noncore', 'f_shell',   'f_pump',
                'f_rec',     'f_cash',    'f_lev',     'f_triangle',
                'f_inv_anomaly']
SIGNAL_NAMES = ['Accrual',  'NonCore', 'Shell',   'RevPump',
                'RecManip', 'CashCrunch', 'Leverage', 'Triangle',
                'InvAnomaly']
WEIGHTS      = [1.95, 0.80, 1.35, 1.75, 1.95, 2.30, 1.40, 1.25, 1.20]
RATIO_COLS   = ['accrual_ratio', 'noncore_ratio', 'shell_ratio',
                'rec_manip_ratio', 'cash_crunch_ratio', 'leverage_ratio']

INDICATOR_DEFS = dict(zip(SIGNAL_COLS, [
    {"label":"Accrual Indicator",             "weight":1.95},
    {"label":"Non-Core Income Indicator",     "weight":0.80},
    {"label":"Shell Company Risk Indicator",  "weight":1.35},
    {"label":"Revenue Pump-and-Drop",         "weight":1.75},
    {"label":"Receivable Manipulation",       "weight":1.95},
    {"label":"Cash Crunch Indicator",         "weight":2.30},
    {"label":"Leverage Risk Indicator",       "weight":1.40},
    {"label":"Earnings Quality Triangle",     "weight":1.25},
]))

# ── Semantic Dictionary ────────────────────────────────────────
semantic_dict = {
    'Flag_Accrual': "Accruals: Phantom profit, negative OCF.",
    'Flag_Channel_Stuffing': "Channel Stuffing: Abnormal increase in receivables compared to revenue.",
    'Flag_Noncore': "Non-Core Profit: >50% profit from non-core operations.",
    'Flag_Desperation': "Desperation: Margin compression, increased leverage, inventory pileup.",
    'Flag_Inventory_Anomaly': "Inventory Anomaly: Inventory growing faster than COGS.",
    'Flag_Shadow_Invest': "Shadow Investment: >40% of assets in LT investments and negative OCF.",
    'Flag_Cash_Crunch': "Cash Crunch: Liquidity covers <20% of short-term debt.",
    'Flag_Insolvency': "Insolvency Risk: Leverage >3.0 or negative equity.",
    'Flag_Core_Efficiency': "Core Efficiency: ROE >15% with strong cash flow.",
    'Flag_Fortress': "Fortress Balance Sheet: Cash covers all debt."
}

# ── Ontology Knowledge Base ────────────────────────────────────
KNOWLEDGE_BASE = {
    "AccrualIndicator": {
        "class":     "FFO:AccrualIndicator",
        "type":      "indicator",
        "label":     "Accrual Indicator",
        "formula":   "(Net_Profit - Operating_Cash_Flow) / Total_Assets",
        "threshold": "> 0 (Net_Profit > 0 AND OCF < 0) | Untitled18: Flag_Accrual",
        "weight":    1.95,
        "signal":    "f_accrual",
        "theory":    "Sloan (1996) accrual anomaly; Beneish (1999) M-score DSRI.",
        "meaning":   ("Positive net profit with negative OCF signals revenues recorded "
                      "without cash collection — aggressive recognition or fabricated receivables."),
        "risk_implication": "3.2x higher restatement probability on Vietnamese listed companies.",
        "data_source": "Income Statement + Cash Flow Statement",
    },
    "NonCoreIncomeIndicator": {
        "class":     "FFO:NonCoreIncomeIndicator",
        "type":      "indicator",
        "label":     "Non-Core Income Indicator",
        "formula":   "(Net_Profit - Operating_Income) / |Net_Profit|",
        "threshold": "> 0.50 | Untitled18: Flag_Noncore",
        "weight":    0.80,
        "signal":    "f_noncore",
        "theory":    "Penman & Zhang (2002) earnings quality; Schilit (2010) GAAP manipulation.",
        "meaning":   "Over 50% of profit from non-core sources (disposals, investment income) "
                     "signals low earnings quality and potential shell structure.",
        "risk_implication": "Secondary signal (w=0.80) but correlated with shell structures.",
        "data_source": "Income Statement — Operating vs. Total Net Profit",
    },
    "ShellRiskIndicator": {
        "class":     "FFO:ShellRiskIndicator",
        "type":      "indicator",
        "label":     "Shadow Investment Indicator",
        "formula":   "LT_Investments / Total_Assets > 0.40 AND OCF < 0",
        "threshold": "> 0.40 (LT_Invest ratio) AND OCF < 0 | Untitled18: Flag_Shadow_Invest",
        "weight":    1.35,
        "signal":    "f_shell",
        "theory":    "Schilit (2010) — shadow investment via off-balance-sheet vehicles; Vietnamese SEC enforcement.",
        "meaning":   "Long-term investment concentration exceeding 40% of total assets combined with "
                     "negative operating cash flow indicates possible asset inflation via related-party "
                     "transactions or circular fund flows.",
        "risk_implication": "Identifies companies parking value in opaque LT investments while bleeding cash — "
                            "common pattern in Vietnamese listed company fraud cases.",
        "data_source": "Balance Sheet — LT Investments, Total Assets + Cash Flow Statement",
    },
    "RevenuePumpIndicator": {
        "class":     "FFO:RevenuePumpIndicator",
        "type":      "indicator",
        "label":     "Revenue Pump-and-Drop Indicator",
        "formula":   "Delta_Rev = (Rev_t - Rev_{t-1}) / Rev_{t-1}",
        "threshold": "Delta_Rev < -0.10 AND OCF < 0 | Derived from Untitled18 temporal features",
        "weight":    1.75,
        "signal":    "f_pump",
        "theory":    "Channel stuffing detection; Marquardt & Wiedman (2004).",
        "meaning":   "Sharp revenue decline with negative OCF is the signature of prior-period "
                     "revenue inflation (channel stuffing, fictitious sales) now reversed.",
        "risk_implication": "Revenue drop >10% YoY + negative OCF = prior revenue inflation.",
        "data_source": "Income Statement YoY + Cash Flow Statement",
    },
    "ReceivableManipulationIndicator": {
        "class":     "FFO:ReceivableManipulationIndicator",
        "type":      "indicator",
        "label":     "Receivable Manipulation Indicator",
        "formula":   "Delta_Rec - Delta_Rev (Ratio_Channel_Stuffing)",
        "threshold": "> 0.20 | Untitled18: Flag_Channel_Stuffing",
        "weight":    1.95,
        "signal":    "f_rec",
        "theory":    "Beneish (1999) DSRI (Days Sales in Receivables Index).",
        "meaning":   "Receivables growing faster than revenues signals fictitious sales "
                     "booked without cash collection — most direct revenue fabrication indicator.",
        "risk_implication": "Highest individual signal precision (67%) in this dataset.",
        "data_source": "Balance Sheet (Receivables) + Income Statement (Revenue) YoY",
    },
    "CashCrunchIndicator": {
        "class":     "FFO:CashCrunchIndicator",
        "type":      "indicator",
        "label":     "Cash Crunch Indicator",
        "formula":   "Cash / ST_Debt (Short_Term_Ratio < 1.0 AND cash_ratio < 0.20)",
        "threshold": "Short_Term_Ratio < 1.0 AND Cash/ST_Debt < 0.20 | Untitled18: Flag_Cash_Crunch",
        "weight":    2.30,
        "signal":    "f_cash",
        "theory":    "Dechow et al. (2011) AAER fraud incentive conditions.",
        "meaning":   "Cannot meet short-term obligations + insufficient cash buffer = acute pressure "
                     "to manipulate financials to maintain credit ratings and avoid covenant violations.",
        "risk_implication": "Highest weight (2.30) — liquidity pressure is primary fraud motive.",
        "data_source": "Balance Sheet — Cash, ST Debt, Current Assets/Liabilities",
    },
    "LeverageRiskIndicator": {
        "class":     "FFO:LeverageRiskIndicator",
        "type":      "indicator",
        "label":     "Leverage Risk Indicator",
        "formula":   "Total_Liabilities / Owners_Equity",
        "threshold": "Leverage > 3.0 OR Owners_Equity < 0 | Untitled18: Flag_Insolvency",
        "weight":    1.40,
        "signal":    "f_lev",
        "theory":    "Debt covenant hypothesis; Watts & Zimmerman (1990); Altman (1968) Z-score.",
        "meaning":   "Over-leveraged companies under extreme creditor pressure. "
                     "Negative equity = technical insolvency — strong incentive to overstate assets.",
        "risk_implication": "Precursor to fraudulent reporting when combined with negative OCF.",
        "data_source": "Balance Sheet — Total Liabilities, Owners Equity",
    },
    "EarningsQualityTriangleIndicator": {
        "class":     "FFO:EarningsQualityTriangleIndicator",
        "type":      "indicator",
        "label":     "Earnings Quality Triangle Indicator",
        "formula":   "Composite: Delta_GM < 0 AND Delta_Lev > 0.10 AND Delta_Inv > 0.15",
        "threshold": "All 3 conditions simultaneous | Untitled18: Flag_Desperation",
        "weight":    1.25,
        "signal":    "f_triangle",
        "theory":    "Triangle fraud theory (Cressey, 1953) adapted to financial ratios.",
        "meaning":   "Simultaneous margin compression + leverage increase + inventory build-up "
                     "suggests channel stuffing under financial deterioration — deliberate multi-dim manipulation.",
        "risk_implication": "Triangle co-occurrence rarely occurs by chance — systematic signal.",
        "data_source": "Income Statement + Balance Sheet YoY deltas",
    },
    "InventoryAnomalyIndicator": {
        "class":     "FFO:InventoryAnomalyIndicator",
        "type":      "indicator",
        "label":     "Inventory Anomaly Indicator",
        "formula":   "Delta_Inv - Delta_COGS (Ratio_Inventory_Anomaly)",
        "threshold": "> 0.20 | Untitled18: Flag_Inventory_Anomaly",
        "weight":    1.20,
        "signal":    "f_inv_anomaly",
        "theory":    "Lev & Thiagarajan (1993) inventory signal; Beneish (1999) DSRI/DEPI component.",
        "meaning":   "When inventory grows significantly faster than cost of goods sold, the company "
                     "may be building up unsellable stock to defer expense recognition, or inflating "
                     "asset values on the balance sheet.",
        "risk_implication": "Inventory build-up without matching COGS increase is a classic channel "
                            "stuffing precursor — often co-occurs with receivable manipulation.",
        "data_source": "Balance Sheet (Inventory) + Income Statement (COGS) YoY deltas",
    },
    "HighRisk": {
        "class":       "FFO:HighRisk",
        "type":        "risk_level",
        "label":       "High Risk",
        "score_range": ">= 3.00",
        "interpretation": ("Multiple fraud signals firing simultaneously. Score >= 3.00 requires at least "
                           "two high-weight signals (e.g., CashCrunch + Accrual) or three moderate signals. "
                           "Requires immediate regulatory scrutiny and enhanced audit procedures."),
    },
    "MediumRisk": {
        "class":       "FFO:MediumRisk",
        "type":        "risk_level",
        "label":       "Medium Risk",
        "score_range": "1.50 – 2.99",
        "interpretation": ("One or two warning signals active. Place on watchlist. "
                           "Heightened scrutiny on specific flagged areas. Monitor across consecutive periods."),
    },
    "LowRisk": {
        "class":       "FFO:LowRisk",
        "type":        "risk_level",
        "label":       "Low Risk",
        "score_range": "< 1.50",
        "interpretation": ("No significant fraud indicators. Financial ratios within normal ranges. "
                           "Standard audit procedures appropriate."),
    },
    "FFO_Methodology": {
        "class":   "FFO:Ontology",
        "type":    "methodology",
        "label":   "FFO v3.0 System Methodology",
        "description": ("The Financial Fraud Detection Ontology v3.0 (FFO v3.0) is a rule-based semantic system "
                         "integrating 8 financial indicators derived from academic fraud detection literature. "
                         "The system was calibrated on Vietnamese listed companies (HOSE/HNX) over 2014–2025. "
                         "Feature engineering pipeline (Untitled18) produces Temporal Ontology features with "
                         "4 assessment pillars: Fraud Risk, Operational Risk, Credit Risk, Value Investing. "
                         "Activation threshold of 3.00 maximises recall for regulatory screening."),
        "academic_references": [
            "Beneish, M.D. (1999). The detection of earnings manipulation. Financial Analysts Journal.",
            "Sloan, R.G. (1996). Do stock prices fully reflect information in accruals? Accounting Review.",
            "Dechow, P. et al. (2011). Predicting material accounting misstatements. CAR.",
            "Cressey, D.R. (1953). Other People's Money. Free Press.",
            "Altman, E.I. (1968). Financial ratios, discriminant analysis. Journal of Finance.",
            "Penman, S. & Zhang, X.J. (2002). Accounting conservatism, quality of earnings. AR.",
        ],
    },
}

# ── LLM Prompt ─────────────────────────────────────────────────
PROMPT_TEMPLATE = """
You are a senior financial forensic analyst specialized in ontology-driven fraud detection
for Vietnamese listed companies.

Generate a STRUCTURED PROFESSIONAL REPORT following EXACTLY the template below.
Replace every [PLACEHOLDER] with specific values derived from the data provided.
All numeric values must be taken directly from the data — do NOT fabricate.
Keep all sections concise and precise.

==============================================================
ONTOLOGY KNOWLEDGE CONTEXT
==============================================================
{context}

==============================================================
COMPANY FINANCIAL DATA
==============================================================

Company              : {symbol}
Fiscal Year          : {year}
Industry             : {industry}
Overall Risk Level   : {risk_label}
Total Risk Score     : {fraud_score}
Alert Threshold      : {threshold}
Exceeds Threshold by : {delta} pts
Report Date          : {report_date}
Data Cutoff          : {data_cutoff}

Ratio Indicators:
  Accrual Ratio          : {accrual_ratio}
  Non-Core Income Ratio  : {noncore_ratio}
  Shell Risk Ratio       : {shell_ratio}
  Receivable Manip Ratio : {rec_manip_ratio}
  Cash Crunch Ratio      : {cash_crunch_ratio}
  Leverage Ratio         : {leverage_ratio}
  Short-Term Ratio       : {str_ratio}

Delta YoY:
  Delta Gross Margin     : {delta_gm}
  Delta Leverage         : {delta_lev}
  Delta Inventory        : {delta_inv}

Operating Cash Flow      : {ocf:,}
Owners Equity            : {owners_equity:,}

Active Signal Pre-Analysis:
{signal_section}

==============================================================
STRICT OUTPUT FORMAT — FILL IN ALL [PLACEHOLDERS]
==============================================================

Company: {symbol}
Fiscal Year: {year}
Industry: [INDUSTRY_DESCRIPTION]
Overall Risk Level: {risk_label}
Total Risk Score: {fraud_score}
Alert Threshold: {threshold}

Executive Summary

This case is classified as {risk_label}, with a total score of {fraud_score} relative to the
threshold of {threshold}, exceeding it by {delta} points.

The three strongest contributing factors are [TOP_SIGNAL_1], [TOP_SIGNAL_2], and [TOP_SIGNAL_3],
which together account for {top3_pct}% of the total score.

The main driver is [MAIN_DRIVER], contributing [MAIN_CONTRIBUTION] to the overall score.
This pattern suggests [MAIN_INTERPRETATION], primarily related to [TARGET_DIMENSION].

At the contextual level, part of the observed signal may also reflect [ALTERNATIVE_EXPLANATION],
particularly under conditions such as [MACRO_CONTEXT], rather than necessarily indicating
reporting distortion.

Overall, the co-occurrence of [PATTERN_1], [PATTERN_2], and [PATTERN_3] increases concern
regarding [TARGET_AREA].

Priority follow-up actions should focus on [CHECK_1], [CHECK_2], and [CHECK_3] within
[TIMEFRAME], with findings reported in [OUTPUT_FORMAT].

Signal-Level Interpretation

For each active signal, provide:

Signal N: [SIGNAL_NAME]
Observed Value: [ratio value from the data above]
Rule Triggered: [threshold condition from the signal pre-analysis above]
Contribution to Total Score: [contribution value and percentage]
Financial Interpretation: [2-3 sentences linking the observed ratio to financial behavior]
Alternative Explanation: [one plausible benign explanation]
Suggested Verification: [one specific audit or data check]

(Repeat for each active signal, ordered by contribution descending)

Risk Synthesis

The company exhibits a combined pattern of [PRIMARY_PATTERN] supported by
[SECONDARY_PATTERN_1] and [SECONDARY_PATTERN_2].
The co-occurrence of these signals strengthens the likelihood of a systemic issue
rather than isolated anomalies, particularly indicating potential deterioration in [TARGET_AREA].

Specifically, the interaction between these patterns suggests that [BRIEF_CAUSAL_EXPLANATION].

While this pattern is consistent with elevated financial reporting risk, alternative
explanations may include [BENIGN_EXPLANATION_1] and [BENIGN_EXPLANATION_2].

Accordingly, this case should be treated as a high-priority analytical flag requiring
further investigation, rather than conclusive evidence of misconduct.

Recommended Follow-up

Provide 3-5 concrete, specific follow-up actions mapped to the active signals above.
Each action must reference: (a) the specific financial statement or account, (b) the
verification method, (c) the responsible party (auditor/regulator/management).
"""

# ── PDF Report Definitions ─────────────────────────────────────
RISK_COLORS = {
    'HIGH RISK':   (colors.HexColor('#fdf0f0'), colors.HexColor('#8b1a1a'), colors.HexColor('#c0392b')),
    'MEDIUM RISK': (colors.HexColor('#fffbf0'), colors.HexColor('#7d5a00'), colors.HexColor('#e67e22')),
    'LOW RISK':    (colors.HexColor('#f0fdf4'), colors.HexColor('#155724'), colors.HexColor('#27ae60')),
}

SIGNAL_AUDIT = {
    'Accrual Indicator': {
        'area':       'Earnings Quality / Cash Flow',
        'assertions': 'Occurrence, Cut-off, Valuation',
        'accounts':   'Revenue, Trade receivables, Operating cash flow',
    },
    'Receivable Manipulation Indicator': {
        'area':       'Revenue Recognition',
        'assertions': 'Occurrence, Completeness, Cut-off',
        'accounts':   'Trade receivables, Revenue, Sales returns',
    },
    'Revenue Pump-and-Drop Indicator': {
        'area':       'Revenue Recognition / Going Concern',
        'assertions': 'Occurrence, Accuracy, Cut-off',
        'accounts':   'Revenue, Operating cash flow, Deferred income',
    },
    'Shell Company Risk Indicator': {
        'area':       'Asset Integrity / Related Parties',
        'assertions': 'Existence, Valuation, Rights & Obligations',
        'accounts':   'LT investments, Receivables, Total assets',
    },
    'Non-Core Income Indicator': {
        'area':       'Earnings Quality',
        'assertions': 'Occurrence, Classification, Presentation',
        'accounts':   'Other income, Operating profit, Net profit',
    },
    'Cash Crunch Indicator': {
        'area':       'Liquidity / Going Concern',
        'assertions': 'Valuation, Completeness',
        'accounts':   'Cash & equivalents, Short-term debt, Current liabilities',
    },
    'Leverage Risk Indicator': {
        'area':       'Debt & Solvency',
        'assertions': 'Completeness, Valuation, Rights & Obligations',
        'accounts':   'Total liabilities, Owners equity, Long-term debt',
    },
    'Earnings Quality Triangle Indicator': {
        'area':       'Multi-Dimensional Earnings Manipulation',
        'assertions': 'Occurrence, Valuation, Cut-off',
        'accounts':   'Gross margin, Inventory, Total debt',
    },
    'Inventory Anomaly Indicator': {
        'area':       'Inventory Valuation / Cost of Sales',
        'assertions': 'Existence, Valuation, Cut-off',
        'accounts':   'Inventory, Cost of goods sold, Gross margin',
    },
}

SECTION_HEADERS = {
    'Executive Summary',
    'Signal-Level Interpretation',
    'Risk Synthesis',
    'Recommended Follow-up',
}

SIGNAL_SUBFIELDS = (
    'Observed Value:',
    'Rule Triggered:',
    'Contribution to Total Score:',
    'Financial Interpretation:',
    'Alternative Explanation:',
    'Suggested Verification:',
)
