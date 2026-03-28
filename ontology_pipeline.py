import pandas as pd
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD
from sklearn.metrics import recall_score, accuracy_score
from config import SIGNAL_COLS, SIGNAL_NAMES, WEIGHTS, RATIO_COLS, INDICATOR_DEFS, OUTPUT_DIR

def run_ontology_pipeline(df, ml_results):
    print("[2/4] FFO Signal Bridge & OWL Graph Construction...")
    
    # M4 - Signal Bridge
    df['f_accrual']     = df['Flag_Accrual']
    df['f_noncore']     = df['Flag_Noncore']
    df['f_shell']       = df['Flag_Shadow_Invest']
    df['f_pump']        = ((df['Delta_Rev'] < -0.10) & (df['Operating_Cash_Flow'] < 0)).astype(int)
    df['f_rec']         = df['Flag_Channel_Stuffing']
    df['f_cash']        = df['Flag_Cash_Crunch']
    df['f_lev']         = df['Flag_Insolvency']
    df['f_triangle']    = df['Flag_Desperation']
    df['f_inv_anomaly'] = df['Flag_Inventory_Anomaly']

    for sig, w in zip(SIGNAL_COLS, WEIGHTS):
        df[f'contrib_{sig}'] = df[sig] * w

    contrib_cols = [f'contrib_{s}' for s in SIGNAL_COLS]
    df['fraud_score'] = df[contrib_cols].sum(axis=1)

    def classify_risk(score):
        if score >= 3.00: return "HighRisk"
        if score >= 1.50: return "MediumRisk"
        return "LowRisk"

    df['risk_level'] = df['fraud_score'].apply(classify_risk)
    df['pred_onto']  = (df['risk_level'] == 'HighRisk').astype(int)

    df['accrual_ratio']    = df['Ratio_Accrual']
    df['noncore_ratio']    = df['Ratio_Noncore']
    df['shell_ratio']      = (df['Rec'] + df['LT_Investments']) / df['Total_Assets'].replace(0, 1e-9)
    df['rec_manip_ratio']  = df['Ratio_Channel_Stuffing']
    df['cash_crunch_ratio']= df['Ratio_Cash_Crunch']
    df['leverage_ratio']   = df['Ratio_Leverage']
    for col in RATIO_COLS:
        df[col] = df[col].clip(-10, 10)

    ONTO_METRICS = {
        "Recall":   recall_score(df['Target_Fraud'], df['pred_onto'], zero_division=0),
        "Accuracy": accuracy_score(df['Target_Fraud'], df['pred_onto']),
    }

    ml_rows = [{"Model": "Ontology Engine (FFO v3)", "Accuracy": ONTO_METRICS['Accuracy']*100, "Recall": ONTO_METRICS['Recall']*100}]
    for r in ml_results:
        ml_rows.append({"Model": r['Model'], "Accuracy": r['Accuracy'], "Recall": r['Recall']})
    df_ml_results = pd.DataFrame(ml_rows).set_index("Model")

    # M3 - OWL
    FFO = Namespace("http://ffo.vn/ontology/v3#")
    g   = Graph()
    g.bind("ffo",  FFO)
    g.bind("owl",  OWL)
    g.bind("rdfs", RDFS)
    g.bind("xsd",  XSD)

    onto_uri = FFO["FFO_v3"]
    g.add((onto_uri, RDF.type, OWL.Ontology))
    g.add((onto_uri, RDFS.label, Literal("Financial Fraud Detection Ontology v3.0")))
    g.add((onto_uri, RDFS.comment, Literal("FFO v3.0: Integrated Ontology + XGBoost + LLM pipeline")))

    base_classes = ["Company", "FinancialPeriod", "FinancialIndicator", "FraudSignal", "FraudScore", "RiskLevel", "FraudReport"]
    for cls in base_classes:
        g.add((FFO[cls], RDF.type, OWL.Class))

    INDICATOR_CLASSES = [
        ("AccrualIndicator", "Flag_Accrual", "f_accrual", 1.95), ("NonCoreIncomeIndicator", "Flag_Noncore", "f_noncore", 0.80),
        ("ShellRiskIndicator", "Flag_Shadow_Invest", "f_shell", 1.35), ("RevenuePumpIndicator", "f_pump_derived", "f_pump", 1.75),
        ("ReceivableManipulationIndicator", "Flag_Channel_Stuffing", "f_rec", 1.95), ("CashCrunchIndicator", "Flag_Cash_Crunch", "f_cash", 2.30),
        ("LeverageRiskIndicator", "Flag_Insolvency", "f_lev", 1.40), ("EarningsQualityTriangleIndicator", "Flag_Desperation", "f_triangle", 1.25),
    ]

    for cls_name, flag_src, sig_col, weight in INDICATOR_CLASSES:
        uri = FFO[cls_name]
        g.add((uri, RDF.type, OWL.Class))
        g.add((uri, RDFS.subClassOf, FFO["FinancialIndicator"]))
        g.add((uri, FFO["hasWeight"], Literal(weight, datatype=XSD.float)))

    for risk, lo, hi in [("HighRisk",3.0,99.0),("MediumRisk",1.5,2.99),("LowRisk",0.0,1.49)]:
        uri = FFO[risk]
        g.add((uri, RDF.type, OWL.Class))
        g.add((uri, RDFS.subClassOf, FFO["RiskLevel"]))

    owl_path = OUTPUT_DIR / "ffo_ontology_v3.owl"
    g.serialize(destination=str(owl_path), format="xml")

    return df, df_ml_results, g, ONTO_METRICS
