import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
import json
from datetime import datetime
from config import OUTPUT_DIR, SIGNAL_COLS, SIGNAL_NAMES, REPORT_DATE, DATA_CUTOFF, KNOWLEDGE_BASE

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#f8f9fa", "axes.grid": True, "grid.alpha": 0.4, "axes.spines.top": False, "axes.spines.right": False})

def run_viz_and_export(df, df_ml_results, g, ONTO_METRICS, all_reports):
    print("[4/4] Initialising Academic Visualisations & Export System...")

    # Fig 1
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Figure 1: Dataset Overview", fontsize=13, fontweight="bold", y=1.02)
    counts = df['Target_Fraud'].value_counts().sort_index()
    axes[0].bar(['Clean (0)','Fraud (1)'], counts.values, color=['#4a90d9','#e74c3c'])
    axes[0].set_title('(A) Target Class Distribution')
    
    fy = df.groupby('Year')['Target_Fraud'].agg(['sum','count'])
    axes[1].bar(fy.index, fy['count'], color='#4a90d9', alpha=0.5, label='Total')
    axes[1].bar(fy.index, fy['sum'], color='#e74c3c', alpha=0.9, label='Fraud')
    axes[1].set_title('(B) Fraud Cases by Year')
    
    rc = df['risk_level'].value_counts()
    axes[2].pie(rc.values, labels=rc.index, colors=[{'LowRisk':'#2ecc71','MediumRisk':'#f39c12','HighRisk':'#e74c3c'}.get(k) for k in rc.index], autopct='%1.1f%%')
    axes[2].set_title('(C) FFO Risk Level Distribution')
    plt.savefig(OUTPUT_DIR/'fig1_dataset_overview.png', dpi=150, bbox_inches='tight'); plt.close()

    # Fig 2
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    co = df[SIGNAL_COLS].T.dot(df[SIGNAL_COLS])
    co.index, co.columns = SIGNAL_NAMES, SIGNAL_NAMES
    sns.heatmap(co, ax=axes[0], mask=np.eye(len(SIGNAL_NAMES), dtype=bool), cmap='YlOrRd', annot=True)
    axes[0].set_title('(A) Signal Co-occurrence Matrix')
    
    sig_rates = pd.DataFrame({'Clean': df[df['Target_Fraud']==0][SIGNAL_COLS].mean()*100, 'Fraud': df[df['Target_Fraud']==1][SIGNAL_COLS].mean()*100}, index=SIGNAL_NAMES)
    x = np.arange(len(SIGNAL_NAMES)); w = 0.35
    axes[1].bar(x-w/2, sig_rates['Clean'], w, color='#4a90d9', label='Clean')
    axes[1].bar(x+w/2, sig_rates['Fraud'], w, color='#e74c3c', label='Fraud')
    axes[1].set_xticks(x); axes[1].set_xticklabels(SIGNAL_NAMES, rotation=45)
    axes[1].set_title('(B) Signal Activation Rate by Label (%)'); axes[1].legend()
    plt.savefig(OUTPUT_DIR/'fig2_signal_analysis.png', dpi=150, bbox_inches='tight'); plt.close()

    # Fig 4 & 5
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.cm.get_cmap('tab10')
    colors = {m: cmap(i) for i, m in enumerate(df_ml_results.index)}
    for ax, metric in zip(axes, ['Recall','Accuracy']):
        vals = df_ml_results[metric].sort_values()
        bars = ax.barh(vals.index, vals.values, color=[colors[m] for m in vals.index])
        ax.set_title(f'{metric} (%)')
    plt.savefig(OUTPUT_DIR/'fig4_model_comparison.png', dpi=150, bbox_inches='tight'); plt.close()

    # Export Metadata Summary
    summary = {
        'system': 'FFO v3.0 — Financial Fraud Detection: Ontology + XGBoost + Groq LLM',
        'dataset': {'rows': len(df), 'companies': df['Symbol'].nunique(), 'fraud': int(df['Target_Fraud'].sum())},
        'ontology': {'triples': len(g), 'kb_documents': len(KNOWLEDGE_BASE)},
        'performance': {'Ontology_Recall': round(ONTO_METRICS['Recall']*100, 2), 'Ontology_Accuracy': round(ONTO_METRICS['Accuracy']*100, 2)},
        'report_date': REPORT_DATE, 'data_cutoff': DATA_CUTOFF, 'generated_at': datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR/'pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
