import pandas as pd
import json
import time
import re
from groq import Groq
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT

from config import GROQ_API_KEY, GROQ_MODEL, KNOWLEDGE_BASE, PROMPT_TEMPLATE, OUTPUT_DIR
from config import SIGNAL_COLS, WEIGHTS, SCORE_THRESHOLD, REPORT_DATE, DATA_CUTOFF
from config import RISK_COLORS, SIGNAL_AUDIT, SECTION_HEADERS, SIGNAL_SUBFIELDS

SIGNAL_TO_KB = {
    "f_accrual": "AccrualIndicator", "f_noncore": "NonCoreIncomeIndicator", "f_shell": "ShellRiskIndicator",
    "f_pump": "RevenuePumpIndicator", "f_rec": "ReceivableManipulationIndicator", "f_cash": "CashCrunchIndicator",
    "f_lev": "LeverageRiskIndicator", "f_triangle": "EarningsQualityTriangleIndicator", "f_inv_anomaly": "InventoryAnomalyIndicator",
}
RISK_TO_KB = {"HighRisk": "HighRisk", "MediumRisk": "MediumRisk", "LowRisk": "LowRisk"}

def build_ontology_context(row: pd.Series) -> str:
    selected_docs = [f"[METHODOLOGY]\n{KNOWLEDGE_BASE['FFO_Methodology']['description']}"]
    risk_key = str(row["risk_level"])
    if risk_key in RISK_TO_KB:
        rd = KNOWLEDGE_BASE[RISK_TO_KB[risk_key]]
        selected_docs.append(f"[RISK LEVEL: {rd['label']} | Score range: {rd['score_range']}]\n{rd['interpretation']}")
    for sig_col, kb_key in SIGNAL_TO_KB.items():
        if row.get(sig_col, 0) == 1:
            doc = KNOWLEDGE_BASE[kb_key]
            selected_docs.append(f"[ACTIVE SIGNAL: {doc['label']}]\nFormula: {doc['formula']}\nMeaning: {doc['meaning']}\nTheory: {doc['theory']}")
    if len(selected_docs) == 2: selected_docs.append("[NO ACTIVE SIGNALS]\nAll indicators below thresholds.")
    return "\n\n" + ("-"*60 + "\n\n").join(selected_docs)

def build_signal_section(signal_details: list) -> str:
    lines = []
    for i, s in enumerate(signal_details, 1):
        lines.extend([f"Signal {i}: {s['name']}", f"  Formula: {s['formula']}", f"  Contribution: {s['contribution']} pts ({s['pct_of_total']}%)", f"  Theory: {s['theory']}", f"  Meaning: {s['meaning']}", ""])
    return "\n".join(lines)

def build_llm_input(row: pd.Series) -> dict:
    active_sigs  = [(s, SIGNAL_TO_KB[s].replace('Indicator',''), WEIGHTS[i]) for i, s in enumerate(SIGNAL_COLS) if row.get(s, 0) == 1]
    active_sigs_sorted = sorted(active_sigs, key=lambda x: x[2], reverse=True)
    fraud_score = float(row['fraud_score'])
    total_contrib = sum(w for _, _, w in active_sigs_sorted)
    
    signal_details = []
    for sig_col, sig_name, w in active_sigs_sorted:
        doc = KNOWLEDGE_BASE[SIGNAL_TO_KB[sig_col]]
        pct = (w / total_contrib * 100) if total_contrib > 0 else 0
        signal_details.append({
            "name": doc['label'], "signal_col": sig_col, "weight": w, "contribution": round(w, 2),
            "pct_of_total": round(pct, 1), "threshold": doc['threshold'], "formula": doc['formula'],
            "theory": doc['theory'], "meaning": doc['meaning']
        })
    
    return {
        "symbol": row['Symbol'], "year": int(row['Year']), "industry": "Vietnamese Listed Company (HOSE/HNX)",
        "fraud_score": round(fraud_score, 3), "risk_level": str(row['risk_level']),
        "risk_label": {"HighRisk":"HIGH RISK","MediumRisk":"MEDIUM RISK","LowRisk":"LOW RISK"}.get(str(row['risk_level']), str(row['risk_level'])),
        "threshold": SCORE_THRESHOLD, "delta": round(fraud_score - SCORE_THRESHOLD, 3),
        "label": int(row['Target_Fraud']), "active_signals": [s['name'] for s in signal_details],
        "signal_details": signal_details, "n_signals": len(signal_details),
        "top3_pct": round(sum(s['pct_of_total'] for s in signal_details[:3]), 1),
        "context": build_ontology_context(row), "report_date": REPORT_DATE, "data_cutoff": DATA_CUTOFF,
        "accrual_ratio": round(float(row['accrual_ratio']), 4), "noncore_ratio": round(float(row['noncore_ratio']), 4),
        "shell_ratio": round(float(row['shell_ratio']), 4), "rec_manip_ratio": round(float(row['rec_manip_ratio']), 4),
        "cash_crunch_ratio": round(float(row['cash_crunch_ratio']), 4), "leverage_ratio": round(float(row['leverage_ratio']), 4),
        "str_ratio": round(float(row['Short_Term_Ratio']), 3), "delta_gm": round(float(row['Delta_Gross_Margin']),4),
        "delta_lev": round(float(row['Delta_Leverage']), 4), "delta_inv": round(float(row['Delta_Inv']), 4),
        "ocf": int(row['Operating_Cash_Flow']), "owners_equity": int(row['Owners_Equity'])
    }

def format_prompt(inp: dict) -> str:
    return PROMPT_TEMPLATE.format(signal_section=build_signal_section(inp['signal_details']), **inp)

def call_groq(prompt: str) -> str:
    if GROQ_API_KEY == 'your_groq_api_key_here' or not GROQ_API_KEY:
        return "[DEMO MODE] LLM skipped due to no API Key."
    try:
        client = Groq(api_key=GROQ_API_KEY)
        return client.chat.completions.create(model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=3000, temperature=0.05).choices[0].message.content
    except Exception as e:
        return f"[ERROR] Groq API call failed: {e}"

def generate_report(row: pd.Series) -> dict:
    inp = build_llm_input(row)
    inp['prompt'] = format_prompt(inp)
    inp['report'] = re.sub(r'\n{3,}', '\n\n', call_groq(inp['prompt'])).strip()
    return inp

# ----------- REPORTLAB PDF LOGIC -----------
def xml_safe(text: str) -> str: return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
def _hr(color='#d1d5db', thickness=0.5, space_before=4, space_after=4): return HRFlowable(width='100%', thickness=thickness, color=colors.HexColor(color), spaceBefore=space_before, spaceAfter=space_after)

def build_pdf(result: dict, output_path) -> None:
    PAGE_W, PAGE_H = A4
    MARGIN = 2.0 * cm
    CONTENT_W = PAGE_W - 2 * MARGIN
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=2.2*cm, title=f"Risk Assessment — {result['symbol']} {result['year']}")
    
    base = getSampleStyleSheet()
    S = {
        'report_title': ParagraphStyle('RT', parent=base['Normal'], fontSize=15, fontName='Helvetica-Bold', textColor=colors.white, leading=19),
        'report_sub': ParagraphStyle('RS', parent=base['Normal'], fontSize=13, textColor=colors.HexColor('#93c5fd'), alignment=TA_RIGHT),
        'section_title': ParagraphStyle('ST', parent=base['Normal'], fontSize=10, fontName='Helvetica-Bold', textColor=colors.HexColor('#0d1b2a'), spaceBefore=10, spaceAfter=5),
        'signal_title': ParagraphStyle('SiT', parent=base['Normal'], fontSize=9, fontName='Helvetica-Bold', textColor=colors.HexColor('#0d1b2a'), spaceBefore=8, spaceAfter=3),
        'field_label': ParagraphStyle('FL', parent=base['Normal'], fontSize=8, fontName='Helvetica-Bold', textColor=colors.HexColor('#374151')),
        'body': ParagraphStyle('B', parent=base['Normal'], fontSize=9, leading=13, spaceAfter=4, alignment=TA_JUSTIFY),
        'cell': ParagraphStyle('C', parent=base['Normal'], fontSize=8),
        'cell_bold': ParagraphStyle('CB', parent=base['Normal'], fontSize=8, fontName='Helvetica-Bold'),
        'cell_hdr': ParagraphStyle('CH', parent=base['Normal'], fontSize=8, fontName='Helvetica-Bold', textColor=colors.white, alignment=TA_CENTER),
        'footer': ParagraphStyle('F', parent=base['Normal'], fontSize=7, textColor=colors.HexColor('#9ca3af'), alignment=TA_CENTER),
        'disclaimer': ParagraphStyle('disc2', parent=base['Normal'], fontSize=7, textColor=colors.HexColor('#d1d5db'), alignment=TA_CENTER),
    }

    def flush_signal_block(block, story):
        if not block: return
        rows = [[Paragraph(xml_safe(lbl), S['field_label']), Paragraph(xml_safe(cnt), S['cell'])] for lbl, cnt in block]
        tbl = Table(rows, colWidths=[4.2 * cm, CONTENT_W - 4.2 * cm])
        tbl.setStyle(TableStyle([('FONTSIZE',(0,0),(-1,-1),8), ('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#f8fafc')), ('GRID',(0,0),(-1,-1),0.25,colors.HexColor('#e2e8f0')), ('VALIGN',(0,0),(-1,-1),'TOP'), ('LEFTPADDING',(0,0),(-1,-1),7)]))
        story.extend([tbl, Spacer(1, 4)])

    risk_bg, risk_text, risk_accent = RISK_COLORS.get(result['risk_label'], (colors.HexColor('#f9fafb'), colors.HexColor('#111827'), colors.HexColor('#6b7280')))
    story = []

    hdr_tbl = Table([[Paragraph('Financial Reporting Risk<br/>Assessment Report', S['report_title']), Paragraph(f"{result['symbol']}  ·  FY {result['year']}", S['report_sub'])]], colWidths=[CONTENT_W * 0.6, CONTENT_W * 0.4])
    hdr_tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#0d1b2a')), ('VALIGN',(0,0),(-1,-1),'MIDDLE'), ('PADDING',(0,0),(-1,-1),12)]))
    story.extend([hdr_tbl, Spacer(1, 12)])

    story.append(Paragraph('Risk Scorecard', S['section_title']))
    # (Omitted verbose PDF Table style details for Scorecard, focusing on LLM mapping)
    story.append(Spacer(1, 12))

    story.extend([_hr(color='#0d1b2a', thickness=1.5), Paragraph('LLM-Generated Forensic Analysis', S['section_title'])])
    
    report_text = result.get('report', '').replace('(Delta Gross Margin)', '(Delta_Rev)')
    lines = report_text.split('\n')
    signal_block = []
    
    for line in lines:
        line = line.strip()
        if not line:
            flush_signal_block(signal_block, story); signal_block = []
            story.append(Spacer(1, 3))
            continue
        if line in SECTION_HEADERS:
            flush_signal_block(signal_block, story); signal_block = []
            story.extend([_hr(color='#e2e8f0'), Paragraph(xml_safe(line), S['section_title'])])
            continue
        if line.startswith('Signal ') and ':' in line:
            flush_signal_block(signal_block, story); signal_block = []
            story.append(Paragraph(xml_safe(line), S['signal_title']))
            continue
        
        matched = next((p for p in SIGNAL_SUBFIELDS if line.startswith(p)), None)
        if matched: signal_block.append((matched.rstrip(':'), line[len(matched):].strip()))
        else: story.append(Paragraph(xml_safe(line), S['body']))

    flush_signal_block(signal_block, story)
    doc.build(story)

def run_report_pipeline(df):
    print("[3/4] Running Groq LLM & Rendering PDF Report...")
    
    test_row = df[df['risk_level'] == 'HighRisk'].sort_values('fraud_score', ascending=False).drop_duplicates('Symbol').iloc[0]
    res_test = generate_report(test_row)
    build_pdf(res_test, OUTPUT_DIR / f"report_{res_test['symbol']}_{res_test['year']}.pdf")
    
    high_risk_df = df[df['risk_level']=='HighRisk'].reset_index(drop=True)
    all_reports = []
    for _, row in high_risk_df.iterrows():
        try:
            r = generate_report(row)
            all_reports.append(r)
            print(f"  [OK] LLM Report generated: {r['symbol']} - {r['year']}")
        except Exception as e: print(f"  [ERR] {row['Symbol']}: {e}")
        time.sleep(0.3)
        
    with open(OUTPUT_DIR / "high_risk_reports.json", "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2, default=str)
        
    return all_reports
