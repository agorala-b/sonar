from config import DATA_PATH, OUTPUT_DIR
from data_ml_pipeline import run_data_ml_pipeline
from ontology_pipeline import run_ontology_pipeline
from report_pipeline import run_report_pipeline
from viz_export_pipeline import run_viz_and_export

def main():
    print("="*70)
    print(" STARTING NEUROSYMBOLIC FRAUD DETECTION PIPELINE (FFO v3)")
    print("="*70)

    try:
        # 1. Feature Engineering & ML
        df, ml_results = run_data_ml_pipeline(DATA_PATH)
        
        # 2. Ontology Bridge & Graph
        df, df_ml_results, g, ONTO_METRICS = run_ontology_pipeline(df, ml_results)
        df.to_csv(OUTPUT_DIR / 'labeled_dataset.csv', index=False)
        df_ml_results.to_csv(OUTPUT_DIR / 'model_comparison.csv')
        
        # 3. LLM Report Generation (PDF + JSON)
        all_reports = run_report_pipeline(df)
        
        # 4. Visualization & Export
        run_viz_and_export(df, df_ml_results, g, ONTO_METRICS, all_reports)
        
        print("\n" + "="*70)
        print("  [COMPLETED] Pipeline executed successfully. Files saved in 'outputs/'.")
        print("="*70)
        
    except FileNotFoundError:
        print(f"\n[ERROR] '{DATA_PATH}' not found.")

if __name__ == "__main__":
    main()
