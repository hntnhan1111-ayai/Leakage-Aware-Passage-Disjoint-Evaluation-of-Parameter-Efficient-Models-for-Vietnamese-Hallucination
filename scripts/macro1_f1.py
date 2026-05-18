import pandas as pd
from sklearn.metrics import f1_score, classification_report

def evaluate_performance(ground_truth_path, predictions_path):
    # 1. Load the files
    df_gt = pd.read_csv(ground_truth_path)
    df_pred = pd.read_csv(predictions_path)
    
    # 2. Merge dataframes on 'id' to ensure correct alignment
    # 'label' comes from vihallu-test.csv and 'predict_label' from your submission
    merged = pd.merge(
        df_gt[['id', 'label']], 
        df_pred[['id', 'predict_label']], 
        on='id', 
        how='inner'
    )
    
    # 3. Handle potential missing IDs
    if len(merged) < len(df_gt):
        print(f"Warning: Only {len(merged)} matches found out of {len(df_gt)} ground truth samples.")
    
    # 4. Calculate Macro-F1 (Average of 3 classes)
    y_true = merged['label']
    y_pred = merged['predict_label']
    
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # 5. Output results
    print("=" * 30)
    print(f"FINAL Macro-F1: {macro_f1:.4f}")
    print("=" * 30)
    
    # Detailed breakdown per class
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred))

# Usage
evaluate_performance('vihallu-test.csv', 'final_submission_scratch.csv')