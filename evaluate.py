import os
import sys
import pandas as pd
from lexsub_pipeline import evaluate_single

def list_to_lower_set(gold_str: str) -> set:
    """Parses a string like 'word|another word|test' to a set of lowercase strings."""
    if pd.isna(gold_str):
        return set()
    return {w.strip().lower() for w in str(gold_str).split('|') if w.strip()}

def main():
    dataset_file = 'lexsub_polysemy_100.ods'
    print(f"Loading dataset: {dataset_file}")
    
    try:
        df = pd.read_excel(dataset_file, engine='odf')
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)

    # Validate columns
    required_cols = ['sentence', 'target', 'gold_substitutes']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Dataset is missing required column '{col}'. Columns found: {df.columns.tolist()}")
            sys.exit(1)

    methods = ['sbert', 'lexconsub', 'xldurel']
    
    report_lines = []
    failure_lines = []
    
    summary_stats = {m: {'correct': 0, 'total': 0} for m in methods}

    report_lines.append("============================================================")
    report_lines.append("   LEXICAL SUBSTITUTION PIPELINE - EVALUATION REPORT")
    report_lines.append(f"   Dataset: {dataset_file}  ({len(df)} templates)")
    report_lines.append("============================================================\n")

    for method in methods:
        print(f"\nEvaluating embedding method: [{method.upper()}]")
        
        failure_lines.append("============================================================")
        failure_lines.append(f"   FAILURES FOR MODEL: {method.upper()}")
        failure_lines.append("============================================================\n")
        
        correct_count = 0
        total_count = 0
        
        for idx, row in df.iterrows():
            sentence = str(row['sentence']).strip()
            target_word = str(row['target']).strip()
            gold_str = str(row['gold_substitutes']).strip()
            
            gold_set = list_to_lower_set(gold_str)
            
            # Print progress indicator (overwrite line)
            print(f"\r  Running row {idx+1:03d}/{len(df)} ...", end='', flush=True)

            try:
                # evaluate_single catches its own prints if verbose=False
                _, top_candidates = evaluate_single(sentence, target_word, embed_method=method, verbose=False)
            except Exception as e:
                import traceback
                print(f"\\nException on row {idx}: {e}")
                traceback.print_exc()
                top_candidates = []
                
            total_count += 1
            
            if top_candidates:
                # The prediction is the best candidate lemma (top 1)
                predicted_lemma = top_candidates[0]['lemma'].lower()
                
                # We consider it correct if the predicted lemma is strictly inside the gold replacements
                # Note: "candidate_scoring.py" handles morphology, but gold is typically lemmas or specific forms.
                # We check the lemma to be safe, but can also check the inflected form.
                predicted_inflected = top_candidates[0]['inflected'].lower()
                
                if predicted_lemma in gold_set or predicted_inflected in gold_set:
                    correct_count += 1
                else:
                    failure_lines.append(f"Row {idx+1}:")
                    failure_lines.append(f"  Sentence  : {sentence}")
                    failure_lines.append(f"  Target    : {target_word}")
                    failure_lines.append(f"  Predicted : {predicted_inflected} / lemma: {predicted_lemma}")
                    failure_lines.append(f"  Gold Subs : {', '.join(gold_set)}")
                    failure_lines.append(f"  Top 3     : {', '.join([c['inflected'] for c in top_candidates[:3]])}\n")
            else:
                failure_lines.append(f"Row {idx+1}:")
                failure_lines.append(f"  Sentence  : {sentence}")
                failure_lines.append(f"  Target    : {target_word}")
                failure_lines.append(f"  Predicted : [NO CANDIDATES GENERATED]")
                failure_lines.append(f"  Gold Subs : {', '.join(gold_set)}\n")
                
        print() # Add newline after progress trace
        
        summary_stats[method]['correct'] = correct_count
        summary_stats[method]['total'] = total_count
        
        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
        report_lines.append(f"Model: {method.upper().ljust(12)} | Accuracy: {accuracy:5.1f}%  ({correct_count}/{total_count})")
        print(f"-> {method.upper()} Done. Accuracy: {accuracy:.1f}%\n")

    # Write output reports
    try:
        with open('eval_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines) + '\n')
            
        with open('eval_failures.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(failure_lines) + '\n')
            
        print("Evaluation completes successfully!")
        print("Summary stats saved to -> eval_report.txt")
        print("Failure traces saved to -> eval_failures.txt")
    except Exception as e:
        print(f"Error writing output files: {e}")

if __name__ == "__main__":
    main()
