import os
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score_func

class NLPValidator:
    def __init__(self, lang="id"):
        self.rouge_eval = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.lang = lang

    def read_files_from_folder(self, folder_path):
        """Membaca semua isi file .txt dalam sebuah folder."""
        texts = []
        filenames = []
        if os.path.exists(folder_path):
            for file in sorted(os.listdir(folder_path)):
                if file.endswith(".txt"):
                    with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                        texts.append(f.read())
                        filenames.append(file)
        return texts, filenames

    def evaluate_multi_refs(self, predictions: list, references: list):
        results = []
        for pred in predictions:
            rouge_results = {"rouge1": [], "rouge2": [], "rougeL": []}
            
            for ref in references:
                s = self.rouge_eval.score(ref, pred)
                rouge_results["rouge1"].append(s['rouge1'].fmeasure)
                rouge_results["rouge2"].append(s['rouge2'].fmeasure)
                rouge_results["rougeL"].append(s['rougeL'].fmeasure)
            
            # BERTScore mendukung perbandingan 1 pred ke banyak refs secara native
            _, _, F1 = bert_score_func([pred], [references], lang=self.lang, verbose=False)

            results.append({
                "rouge1": max(rouge_results["rouge1"]),
                "rouge2": max(rouge_results["rouge2"]),
                "rougeL": max(rouge_results["rougeL"]),
                "bertscore_f1": F1.item()
            })
        return results

if __name__ == "__main__":
    validator = NLPValidator(lang="id")
    
    # 1. Ambil semua referensi (Ground Truth) dari folder materials
    refs, _ = validator.read_files_from_folder("gt_ct")
    
    # 2. Ambil semua hasil model dari folder hasil_model
    preds, pred_names = validator.read_files_from_folder("hasil_model")
    
    if preds and refs:
        skor_per_model = validator.evaluate_multi_refs(preds, refs)
        
        for name, skor in zip(pred_names, skor_per_model):
            print(f"File: {name} | R1: {skor['rouge1']:.4f} | R2: {skor['rouge2']:.4f} | RL: {skor['rougeL']:.4f} | BERT: {skor['bertscore_f1']:.4f}")
    else:
        print("Pastikan folder 'materials' dan 'hasil_model' berisi file .txt")