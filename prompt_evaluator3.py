import os
import csv
import numpy as np
from pathlib import Path
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
    refs, _ = validator.read_files_from_folder("materials")
    
    # 2. Ambil semua hasil model dari folder hasil_model
    folder_preds = "test_hasil"
    preds, pred_names = validator.read_files_from_folder(folder_preds)
    
    if preds and refs:
        print("[INFO] Memulai evaluasi NLP Metrics...")
        skor_per_model = validator.evaluate_multi_refs(preds, refs)
        
        # --- PROSES SIMPAN KE FILE CSV ---
        # Menentukan path output file laporan
        output_dir = Path("test_evaluasi")
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_file_path = output_dir / "rekap_metriks_nlp_engineered_llama3.csv"
        
        print(f"[INFO] Menulis hasil ke {csv_file_path}...")
        
        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Tulis Header Kolom
            writer.writerow(["Nama File", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore F1"])
            
            # Tulis Data dan Cetak di Terminal
            for name, skor in zip(pred_names, skor_per_model):
                # Tulis ke baris CSV
                writer.writerow([
                    name, 
                    f"{skor['rouge1']:.4f}", 
                    f"{skor['rouge2']:.4f}", 
                    f"{skor['rougeL']:.4f}", 
                    f"{skor['bertscore_f1']:.4f}"
                ])
                # Cetak ke terminal sebagai log aktif
                print(f"File: {name} | R1: {skor['rouge1']:.4f} | R2: {skor['rouge2']:.4f} | RL: {skor['rougeL']:.4f} | BERT: {skor['bertscore_f1']:.4f}")
                
        print(f"[DONE] File laporan berhasil dibuat di: {csv_file_path}")
    else:
        print("Pastikan folder 'gt_ct' dan 'hasil_raw_ollama/llama3_8b' berisi file .txt")