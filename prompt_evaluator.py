from rouge_score import rouge_scorer
from bert_score import score as bert_score_func
import torch
import os

class NLPValidator:
    def __init__(self, lang="id"):
        # Inisialisasi ROUGE (1, 2, L)
        self.rouge_eval = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.lang = lang

    def calculate_metrics(self, prediction: str, reference: str):
        # 1. Hitung ROUGE
        rouge_scores = self.rouge_eval.score(reference, prediction)
        
        # 2. Hitung BERTScore
        # Rescale_with_baseline=True membantu menormalisasi skor agar lebih mudah dibaca
        P, R, F1 = bert_score_func([prediction], [reference], lang=self.lang, verbose=False)
        
        return {
            "rouge1": rouge_scores['rouge1'].fmeasure,
            "rouge2": rouge_scores['rouge2'].fmeasure,
            "rougeL": rouge_scores['rougeL'].fmeasure,
            "bertscore_f1": F1.item()
        }
        
if __name__ == "__main__":
    validator = NLPValidator(lang="id")
    
    # Membaca Ground Truth langsung dari file .txt agar tidak ada error import
    file_path = "gt_ct/01_01.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            ref = f.read()
    else:
        print(f"Error: File {file_path} tidak ditemukan!")
        ref = ""
    
    pred = """Dalam mata kuliah Berpikir Komputasional, tujuan utama bagi mahasiswa non-Ilmu Komputer adalah:

Mengembangkan kemampuan berpikir kritis dan analisis: Mereka dapat mengidentifikasi dan memecahkan masalah dengan cara yang logis dan sistematis.
Mengembangkan kemampuan berpikir abstrak: Mereka dapat mengidentifikasi dan mengembangkan konsep-konsep yang lebih tinggi dan abstrak, serta menerapkan konsep-konsep tersebut pada berbagai konteks.
Mengembangkan kemampuan pengambilan keputusan: Mereka dapat mengidentifikasi dan menghitung risiko, serta membuat keputusan yang cerdas berdasarkan data dan informasi.
Mengembangkan kemampuan komunikasi: Mereka dapat mengkomunikasikan gagasan dan ide-ide mereka dengan cara yang jelas dan efektif.
Mengembangkan kemampuan kolaborasi: Mereka dapat bekerja sama dengan orang lain untuk mencapai tujuan, serta menerima dan menghargai kontribusi orang lain."""

    # Pastikan variabel hasil sejajar dengan baris di atasnya
    hasil = validator.calculate_metrics(pred, ref)
    print(f"Hasil Evaluasi: {hasil}")

# Contoh Penggunaan:
# validator = NLPValidator(lang="id")
# result = validator.calculate_metrics("Jawaban mahasiswa", "Kunci jawaban referensi")
# print(result)