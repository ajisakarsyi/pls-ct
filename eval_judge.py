#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import re
import sys
import time
from pathlib import Path
from google import genai
from google.genai import types
import pandas as pd

def load_questions_only(path):
    """Memuat nomor soal, teks soal, konsep, dan gaya kognitif dari JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File soal tidak ditemukan: {path}")
    
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("questions") or data.get("data") or data
        
    questions_dict = {}
    for idx, item in enumerate(data, start=1):
        if isinstance(item, dict):
            nomor = item.get("nomor") or item.get("no") or item.get("id") or idx
            soal = item.get("soal") or item.get("question") or ""
            gaya = item.get("gaya_kognitif") or item.get("gaya") or "Umum"
            konsep = item.get("konsep") or item.get("materi") or "Computational Thinking"
            
            questions_dict[int(nomor)] = {
                "soal": soal,
                "gaya": gaya,
                "konsep": konsep
            }
            
    return questions_dict

def evaluate_pairwise(client, model_judge, data_soal, ground_truth_text, ans_A, ans_B):
    """Mengirim prompt komparatif ke Gemini API dengan penanganan otomatis untuk error 503 dan 429."""
    
    prompt_judge = f"""Role: Anda adalah Pakar Evaluasi Pedagogi dan Ahli Linguistik Komputasional. Tugas Anda adalah melakukan evaluasi komparatif antara Model A dan Model B dalam konteks Pembelajaran Ilmu Komputer yang Dipersonalisasi.

Konteks:
Gaya Kognitif: {data_soal['gaya']}
Materi: {data_soal['konsep']}

Teks Referensi (Ground Truth):
{ground_truth_text}

Rubrik Penilaian (Skala 0, 1, 2):
0 (Tidak Tepat): Jawaban mengandung halusinasi (informasi yang secara faktual salah atau bertentangan dengan GT), atau gagal total mengikuti format.
1 (Cukup/Parsial): Jawaban benar secara umum tetapi terasa sangat generik, kurang memberikan kedalaman teknis yang ada di GT, atau gagal mengadopsi gaya kognitif {data_soal['gaya']} meskipun instruksi format diikuti.
2 (Sangat Tepat): Penjelasan model selaras dengan substansi GT (tidak harus menyebutkan seluruh isi GT, selama poin yang diambil tidak melenceng), berhasil mempersonalisasi konten sesuai gaya {data_soal['gaya']}, dan memenuhi semua batasan format (maksimal 3 poin/paragraf).

Tugas Evaluasi:
Bandingkan Model A dan Model B berdasarkan teks referensi yang tersedia.
Berikan skor (0/1/2) serta alasan singkat untuk 4 dimensi:
1. Akurasi Substansial
2. Kesesuaian Gaya
3. Nilai Pedagogis
4. Kepatuhan Format

Pilih pemenang berdasarkan akumulasi skor dan justifikasi kritis.

Input Jawaban:
Model A: {ans_A}
Model B: {ans_B}

Format Output:
WAJIB mengembalikan data dalam struktur JSON murni seperti ini (tanpa markdown backticks):
{{
  "skor_A": {{
    "akurasi": 2, "gaya": 1, "pedagogis": 2, "format": 2
  }},
  "skor_B": {{
    "akurasi": 2, "gaya": 2, "pedagogis": 2, "format": 1
  }},
  "alasan_A": "Isi alasan singkat evaluasi Model A di sini.",
  "alasan_B": "Isi alasan singkat evaluasi Model B di sini.",
  "pemenang": "Model B"
}}
"""

    max_retries = 10
    base_delay = 5  # Jeda awal default dalam detik

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_judge,
                contents=prompt_judge,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                ),
            )
            return json.loads(response.text.strip())
            
        except Exception as e:
            err_msg = str(e)
            
            # Kasus 1: Terkena Rate Limit Quota Token (429)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                # Default tunggu 50 detik karena limitasi token gratis adalah per menit
                wait_time = 50 
                
                # Coba ekstrak rekomendasi waktu dari error jika ada (misal: "Please retry in 46.5s")
                match = re.search(r"retry in ([\d\.]+)\s*s", err_msg)
                if match:
                    wait_time = int(float(match.group(1))) + 2 # Ditambah buffer 2 detik aman
                
                print(f"         [RATE-LIMIT] Batas token terlampaui (429). Menunggu {wait_time} detik sebelum mencoba kembali... (Percobaan {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                
            # Kasus 2: Server Overload Sementara (503)
            elif "503" in err_msg or "UNAVAILABLE" in err_msg:
                wait_time = base_delay * (2 ** attempt)
                print(f"         [SERVER-BUSY] Server sibuk (503). Mencoba kembali dalam {wait_time} detik... (Percobaan {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                
            # Kasus Lainnya (Error Fatal / Putus Koneksi)
            else:
                if attempt < max_retries - 1:
                    print(f"         [WARN] Terjadi kendala: {err_msg}. Mencoba kembali dalam 5 detik...")
                    time.sleep(5)
                else:
                    return {
                        "error": f"Gagal mengevaluasi setelah beberapa percobaan: {err_msg}",
                        "skor_A": {"akurasi": 0, "gaya": 0, "pedagogis": 0, "format": 0},
                        "skor_B": {"akurasi": 0, "gaya": 0, "pedagogis": 0, "format": 0},
                        "pemenang": "Error"
                    }

def main():
    parser = argparse.ArgumentParser(description="Automated Pairwise LLM-as-a-Judge menggunakan Gemini & Ground Truth Gabungan")
    parser.add_argument("--api-key", required=True, help="Masukkan Gemini API Key Anda")
    parser.add_argument("--model-judge", default="gemini-2.5-flash")
    parser.add_argument("--questions-json", default="soal_ct.json", help="JSON berisi daftar soal, gaya, konsep")
    parser.add_argument("--gt-file", default="merged_files.txt", help="File gabungan seluruh Ground Truth")
    
    # Path folder jawaban Model A dan Model B
    parser.add_argument("--dir-A", default="hasil_raw_ollama/llama3", help="Folder Jawaban Model A")
    parser.add_argument("--dir-B", default="hasil_prompt_engineering/llama3", help="Folder Jawaban Model B")
    
    parser.add_argument("--output-csv", default="rekap_komparatif_skripsi.csv")
    args = parser.parse_args()

    # Inisialisasi client Gemini
    client = genai.Client(api_key=args.api_key)
    
    # Load Daftar Soal (Tanpa Kunci)
    try:
        questions = load_questions_only(args.questions_json)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    # Load Teks Ground Truth Utama dari merged_files.txt
    gt_path = Path(args.gt_file)
    if not gt_path.exists():
        print(f"[ERROR] File Ground Truth tidak ditemukan: {args.gt_file}")
        return 1
    
    print(f"[INFO] Memuat Teks Referensi (Ground Truth) dari {args.gt_file}...")
    ground_truth_text = gt_path.read_text(encoding="utf-8").strip()

    dir_a = Path(args.dir_A)
    dir_b = Path(args.dir_B)

    if not dir_a.exists() or not dir_b.exists():
        print("[ERROR] Salah satu folder input (Model A atau Model B) tidak ditemukan!")
        return 1

    rekap_evaluasi = []
    print(f"[INFO] Memulai evaluasi komparatif 30 Soal dengan {args.model_judge}...\n")

    for nomor_soal in sorted(questions.keys()):
        file_A = list(dir_a.glob(f"soal{nomor_soal}_*.txt"))
        file_B = list(dir_b.glob(f"soal{nomor_soal}_*.txt"))

        if not file_A or not file_B:
            print(f"[SKIP] Soal {nomor_soal} tidak lengkap di kedua folder.")
            continue

        print(f"[JUDGE] Mengevaluasi Soal {nomor_soal}...")
        
        ans_A = file_A[0].read_text(encoding="utf-8").strip()
        ans_B = file_B[0].read_text(encoding="utf-8").strip()
        data_soal = questions[nomor_soal]

        # Kirim data ke Gemini dengan Ground Truth dari merged_files.txt
        res = evaluate_pairwise(client, args.model_judge, data_soal, ground_truth_text, ans_A, ans_B)
        
        if "error" in res:
            print(f"         -> Terjadi Error: {res['error']}")
            continue

        print(f"         -> Pemenang: {res['pemenang']}")

        total_A = sum(res['skor_A'].values())
        total_B = sum(res['skor_B'].values())

        rekap_evaluasi.append({
            "No Soal": nomor_soal,
            "Gaya Kognitif": data_soal['gaya'],
            "Materi": data_soal['konsep'],
            "Total Skor A": total_A,
            "Total Skor B": total_B,
            "Skor Detail A (A,G,P,F)": f"{res['skor_A']['akurasi']},{res['skor_A']['gaya']},{res['skor_A']['pedagogis']},{res['skor_A']['format']}",
            "Skor Detail B (A,G,P,F)": f"{res['skor_B']['akurasi']},{res['skor_B']['gaya']},{res['skor_B']['pedagogis']},{res['skor_B']['format']}",
            "Justifikasi A": res.get("alasan_A", ""),
            "Justifikasi B": res.get("alasan_B", ""),
            "Pemenang": res['pemenang']
        })
        
        time.sleep(1)

    # Simpan rekap penilaian ke CSV
    df = pd.DataFrame(rekap_evaluasi)
    df.to_csv(args.output_csv, index=False, encoding="utf-8")
    
    print(f"\n[DONE] Seluruh 30 Soal selesai dinilai! Hasil akhir tersimpan di: {args.output_csv}")
    print(f"[INFO] Dominasi Pemenang:\n{df['Pemenang'].value_counts()}")
    return 0

if __name__ == "__main__":
    sys.exit(main())