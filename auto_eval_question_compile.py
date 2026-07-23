#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import re
import sys
import time
from pathlib import Path
import requests

def safe_name(name):
    name = re.sub(r"[^\w\-.]+", "_", str(name).strip(), flags=re.UNICODE)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "model"

def load_questions(path, start, end):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File soal tidak ditemukan: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("questions") or data.get("data") or data

    if not isinstance(data, list):
        raise ValueError("Format soal harus list, atau dict dengan key 'questions'/'data'.")

    questions = []
    for idx, item in enumerate(data, start=1):
        if isinstance(item, str):
            nomor = idx
            soal = item
        elif isinstance(item, dict):
            nomor = item.get("nomor") or item.get("no") or item.get("id") or item.get("number") or idx
            soal = item.get("soal") or item.get("question") or item.get("pertanyaan") or item.get("text") or item.get("query")
        else:
            continue

        try:
            nomor = int(nomor)
        except Exception:
            nomor = idx

        if soal and start <= nomor <= end:
            questions.append({"nomor": nomor, "soal": str(soal).strip()})

    questions.sort(key=lambda x: x["nomor"])
    if not questions:
        raise ValueError(f"Tidak ada soal pada rentang {start}-{end}.")

    return questions

def post_chat(base_url, soal, timeout):
    url = base_url.rstrip("/") + "/chat"
    response = requests.post(url, json={"message": soal}, timeout=timeout)
    response.raise_for_status()

    try:
        return response.json()
    except Exception:
        return {"response": response.text}

def extract_answer(data):
    """
    Mengambil jawaban utama (reply) dan pertanyaan lanjutan (followup_question)
    dari respons API.
    """
    if isinstance(data, dict):
        reply = data.get("reply", "").strip()
        followup = data.get("followup_question", "").strip()
        
        # Fallback jika endpoint mengembalikan struktur default lain
        if not reply:
            for key in ["response", "answer", "message", "content", "result", "output"]:
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    reply = value.strip()
                    break
        return reply, followup
        
    return str(data).strip(), ""

def main():
    parser = argparse.ArgumentParser(
        description="Automation Compile: Kirim soal ke /chat dan kompilasi hasil ke 1 file besar dengan Follow-Up."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--questions-json", default="soal_ct.json")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-root", default="hasil_prompt_engineering")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--chat-timeout", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    model_safe = safe_name(args.model_name)
    questions = load_questions(args.questions_json, args.start, args.end)

    output_dir = Path(args.output_root) / model_safe
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Server: {base_url}")
    print(f"[INFO] Jumlah soal: {len(questions)}")
    print(f"[INFO] Mode: Compile All-in-One File dengan Follow-Up\n")

    compiled_results = []

    for item in questions:
        nomor = item["nomor"]
        soal = item["soal"]
        out_file = output_dir / f"soal{nomor}_{model_safe}.txt"

        # Jika skip-existing aktif dan file satuan sudah ada, coba baca data lokal
        if args.skip_existing and out_file.exists():
            print(f"[SKIP] Mengambil data lokal untuk Soal {nomor}")
            content = out_file.read_text(encoding="utf-8")
            
            # Parsing sederhana untuk memisahkan jawaban dan follow-up di file cadangan lama/lokal
            reply, followup = content, ""
            if "=== JAWABAN TUTOR ===" in content and "=== FOLLOW-UP QUESTION ===" in content:
                parts = content.split("=== FOLLOW-UP QUESTION ===")
                reply = parts[0].replace("=== JAWABAN TUTOR ===", "").strip()
                followup = parts[1].strip()
                
            compiled_results.append((nomor, reply, followup))
            continue

        try:
            print(f"[CHAT] Mengirim Soal {nomor}...")
            data = post_chat(base_url, soal, args.chat_timeout)
            reply, followup = extract_answer(data)

            # Simpan backup dokumen satuan dengan format baru yang informatif
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(f"=== JAWABAN TUTOR ===\n{reply}\n\n")
                f.write(f"=== FOLLOW-UP QUESTION ===\n{followup}\n")
            
            # Simpan ke list memori untuk kompilasi akhir
            compiled_results.append((nomor, reply, followup))
            print(f"[OK] Soal {nomor} selesai")
            print()

            if args.sleep > 0:
                time.sleep(args.sleep)

        except KeyboardInterrupt:
            print("\n[STOP] Dihentikan pengguna.")
            return 130
        except Exception as exc:
            print(f"[ERROR] Soal {nomor} gagal: {exc}")
            error_file = output_dir / f"soal{nomor}_{model_safe}_ERROR.txt"
            error_file.write_text(str(exc), encoding="utf-8")
            compiled_results.append((nomor, f"ERROR: {exc}", "ERROR"))
            print()

    # --- PROSES KOMPILASI KE SATU FILE BESAR ---
    if compiled_results:
        compiled_file_path = Path(args.output_root) / f"SEMUA_JAWABAN_{model_safe}.txt"
        
        with open(compiled_file_path, "w", encoding="utf-8") as f:
            f.write(f"=== REKAPITULASI JAWABAN & FOLLOW-UP MODEL: {args.model_name} ===\n")
            f.write(f"Total Soal: {len(compiled_results)}\n")
            f.write("=" * 50 + "\n\n")
            
            for nomor, jawaban, followup in sorted(compiled_results, key=lambda x: x[0]):
                f.write(f"--- SOAL NOMOR {nomor} ---\n")
                f.write(f"=== JAWABAN TUTOR ===\n{jawaban}\n\n")
                f.write(f"=== FOLLOW-UP QUESTION ===\n{followup}\n")
                f.write("-" * 40 + "\n\n")
                
        print(f"[DONE] Semua proses selesai!")
        print(f"[INFO] File kompilasi berhasil diperbarui di: {compiled_file_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())