#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import re
import sys
import time
from pathlib import Path
import requests
import ollama

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

def post_ollama_raw(host_url, model_name, soal):
    # Memanfaatkan library resmi Ollama
    client = ollama.Client(host=host_url)

    # Memanggil model secara RAW tanpa system prompt/options tambahan
    response = client.generate(model=model_name, prompt=soal)

    return response.get("response", "").strip()

def main():
    parser = argparse.ArgumentParser(
        description="Automation Ollama Raw: Kirim soal langsung ke Ollama API tanpa Prompt Engineering."
    )
    # Default port default Ollama adalah 11434
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--questions-json", default="soal_ct.json")
    parser.add_argument("--model-name", required=True, help="Contoh: llama3 atau llama3:8b")
    parser.add_argument("--output-root", default="hasil_raw_ollama")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    model_safe = safe_name(args.model_name)
    questions = load_questions(args.questions_json, args.start, args.end)

    output_dir = Path(args.output_root) / model_safe
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Ollama URL: {args.base_url}")
    print(f"[INFO] Model Nama: {args.model_name}")
    print(f"[INFO] Jumlah Soal: {len(questions)}")
    print(f"[INFO] Output Folder: {output_dir}")
    print("[INFO] Mode: RAW (Tanpa Prompt Engineering)\n")

    for item in questions:
        nomor = item["nomor"]
        soal = item["soal"]
        out_file = output_dir / f"soal{nomor}_{model_safe}.txt"

        if args.skip_existing and out_file.exists():
            print(f"[SKIP] Soal {nomor} sudah ada -> {out_file.name}")
            continue

        try:
            print(f"[OLLAMA] Mengirim Soal {nomor} -> {out_file.name}")

            # Panggil fungsi yang baru
            raw_answer = post_ollama_raw(args.base_url, args.model_name, soal)

            out_file.write_text(raw_answer + "\n", encoding="utf-8")
            print(f"[OK] Soal {nomor} selesai.")
            print()

            if args.sleep > 0:
                time.sleep(args.sleep)

        except KeyboardInterrupt:
            print("\n[STOP] Dihentikan oleh pengguna.")
            return 130
        except Exception as exc:
            print(f"[ERROR] Soal {nomor} gagal: {exc}")
            error_file = output_dir / f"soal{nomor}_{model_safe}_ERROR.txt"
            error_file.write_text(str(exc), encoding="utf-8")
            print()

    print("[DONE] Pengujian selesai.")
    return 0

if __name__ == "__main__":
    sys.exit(main())