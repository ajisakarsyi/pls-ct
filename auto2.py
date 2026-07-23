#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
auto_eval_chat_only_soal_1_30.py

Automation ini hanya mengirim soal ke endpoint /chat.
Tidak memanggil /retrieve.
Tidak membuat prompt baru.
Tidak mengubah prompt engineering.
Tidak membaca materials langsung.

Retrieval, prompt engineering, model, dan konfigurasi lain sepenuhnya diproses oleh ollamaapi.py.
'''

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

    # Hanya kirim soal mentah ke /chat.
    # Prompt, RAG, dan processing lain diurus oleh ollamaapi.py.
    response = requests.post(url, json={"message": soal}, timeout=timeout)
    response.raise_for_status()

    try:
        return response.json()
    except Exception:
        return {"response": response.text}


def extract_answer(data):
    if isinstance(data, str):
        return data.strip()

    if isinstance(data, dict):
        for key in ["response", "answer", "message", "content", "result", "output", "reply"]:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()

        return json.dumps(data, ensure_ascii=False, indent=2).strip()

    return str(data).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Automation chat-only: kirim soal ke /chat, simpan jawaban saja."
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
    print(f"[INFO] File soal: {args.questions_json}")
    print(f"[INFO] Jumlah soal: {len(questions)}")
    print(f"[INFO] Output: {output_dir}")
    print("[INFO] Mode: chat-only")
    print("[INFO] Automation hanya mengirim soal ke /chat.")
    print("[INFO] Retrieval dan prompt engineering sepenuhnya diproses oleh ollamaapi.py.")
    print()

    for item in questions:
        nomor = item["nomor"]
        soal = item["soal"]
        out_file = output_dir / f"soal{nomor}_{model_safe}.txt"

        if args.skip_existing and out_file.exists():
            print(f"[SKIP] Soal {nomor} sudah ada -> {out_file.name}")
            continue

        try:
            print(f"[CHAT] Soal {nomor} -> {out_file.name}")
            data = post_chat(base_url, soal, args.chat_timeout)
            answer = extract_answer(data).strip()

            out_file.write_text(answer + "\n", encoding="utf-8")
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
            print()

    print("[DONE] Semua proses selesai.")
    return 0


if __name__ == "__main__":
    sys.exit(main())