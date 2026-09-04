"""Тонкая CLI-обёртка над server.prepare_labeling_job (разметка голосов).

Запуск: .venv\\Scripts\\python.exe prepare_speakers.py <audio> [job-name] [chunk_seconds]
Вся логика — в server.py (prepare_labeling_job), здесь только вызов и печать.
"""
import json
import sys

import server

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    audio = sys.argv[1]
    job_name = sys.argv[2] if len(sys.argv) > 2 else ""
    chunk_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 0
    r = server.prepare_labeling_job(audio, chunk_sec, job_name)
    print(json.dumps(r, ensure_ascii=False, indent=2))
