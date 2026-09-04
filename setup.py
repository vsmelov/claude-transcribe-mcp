#!/usr/bin/env python3
"""Проверка готовности к запуску: зависимости, ffmpeg, .env, модель эмбеддингов.

    python setup.py                   # только проверить
    python setup.py --download-model  # ещё и скачать модель для базы голосов (~27 МБ)

Ничего не отправляет наружу; единственная сетевая операция — скачивание модели,
и только по явному флагу.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "campplus_zh_en_advanced.onnx"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/"
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
)
MODEL_BYTES = 28_281_164
OK, WARN, BAD = "[ok]", "[ ! ]", "[x ]"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

problems: list[str] = []


def say(mark: str, text: str, hint: str = "") -> None:
    print(f" {mark} {text}")
    if hint:
        print(f"      {hint}")


def fail(text: str, hint: str) -> None:
    say(BAD, text, hint)
    problems.append(text)


def check_python() -> None:
    v = sys.version_info
    if v >= (3, 10):
        say(OK, f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        fail(f"Python {v.major}.{v.minor} — нужен 3.10+", "поставь свежий Python")


def check_deps() -> None:
    missing = [
        pkg
        for mod, pkg in (
            ("mcp", "mcp"),
            ("openai", "openai"),
            ("dotenv", "python-dotenv"),
            ("numpy", "numpy"),
        )
        if importlib.util.find_spec(mod) is None
    ]
    if missing:
        fail(
            "не установлены зависимости: " + ", ".join(missing),
            f'"{sys.executable}" -m pip install -r requirements.txt',
        )
    else:
        say(OK, "зависимости установлены")

    if importlib.util.find_spec("sherpa_onnx") is None:
        say(WARN, "sherpa-onnx не установлен — база голосов работать не будет",
            "транскрибация без неё работает; ставится тем же requirements.txt")
    else:
        say(OK, "sherpa-onnx установлен")


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        fail(
            "ffmpeg не найден в PATH",
            "нужен обязательно: длинные файлы режутся им на куски. ffmpeg.org/download",
        )
        return
    try:
        out = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=15, check=False
        )
        first = (out.stdout or "").splitlines()[0] if out.stdout else "ffmpeg"
        say(OK, first[:60])
    except Exception:
        say(WARN, "ffmpeg есть в PATH, но не отвечает")


def check_env() -> None:
    env_path, example = ROOT / ".env", ROOT / ".env.example"
    if not env_path.exists():
        if not example.exists():
            fail("нет ни .env, ни .env.example", "репозиторий склонирован не полностью?")
            return
        shutil.copyfile(example, env_path)
        say(WARN, "создан .env из .env.example", f"впиши свой ключ: {env_path}")

    key = ""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("OPENAI_API_KEY="):
            key = line.partition("=")[2].strip()
    if key and not key.startswith("sk-..."):
        say(OK, "OPENAI_API_KEY задан")
    else:
        fail("OPENAI_API_KEY не заполнен в .env", "ключ с https://platform.openai.com/api-keys")


def check_model(download: bool) -> None:
    if MODEL.exists() and MODEL.stat().st_size > 1_000_000:
        say(OK, f"модель эмбеддингов на месте ({MODEL.stat().st_size // 1024 // 1024} МБ)")
        return
    if not download:
        say(
            WARN,
            "модели эмбеддингов нет — транскрибация работает, база голосов нет",
            f'скачать: "{sys.executable}" setup.py --download-model',
        )
        return

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL.with_suffix(".onnx.part")
    print(f"      качаю {MODEL_BYTES // 1024 // 1024} МБ с github.com/k2-fsa/sherpa-onnx ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        size = tmp.stat().st_size
        if size < 1_000_000:
            tmp.unlink(missing_ok=True)
            fail(f"скачался обрезанный файл ({size} байт)", "проверь сеть и повтори")
            return
        tmp.replace(MODEL)
        say(OK, f"модель скачана: {MODEL.relative_to(ROOT)} ({size // 1024 // 1024} МБ)")
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        fail(f"не удалось скачать модель: {exc}", f"скачай вручную: {MODEL_URL}\n      и положи в {MODEL}")


def main() -> int:
    download = "--download-model" in sys.argv
    print(f"\nclaude-transcribe-mcp — проверка окружения\n{ROOT}\n")
    check_python()
    check_deps()
    check_ffmpeg()
    check_env()
    check_model(download)

    if problems:
        print(f"\nНе готово, {len(problems)} пункт(ов):")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("\nВсё на месте.\n\nПодключить к Claude Code:\n")
    print(f'  claude mcp add --scope user transcribe -- "{sys.executable}" "{ROOT / "server.py"}"\n')
    print("Помни: transcribe — всегда платное действие. Перед запуском смотри estimate().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
