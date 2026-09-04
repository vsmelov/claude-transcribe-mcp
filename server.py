"""Transcribe MCP — локальные аудиофайлы -> текст через OpenAI STT.

MCP-сервер (stdio) + CLI для отладки:
    python server.py                      # запуск MCP-сервера
    python server.py estimate <file>      # оценка длительности/чанков/стоимости
    python server.py transcribe <file> [--model M] [--language ru] [--output PATH]
                                           [--expected-speakers N] [--known-speakers JSON]
    python server.py remerge <chunks.json> [--expected-speakers N] [--merge-model M]
    python server.py usage                # расход: сегодня / за всё время

Большие файлы (>24 MB или >15 мин) режутся ffmpeg'ом на 10-минутные куски
с перекрытием 5 с, транскрибируются параллельно и склеиваются дешёвой LLM:
для обычного текста — дедупликация стыков, для диаризации — глобальная
кластеризация меток спикеров (см. merge_diarized_global).

Сырые данные по кускам сохраняются рядом с транскриптом в <имя>.transcript.chunks.json —
это позволяет пересобрать транскрипт (remerge) с другими параметрами склейки БЕЗ
повторной оплаты транскрибации.

Прогресс печатается по мере готовности кусков (номер/всего, прошедшее время, ETA),
каждый платный вызов пишется в usage_ledger.jsonl рядом со скриптом — оттуда считаются
расходы за сегодня и за всё время.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from mcp.server import MCPServer
from openai import OpenAI

load_dotenv(Path(__file__).with_name(".env"))

# httpx/openai по умолчанию логируют "HTTP Request: POST ..." на INFO — это неинформативный шум,
# вместо него ниже печатается предметный прогресс (кусок X/N, ETA, живая стоимость)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

AUDIO_EXTS = {".mp3", ".mp4", ".m4a", ".wav", ".webm", ".ogg", ".oga", ".flac", ".mpga", ".mpeg"}
CHUNK_SECONDS = 600          # 10 минут
OVERLAP_SECONDS = 5          # перекрытие между чанками
MAX_SINGLE_BYTES = 24 * 1024 * 1024   # лимит API — 25 MB, берём с запасом
MAX_SINGLE_SECONDS = 900     # >15 мин — режем, у 4o-моделей лимит ~25 мин
PARALLEL_CHUNKS = 3
DEFAULT_STT_MODEL = os.environ.get("STT_MODEL", "gpt-transcribe")
DEFAULT_MERGE_MODEL = os.environ.get("MERGE_MODEL", "gpt-5-mini")
KNOWN_SPEAKERS_MAX = 4        # лимит OpenAI API
KNOWN_SPEAKER_MIN_SEC = 1.0   # мягкая проверка (API просит 2-10с)
KNOWN_SPEAKER_MAX_SEC = 12.0
MIME_BY_EXT = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".mp4": "audio/mp4",
    ".ogg": "audio/ogg", ".oga": "audio/ogg", ".flac": "audio/flac", ".webm": "audio/webm",
    ".mpga": "audio/mpeg", ".mpeg": "audio/mpeg",
}
LEDGER_PATH = Path(__file__).with_name("usage_ledger.jsonl")
_ledger_lock = threading.Lock()

# Цены сверены с https://developers.openai.com/api/docs/pricing (2026-08-14)
MODELS: dict[str, dict] = {
    "gpt-transcribe": {
        "per_min": 0.0045,
        "diarization": False,
        "notes": "Новейшая STT-модель OpenAI, лучшее качество, дешевле 4o. Рекомендуется по умолчанию.",
    },
    "gpt-4o-transcribe": {
        "per_min": 0.006,
        "diarization": False,
        "notes": "Проверенный флагман предыдущего поколения (WER ~4.1%).",
    },
    "gpt-4o-mini-transcribe": {
        "per_min": 0.003,
        "diarization": False,
        "notes": "Самая дешёвая из приличных. Для черновиков и массовой обработки.",
    },
    "gpt-4o-transcribe-diarize": {
        "per_min": 0.006,
        "diarization": True,
        "notes": ("Разделение по спикерам (кто когда говорил). Без known_speakers метки "
                  "анонимные (S1, S2...) и склеиваются эвристикой — см. expected_speakers."),
    },
    "whisper-1": {
        "per_min": 0.006,
        "diarization": False,
        "notes": "Legacy Whisper. Смысла нет: gpt-transcribe точнее и дешевле.",
    },
}
# $ за 1M токенов (in, out) для моделей склейки — те же источники, что и MODELS выше
TEXT_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
}

INSTRUCTIONS = f"""Транскрибация локальных аудиофайлов через OpenAI STT.

Вход: абсолютный путь к локальному аудиофайлу ({", ".join(sorted(e.lstrip(".") for e in AUDIO_EXTS))}).
Размер не ограничен: файлы больше 24 MB или длиннее 15 минут автоматически режутся
ffmpeg'ом на куски по {CHUNK_SECONDS // 60} минут (перекрытие {OVERLAP_SECONDS} с) и склеиваются
дешёвой LLM ({DEFAULT_MERGE_MODEL}): для текста — дедупликация стыков, для диаризации —
глобальная кластеризация меток спикеров разом по всей записи (не по стыкам), что заметно
устойчивее к раздуванию числа спикеров, чем попарная склейка соседних кусков.

Инструменты:
- list_models — модели и стоимость за минуту речи
- estimate(file_path) — длительность, число кусков, стоимость по каждой модели
- transcribe(file_path, ...) — транскрипт пишется рядом с аудио в <имя>.transcript.md
  (или в output_path), в ответе — путь и превью. Диаризация: model=gpt-4o-transcribe-diarize.
  ВСЕГДА платное действие (реальные $) — уточните у пользователя параметры и стоимость
  (estimate) перед запуском, если они не были явно оговорены.
  - expected_speakers: если знаете точное/примерное число участников — передайте, склейка
    станет заметно строже (модель будет объединять неуверенные метки, а не плодить новых).
  - known_speakers: JSON-список до {KNOWN_SPEAKERS_MAX} человек с именем и коротким (2-10с)
    референс-клипом голоса — [{{"name": "Anna", "reference": "C:\\\\path\\\\anna.wav"}}]. Эти
    голоса получат настоящее имя вместо S1/S2 и не участвуют в кластеризации (совпадают
    по конструкции), что и даёт меньше спикеров, и подписывает их правильно.
- remerge(chunks_json_path, ...) — пересобрать транскрипт с другими expected_speakers/
  known_speakers БЕЗ повторной оплаты транскрибации (только дешёвая LLM-склейка), используя
  сырые данные из <транскрипт>.chunks.json (создаётся автоматически при каждом transcribe).
- usage() — расход по журналу usage_ledger.jsonl: за сегодня и за всё время.

База голосов (voiceprints) и разметка:
- voiceprints/ — база известных голосов: registry.json (имя, алиасы, сэмплы) + аудиофайлы,
  всё обозримо глазами. list_voiceprints() — вся база; find_voiceprint("Аня с курсов")
  резолвит алиас в человека (-> Anna Ivanova) и показывает его сэмплы.
- В transcribe(known_speakers=...) можно передать просто имена/алиасы через запятую —
  референсы подставятся из базы автоматически.
- ВАЖНО, порядок работы с диаризацией: НЕ угадывай спикеров сам. Перед запуском явно
  спроси пользователя, какие люди ожидаются в записи, и проверь через find_voiceprint,
  что на каждого есть сэмпл. Если сэмплов нет — предложи разметку:
  prepare_speaker_labeling(file) (ПЛАТНО: диаризация одного куска) нарежет сольные
  фрагменты голосов, дай их пользователю послушать и назвать людей, затем
  register_voiceprint сохранит голоса в базу для всех будущих записей.
- Авто-опознание по базе (локально, бесплатно, sherpa-onnx + косинус): разметка сама
  матчит каждый найденный голос против базы (поле match: strong >= 0.7 — почти наверняка
  этот человек, weak 0.5-0.7 — гипотеза). strong предлагай пользователю как готовый ответ
  на подтверждение, weak/нет — спрашивай как обычно. Отдельный клип можно проверить
  тулом match_voice(file). Это и есть гибрид «большая база голосов + <=4 референса в API».

Прогресс печатается по мере готовности кусков (номер/всего, ETA, живая стоимость этого
запуска) — виден в логах процесса, не в ответе синхронного tool-вызова.

Требуется ffmpeg в PATH и OPENAI_API_KEY в .env рядом с server.py."""

mcp = MCPServer("transcribe", instructions=INSTRUCTIONS)

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY не задан — положите его в .env рядом с server.py")
        _client = OpenAI()
    return _client


# ------------------------------------------------------------------- расходы

class CostTracker:
    """Живая сумма расходов текущего запуска (потокобезопасно — куски идут параллельно)."""
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.usd = 0.0

    def add(self, amount: float) -> float:
        with self._lock:
            self.usd += amount
            return self.usd


def record_cost(tracker: CostTracker | None, usd: float, kind: str, model: str, file_label: str) -> float:
    """Прибавляет к живому счётчику запуска и дописывает строку в usage_ledger.jsonl."""
    run_total = tracker.add(usd) if tracker is not None else usd
    entry = {"ts": time.time(), "usd": round(usd, 6), "kind": kind, "model": model, "file": file_label}
    with _ledger_lock:
        with open(LEDGER_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return run_total


def ledger_summary() -> tuple[float, float]:
    """(потрачено сегодня, потрачено за всё время) по локальному календарному дню."""
    if not LEDGER_PATH.exists():
        return 0.0, 0.0
    today = datetime.now().date()
    today_sum = total_sum = 0.0
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_sum += e.get("usd", 0.0)
            if datetime.fromtimestamp(e["ts"]).date() == today:
                today_sum += e.get("usd", 0.0)
    return today_sum, total_sum


def chat_call_cost(merge_model: str, usage) -> float:
    if usage is None:
        return 0.0
    price_in, price_out = TEXT_MODEL_PRICING.get(merge_model, (0.0, 0.0))
    return (getattr(usage, "prompt_tokens", 0) / 1e6 * price_in
            + getattr(usage, "completion_tokens", 0) / 1e6 * price_out)


# ---------------------------------------------------------------- ffmpeg utils

def ffprobe_duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except FileNotFoundError:
        raise RuntimeError("ffprobe/ffmpeg не найдены в PATH")
    return float(out)


@dataclass
class Chunk:
    index: int
    path: Path
    offset: float    # глобальное время начала куска в исходном файле, сек
    duration: float  # длительность самого куска, сек (для точного счёта стоимости)


def split_audio(src: Path, duration: float, tmpdir: Path) -> list[Chunk]:
    """Режет файл на куски по CHUNK_SECONDS с перекрытием OVERLAP_SECONDS (stream copy, без перекодирования)."""
    n = max(1, -(-int(duration) // CHUNK_SECONDS))  # ceil
    chunks = []
    for i in range(n):
        start = max(0.0, i * CHUNK_SECONDS - (OVERLAP_SECONDS if i > 0 else 0))
        seg_len = duration - start if i == n - 1 else CHUNK_SECONDS + (OVERLAP_SECONDS if i > 0 else 0)
        out = tmpdir / f"chunk_{i:03d}{src.suffix}"
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(start)]
        if i < n - 1:
            cmd += ["-t", str(seg_len)]
        cmd += ["-i", str(src), "-c", "copy", str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg не смог вырезать кусок {i}: {proc.stderr[:500]}")
        chunks.append(Chunk(index=i, path=out, offset=start, duration=seg_len))
    return chunks


# --------------------------------------------------------------- known speakers

VOICEPRINTS_DIR = Path(__file__).with_name("voiceprints")
REGISTRY_PATH = VOICEPRINTS_DIR / "registry.json"

_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def _slug(name: str) -> str:
    latin = "".join(_TRANSLIT.get(ch, ch) for ch in name.casefold())
    import re as _re
    words = _re.findall(r"[a-z0-9]+", latin)
    return "-".join(words) or "person"


def audio_to_data_uri(path: Path) -> str:
    mime = MIME_BY_EXT.get(path.suffix.lower(), "audio/mpeg")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def load_registry() -> list[dict]:
    if not REGISTRY_PATH.is_file():
        return []
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def save_registry(people: list[dict]) -> None:
    VOICEPRINTS_DIR.mkdir(exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(people, ensure_ascii=False, indent=2), encoding="utf-8")


def _norm_name(s: str) -> str:
    return " ".join(s.casefold().replace("ё", "е").split())


def find_voiceprints_data(query: str) -> list[dict]:
    """Кандидаты из базы по имени/алиасу, лучшие первыми. 'Аня с курсов' -> Anna Ivanova."""
    q = _norm_name(query)
    scored = []
    for p in load_registry():
        variants = [p["name"], *p.get("aliases", [])]
        best = 0
        for v in variants:
            nv = _norm_name(v)
            if q == nv:
                best = max(best, 100)
            elif q in nv or nv in q:
                best = max(best, 80)
            else:
                overlap = set(q.split()) & set(nv.split())
                if overlap:
                    best = max(best, 40 + 20 * len(overlap))
        if best:
            scored.append((best, p))
    return [p for _, p in sorted(scored, key=lambda x: -x[0])]


def _validate_reference(name: str, p: Path) -> None:
    if not p.is_absolute() or not p.exists():
        raise ValueError(f"known_speakers: reference должен быть абсолютным путём к существующему файлу: {p}")
    dur = ffprobe_duration(p)
    if not (KNOWN_SPEAKER_MIN_SEC <= dur <= KNOWN_SPEAKER_MAX_SEC):
        raise ValueError(
            f"known_speakers: клип {name!r} длится {dur:.1f}с, OpenAI просит 2-10с "
            f"(допускаем {KNOWN_SPEAKER_MIN_SEC}-{KNOWN_SPEAKER_MAX_SEC}с) — обрежьте клип покороче")


def parse_known_speakers(raw: str) -> list[tuple[str, Path]]:
    """Два формата: JSON [{"name","reference"}...] ИЛИ имена/алиасы через запятую,
    тогда референсы берутся из базы voiceprints (см. find_voiceprints_data)."""
    raw = raw.strip()
    if not raw:
        return []
    result: list[tuple[str, Path]] = []
    if raw.startswith("["):
        try:
            items = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"known_speakers должен быть валидным JSON-списком: {e}")
        if not isinstance(items, list):
            raise ValueError("known_speakers должен быть JSON-списком объектов {name, reference}")
        for it in items:
            name, ref = str(it.get("name", "")).strip(), str(it.get("reference", "")).strip()
            if not name or not ref:
                raise ValueError(f"known_speakers: нужны и name, и reference: {it}")
            result.append((name, Path(ref)))
    else:
        for q in (part.strip() for part in raw.split(",") if part.strip()):
            matches = find_voiceprints_data(q)
            if not matches:
                raise ValueError(
                    f"Голос {q!r} не найден в базе voiceprints. Варианты: сделать разметку "
                    f"(prepare_speaker_labeling + register_voiceprint) или передать JSON с reference.")
            person = matches[0]
            refs = [s for s in person.get("samples", []) if s.get("kind") != "embed"]
            if not refs:
                raise ValueError(f"У {person['name']} в базе нет коротких (2-10с) API-референсов")
            result.append((person["name"], VOICEPRINTS_DIR / refs[0]["file"]))
    if len(result) > KNOWN_SPEAKERS_MAX:
        raise ValueError(f"known_speakers: максимум {KNOWN_SPEAKERS_MAX} человек (у API), передано {len(result)}")
    for name, p in result:
        _validate_reference(name, p)
    return result


def register_voiceprint_files(name: str, sample_paths: list[Path], aliases: list[str],
                              source: str = "", embed_paths: list[Path] | None = None) -> dict:
    """Кладёт сэмплы голоса в voiceprints/<slug>/ и обновляет registry.json.

    sample_paths — короткие (2-10с) клипы, годные как API-референсы known_speakers.
    embed_paths — длинные (до ~{EMBED_CLIP_MAX_SEC:.0f}с) куски сольной речи только для
    локального матчинга (kind="embed"); в API они не отправляются."""
    people = load_registry()
    person = next((p for p in people if _norm_name(p["name"]) == _norm_name(name)), None)
    if person is None:
        person = {"name": name, "slug": _slug(name), "aliases": [], "samples": []}
        people.append(person)
    for a in aliases:
        if a and a not in person["aliases"] and _norm_name(a) != _norm_name(name):
            person["aliases"].append(a)
    pdir = VOICEPRINTS_DIR / person["slug"]
    pdir.mkdir(parents=True, exist_ok=True)
    for src in sample_paths:
        n = sum(1 for s in person["samples"] if s.get("kind") != "embed") + 1
        dst = pdir / f"sample_{n}{src.suffix}"
        dst.write_bytes(src.read_bytes())
        person["samples"].append({
            "file": f"{person['slug']}/{dst.name}",
            "duration_sec": round(ffprobe_duration(dst), 1),
            "source": source or str(src),
            "added_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        })
    for src in (embed_paths or []):
        n = sum(1 for s in person["samples"] if s.get("kind") == "embed") + 1
        dst = pdir / f"embed_{n}{src.suffix}"
        dst.write_bytes(src.read_bytes())
        person["samples"].append({
            "file": f"{person['slug']}/{dst.name}",
            "kind": "embed",
            "duration_sec": round(ffprobe_duration(dst), 1),
            "source": source or str(src),
            "added_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        })
    save_registry(people)
    try:
        ensure_embeddings()  # локальный эмбеддинг нового голоса для будущего авто-матча
    except Exception as exc:
        person["embedding_note"] = f"эмбеддинг не посчитан: {exc}"
    return person


# ---------------------------------------------------- голосовые эмбеддинги (локально)

EMBED_MODEL_PATH = Path(__file__).with_name("models") / "campplus_zh_en_advanced.onnx"
EMBEDDINGS_PATH = VOICEPRINTS_DIR / "embeddings.json"  # {sample_rel_path: [192 float]}
# калибровка на реальных сэмплах: свой ~0.89, чужой ~0.25
EMBED_STRONG = 0.70   # выше — уверенно тот же голос
EMBED_WEAK = 0.50     # 0.5-0.7 — «похоже, но подтверди у пользователя»
# для ЛОКАЛЬНОГО матчинга лимит API 2-10с не действует: эмбеддинг по длинному куску
# сольной речи заметно стабильнее, чем по 8-секундному референсу
EMBED_CLIP_MAX_SEC = 45.0

_embed_extractor = None


def _extractor():
    global _embed_extractor
    if _embed_extractor is None:
        import sherpa_onnx
        if not EMBED_MODEL_PATH.exists():
            raise RuntimeError(
                f"Модель эмбеддингов не найдена: {EMBED_MODEL_PATH}. Скачайте "
                f"campplus_zh_en_16k-common_advanced с github.com/k2-fsa/sherpa-onnx "
                f"releases (тег speaker-recongition-models, ~27 МБ)")
        _embed_extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMBED_MODEL_PATH), num_threads=2))
    return _embed_extractor


def embed_audio(path: Path) -> list[float]:
    """Нормированный эмбеддинг голоса (192 float) из аудио любого формата. ~100мс CPU."""
    import wave

    import numpy as np
    ext = _extractor()
    with tempfile.TemporaryDirectory(prefix="embed_") as td:
        wav = Path(td) / "a.wav"
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-ar", "16000", "-ac", "1", str(wav)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg при конвертации для эмбеддинга: {proc.stderr[:300]}")
        with wave.open(str(wav), "rb") as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = data.astype(np.float32) / 32768.0
    s = ext.create_stream()
    s.accept_waveform(16000, audio)
    s.input_finished()
    v = np.array(ext.compute(s), dtype=np.float64)
    v /= (np.linalg.norm(v) or 1.0)
    return [round(float(x), 6) for x in v]


def load_embeddings() -> dict:
    if not EMBEDDINGS_PATH.is_file():
        return {}
    return json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))


def ensure_embeddings() -> dict:
    """Досчитывает эмбеддинги для всех сэмплов базы, которых ещё нет в embeddings.json."""
    emb = load_embeddings()
    changed = False
    for p in load_registry():
        for s in p.get("samples", []):
            if s["file"] not in emb:
                emb[s["file"]] = embed_audio(VOICEPRINTS_DIR / s["file"])
                changed = True
    if changed:
        VOICEPRINTS_DIR.mkdir(exist_ok=True)
        EMBEDDINGS_PATH.write_text(json.dumps(emb, ensure_ascii=False), encoding="utf-8")
    return emb


def match_embedding(vec: list[float]) -> list[tuple[str, float]]:
    """Матч вектора против всей базы: [(имя, косинус)], лучшие первыми.

    Голос человека — среднее его нормированных векторов (enrollment averaging):
    длинные embed-клипы и короткие референсы усредняются в один эталон."""
    import numpy as np
    v = np.array(vec)
    emb = ensure_embeddings()
    best: dict[str, float] = {}
    for p in load_registry():
        vecs = [np.array(emb[s["file"]]) for s in p.get("samples", []) if s["file"] in emb]
        if vecs:
            mean = np.mean(vecs, axis=0)
            mean /= (np.linalg.norm(mean) or 1.0)
            best[p["name"]] = float(mean @ v)
    return sorted(best.items(), key=lambda kv: -kv[1])


# ----------------------------------------- пере-опознание спикеров по эмбеддингам

CLUSTER_JOIN = 0.70      # косинус, при котором две метки считаются одним человеком
MIN_EMBED_RUN_SEC = 1.5  # сольный кусок короче — эмбеддинг не считаем


def _runs_of(segs: list[Seg]) -> dict[str, list[dict]]:
    """Непрерывные сольные отрезки по каждой метке (паузы < MAX_GAP_SEC)."""
    runs: dict[str, list[dict]] = {}
    cur = None
    for s in sorted(segs, key=lambda x: x.start):
        if cur and s.speaker == cur["speaker"] and s.start - cur["end"] <= MAX_GAP_SEC:
            cur["end"] = max(cur["end"], s.end)
        else:
            cur = {"speaker": s.speaker, "start": s.start, "end": s.end}
            runs.setdefault(s.speaker, []).append(cur)
    return runs


def resolve_speakers_by_embeddings(chunks_json_path: str, output_path: str = "") -> dict:
    """Опознание спикеров ПОСЛЕ транскрибации, локально и бесплатно.

    Для каждой сырой чанк-метки берётся её длиннейший сольный кусок из исходного
    аудио -> эмбеддинг -> жадная кластеризация меток по косинусу через всю запись
    (детерминированно, без LLM-угадывания) -> кластеры матчатся против базы
    voiceprints (>= EMBED_STRONG — имя из базы, иначе S1/S2...). Имена, известные
    ещё с прогона (known_speakers), не трогаются. Переписывает транскрипт.
    """
    import numpy as np
    p = Path(chunks_json_path)
    if not p.is_absolute() or not p.exists():
        raise ValueError(f"chunks_json_path не найден: {chunks_json_path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not data.get("diarize"):
        raise ValueError("Это не диаризованный прогон — опознавать нечего")
    src = Path(data["source"])
    known = set(data.get("known_speakers", []))
    out = Path(output_path) if output_path else Path(data["source"]).with_suffix(".transcript.md")
    clips_dir = Path(__file__).with_name("workspace") / f"resolve_{_slug(src.stem)[:40]}"
    clips_dir.mkdir(parents=True, exist_ok=True)

    chunks_segs = [[Seg(**d) for d in segs] for segs in data["segments_per_chunk"]]

    # эмбеддинг длиннейшего сольного куска каждой метки
    per_label: dict[str, dict] = {}
    for segs in chunks_segs:
        for label, runs in _runs_of(segs).items():
            if label in known:
                continue
            longest = max(runs, key=lambda r: r["end"] - r["start"])
            dur = longest["end"] - longest["start"]
            info = {"spoke": sum(r["end"] - r["start"] for r in runs),
                    "first": min(r["start"] for r in runs), "vec": None, "clip": None}
            if dur >= MIN_EMBED_RUN_SEC:
                clip = clips_dir / f"{label.replace(':', '_')}.m4a"
                try:
                    _cut_audio(src, longest["start"], min(dur, EMBED_CLIP_MAX_SEC), clip)
                    info["vec"] = np.array(embed_audio(clip))
                    info["clip"] = str(clip)
                except Exception:
                    pass
            per_label[label] = info

    # жадная кластеризация: длинноговорящие первыми, центроид = среднее нормированных
    clusters: list[dict] = []
    for label, info in sorted(per_label.items(), key=lambda kv: -kv[1]["spoke"]):
        if info["vec"] is None:
            continue
        v = info["vec"]
        best_i, best_cos = -1, 0.0
        for i, c in enumerate(clusters):
            cen = c["sum"] / np.linalg.norm(c["sum"])
            cos = float(cen @ v)
            if cos > best_cos:
                best_i, best_cos = i, cos
        if best_i >= 0 and best_cos >= CLUSTER_JOIN:
            c = clusters[best_i]
            c["members"].append(label)
            c["sum"] = c["sum"] + v
            c["spoke"] += info["spoke"]
            c["first"] = min(c["first"], info["first"])
        else:
            clusters.append({"members": [label], "sum": v.copy(),
                             "spoke": info["spoke"], "first": info["first"],
                             "clip": info["clip"]})

    # имена кластерам: матч центроида против базы, иначе S1/S2 в порядке появления
    mapping: dict[str, str] = {}
    summary = []
    counter = 0
    for c in sorted(clusters, key=lambda c: c["first"]):
        cen = c["sum"] / np.linalg.norm(c["sum"])
        matches = match_embedding([float(x) for x in cen])
        if matches and matches[0][1] >= EMBED_STRONG:
            name, score = matches[0]
        else:
            counter += 1
            name, score = f"S{counter}", (matches[0][1] if matches else 0.0)
        for lab in c["members"]:
            mapping[lab] = name
        summary.append({"name": name, "spoke_sec": round(c["spoke"]),
                        "labels": c["members"], "best_base_cos": round(score, 3),
                        "listen": c["clip"]})
    # метки без эмбеддинга (микро-реплики) — анонимной кучкой, чтобы не наврать
    for label, info in per_label.items():
        if label not in mapping:
            mapping[label] = "S?"

    # сборка финальных сегментов с выкидыванием дублей перекрытия (как в merge)
    result: list[Seg] = []
    for i, segs in enumerate(chunks_segs):
        if i > 0:
            boundary = i * CHUNK_SECONDS
            segs = [s for s in segs if s.end > boundary + 0.25]
        for s in segs:
            result.append(Seg(s.start, s.end, s.speaker if s.speaker in known
                              else mapping.get(s.speaker, "S?"), s.text))
    result.sort(key=lambda s: s.start)
    body = format_diarized(result)
    n_speakers = len({s.speaker for s in result})
    write_transcript(out, src.name, str(src), data["duration"], data["model"],
                     "embeddings (локально)", len(chunks_segs), 0.0, n_speakers, body)
    return {"output": str(out), "speakers": n_speakers,
            "clusters": summary, "known_kept": sorted(known)}


# ------------------------------------------------------- разметка голосов (labeling)

MIN_SOLO_SEC = 3.5    # рана короче — не годится как референс (API просит 2-10с)
MAX_REF_SEC = 8.0     # референс не длиннее этого (лимит API 10с, с запасом)
MAX_GAP_SEC = 0.8     # паузы внутри раны одного спикера
SAMPLES_PER_SPEAKER = 2


def _cut_audio(src: Path, start: float, dur: float, out: Path) -> None:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}",
         "-i", str(src), "-c", "copy", str(out)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg: {proc.stderr[:300]}")


def prepare_labeling_job(file_path: str, chunk_seconds: float = 0, job_name: str = "") -> dict:
    """Диаризует ОДИН кусок файла и нарезает сольные фрагменты каждого голоса,
    чтобы человек их послушал и назвал людей. Платно: chunk_seconds * цена диаризации."""
    src = validate_input(file_path)
    chunk_seconds = chunk_seconds or float(CHUNK_SECONDS)
    job_name = job_name or f"{datetime.now().strftime('%Y-%m-%d')}_{_slug(src.stem)[:40]}"
    job_dir = Path(__file__).with_name("workspace") / job_name
    for sub in ("chunks", "diarized", "speakers"):
        (job_dir / sub).mkdir(parents=True, exist_ok=True)

    chunk_path = job_dir / "chunks" / f"chunk_000{src.suffix}"
    _cut_audio(src, 0, chunk_seconds, chunk_path)
    model = "gpt-4o-transcribe-diarize"
    segs = transcribe_chunk_diarized(Chunk(0, chunk_path, 0.0, chunk_seconds), model, "")
    cost = chunk_seconds / 60 * MODELS[model]["per_min"]
    record_cost(None, cost, "stt", model, f"{job_name}/chunk_000")
    (job_dir / "diarized" / "chunk_000.segments.json").write_text(
        json.dumps([vars(s) for s in segs], ensure_ascii=False, indent=2), encoding="utf-8")

    # сольные раны по спикерам
    runs_by_speaker: dict[str, list[dict]] = {}
    cur = None
    for s in sorted(segs, key=lambda x: x.start):
        if cur and s.speaker == cur["speaker"] and s.start - cur["end"] <= MAX_GAP_SEC:
            cur["end"] = max(cur["end"], s.end)
            cur["text"] += " " + s.text
        else:
            cur = {"speaker": s.speaker, "start": s.start, "end": s.end, "text": s.text}
            runs_by_speaker.setdefault(s.speaker, []).append(cur)

    speakers_map = []
    for label, runs in sorted(runs_by_speaker.items()):
        good = sorted((r for r in runs if r["end"] - r["start"] >= MIN_SOLO_SEC),
                      key=lambda r: r["end"] - r["start"], reverse=True)
        picked = []
        for i, r in enumerate(good[:SAMPLES_PER_SPEAKER], 1):
            dur = min(r["end"] - r["start"], MAX_REF_SEC)
            out = job_dir / "speakers" / f"{label.replace(':', '_')}_sample_{i}{src.suffix}"
            _cut_audio(src, r["start"], dur, out)
            picked.append({"file": str(out), "start_sec": round(r["start"], 1),
                           "duration_sec": round(dur, 1), "says": r["text"][:200]})
        entry = {
            "label": label, "name": None,
            "spoke_sec_in_chunk": round(sum(r["end"] - r["start"] for r in runs)),
            "samples": picked,
        }
        # длинный сольный кусок (до EMBED_CLIP_MAX_SEC) — отдельно, только для эмбеддинга:
        # он стабильнее короткого референса и при регистрации уйдёт в базу как kind=embed
        if good:
            dur = min(good[0]["end"] - good[0]["start"], EMBED_CLIP_MAX_SEC)
            embed_clip = job_dir / "speakers" / f"{label.replace(':', '_')}_embed{src.suffix}"
            _cut_audio(src, good[0]["start"], dur, embed_clip)
            entry["embed_clip"] = str(embed_clip)
            entry["embed_clip_sec"] = round(dur, 1)
        # авто-матч против базы voiceprints (локально, бесплатно): сильный матч — почти
        # наверняка этот человек, слабый — гипотеза, подтвердить у пользователя
        if entry.get("embed_clip") or picked:
            try:
                clip = Path(entry.get("embed_clip") or picked[0]["file"])
                matches = match_embedding(embed_audio(clip))
                if matches and matches[0][1] >= EMBED_WEAK:
                    entry["match"] = {"name": matches[0][0], "score": round(matches[0][1], 3),
                                      "confidence": "strong" if matches[0][1] >= EMBED_STRONG else "weak"}
            except Exception as exc:
                entry["match_error"] = str(exc)[:200]
        speakers_map.append(entry)
    (job_dir / "speakers" / "speakers.json").write_text(
        json.dumps(speakers_map, ensure_ascii=False, indent=2), encoding="utf-8")
    (job_dir / "job.json").write_text(json.dumps({
        "source": str(src),
        "created_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "awaiting_speaker_identification",
        "chunk_seconds": chunk_seconds,
        "diarize_model": model,
        "cost_so_far_usd": round(cost, 4),
        "next_step": ("человек слушает speakers/*_sample_* и называет людей, затем "
                      "register_voiceprint и полный transcribe с known_speakers"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"job_dir": str(job_dir), "cost_usd": round(cost, 4), "speakers": speakers_map}


# ------------------------------------------------------------- транскрибация

@dataclass
class Seg:
    start: float  # глобальные секунды
    end: float
    speaker: str
    text: str


def transcribe_chunk_text(chunk: Chunk, model: str, language: str) -> str:
    kwargs: dict = {"model": model, "response_format": "text"}
    if language:
        kwargs["language"] = language
    with open(chunk.path, "rb") as f:
        r = client().audio.transcriptions.create(file=f, **kwargs)
    return r if isinstance(r, str) else getattr(r, "text", str(r))


def transcribe_chunk_diarized(chunk: Chunk, model: str, language: str,
                               known: list[tuple[str, Path]] | None = None) -> list[Seg]:
    kwargs: dict = {"model": model, "response_format": "diarized_json", "chunking_strategy": "auto"}
    if language:
        kwargs["language"] = language
    known_names = {n for n, _ in (known or [])}
    if known:
        kwargs["known_speaker_names"] = [n for n, _ in known]
        kwargs["known_speaker_references"] = [audio_to_data_uri(p) for _, p in known]
    with open(chunk.path, "rb") as f:
        r = client().audio.transcriptions.create(file=f, **kwargs)
    data = r.model_dump() if hasattr(r, "model_dump") else json.loads(str(r))
    segs = []
    for s in data.get("segments", []):
        raw_label = str(s.get("speaker", "?"))
        # известные голоса модель возвращает как есть (совпадают между кусками сами по себе,
        # без кластеризации); безымянные метки чанк-локальны — префиксуем индексом куска,
        # чтобы не путать "A" из чанка 0 с "A" из чанка 3, и разрешаем их позже одним LLM-проходом
        label = raw_label if raw_label in known_names else f"{chunk.index}:{raw_label}"
        segs.append(Seg(
            start=float(s.get("start", 0)) + chunk.offset,
            end=float(s.get("end", 0)) + chunk.offset,
            speaker=label,
            text=str(s.get("text", "")).strip(),
        ))
    return segs


def run_chunks_with_progress(chunks: list[Chunk], worker, model: str, per_min: float,
                             tracker: CostTracker, file_label: str) -> list:
    """Гоняет worker(chunk) в пуле потоков, печатая по завершении каждого куска:
    номер/всего, прошедшее время, ETA, стоимость куска и накопленную стоимость запуска."""
    n = len(chunks)
    results: list = [None] * n
    done = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=PARALLEL_CHUNKS) as pool:
        futures = {pool.submit(worker, c): c for c in chunks}
        for fut in as_completed(futures):
            c = futures[fut]
            results[c.index] = fut.result()
            cost = c.duration / 60 * per_min
            run_total = record_cost(tracker, cost, "stt", model, file_label)
            done += 1
            elapsed = time.monotonic() - t0
            eta = elapsed / done * (n - done)
            print(f"[STT] {done}/{n} кусков — прошло {fmt_ts(elapsed)}, ETA ~{fmt_ts(eta)} — "
                  f"кусок ${cost:.3f}, в этом запуске ${run_total:.3f}", flush=True)
    return results


# ------------------------------------------------------------------- склейка

STITCH_PROMPT = """Два фрагмента одного непрерывного транскрипта: конец части A и начало части B.
Аудио резалось с перекрытием ~{overlap} секунд, поэтому на стыке текст частично продублирован.
Верни ТОЛЬКО итоговый текст стыка (замену обоим фрагментам): убери дубль, сохрани каждое
слово ровно один раз, ничего не перефразируй и не добавляй.

=== Конец A ===
{tail}
=== Начало B ===
{head}"""


def stitch_texts(texts: list[str], merge_model: str,
                 tracker: CostTracker | None = None, file_label: str = "") -> str:
    """Склеивает тексты кусков, убирая дубль на стыках дешёвой LLM (fallback — простая конкатенация)."""
    acc = texts[0].strip()
    n_joins = len(texts) - 1
    for idx, nxt in enumerate(texts[1:], 1):
        nxt = nxt.strip()
        tail, head = acc[-400:], nxt[:400]
        try:
            r = client().chat.completions.create(
                model=merge_model,
                messages=[{"role": "user", "content": STITCH_PROMPT.format(
                    overlap=OVERLAP_SECONDS, tail=tail, head=head)}],
            )
            cost = chat_call_cost(merge_model, getattr(r, "usage", None))
            run_total = record_cost(tracker, cost, "merge", merge_model, file_label)
            print(f"[merge] стык {idx}/{n_joins} — ${cost:.4f}, в этом запуске ${run_total:.3f}", flush=True)
            junction = (r.choices[0].message.content or "").strip()
            expected = len(tail) + len(head)
            if 0.25 * expected <= len(junction) <= 1.3 * expected:
                acc = acc[:-400] + junction + nxt[400:]
                continue
        except Exception:
            pass
        acc = acc + "\n\n" + nxt  # fallback: дубль ~5 с останется, но текст не потеряем
    return acc


CLUSTER_PROMPT = """This is one continuous conversation, split into chunks for processing.
Speaker tags below were assigned independently PER CHUNK, so the SAME physical person almost
always gets a DIFFERENT tag in each chunk (and can even get split into two tags within one
chunk if the automatic diarizer stumbled) — tags are not reliable identity, just raw signal.
Known/named speakers are already resolved correctly and excluded from this list: {known}.

Your job: cluster the tags below into the real distinct people actually present. {hint}

Rule of thumb: short interjections (single words like "yes", "yeah", "ok", "thanks", brief
overlapping replies) near another tag's line in time are extremely likely to belong to the
SAME small set of active participants, not a brand-new person. Only split out a new person when
the content clearly justifies it (a name/role is introduced, or the voice/topic is unrelated to
existing clusters and far from them in time). When in doubt, merge into an existing cluster.

Tags (id | first appearance | total time spoken | sample lines):
{items}

Return ONLY a JSON object mapping every id above to a cluster key. Reuse the SAME short key
string (e.g. "1", "2", "3") for ids you believe are the same person — use as few distinct keys
as the evidence supports."""


def merge_diarized_global(chunks_segs: list[list[Seg]], merge_model: str,
                           expected_speakers: int = 0, known_names: set[str] | None = None,
                           tracker: CostTracker | None = None, file_label: str = "") -> list[Seg]:
    """Склеивает диаризованные куски одним глобальным проходом кластеризации меток
    (вместо попарной склейки по стыкам) — устойчивее к раздуванию числа спикеров,
    т.к. видит все метки разом и по умолчанию склонна объединять, а не плодить новых.
    """
    known_names = known_names or set()
    result: list[Seg] = []
    for i, segs in enumerate(chunks_segs):
        if i > 0:
            boundary = i * CHUNK_SECONDS
            segs = [s for s in segs if s.end > boundary + 0.25]  # выкидываем дубль перекрытия
        result.extend(segs)
    result.sort(key=lambda s: s.start)

    labels_order: list[str] = []
    profiles: dict[str, dict] = {}
    for s in result:
        if s.speaker not in profiles:
            profiles[s.speaker] = {"start": s.start, "dur": 0.0, "samples": []}
            labels_order.append(s.speaker)
        profiles[s.speaker]["dur"] += max(0.0, s.end - s.start)
        if len(profiles[s.speaker]["samples"]) < 2 and s.text:
            profiles[s.speaker]["samples"].append(s.text[:160])

    unclustered = [l for l in labels_order if l not in known_names]
    mapping: dict[str, str] = {l: l for l in known_names & set(labels_order)}
    if unclustered:
        items = "\n".join(
            f'- id="{l}" | start={fmt_ts(profiles[l]["start"])} | spoke={profiles[l]["dur"]:.0f}s | '
            f'sample="{" / ".join(profiles[l]["samples"]) or "..."}"'
            for l in unclustered
        )
        hint = (f"There are EXACTLY {expected_speakers} real speakers among these unresolved ids "
                f"(named speakers above are separate and already handled) — merge down to at most that many."
                if expected_speakers else
                "Prefer the SMALLEST plausible number of distinct people consistent with the evidence.")
        try:
            print(f"[merge] кластеризация {len(unclustered)} меток спикеров по всей записи...", flush=True)
            r = client().chat.completions.create(
                model=merge_model,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": CLUSTER_PROMPT.format(
                    known=", ".join(sorted(known_names)) or "none", hint=hint, items=items)}],
            )
            cost = chat_call_cost(merge_model, getattr(r, "usage", None))
            run_total = record_cost(tracker, cost, "merge", merge_model, file_label)
            print(f"[merge] кластеризация спикеров готова — ${cost:.4f}, в этом запуске ${run_total:.3f}", flush=True)
            raw = json.loads(r.choices[0].message.content or "{}")
            for l in unclustered:
                mapping[l] = str(raw.get(l, l))
        except Exception:
            for l in unclustered:
                mapping[l] = l  # fallback: без объединения, как раньше — не хуже статус-кво

    canon: dict[str, str] = {}
    final: dict[str, str] = {}
    counter = 0
    for l in labels_order:
        if l in known_names:
            final[l] = l
            continue
        cl = mapping.get(l, l)
        if cl not in canon:
            counter += 1
            canon[cl] = f"S{counter}"
        final[l] = canon[cl]

    return [Seg(s.start, s.end, final.get(s.speaker, s.speaker), s.text) for s in result]


def fmt_ts(sec: float) -> str:
    sign = "-" if sec < 0 else ""
    sec = abs(sec)
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{sign}{h:02d}:{m:02d}:{s:02d}"


def format_diarized(segs: list[Seg]) -> str:
    """Markdown: реплики подряд от одного спикера объединяются в один блок."""
    lines = []
    cur_speaker, cur_start, cur_texts = None, 0.0, []
    for s in segs:
        if s.speaker != cur_speaker:
            if cur_texts:
                lines.append(f"**[{fmt_ts(cur_start)}] {cur_speaker}:** {' '.join(cur_texts)}")
            cur_speaker, cur_start, cur_texts = s.speaker, s.start, []
        cur_texts.append(s.text)
    if cur_texts:
        lines.append(f"**[{fmt_ts(cur_start)}] {cur_speaker}:** {' '.join(cur_texts)}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------- core

def validate_input(file_path: str) -> Path:
    p = Path(file_path)
    if not p.is_absolute():
        raise ValueError(f"Нужен абсолютный путь, получено: {file_path}")
    if not p.exists():
        raise ValueError(f"Файл не найден: {p}")
    if p.suffix.lower() not in AUDIO_EXTS:
        raise ValueError(f"Неподдерживаемое расширение {p.suffix}. Можно: {', '.join(sorted(AUDIO_EXTS))}")
    return p


def write_transcript(out: Path, src_name: str, src_path: str, duration: float, model: str,
                     merge_model: str, n_chunks: int, cost: float, n_speakers: int | None, body: str) -> None:
    header = "\n".join([
        f"# Транскрипт: {src_name}",
        "",
        f"- Источник: `{src_path}`",
        f"- Длительность: {fmt_ts(duration)}",
        f"- Модель: {model}" + (f" + {merge_model} (склейка)" if n_chunks > 1 else ""),
        f"- Кусков: {n_chunks}" + (f" × {CHUNK_SECONDS // 60} мин, перекрытие {OVERLAP_SECONDS} с" if n_chunks > 1 else ""),
        *([f"- Спикеров: {n_speakers}"] if n_speakers else []),
        f"- Стоимость: ~${cost:.2f}",
        f"- Создан: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
    ])
    out.write_text(header + body + "\n", encoding="utf-8")


def transcribe_file(file_path: str, model: str = "", language: str = "",
                    output_path: str = "", merge_model: str = "",
                    expected_speakers: int = 0, known_speakers: str = "") -> dict:
    src = validate_input(file_path)
    model = model or DEFAULT_STT_MODEL
    merge_model = merge_model or DEFAULT_MERGE_MODEL
    if model not in MODELS:
        raise ValueError(f"Неизвестная модель {model}. Доступны: {', '.join(MODELS)}")
    diarize = MODELS[model]["diarization"]
    known = parse_known_speakers(known_speakers) if known_speakers else []
    if known and not diarize:
        raise ValueError("known_speakers работает только с диаризующей моделью (gpt-4o-transcribe-diarize)")
    known_names = {n for n, _ in known}

    duration = ffprobe_duration(src)
    size = src.stat().st_size
    out = Path(output_path) if output_path else src.with_suffix(".transcript.md")
    chunks_json = out.with_suffix(".chunks.json")

    tracker = CostTracker()
    needs_split = size > MAX_SINGLE_BYTES or duration > MAX_SINGLE_SECONDS
    with tempfile.TemporaryDirectory(prefix="transcribe_") as td:
        if needs_split:
            chunks = split_audio(src, duration, Path(td))
        else:
            chunks = [Chunk(index=0, path=src, offset=0.0, duration=duration)]
        print(f"[STT] {src.name}: {fmt_ts(duration)}, {len(chunks)} кусков, модель {model}", flush=True)

        worker = ((lambda c: transcribe_chunk_diarized(c, model, language, known)) if diarize
                 else (lambda c: transcribe_chunk_text(c, model, language)))
        per_chunk = run_chunks_with_progress(chunks, worker, model, MODELS[model]["per_min"],
                                             tracker, src.name)

    # сырые данные по кускам — на диск, чтобы можно было пересобрать (remerge) без повторной оплаты
    sidecar = {
        "source": str(src), "model": model, "diarize": diarize, "duration": duration,
        "known_speakers": sorted(known_names),
    }
    if diarize:
        sidecar["segments_per_chunk"] = [[asdict(s) for s in segs] for segs in per_chunk]
    else:
        sidecar["texts"] = per_chunk
    chunks_json.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")

    if diarize:
        segs = merge_diarized_global(per_chunk, merge_model, expected_speakers, known_names,
                                     tracker, src.name)
        body = format_diarized(segs)
        n_speakers = len({s.speaker for s in segs})
    else:
        body = stitch_texts(per_chunk, merge_model, tracker, src.name) if len(per_chunk) > 1 else per_chunk[0].strip()
        n_speakers = None

    write_transcript(out, src.name, str(src), duration, model, merge_model, len(per_chunk),
                     tracker.usd, n_speakers, body)
    today_usd, total_usd = ledger_summary()
    print(f"[done] {out} — в этом запуске ${tracker.usd:.3f}, сегодня ${today_usd:.3f}, всего ${total_usd:.3f}",
          flush=True)
    return {
        "output": str(out),
        "chunks_json": str(chunks_json),
        "duration": fmt_ts(duration),
        "chunks": len(per_chunk),
        "speakers": n_speakers,
        "model": model,
        "cost_run_usd": round(tracker.usd, 3),
        "cost_today_usd": round(today_usd, 3),
        "cost_alltime_usd": round(total_usd, 3),
        "preview": body[:400],
    }


def remerge_file(chunks_json_path: str, expected_speakers: int = 0, merge_model: str = "",
                 output_path: str = "") -> dict:
    p = Path(chunks_json_path)
    if not p.is_absolute() or not p.exists():
        raise ValueError(f"chunks_json_path должен быть абсолютным путём к существующему файлу: {chunks_json_path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    merge_model = merge_model or DEFAULT_MERGE_MODEL
    src_path = data["source"]
    out = Path(output_path) if output_path else Path(src_path).with_suffix(".transcript.md")
    tracker = CostTracker()
    file_label = Path(src_path).name

    if data["diarize"]:
        chunks_segs = [[Seg(**d) for d in segs] for segs in data["segments_per_chunk"]]
        known_names = set(data.get("known_speakers", []))
        segs = merge_diarized_global(chunks_segs, merge_model, expected_speakers, known_names,
                                     tracker, file_label)
        body = format_diarized(segs)
        n_speakers = len({s.speaker for s in segs})
        n_chunks = len(chunks_segs)
    else:
        texts = data["texts"]
        body = stitch_texts(texts, merge_model, tracker, file_label) if len(texts) > 1 else texts[0].strip()
        n_speakers = None
        n_chunks = len(texts)

    write_transcript(out, file_label, src_path, data["duration"], data["model"],
                     merge_model, n_chunks, tracker.usd, n_speakers, body)
    today_usd, total_usd = ledger_summary()
    print(f"[done] {out} — пересборка ${tracker.usd:.4f}, сегодня ${today_usd:.3f}, всего ${total_usd:.3f}",
          flush=True)
    return {
        "output": str(out), "speakers": n_speakers, "model": data["model"], "merge_model": merge_model,
        "cost_run_usd": round(tracker.usd, 4), "cost_today_usd": round(today_usd, 3),
        "cost_alltime_usd": round(total_usd, 3), "preview": body[:400],
    }


# ------------------------------------------------------------------ MCP tools

@mcp.tool()
def list_models() -> str:
    """Список STT-моделей и стоимость обработки минуты речи."""
    rows = [f"| {m} | ${v['per_min']:.4f}/мин (${v['per_min'] * 60:.2f}/час) | "
            f"{'да' if v['diarization'] else 'нет'} | {v['notes']} |" for m, v in MODELS.items()]
    return ("| Модель | Цена | Диаризация | Заметки |\n|---|---|---|---|\n" + "\n".join(rows)
            + f"\n\nПо умолчанию: {DEFAULT_STT_MODEL}. Склейка кусков: {DEFAULT_MERGE_MODEL} (копейки, ~$0.001 за LLM-вызов).")


@mcp.tool()
def estimate(file_path: str) -> str:
    """Оценка файла перед транскрибацией: длительность, число кусков, стоимость по каждой модели."""
    src = validate_input(file_path)
    duration = ffprobe_duration(src)
    size_mb = src.stat().st_size / 1024 / 1024
    needs_split = src.stat().st_size > MAX_SINGLE_BYTES or duration > MAX_SINGLE_SECONDS
    n = max(1, -(-int(duration) // CHUNK_SECONDS)) if needs_split else 1
    costs = "\n".join(f"- {m}: ~${duration / 60 * v['per_min']:.2f}" for m, v in MODELS.items())
    return (f"{src.name}: {fmt_ts(duration)}, {size_mb:.1f} MB, кусков: {n}\n\nСтоимость:\n{costs}")


@mcp.tool()
def usage() -> str:
    """Расход по журналу usage_ledger.jsonl: за сегодня (по локальной дате) и за всё время."""
    today_usd, total_usd = ledger_summary()
    return f"Потрачено сегодня: ${today_usd:.3f}\nПотрачено за всё время: ${total_usd:.3f}"


@mcp.tool()
def list_voiceprints() -> str:
    """База известных голосов: кто есть, под какими алиасами, сколько сэмплов."""
    people = load_registry()
    if not people:
        return ("База voiceprints пуста. Разметка: prepare_speaker_labeling(file) -> "
                "человек слушает и называет -> register_voiceprint.")
    lines = []
    for p in people:
        aliases = ", ".join(p.get("aliases", [])) or "-"
        lines.append(f"- {p['name']} (алиасы: {aliases}) — сэмплов: {len(p.get('samples', []))}")
    return "\n".join(lines)


@mcp.tool()
def find_voiceprint(query: str) -> str:
    """Найти человека в базе голосов по имени или алиасу ('Аня с курсов' -> Anna Ivanova).

    Используй ПЕРЕД диаризацией с known_speakers: пользователь называет ожидаемых
    участников — проверь, что на каждого есть сэмпл. Если нет — предложи разметку.
    """
    matches = find_voiceprints_data(query)
    if not matches:
        return (f"{query!r} в базе не найден. Предложи пользователю разметку: "
                f"prepare_speaker_labeling(file), человек слушает сэмплы и называет людей, "
                f"затем register_voiceprint.")
    out = []
    for p in matches[:3]:
        samples = "; ".join(f"{s['file']} ({s['duration_sec']}с)" for s in p.get("samples", []))
        out.append(f"{p['name']} (алиасы: {', '.join(p.get('aliases', [])) or '-'})\n  сэмплы: {samples or 'нет'}")
    return "\n".join(out)


@mcp.tool()
def register_voiceprint(name: str, sample_files: str, aliases: str = "", source: str = "",
                        embed_files: str = "") -> str:
    """Сохранить голос человека в базу voiceprints (для будущих known_speakers).

    Args:
        name: каноничное имя (например "Anna Ivanova")
        sample_files: пути к коротким аудио-сэмплам (2-10с каждый, для API) через запятую
        aliases: альтернативные написания через запятую ("Анна Иванова, Аня с курсов")
        source: откуда сэмплы (файл/запись), для истории
        embed_files: пути к длинным (до 45с) кускам сольной речи через запятую — только
            для локального матчинга, точность опознания заметно выше; разметка
            (prepare_speaker_labeling) кладёт такой кусок в *_embed рядом с сэмплами
    """
    paths = [Path(p.strip()) for p in sample_files.split(",") if p.strip()]
    epaths = [Path(p.strip()) for p in embed_files.split(",") if p.strip()]
    for p in paths + epaths:
        if not p.is_absolute() or not p.exists():
            raise ValueError(f"Сэмпл не найден: {p}")
    person = register_voiceprint_files(
        name, paths, [a.strip() for a in aliases.split(",") if a.strip()], source, epaths)
    return (f"Сохранено: {person['name']} (алиасы: {', '.join(person['aliases']) or '-'}), "
            f"сэмплов теперь {len(person['samples'])}. В transcribe можно передавать "
            f"known_speakers=\"{person['name']}\" — референс подставится из базы.")


@mcp.tool()
def resolve_speakers(chunks_json_path: str, output_path: str = "") -> str:
    """Опознать/пересобрать спикеров уже готового диаризованного транскрипта — локально,
    БЕСПЛАТНО, без LLM: эмбеддинг длиннейшего сольного куска каждой чанк-метки, жадная
    кластеризация по косинусу через всю запись, матч кластеров против базы voiceprints.
    Строже и надёжнее LLM-склейки, особенно на записях с шумом/музыкой.

    Args:
        chunks_json_path: путь к <транскрипт>.chunks.json от прогона transcribe
        output_path: куда писать; пусто = поверх исходного .transcript.md
    """
    r = resolve_speakers_by_embeddings(chunks_json_path, output_path)
    lines = [f"Готово: {r['output']} — спикеров {r['speakers']}"]
    if r["known_kept"]:
        lines.append(f"Сохранены известные: {', '.join(r['known_kept'])}")
    for c in r["clusters"]:
        listen = f" | послушать: {c['listen']}" if c["listen"] and c["name"].startswith("S") else ""
        lines.append(f"- {c['name']}: ~{c['spoke_sec']}с речи, меток {len(c['labels'])}, "
                     f"матч с базой {c['best_base_cos']}{listen}")
    lines.append("Кластеры S1/S2... — незнакомцы: дай пользователю послушать клипы и, "
                 "если он их назовёт, register_voiceprint + повторный resolve_speakers.")
    return "\n".join(lines)


@mcp.tool()
def match_voice(sample_file: str) -> str:
    """Определить, чей голос в аудиоклипе, по базе voiceprints. Локально и бесплатно
    (sherpa-onnx эмбеддинг + косинус), ~100мс. Клип: желательно 3-10с одного голоса.

    Args:
        sample_file: абсолютный путь к аудиоклипу
    """
    p = Path(sample_file)
    if not p.is_absolute() or not p.exists():
        raise ValueError(f"Файл не найден: {sample_file}")
    matches = match_embedding(embed_audio(p))
    if not matches:
        return "База voiceprints пуста — сравнивать не с кем."
    lines = []
    for name, score in matches[:5]:
        verdict = ("тот же голос" if score >= EMBED_STRONG
                   else "похоже, подтвердить" if score >= EMBED_WEAK else "не он")
        lines.append(f"- {name}: {score:.3f} ({verdict})")
    return "\n".join(lines)


@mcp.tool()
def prepare_speaker_labeling(file_path: str, chunk_seconds: float = 0) -> str:
    """Разметка голосов: диаризует ПЕРВЫЙ кусок файла и нарезает сольные фрагменты
    каждого голоса, чтобы пользователь их послушал и назвал людей.

    ПЛАТНО (диаризация одного куска, ~$0.06 за 10 мин) — согласуй с пользователем.
    После его ответов: register_voiceprint(name, sample_files=...) для каждого голоса,
    затем полный transcribe(known_speakers="Имя1, Имя2").

    Args:
        file_path: абсолютный путь к аудио
        chunk_seconds: сколько секунд с начала анализировать (0 = стандартный кусок 600с)
    """
    r = prepare_labeling_job(file_path, chunk_seconds)
    lines = [f"Разметка готова: {r['job_dir']} (стоило ${r['cost_usd']})",
             "Авто-матч по базе уже сделан; strong-матчи можно предложить пользователю "
             "как готовый ответ, weak и без матча — дать послушать и спросить:"]
    for sp in r["speakers"]:
        if not sp["samples"]:
            lines.append(f"- {sp['label']}: говорил ~{sp['spoke_sec_in_chunk']}с, "
                         f"сольных фрагментов нет (вероятно фоновый голос)")
            continue
        m = sp.get("match")
        guess = (f" -> авто-матч: {m['name']} ({m['score']}, {m['confidence']})" if m
                 else " -> в базе не найден")
        lines.append(f"- {sp['label']}: говорил ~{sp['spoke_sec_in_chunk']}с{guess}")
        for s in sp["samples"]:
            lines.append(f"    {s['file']} — «{s['says'][:120]}»")
    return "\n".join(lines)


@mcp.tool()
def transcribe(file_path: str, model: str = "", language: str = "",
               output_path: str = "", merge_model: str = "",
               expected_speakers: int = 0, known_speakers: str = "") -> str:
    """Транскрибирует аудиофайл, пишет результат рядом с ним в <имя>.transcript.md.
    ВСЕГДА платное действие — согласуйте параметры/стоимость с пользователем перед вызовом,
    если это не было явно оговорено заранее.

    Args:
        file_path: абсолютный путь к аудиофайлу (m4a, mp3, wav, ogg, webm, flac...)
        model: STT-модель (см. list_models); пусто = gpt-transcribe;
               для разделения по спикерам — gpt-4o-transcribe-diarize
        language: ISO-код языка (ru, en...); пусто = автоопределение
        output_path: куда писать транскрипт; пусто = рядом с аудио
        merge_model: LLM для склейки кусков; пусто = gpt-5-mini
        expected_speakers: только для диаризации — если известно число участников (без учёта
            known_speakers), передайте его: склейка станет заметно строже и перестанет плодить
            лишних спикеров на неуверенных стыках. 0 = не ограничивать явно.
        known_speakers: только для диаризации — JSON-список до 4 человек с именем и коротким
            (2-10с) референс-клипом голоса: [{"name": "Anna", "reference": "C:\\\\clip.wav"}].
            Эти голоса получат настоящее имя и не путаются между кусками по построению.
    """
    r = transcribe_file(file_path, model, language, output_path, merge_model,
                        expected_speakers, known_speakers)
    lines = [f"Готово: {r['output']}",
             f"Длительность {r['duration']}, кусков {r['chunks']}, модель {r['model']}"]
    if r["speakers"]:
        lines.append(f"Спикеров: {r['speakers']}")
    lines.append(f"Стоимость: этот запуск ${r['cost_run_usd']}, сегодня ${r['cost_today_usd']}, "
                 f"всего ${r['cost_alltime_usd']}")
    lines.append(f"Сырые данные для пересборки (remerge, без повторной оплаты): {r['chunks_json']}")
    lines.append(f"\nНачало:\n{r['preview']}")
    return "\n".join(lines)


@mcp.tool()
def remerge(chunks_json_path: str, expected_speakers: int = 0, merge_model: str = "",
           output_path: str = "") -> str:
    """Пересобирает транскрипт из уже полученных сырых данных БЕЗ повторной оплаты транскрибации
    (только копеечная LLM-склейка).

    Полезно, если диаризация дала слишком много/мало спикеров — подберите expected_speakers
    или другой merge_model и пересоберите за копейки.

    Args:
        chunks_json_path: путь к <транскрипт>.chunks.json, создаётся автоматически при transcribe
        expected_speakers: желаемое (примерное) число реальных спикеров, кроме known_speakers
        merge_model: LLM для склейки; пусто = gpt-5-mini
        output_path: куда писать; пусто = поверх исходного <имя>.transcript.md
    """
    r = remerge_file(chunks_json_path, expected_speakers, merge_model, output_path)
    lines = [f"Готово: {r['output']} (модель склейки: {r['merge_model']})"]
    if r["speakers"]:
        lines.append(f"Спикеров: {r['speakers']}")
    lines.append(f"Стоимость пересборки: ${r['cost_run_usd']} | сегодня ${r['cost_today_usd']} | "
                 f"всего ${r['cost_alltime_usd']}")
    lines.append(f"\nНачало:\n{r['preview']}")
    return "\n".join(lines)


# ------------------------------------------------------------------------ CLI

def _cli() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1]
    if cmd == "usage":
        print(usage())
        return
    path = sys.argv[2]
    opts = dict(zip([a.lstrip("-") for a in sys.argv[3::2]], sys.argv[4::2]))
    if cmd == "estimate":
        print(estimate(path))
    elif cmd == "transcribe":
        r = transcribe_file(path, opts.get("model", ""), opts.get("language", ""),
                            opts.get("output", ""), opts.get("merge-model", ""),
                            int(opts.get("expected-speakers", 0)), opts.get("known-speakers", ""))
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif cmd == "remerge":
        r = remerge_file(path, int(opts.get("expected-speakers", 0)),
                         opts.get("merge-model", ""), opts.get("output", ""))
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif cmd == "resolve":
        r = resolve_speakers_by_embeddings(path, opts.get("output", ""))
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        sys.exit(f"Неизвестная команда: {cmd}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _cli()
    else:
        mcp.run()
