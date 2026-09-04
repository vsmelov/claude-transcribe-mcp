# transcribe-mcp

MCP-сервер для транскрибации локальных аудиофайлов через OpenAI STT.
Написан Claude Code для собственного использования.

## Что умеет

- **list_models** — STT-модели и цена за минуту речи (сверено с прайсом OpenAI 2026-08-14):

  | Модель | $/мин | Диаризация |
  |---|---|---|
  | `gpt-transcribe` (default) | 0.0045 | нет |
  | `gpt-4o-transcribe` | 0.006 | нет |
  | `gpt-4o-mini-transcribe` | 0.003 | нет |
  | `gpt-4o-transcribe-diarize` | 0.006 | **да** (S1, S2, ... + таймкоды) |
  | `whisper-1` | 0.006 | нет |

- **estimate(file_path)** — длительность, число кусков, стоимость по каждой модели.
- **transcribe(file_path, model, language, output_path, merge_model)** — транскрипт
  пишется рядом с аудио в `<имя>.transcript.md`.

## Вход

Абсолютный путь к локальному файлу: flac, m4a, mp3, mp4, mpeg, mpga, oga, ogg, wav, webm.

## Большие файлы

Лимит API — 25 MB / ~25 мин на запрос, поэтому файлы >24 MB или >15 мин режутся
ffmpeg'ом (stream copy, без перекодирования) на куски по 10 минут с перекрытием 5 с,
транскрибируются в 3 потока и склеиваются дешёвой LLM (`gpt-5-mini`, ~$0.001 на стык):

- обычный текст — LLM убирает продублированный текст на стыке (fallback — конкатенация);
- диаризация — метки спикеров в каждом куске свои, LLM сшивает их в сквозные S1..Sn
  по совпадающим репликам в перекрытии (fallback — куски получают отдельных спикеров).

Известное ограничение: при диаризации сегмент, попавший ровно на границу куска,
может дать небольшой дубль текста.

## Установка

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Ключ — в `.env` рядом с `server.py` (`OPENAI_API_KEY=...`). Требуется ffmpeg в PATH.

Регистрация в Claude Code — в `~/.claude.json` → `mcpServers`:

```json
"transcribe": {
  "command": "C:\\path\\to\\transcribe-mcp\\.venv\\Scripts\\python.exe",
  "args": ["C:\\path\\to\\transcribe-mcp\\server.py"]
}
```

## CLI для отладки (без MCP)

```
.venv\Scripts\python server.py estimate <file>
.venv\Scripts\python server.py transcribe <file> [--model gpt-4o-transcribe-diarize] [--language ru] [--output PATH]
```

## База голосов и авто-опознание

- `voiceprints/` — база известных голосов: `registry.json` (имя, алиасы, сэмплы),
  `embeddings.json` (192-float векторы), аудио по папкам. Всё смотрится глазами.
- `known_speakers` в `transcribe` принимает просто имена/алиасы через запятую —
  референсы подставляются из базы («Аня с курсов» → Anna Ivanova).
- Разметка нового файла: `prepare_speaker_labeling` (диаризует первый кусок, режет
  сольные фрагменты) → авто-матч каждого голоса против базы (sherpa-onnx CAM++,
  локально, ~100мс/клип; свой ~0.89 косинус, чужой ~0.25) → незнакомые голоса
  слушает и называет человек → `register_voiceprint`.
- Модель эмбеддингов: `models/campplus_zh_en_advanced.onnx` (27 МБ), скачать с
  [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models)
  (`3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx`).
- Лимит OpenAI API — 4 known_speakers на запрос; большая база живёт локально и
  используется для выбора нужной четвёрки под конкретную запись.
