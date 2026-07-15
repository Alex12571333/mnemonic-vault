# Mnemonic Vault

[![Tests](https://github.com/Alex12571333/mnemonic-vault/actions/workflows/tests.yml/badge.svg)](https://github.com/Alex12571333/mnemonic-vault/actions/workflows/tests.yml)

![Mnemonic Vault — portable file-first memory architecture](assets/mnemonic-vault-hero.png)

> Одна переносимая папка. Неизменяемая история. Быстрый гибридный поиск.

Переносимая файловая долговременная память для агентов. Полные разговоры остаются
append-only файлами, тематические summary хранятся в Markdown, а SQLite используется
только как восстанавливаемый каталог и поисковый индекс.

## Инварианты

- `transcript.jsonl` — источник истины и никогда не заменяется summary.
- Topic Markdown обновляется атомарно через временный файл, `fsync` и `rename`.
- Сообщение сначала дописывается и синхронизируется на диск, затем меняются метаданные.
- `catalog.sqlite` можно удалить и пересоздать командой `rebuild-index`.
- Memory LLM вызывается только фоновым summarizer; retrieval не использует генеративную LLM.
- Пути в SQLite относительны к `data/`, поэтому папку проекта можно переносить.
- Векторный KNN выполняет `sqlite-vec`; exact cosine scan остаётся fallback при
  недоступном native-расширении.

## Быстрый запуск

Требуется Python 3.11–3.13.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py serve --host 0.0.0.0 --port 8765
```

Проверенная локальная схема развёртывания:

- приложение и FastEmbed: хост `192.168.0.14`;
- Memory LLM Qwen/vLLM: `http://192.168.0.10:8000/v1`;
- embedding-модель: `sentence-transformers/all-MiniLM-L6-v2`, 384 измерения;
- cache embedding-модели: `data/models/` внутри переносимой папки.

FastEmbed запускается в процессе приложения и не требует отдельного HTTP-сервиса.
При необходимости его можно заменить OpenAI-compatible endpoint через конфигурацию.

Те же параметры можно переопределить без редактирования файла:

```bash
export ETERNAL_MEMORY_LLM_BASE_URL=http://192.168.0.10:8000/v1
export ETERNAL_MEMORY_LLM_MODEL=exact-memory-model-id
export ETERNAL_MEMORY_EMBEDDINGS_PROVIDER=openai-compatible
export ETERNAL_MEMORY_EMBEDDINGS_BASE_URL=http://192.168.0.14:PORT/v1
export ETERNAL_MEMORY_EMBEDDINGS_MODEL=exact-embedding-model-id
```

## API

Основные endpoints:

```text
POST /v1/sessions/start
POST /v1/sessions/{session_id}/messages
POST /v1/sessions/{session_id}/end
POST /v1/memory/search
GET  /v1/memory/topics/{topic_id}
POST /v1/memory/topics/{topic_id}/expand
GET  /v1/sessions/{session_id}/turns?from=1&to=20
```

Пример:

```bash
curl -sS http://127.0.0.1:8765/v1/sessions/start \
  -H 'content-type: application/json' \
  -d '{"agent":"openclaw"}'

curl -sS http://127.0.0.1:8765/v1/memory/search \
  -H 'content-type: application/json' \
  -d '{"query":"какая команда для DFlash?","include_sources":"auto"}'
```

## Фоновые jobs и обслуживание

API запускает job runner автоматически. Его можно запускать отдельно:

```bash
python run.py process-jobs
python run.py rebuild-index
python run.py rebuild-index --with-embeddings
python run.py reembed-all
python run.py resummarize --session session-a83f
```

Незавершённые jobs при старте переводятся обратно в `pending`. После трёх неудачных
попыток job получает статус `failed`; исходный transcript при этом уже сохранён.

## Перенос и резервная копия

Остановите процесс или сделайте SQLite checkpoint, затем скопируйте всю папку:

```bash
cp -a eternal-memory eternal-memory-backup
```

Если каталог был скопирован во время записи или оказался повреждён:

```bash
python run.py rebuild-index
```

## Запуск как user service на Linux

В репозитории есть готовый unit `deploy/mnemonic-vault.service`. По умолчанию он
ожидает проект в `~/mnemonic-vault` и слушает только loopback `127.0.0.1:8765`:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/mnemonic-vault.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now mnemonic-vault.service
curl -sS http://127.0.0.1:8765/health
```

## Тесты

```bash
python -m unittest discover -s tests -v
```
