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
.venv/bin/python run.py serve --host 127.0.0.1 --port 8765
```

Проверенная локальная схема развёртывания:

- приложение и FastEmbed: хост `192.168.0.14`;
- Memory LLM Qwen/vLLM: `http://192.168.0.10:8000/v1`;
- embedding-модель: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,
  384 измерения;
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
POST /v1/memory/search-transcript
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

Для lossless-интеграций сообщение также содержит стабильный ID события:

```json
{
  "role": "user",
  "content": "Какой фикс использовали для DFlash?",
  "external_event_id": "agent-turn-7f1a"
}
```

Повтор с той же парой `session_id + external_event_id` возвращает уже сохранённое
сообщение и не добавляет вторую строку в transcript.

## Надёжная доставка

OpenClaw и Hermes сначала делают `fsync` события в переносимый spool:

```text
data/spool/openclaw.jsonl
data/spool/openclaw.dead-letter.jsonl
data/spool/hermes.jsonl
data/spool/hermes.dead-letter.jsonl
```

После подтверждения API в spool дописывается отметка `delivered`. Неподтверждённые
события повторяются после восстановления Vault или перезапуска агента. Идемпотентность
API исключает дубли, если сервер сохранил сообщение, но HTTP-ответ потерялся.

Идентификатор Vault-сессии строится из постоянного имени установки агента и внешнего
ID чата, поэтому рестарт процесса не дробит историю. Постоянные ошибки `400/413/422`
карантинируются в dead-letter и не блокируют следующие события; `401/403` останавливают
доставку до исправления конфигурации, сетевые ошибки и `5xx` повторяются. Append в уже
завершённую сессию автоматически переносится в детерминированную recovery-сессию.

## Relevance и transcript index

Абсолютный lexical coverage / cosine gate отбрасывает слабые кандидаты до RRF.
RRF используется только для порядка уже релевантных тем. Полный архив ищется через
восстанавливаемый `messages_fts`; `search_transcript` больше не загружает все JSONL.
`rebuild-index` восстанавливает topic index, message index и event-id каталог из файлов.

## Фоновые jobs и обслуживание

API запускает job runner автоматически. Его можно запускать отдельно:

```bash
python run.py process-jobs
python run.py rebuild-index
python run.py rebuild-index --with-embeddings
python run.py reembed-all
python run.py resummarize --session session-a83f
python run.py retry-failed
python run.py retry-failed --session session-a83f
```

Незавершённые jobs при старте переводятся обратно в `pending`. После трёх неудачных
попыток job получает статус `failed`; исходный transcript при этом уже сохранён.

## Нативные интеграции агентов

Версия 0.3 включает два lossless-адаптера:

- OpenClaw memory-slot plugin с lifecycle hooks, шестью memory tools и встроенным skill;
- Hermes Agent `MemoryProvider` с persistent spool, bounded prefetch и теми же tools.

Оба адаптера автоматически сохраняют ходы, подмешивают только небольшой релевантный
контекст и позволяют раскрывать исходные transcript ranges для точных значений.
Установка и проверка описаны в [документации интеграций](docs/native-integrations.md).

## Перенос и резервная копия

Остановите процесс или сделайте SQLite checkpoint, затем скопируйте всю папку:

```bash
cp -a eternal-memory eternal-memory-backup
```

Если каталог был скопирован во время записи или оказался повреждён:

```bash
python run.py rebuild-index
```

## Доступ из LAN

Без bearer token приложение отказывается слушать любой не-loopback адрес. Для LAN:

```bash
export MNEMONIC_VAULT_API_TOKEN='use-a-long-random-secret'
.venv/bin/python run.py serve --host 192.168.0.14 --port 8765
```

Клиенты должны передавать `Authorization: Bearer …`. Размер сообщения и поискового
запроса, а также общий размер HTTP-body ограничиваются секцией `api` в
`config/config.yaml`.

## Оценка embedding на собственной памяти

Скопируйте `benchmarks/retrieval-ru.example.jsonl`, замените topic IDs и добавьте
точные, перефразированные, RU/EN, короткие и отрицательные запросы. Затем:

```bash
python run.py evaluate-retrieval --dataset benchmarks/retrieval-ru.jsonl --top-k 5
```

Команда считает recall@5 и долю корректно отвергнутых нерелевантных запросов.
Это позволяет сравнить MiniLM, multilingual E5 и BGE-M3 на реальной памяти.
В проверенной конфигурации multilingual MiniLM заменил English-only MiniLM:
на реальных RU-темах он отделил релевантные запросы (`0.38–0.76`) от контрольных
нерелевантных (`−0.02–0.10`) при absolute gate `0.35`.

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
cd integrations/openclaw/mnemonic-vault && npm ci && npm run check
```
