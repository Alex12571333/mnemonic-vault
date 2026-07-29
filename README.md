# Mnemonic Vault

[![Tests](https://github.com/Alex12571333/mnemonic-vault/actions/workflows/tests.yml/badge.svg)](https://github.com/Alex12571333/mnemonic-vault/actions/workflows/tests.yml)

![Mnemonic Vault — portable file-first memory architecture](assets/mnemonic-vault-hero.png)

> Одна переносимая папка. Неизменяемая история. Быстрый гибридный поиск.

Переносимая файловая долговременная память для агентов. Полные разговоры остаются
append-only файлами, тематические summary хранятся в Markdown, а SQLite используется
только как восстанавливаемый каталог и поисковый индекс.

## Инварианты

- `transcript.jsonl` — источник истины и никогда не заменяется summary.
- `explicit-memory.jsonl` — append-only журнал только прямых пользовательских
  команд «запомни»; SQLite хранит его восстанавливаемое текущее представление.
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
POST /v1/memory/remember
POST /v1/memory/search-transcript
GET  /v1/memory/explicit/{memory_id}
GET  /v1/memory/topics/{topic_id}
POST /v1/memory/topics/{topic_id}/expand
GET  /v1/memory/global-topics
GET  /v1/memory/global-topics/{global_topic_id}
GET  /v1/sessions/{session_id}/turns?from=1&to=20
GET  /v1/sessions/{session_id}/aliases
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

Произвольный `external_event_id` идемпотентен внутри сессии. Нативные адаптеры
создают детерминированные `event-*` из постоянного ID установки, внешнего ID чата,
роли и ID/номера хода; такие события идемпотентны глобально, в том числе после
рестарта агента или перенаправления в recovery-сессию.

## Пятый триггер: явное «запомни»

Четыре фоновых триггера summary остаются прежними: число сообщений, число токенов,
idle timeout и `session_end`. Пятый триггер не ждёт Memory LLM:

```text
точное сообщение пользователя
→ fsync transcript.jsonl
→ fsync data/explicit-memory.jsonl
→ SQLite FTS
→ факт уже доступен retrieval
→ priority summary-job
→ позднее включение в session topic
```

Если embedding или Qwen/vLLM недоступны, успешная явная запись всё равно сразу
находится лексически. Embedding достраивается командой `reembed-all`, а summary-job
остаётся в очереди до восстановления Memory LLM.

```bash
curl -sS http://127.0.0.1:8765/v1/memory/remember \
  -H 'content-type: application/json' \
  -d '{
    "verbatim":"Запомни: production Mnemonic Vault работает на 192.168.0.14",
    "normalized":"Production Mnemonic Vault работает на 192.168.0.14",
    "kind":"configuration",
    "scope":{"type":"project","id":"mnemonic-vault"},
    "idempotency_key":"event-production-server-14"
  }'
```

Допустимые `kind`: `fact`, `preference`, `decision`, `configuration`, `identity`,
`constraint`, `task`, `correction`. Scope бывает `global`, `agent`, `project` и
`session`; у всех, кроме `global`, обязателен `id`. В v0.5.0 явные записи всегда
`author=user`: агент не может самовольно создавать глобальные воспоминания.

`verbatim` хранит точные слова пользователя, а `normalized` — отдельную поисковую
формулировку. Повтор с тем же `idempotency_key` возвращает прежний `memory_id`.
Новый факт с `supersedes` не переписывает старый: прежняя запись получает
производный статус `superseded` и остаётся доступной для исторических запросов.
`/v1/memory/search` сохраняет совместимые поля `topics` и `explicit_memories`, а
также возвращает общий ранжированный список `results` с типами `topic` и
`explicit_memory`.

Гарантированные пути без решения LLM:

```bash
python run.py remember "Production работает на 192.168.0.14" \
  --kind configuration --scope-type project --scope-id mnemonic-vault
```

OpenClaw также поддерживает `/remember`. Без selector команда использует scope
`global`; более узкий selector задаётся явно:

```text
/remember project:mnemonic-vault Production работает на 192.168.0.14
```

## Общая память и scope-aware ranking

Все агенты читают один архив, один `explicit-memory.jsonl` и один SQLite-каталог.
Scope — метка релевантности, а не граница доступа. По умолчанию
`scope_mode=boost`: совпавший project получает `+0.12`, текущая session `+0.10`,
текущий agent `+0.08`, global `+0.05`. Другие project/agent записи остаются
обычными кандидатами; чужая session получает сильное понижение, но не скрывается.

Автоматический recall Corax, OpenClaw и Hermes передаёт свои стабильные
agent/session scope. Необязательный project задаётся через
`MNEMONIC_VAULT_PROJECT_ID` или конфигурацию адаптера.

```json
{
  "query": "Где production Vault?",
  "context_scopes": [
    {"type": "agent", "id": "openclaw-main"},
    {"type": "project", "id": "mnemonic-vault"},
    {"type": "session", "id": "session-openclaw-..."}
  ],
  "scope_mode": "boost"
}
```

`include_all_scopes=true` отключает понижение чужих session-записей для явного
широкого исторического поиска. Настоящая фильтрация включается только явно:

```json
{
  "query": "production",
  "scope": {"type": "project", "id": "mnemonic-vault"},
  "scope_mode": "strict"
}
```

## Надёжная доставка

Corax, OpenClaw и Hermes сначала делают `fsync` события в переносимый spool:

```text
<corax-data>/mnemonic-vault/spool/corax.jsonl
data/spool/openclaw.jsonl
data/spool/openclaw.dead-letter.jsonl
data/spool/hermes.jsonl
data/spool/hermes.dead-letter.jsonl
```

После подтверждения API в spool дописывается отметка `delivered`. Неподтверждённые
события повторяются после восстановления Vault или перезапуска агента. Один и тот же
заново эмитированный lifecycle hook получает тот же ID, поэтому повтор не создаёт дубль.

Идентификатор Vault-сессии строится из постоянного имени установки агента и внешнего
ID чата, поэтому рестарт процесса не дробит историю. Постоянные ошибки `400/413/422`
карантинируются в dead-letter и не блокируют следующие события; `401/403` останавливают
доставку до исправления конфигурации, сетевые ошибки и `5xx` повторяются. Append в уже
завершённую сессию автоматически переносится в детерминированную recovery-сессию.
Redirect хранится в spool, поэтому следующие сообщения и `session_end` продолжают и
закрывают recovery-сессию даже после рестарта процесса.

## Relevance и transcript index

Абсолютный lexical coverage / cosine gate отбрасывает слабые кандидаты до RRF.
RRF используется только для порядка уже релевантных тем. Полный архив ищется через
восстанавливаемый `messages_fts`; `search_transcript` больше не загружает все JSONL.
`rebuild-index` восстанавливает topic index, message index и event-id каталог из файлов.
Явный год или дата извлекается regex и ограничивает кандидатов по
`session_started_at` до BM25/vector top-k, поэтому нужная историческая версия не
вытесняется более новыми результатами.

## Глобальные проекции тем

Сессионные topic-файлы никогда не объединяются и не удаляются. Когда у одной темы
накопится достаточно версий, отдельная offline-команда может построить производный
слой:

```text
data/global-topics/<global-topic-id>/
├── current.md
├── timeline.md
└── sources.json
```

`current.md` содержит последний сессионный snapshot по `topic.updated_at`, но не
синтезирует все исторические сведения в единое «текущее состояние». `timeline.md`
показывает время обновления темы и старта исходной сессии, а `sources.json` хранит
исходные topic IDs. В проекцию входят только полностью обработанные finalized-сессии.
Кластеризация использует консервативное complete-link соответствие, детерминирована и
не вызывает LLM. По умолчанию требуется 12 версий, а rebuild не запускается
автоматически:

```bash
python run.py rebuild-global-topics --dry-run
python run.py rebuild-global-topics
python run.py remember "точный факт" --kind fact --scope-type global
```

После обновления с 0.4.0 существующие производные проекции нужно один раз
перестроить этой командой. Старый формат до rebuild не участвует в retrieval, а
совпадающие global IDs сохраняются по исходным topic IDs.

Проекцию можно удалить и полностью восстановить из session topics. Поисковые карточки
получают `global_topic_id` только после её создания; `memory_open_global_topic`
открывает ограниченный snapshot/timeline пакет, не заменяя source-level retrieval.
Параметры `max_timeline_entries` и `total_token_budget` ограничивают весь tool output,
включая current, timeline, sources и метаданные.

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
python run.py migrate-session-ids --dry-run
python run.py migrate-session-ids
python run.py rebuild-global-topics --dry-run
python run.py rebuild-global-topics
```

Незавершённые jobs при старте переводятся обратно в `pending`. После трёх неудачных
попыток job получает статус `failed`; исходный transcript при этом уже сохранён.

`migrate-session-ids` группирует старые папки 0.3.0 по `agent + external_session_id`
и пишет переносимый `data/session-aliases.json`. Transcript-файлы не объединяются,
не перенумеровываются и не изменяются; scoped transcript search по каноническому ID
читает все физические папки alias-группы. Для нестандартной установки используйте
`--agent-instance openclaw=my-stable-id` или аналогичный override для Hermes.

## Нативные интеграции агентов

В проект входят три lossless-адаптера:

- Corax `memory_provider` с собственным `agent.memoryloop/v1`, persistent spool
  и автоматическим выбором вместо generic memory loop;
- OpenClaw memory-slot plugin с lifecycle hooks, восемью memory tools,
  гарантированной `/remember` command и встроенным skill;
- Hermes Agent `MemoryProvider` с persistent spool, bounded prefetch и теми же
  восемью tools, включая `memory_remember`.

Все адаптеры автоматически сохраняют ходы, подмешивают только небольшой релевантный
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
