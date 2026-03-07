# TinyGPT — Tool Use Extension

## Что было добавлено

### Новые файлы:
```
tiny-gpt/
├── tools/
│   ├── __init__.py          # Импорт всех инструментов
│   ├── calculator.py        # Безопасный AST-калькулятор
│   ├── code_runner.py       # Выполнение Python/Go кода
│   ├── registry.py          # Реестр инструментов
│   ├── parser.py            # Парсинг tool-вызовов
│   └── generation.py        # Генерация с tool-поддержкой
├── generate_tool_data.py    # Генерация тренировочных данных
├── config.py                # +3 tool токена
├── dataset.py               # +tool_call и tool_result роли
├── model.py                 # +stop_token_ids в generate()
└── chat.py                  # +интерфейс с инструментами
```

### Специальные токены:
- `<|tool_start|>` — начало вызова инструмента
- `<|tool_end|>` — конец вызова/результата
- `<|tool_result|>` — начало результата выполнения

---

## Быстрый старт

### 1. Переобучи токенизатор с новыми токенами
```bash
cd tiny-gpt
python train_tokenizer.py --vocab_size 8000
```

### 2. Проверь, что токены добавлены
```bash
python -c "
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load('tokenizer/tok.model')
for t in ['<|tool_start|>','<|tool_end|>','<|tool_result|>']:
    print(f'{t} -> {sp.PieceToId(t)} (unk={sp.unk_id()})')
"
```

### 3. Претрейн модели (как обычно)
```bash
python pretrain.py --epochs 10 --batch_size 2 --grad_accum 8
```

### 4. Сгенерируй данные для tool use
```bash
python generate_tool_data.py --output data/finetune/tool_chat.jsonl --count 3000
```

### 5. Файнтьюн на tool-данных
```bash
python finetune.py \
    --data data/finetune/tool_chat.jsonl \
    --checkpoint checkpoints/best_pretrain.pt \
    --epochs 5 \
    --lr 5e-5
```

### 6. Чат с инструментами
```bash
# С инструментами (калькулятор спрашивает подтверждение для кода)
python chat.py --mode chat --checkpoint checkpoints/best_finetune.pt --tools --verbose

# С авто-подтверждением кода
python chat.py --mode chat --checkpoint checkpoints/best_finetune.pt --tools --auto-approve

# Без инструментов (обычный чат)
python chat.py --mode chat --checkpoint checkpoints/best_finetune.pt --no-tools
```

---

## Пример сессии

```
$ python chat.py --mode chat --checkpoint checkpoints/best_finetune.pt --tools

Device: cuda
  Tools enabled: calculator, python, go
Model: 13,824,000 params

═══ TinyGPT Chat ═══

You> What is 345 * 872?

  🔧 Tool call: calculator
     Args: {'expression': '345 * 872'}
     Result: 300840

Bot> 345 × 872 equals 300,840.

You> What is the square root of 289?

  🔧 Tool call: calculator
     Args: {'expression': 'sqrt(289)'}
     Result: 17

Bot> The square root of 289 is 17.

You> What is the capital of France?

Bot> The capital of France is Paris.

You> quit
Bye!
```

---

## Как это работает

### Архитектура
```
┌──────────────┐     ┌─────────────────┐     ┌────────────┐
│   User Input │────▶│   TinyGPT Model │────▶│   Output   │
└──────────────┘     └─────────────────┘     └─────┬──────┘
                                                    │
                    ┌───────────────────────────────▼──────────┐
                    │  ToolParser: detect <|tool_start|>       │
                    │  Extract: tool_name + {JSON args}        │
                    └───────────────────────────────┬──────────┘
                                                    │
                    ┌───────────────────────────────▼──────────┐
                    │  ToolRegistry.execute(tool_name, args)   │
                    │  → calculator.evaluate()                 │
                    │  → python_runner.run()                   │
                    └───────────────────────────────┬──────────┘
                                                    │
                    ┌───────────────────────────────▼──────────┐
                    │  Inject: <|tool_result|> result <|tool_end│
                    └───────────────────────────────┬──────────┘
                                                    │
                    ┌───────────────────────────────▼──────────┐
                    │  Model continues generation              │
                    └──────────────────────────────────────────┘
```

### Формат данных
```jsonl
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is 345 * 872?"},
  {"role": "assistant", "content": "Let me calculate."},
  {"role": "tool_call", "content": "calculator\n{\"expression\":\"345*872\"}"},
  {"role": "tool_result", "content": "300840"},
  {"role": "assistant", "content": "345 × 872 equals 300,840."}
]}
```

### Masking в loss
| Роль | Обучается? | Label |
|------|------------|-------|
| system | ❌ | -100 |
| user | ❌ | -100 |
| assistant | ✅ | token_id |
| tool_call | ✅ | token_id (модель учится вызывать) |
| tool_result | ❌ | -100 (система инжектит) |

---

## Добавление своих инструментов

1. **Создай обработчик** в `tools/registry.py`:
```python
def my_tool_handler(args: dict) -> str:
    value = args.get("value", "")
    return f"Processed: {value}"

# В _register_defaults():
self.register(
    name="my_tool",
    handler=my_tool_handler,
    description="Processes values. Args: {\"value\": \"...\"}",
    confirm=False,
)
```

2. **Сгенерируй данные** с примерами вызова нового инструмента

3. **Файнтьюнь** модель на новых данных

---

## Почему это работает для маленьких моделей

| Возможность | Без инструментов | С инструментами |
|-------------|------------------|-----------------|
| 7 × 8 | Часто ошибается | Всегда верно |
| 345 × 872 | Почти всегда ошибается | Всегда верно |
| sqrt(289) | Обычно ошибается | Всегда верно |
| factorial(20) | Не может | Всегда верно |
| Код | Не может | Выполняется |
| VRAM | 0 MB доп. | 0 MB (CPU) |
| Скорость | Обычная | +50ms на вызов |

**Ключевая идея:** инструменты превращают слабость модели (ненадёжные вычисления) в задачу классификации (когда вызывать инструмент), которую маленькая модель может научиться решать надёжно.
