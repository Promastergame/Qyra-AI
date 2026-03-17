```markdown
# 🚀 Qyra AI

**Крупномасштабная языковая модель для русского языка с оптимизацией под CPU**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-orange)]()

---

## 📋 О проекте

**Qyra AI** — это современная языковая модель, разработанная специально для работы с русскоязычными текстами. Проект фокусируется на:

- ✅ **Оптимизация под CPU** — обучение и инференс на обычных процессорах без GPU
- ✅ **Малый размер** — от 12 до 60 миллионов параметров
- ✅ **Расширяемая архитектура** — поддержка инструментов и инструментов вызова
- ✅ **Гибкое обучение** — претрейнинг и дообучение на собственных данных

---

## 🏗️ Архитектура

### Модельные пресеты (CPU-friendly)

| Размер | d_model | n_heads | n_layers | d_ff | ОЗУ (примерно) |
|--------|---------|---------|----------|------|----------------|
| **12M** | 384 | 6 | 6 | 1536 | ~2-3 GB |
| **20M** | 416 | 8 | 8 | 1664 | ~3-4 GB |
| **35M** | 448 | 10 | 8 | 2048 | ~5-6 GB |
| **60M** | 512 | 8 | 16 | 2560 | ~8-10 GB |

### Ключевые особенности архитектуры

- **GQA (Grouped Query Attention)** — эффективное использование памяти
- **QK-Norm** — стабилизация обучения
- **RMSNorm** — быстрая нормализация
- **SwiGLU** — улучшенная активация
- **RoPE** — Rotary Position Embeddings
- **Weight Tying** — уменьшение параметров

---

## 📁 Структура проекта

```
Qyra-AI/
├── qyra/                    # Исходный код модели
├── config.py                # Конфигурация архитектуры и обучения
├── train_ru.py              # Универсальный скрипт обучения
├── merge_ru_datasets.py     # Объединение и обработка датасетов
├── data/
│   ├── raw/                # Исходные данные
│   └── finetune/           # Данные для дообучения
├── checkpoints/            # Чекпоинты модели
├── tokenizer/              # Токенизатор
└── README.md               # Эта документация
```

---

## 🛠️ Быстрый старт

### 1. Установка зависимостей

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets sentencepiece
```

### 2. Подготовка данных

Объединение датасетов в единый формат:

```bash
python merge_ru_datasets.py \
  --output data/finetune/chat_merged.jsonl \
  --inputs data/chat1.jsonl data/chat2.jsonl \
  --min_turns 2 \
  --shuffle \
  --seed 42
```

### 3. Обучение модели

**Претрейнинг (с нуля):**

```bash
python train_ru.py \
  --mode pretrain \
  --model_size 12M \
  --data_dir data/raw \
  --max_tokens 1B \
  --batch_size 4 \
  --grad_accum 4
```

**Дообучение (fine-tuning):**

```bash
python train_ru.py \
  --mode finetune \
  --model_size 12M \
  --data data/finetune/chat_merged.jsonl \
  --checkpoint checkpoints/best_pretrain.pt \
  --epochs 5 \
  --batch_size 4
```

---

## ⚙️ Конфигурация

Все настройки находятся в `config.py`:

### ModelConfig
```python
@dataclass
class ModelConfig:
    vocab_size: int = 8000
    max_seq_len: int = 256      # Для скорости на CPU
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    d_ff: int = 1536
    # ... и другие параметры
```

### PretrainConfig / FinetuneConfig
- `batch_size` — размер батча (рекомендуется 2-4 для CPU)
- `grad_accum_steps` — шаги накопления градиента
- `lr` — скорость обучения (6e-4 для претрейна, 5e-5 для финайна)
- `max_steps` / `epochs` — лимиты обучения
- `use_gradient_checkpointing` — экономия памяти

---

## 🔧 Специальные токены

Модель использует специальные токены для структурирования диалогов:

| Токен | Назначение |
|-------|------------|
| `<|system|>` | Системное сообщение |
| `<|user|>` | Сообщение пользователя |
| `<|assistant|>` | Ответ ассистента |
| `<|end|>` | Конец последовательности |
| `<|tool_start|>` | Начало вызова инструмента |
| `<|tool_end|>` | Конец вызова инструмента |
| `<|tool_result|>` | Результат инструмента |

---

## 📊 Формат данных

### Для претрейна
Текстовые файлы в папке `data/raw/`. Модель обучается на сыром тексте.

### Для дообучения (chat)
JSONL формат с полем `messages`:

```json
{
  "messages": [
    {"role": "user", "content": "Привет!"},
    {"role": "assistant", "content": "Привет! Как дела?"}
  ]
}
```

Поддерживаются также ключи: `conversation`, `dialog`, `dialogue`.

---

## 🎯 Цели проекта

1. **Доступность** — обучение на обычных компьютерах без GPU
2. **Качество** — современные архитектуры (GQA, QK-Norm, RMSNorm)
3. **Гибкость** — поддержка инструментов и инструментов вызова
4. **Русский язык** — оптимизация под кириллицу и русские контексты
5. **Открытость** — весь код и веса открыты

---

## 📈 Планы развития

- [ ] Увеличение словаря до 32K токенов
- [ ] Поддержка мультимодальности (изображения)
- [ ] Улучшенный токенизатор на основе SentencePiece
- [ ] Quantization для ещё более быстрого инференса
- [ ] Web-интерфейс для тестирования

---

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для новой функции
3. Внесите изменения
4. Откройте Pull Request

---

## 📄 Лицензия

MIT License — свободное использование и модификация.

---

## 📞 Контакты

- **GitHub**: [Promastergame/Qyra-AI](https://github.com/Promastergame/Qyra-AI)
- **Issues**: [Сообщить о проблеме](https://github.com/Promastergame/Qyra-AI/issues)

---
