# Система потоковой аналитики пользовательских событий с онлайн-прогнозированием вероятности покупки

## Описание проекта
Проект представляет собой MVP системы для анализа пользовательского поведения в e-commerce в режиме реального времени.

Система обрабатывает поток пользовательских событий (просмотры, клики, добавления в корзину) и формирует поведенческие признаки, которые используются для аналитики и ML-прогнозирования вероятности покупки.

Система поддерживает онлайн-инференс ML-модели в реальном времени.

Реализован полный pipeline обработки данных:
генерация событий → Kafka → stream processing → feature engineering → ML (CatBoost) → ClickHouse → Grafana dashboard

---

## Архитектура системы

Общий поток данных:

Event Generator → Kafka → Stream Processor → Feature Builder → ML → ClickHouse → Dashboard

---

## Архитектура потокового конвейера (Data Pipeline)

<p align="center">
  <img src="docs/data_pipeline.png" alt="Data Pipeline Architecture" width="400"/>
</p>

Pipeline включает:
- генерацию пользовательских событий (synthetic data);
- передачу через Apache Kafka (topic: user_events);
- потоковую обработку данных (consumer);
- формирование stateful-признаков (сессии и пользовательская история);
- расчёт feature payload;
- вызов ML-модели для онлайн-прогнозирования;
- запись данных в ClickHouse;
- сохранение артефактов в jsonl (для дебага).

---

## Архитектура ML-пайплайна

<p align="center">
  <img src="docs/ml_pipeline.png" alt="ML Pipeline Architecture" width="400"/>
</p>

ML pipeline включает:
- генерацию синтетического датасета;
- ML-модель (CatBoost) для онлайн-прогнозирования вероятности покупки;
- нормализацию и масштабирование признаков;
- сохранение артефактов модели (weights, scaler, schema);
- онлайн-инференс (predict module);
- расчёт вероятности покупки;
- визуализацию через Streamlit dashboard.

---

## Разделение проекта

Проект реализуется как совместная разработка с разделением на два направления:

### Data Pipeline (Data Engineering)
- генерация событий  
- Kafka producer / consumer  
- потоковая обработка  
- stateful-агрегации  
- feature engineering  
- интеграция с ML  
- запись в ClickHouse  

### ML Pipeline (Machine Learning)
- использование обученной модели CatBoost (model.cbm);
- загрузку feature schema (feature_schema.json);
- формирование признаков из потокового конвейера;
- онлайн-инференс (predict module);
- расчёт вероятности покупки;
- передачу результатов в ClickHouse;
- визуализацию через Grafana dashboard.

---

## Технологический стек

### Data Pipeline
- Python  
- Apache Kafka  
- kafka-python  
- ClickHouse  
- SQL  

### ML Pipeline
- Python  
- CatBoost (предобученная модель)
- pandas / numpy  

### Общие инструменты
- Grafana
- Docker / Docker Compose  

---

## Статус проекта

Проект находится на стадии **рабочего MVP**.

### Реализовано:
- генерация событий;
- Kafka producer;
- Kafka consumer;
- потоковая обработка событий;
- stateful feature engineering (сессии + пользовательская история);
- формирование feature payload;
- онлайн-инференс ML-модели;
- запись в ClickHouse:
  - raw_events
  - feature_payloads
  - predictions
- сохранение данных в jsonl-файлы;
- интеграция ML-модели CatBoost;
- онлайн-инференс вероятности покупки;
- запись результатов в ClickHouse;
- визуализация данных в Grafana:
  - метрики в реальном времени
  - динамика событий
  - распределение поведения пользователей

### Планируется:
- оконные агрегации (5 / 15 / 60 минут);
- API слой (FastAPI);
- масштабирование Kafka (consumer groups, partitions).

### Визуализация данных

Для анализа результатов используется Grafana dashboard, включающий:

- общее число предсказаний;
- среднюю вероятность покупки;
- динамику событий в реальном времени;
- распределение типов событий;
- топ пользовательских сессий по вероятности покупки;
- таблицу последних предсказаний.

Dashboard обновляется в реальном времени.

---

## Структура репозитория

```
real-time-ecommerce-analytics/
├── producer/     # генерация событий и Kafka producer
├── processor/    # Kafka consumer и обработка потока
├── ml/           # модель, обучение и инференс
├── storage/      # ClickHouse клиент и схемы
├── api/          # backend/API (в разработке)
├── dashboard/    # Streamlit dashboard
├── docs/         # архитектурные схемы
├── config/       # конфигурация проекта
├── artifacts/    # результаты работы (jsonl, модели)
├── scripts/      # вспомогательные скрипты
├── tests/        # тесты
├── README.md
├── requirements.txt
└── docker-compose.yml
```

---

## Участники проекта

- [AlexandrShadrukhin](https://github.com/AlexandrShadrukhin) (Шадрухин Александр) — Data Pipeline / Data Engineering  
- [PKS339057](https://github.com/PKS339057) (Пряничников Кирилл) — ML Pipeline / Machine Learning

---

## Запуск проекта

### 1. Запуск инфраструктуры
docker compose up -d

### 2. Установка зависимостей
pip install -r requirements.txt

### 3. Отправка событий в Kafka
python -m producer.kafka_producer

### 4. Запуск stream processing
python -m processor.stream_consumer

### 5. Проверка данных в ClickHouse
docker exec -it rtea-clickhouse clickhouse-client --user app --password app_password --query "SELECT count() FROM predictions"

---

## Репозиторий

Проект реализован как рабочий MVP системы потоковой аналитики с поддержкой онлайн-инференса ML-модели и визуализацией в реальном времени.
