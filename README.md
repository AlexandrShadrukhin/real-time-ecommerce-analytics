# Система потоковой аналитики пользовательских событий с онлайн-прогнозированием вероятности покупки

## Описание
MVP системы для анализа пользовательских действий в реальном времени (клики, просмотры, корзина) с онлайн-прогнозированием вероятности покупки.

## Архитектура
Event Generator → Kafka → Stream Processor → Feature Builder → ML → ClickHouse → Dashboard

## Стек
- Python
- Kafka
- FastAPI
- ClickHouse
- LightGBM / scikit-learn
- Streamlit
- Docker

## Статус
MVP в разработке

## Структура
- producer — генерация событий  
- processor — обработка потока  
- ml — модель  
- storage — ClickHouse  
- api — backend  
- dashboard — визуализация  