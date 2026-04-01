# Структура данных

## Событие
- `event_id`: `str`
- `user_id`: `int`
- `event_type`: `view | click | add_to_cart | purchase`
- `timestamp`: ISO-8601 datetime
- `price`: `float`

## Online feature contract
- `event_id`: `str`
- `user_id`: `int`
- `timestamp`: ISO-8601 datetime
- `price`: `float`
- `is_view`: `0 | 1`
- `is_click`: `0 | 1`
- `is_cart`: `0 | 1`
- `hour_of_day`: `0..23`

## Offline training target
- `label`: `0 | 1`
- `label = 1` означает, что наблюдение конвертируется в покупку

## Ответ inference-сервиса
- `event_id`: `str`
- `user_id`: `int`
- `purchase_probability`: `float`
- `model_version`: `str`
