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
- `price_to_user_mean`: `float`
- `hour_of_day`: `0..23`
- `is_view`: `0 | 1`
- `is_click`: `0 | 1`
- `is_cart`: `0 | 1`
- `session_event_index`: `int >= 1`
- `session_click_count`: `int >= 0`
- `session_cart_count`: `int >= 0`
- `user_total_clicks`: `int >= 0`
- `user_total_carts`: `int >= 0`
- `user_total_purchases`: `int >= 0`

## Feature semantics
- `price_to_user_mean` compares current price with the user's historical price anchor.
- `session_event_index`, `session_click_count`, `session_cart_count` describe the current session state.
- `user_total_clicks`, `user_total_carts`, `user_total_purchases` capture historical user intent before the current session.

## Offline training target
- `label`: `0 | 1`
- `label = 1` означает, что наблюдение конвертируется в покупку

## Ответ inference-сервиса
- `event_id`: `str`
- `user_id`: `int`
- `purchase_probability`: `float`
- `model_version`: `str`
