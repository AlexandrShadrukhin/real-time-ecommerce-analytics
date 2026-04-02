CREATE TABLE IF NOT EXISTS raw_events
(
    event_id String,
    user_id UInt32,
    session_id String,
    product_id UInt32,
    event_type String,
    timestamp DateTime64(6),
    price Float64,
    category String,
    device_type String,
    source String,
    ingested_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (user_id, timestamp, event_id);

CREATE TABLE IF NOT EXISTS feature_payloads
(
    event_id String,
    user_id UInt32,
    timestamp DateTime64(6),
    price Float64,
    price_to_user_mean Float64,
    hour_of_day UInt8,
    is_view UInt8,
    is_click UInt8,
    is_cart UInt8,
    session_event_index UInt32,
    session_click_count UInt32,
    session_cart_count UInt32,
    user_total_clicks UInt32,
    user_total_carts UInt32,
    user_total_purchases UInt32,
    ingested_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (user_id, timestamp, event_id);

CREATE TABLE IF NOT EXISTS predictions
(
    event_id String,
    user_id UInt32,
    purchase_probability Float64,
    model_version String,
    timestamp DateTime64(6),
    ingested_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (user_id, timestamp, event_id);