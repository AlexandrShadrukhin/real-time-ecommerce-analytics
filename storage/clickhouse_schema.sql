CREATE TABLE IF NOT EXISTS raw_events
(
    event_id String,
    user_id UInt32,
    session_id String,
    product_id UInt32,
    event_type String,
    timestamp DateTime64(6),
    price Float64,
    category_code Nullable(String),
    brand Nullable(String),
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
    session_id String,
    timestamp DateTime64(6),
    n_views Float64,
    n_carts Float64,
    uniq_products Float64,
    uniq_categories Float64,
    mean_price Float64,
    max_price Float64,
    price_std Float64,
    uniq_brands Float64,
    missing_brand Float64,
    missing_category_code Float64,
    prefix_duration_sec Float64,
    has_cart_in_prefix Float64,
    cart_view_ratio Float64,
    same_product_repeat_count Float64,
    same_category_repeat_count Float64,
    event_1_is_cart Float64,
    event_2_is_cart Float64,
    event_3_is_cart Float64,
    time_1_to_2_sec Float64,
    time_2_to_3_sec Float64,
    ingested_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (user_id, timestamp, event_id);

CREATE TABLE IF NOT EXISTS predictions
(
    event_id String,
    user_id UInt32,
    session_id String,
    purchase_probability Float64,
    model_version String,
    timestamp DateTime64(6),
    ingested_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (user_id, timestamp, event_id);