#!/usr/bin/env python3
"""
Regenerate balanced train/test datasets with consistent column naming.
Ensures similar distribution of query types and column patterns in both sets.
"""

import os
import json
import random
import pathlib
import time
from typing import List, Dict, Any
from collections import Counter

import duckdb
import jsonlines
from dotenv import load_dotenv
from fireworks import LLM


# Standard column aliases to use consistently
STANDARD_ALIASES = {
    # Counts
    "airline_count": ["airlines", "airline_count", "num_airlines", "total_airlines"],
    "airport_count": ["airports", "airport_count", "num_airports", "total_airports"],
    "route_count": ["routes", "route_count", "num_routes", "total_routes"],
    "destination_count": ["destinations", "destination_count", "num_destinations"],
    
    # Aggregates
    "avg_altitude": ["avg_altitude", "average_altitude", "mean_altitude"],
    "avg_latitude": ["avg_latitude", "average_latitude", "mean_lat"],
    "avg_longitude": ["avg_longitude", "average_longitude", "mean_lon"],
    
    # Entity names
    "country": ["country", "country_name"],
    "city": ["city", "city_name"],
    "airline": ["airline", "airline_name"],
    "airport": ["airport", "airport_name"],
    "timezone": ["timezone", "tz", "timezone_region"],
}


# Query templates with consistent column naming
QUERY_TEMPLATES = [
    # Country-based queries
    {
        "category": "country_airlines",
        "templates": [
            "SELECT country, COUNT(*) AS airline_count FROM airlines GROUP BY country HAVING COUNT(*) > {n} ORDER BY airline_count DESC LIMIT {limit}",
            "SELECT country, COUNT(*) AS airline_count, COUNT(CASE WHEN active = 'Y' THEN 1 END) AS active_count FROM airlines GROUP BY country HAVING COUNT(*) > {n} ORDER BY active_count DESC LIMIT {limit}",
            "SELECT country, COUNT(DISTINCT airline_id) AS airline_count FROM airlines WHERE active = 'Y' GROUP BY country ORDER BY airline_count DESC LIMIT {limit}",
        ]
    },
    {
        "category": "country_airports",
        "templates": [
            "SELECT country, COUNT(*) AS airport_count FROM airports GROUP BY country HAVING COUNT(*) > {n} ORDER BY airport_count DESC LIMIT {limit}",
            "SELECT country, COUNT(*) AS airport_count, ROUND(AVG(altitude), 2) AS avg_altitude FROM airports GROUP BY country HAVING COUNT(*) > {n} ORDER BY avg_altitude DESC LIMIT {limit}",
            "SELECT country, COUNT(*) AS airport_count, ROUND(AVG(latitude), 2) AS avg_latitude FROM airports GROUP BY country ORDER BY airport_count DESC LIMIT {limit}",
        ]
    },
    # City-based queries
    {
        "category": "city_airports",
        "templates": [
            "SELECT city, country, COUNT(*) AS airport_count FROM airports GROUP BY city, country HAVING COUNT(*) > 1 ORDER BY airport_count DESC LIMIT {limit}",
            "SELECT city, COUNT(*) AS airport_count, ROUND(AVG(altitude), 2) AS avg_altitude FROM airports GROUP BY city HAVING COUNT(*) > 1 ORDER BY avg_altitude DESC LIMIT {limit}",
        ]
    },
    # Timezone queries
    {
        "category": "timezone",
        "templates": [
            "SELECT timezone, COUNT(*) AS airport_count FROM airports GROUP BY timezone HAVING COUNT(*) > {n} ORDER BY airport_count DESC LIMIT {limit}",
            "SELECT timezone, COUNT(*) AS airport_count, ROUND(AVG(altitude), 2) AS avg_altitude FROM airports GROUP BY timezone HAVING COUNT(*) > {n} ORDER BY avg_altitude DESC LIMIT {limit}",
            "SELECT timezone, COUNT(*) AS airport_count, ROUND(AVG(longitude), 2) AS avg_longitude FROM airports GROUP BY timezone HAVING COUNT(*) > {n} ORDER BY airport_count DESC LIMIT {limit}",
        ]
    },
    # Route queries
    {
        "category": "routes_airlines",
        "templates": [
            "SELECT a.name AS airline, COUNT(*) AS route_count FROM routes r JOIN airlines a ON r.airline_id = a.airline_id WHERE a.active = 'Y' GROUP BY a.name ORDER BY route_count DESC LIMIT {limit}",
            "SELECT a.country, COUNT(*) AS route_count FROM routes r JOIN airlines a ON r.airline_id = a.airline_id GROUP BY a.country ORDER BY route_count DESC LIMIT {limit}",
        ]
    },
    {
        "category": "routes_airports",
        "templates": [
            "SELECT source_airport, destination_airport, COUNT(*) AS route_count FROM routes GROUP BY source_airport, destination_airport HAVING COUNT(*) > 1 ORDER BY route_count DESC LIMIT {limit}",
            "SELECT ap.city, COUNT(DISTINCT r.airline_id) AS airline_count FROM routes r JOIN airports ap ON r.source_airport_id = ap.airport_id GROUP BY ap.city HAVING COUNT(DISTINCT r.airline_id) > {n} ORDER BY airline_count DESC LIMIT {limit}",
        ]
    },
    # Equipment queries
    {
        "category": "equipment",
        "templates": [
            "SELECT equipment, COUNT(*) AS route_count FROM routes WHERE equipment IS NOT NULL GROUP BY equipment ORDER BY route_count DESC LIMIT {limit}",
            "SELECT equipment, COUNT(DISTINCT airline_id) AS airline_count FROM routes WHERE equipment IS NOT NULL GROUP BY equipment HAVING COUNT(DISTINCT airline_id) > {n} ORDER BY airline_count DESC LIMIT {limit}",
        ]
    },
    # Join queries
    {
        "category": "complex_joins",
        "templates": [
            "SELECT a.country, COUNT(DISTINCT a.airline_id) AS airline_count, COUNT(DISTINCT r.destination_airport_id) AS destination_count FROM airlines a JOIN routes r ON a.airline_id = r.airline_id WHERE a.active = 'Y' GROUP BY a.country HAVING COUNT(DISTINCT a.airline_id) > 1 ORDER BY destination_count DESC LIMIT {limit}",
            "SELECT ap.country, COUNT(DISTINCT ap.airport_id) AS airport_count, COUNT(DISTINCT r.airline_id) AS airline_count FROM airports ap LEFT JOIN routes r ON ap.airport_id = r.source_airport_id GROUP BY ap.country HAVING COUNT(DISTINCT ap.airport_id) > {n} ORDER BY airline_count DESC LIMIT {limit}",
        ]
    },
    # Aggregate queries with conditions
    {
        "category": "conditional_aggregates", 
        "templates": [
            "SELECT country, COUNT(*) AS airport_count, COUNT(CASE WHEN altitude > 1000 THEN 1 END) AS high_altitude_count FROM airports GROUP BY country HAVING COUNT(*) > {n} ORDER BY high_altitude_count DESC LIMIT {limit}",
            "SELECT country, COUNT(*) AS airline_count, COUNT(CASE WHEN iata IS NOT NULL THEN 1 END) AS with_iata_count FROM airlines GROUP BY country HAVING COUNT(*) > {n} ORDER BY with_iata_count DESC LIMIT {limit}",
            "SELECT country, COUNT(*) AS airline_count, COUNT(CASE WHEN callsign IS NOT NULL THEN 1 END) AS with_callsign_count FROM airlines GROUP BY country HAVING COUNT(*) > {n} ORDER BY with_callsign_count DESC LIMIT {limit}",
        ]
    },
    # Geographic queries
    {
        "category": "geographic",
        "templates": [
            "SELECT country, ROUND(AVG(latitude), 2) AS avg_latitude, ROUND(AVG(longitude), 2) AS avg_longitude FROM airports GROUP BY country ORDER BY avg_latitude DESC LIMIT {limit}",
            "SELECT country, ROUND(AVG(latitude), 2) AS avg_latitude, ROUND(AVG(longitude), 2) AS avg_longitude FROM airports GROUP BY country ORDER BY country",
        ]
    },
]


def generate_queries_from_templates(num_queries: int = 300) -> List[str]:
    """Generate SQL queries from templates with varied parameters."""
    queries = []
    
    # Generate multiple variations per template
    for template_group in QUERY_TEMPLATES:
        category = template_group["category"]
        templates = template_group["templates"]
        
        for template in templates:
            # Generate variations with different parameters
            for n in [1, 2, 3]:
                for limit in [5, 6, 7, 8, 10]:
                    try:
                        query = template.format(n=n, limit=limit)
                        queries.append({"query": query, "category": category})
                    except KeyError:
                        # Template doesn't use all params
                        try:
                            query = template.format(n=n, limit=limit)
                        except:
                            query = template
                        queries.append({"query": query, "category": category})
    
    # Deduplicate and shuffle
    seen = set()
    unique_queries = []
    for q in queries:
        if q["query"] not in seen:
            seen.add(q["query"])
            unique_queries.append(q)
    
    random.shuffle(unique_queries)
    return unique_queries[:num_queries]


def execute_query_get_ground_truth(con, query: str) -> List[Dict[str, Any]] | None:
    """Execute query and return ground truth, or None if failed."""
    try:
        df = con.sql(query).df()
        if df.empty:
            return None
        if len(df) > 100:  # Skip very large results
            return None
        # Convert to records
        import pandas as pd
        df = df.astype(object).where(pd.notna(df), None)
        records = df.to_dict("records")
        return records
    except Exception as e:
        return None


def generate_nl_question(llm, schema_md: str, query: str) -> str | None:
    """Generate natural language question from SQL query."""
    nl_template = f"""
Translate the SQL query into a natural language business question that would produce it.
Be precise and faithful to the SQL intent. Include specific numbers from the query (like "more than 2" or "top 7").

Schema:
{schema_md}

SQL:
{query}

Return only the question text, nothing else.
""".strip()
    
    try:
        resp = llm.chat.completions.create(
            messages=[{"role": "user", "content": nl_template}],
            temperature=0.3,
            max_tokens=500,
        )
        nl = (resp.choices[0].message.content or "").strip()
        return nl if nl else None
    except Exception as e:
        print(f"  NL generation error: {e}")
        return None


def stratified_split(data: List[Dict], train_ratio: float = 0.77) -> tuple:
    """Split data ensuring each category is represented proportionally in both sets."""
    # Group by category
    by_category = {}
    for item in data:
        cat = item.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(item)
    
    train = []
    test = []
    
    for cat, items in by_category.items():
        random.shuffle(items)
        split_idx = max(1, int(len(items) * train_ratio))
        train.extend(items[:split_idx])
        test.extend(items[split_idx:])
    
    random.shuffle(train)
    random.shuffle(test)
    
    return train, test


def main():
    load_dotenv()
    
    random.seed(42)
    
    root = pathlib.Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    synth_db = str(data_dir / "synthetic_openflights.db")
    out_train = root / "datasets" / "final_rft_sql_train_data.jsonl"
    out_test = root / "datasets" / "final_rft_sql_test_data.jsonl"
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    
    # Target sizes
    TARGET_TRAIN = 200
    TARGET_TEST = 60
    TOTAL_NEEDED = TARGET_TRAIN + TARGET_TEST
    
    print(f"Target: {TARGET_TRAIN} train, {TARGET_TEST} test examples")
    
    # Initialize LLM
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY is not set")
    llm = LLM(model="accounts/fireworks/models/deepseek-v3p1-terminus", deployment_type="serverless", api_key=api_key)
    
    # Load schema
    with duckdb.connect(synth_db, read_only=True) as con:
        schema_md = con.sql("DESCRIBE;").df().to_markdown(index=False)
    
    system_prompt = f"""You are an expert SQL data analyst.
Write a single DuckDB SQL query to answer the user's question based on the schema.
Return only the SQL text, no explanations, and avoid duplicates via GROUP BY when needed.

Schema:
{schema_md}
""".strip()
    
    # Generate queries from templates
    print("\n1. Generating SQL queries from templates...")
    query_items = generate_queries_from_templates(num_queries=400)
    print(f"   Generated {len(query_items)} unique queries")
    
    # Execute queries and get ground truth
    print("\n2. Executing queries to get ground truth...")
    valid_items = []
    with duckdb.connect(synth_db, read_only=True) as con:
        for i, item in enumerate(query_items):
            gt = execute_query_get_ground_truth(con, item["query"])
            if gt:
                item["ground_truth"] = gt
                valid_items.append(item)
            if (i + 1) % 50 == 0:
                print(f"   Processed {i+1}/{len(query_items)}, valid: {len(valid_items)}")
    
    print(f"   Valid queries with results: {len(valid_items)}")
    
    if len(valid_items) < TOTAL_NEEDED:
        print(f"⚠️  Only {len(valid_items)} valid queries, need {TOTAL_NEEDED}")
    
    # Generate NL questions
    print("\n3. Generating natural language questions...")
    final_items = []
    for i, item in enumerate(valid_items[:TOTAL_NEEDED + 50]):  # Extra buffer
        nl = generate_nl_question(llm, schema_md, item["query"])
        if nl:
            final_items.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": nl},
                    {"role": "assistant", "content": item["query"]},
                ],
                "ground_truth": item["ground_truth"],
                "category": item["category"],
            })
        if (i + 1) % 20 == 0:
            print(f"   Generated {len(final_items)} NL questions from {i+1} queries...")
        time.sleep(0.2)
        
        if len(final_items) >= TOTAL_NEEDED:
            break
    
    print(f"   Total examples with NL questions: {len(final_items)}")
    
    # Stratified split
    print("\n4. Performing stratified train/test split...")
    train, test = stratified_split(final_items, train_ratio=TARGET_TRAIN / TOTAL_NEEDED)
    
    # Adjust sizes
    train = train[:TARGET_TRAIN]
    test = test[:TARGET_TEST]
    
    # Verify distribution
    print("\n5. Verifying distribution...")
    train_cats = Counter(item["category"] for item in train)
    test_cats = Counter(item["category"] for item in test)
    
    print(f"\n   Category distribution:")
    print(f"   {'Category':<25} {'Train':>8} {'Test':>8}")
    print(f"   {'-'*45}")
    all_cats = set(train_cats.keys()) | set(test_cats.keys())
    for cat in sorted(all_cats):
        print(f"   {cat:<25} {train_cats.get(cat, 0):>8} {test_cats.get(cat, 0):>8}")
    
    # Remove category field before saving (not needed in final data)
    for item in train:
        item.pop("category", None)
    for item in test:
        item.pop("category", None)
    
    # Save
    print(f"\n6. Saving datasets...")
    with jsonlines.open(out_train, mode="w") as w:
        w.write_all(train)
    with jsonlines.open(out_test, mode="w") as w:
        w.write_all(test)
    
    print(f"\n✅ Done!")
    print(f"   Train: {out_train} ({len(train)} examples)")
    print(f"   Test:  {out_test} ({len(test)} examples)")
    
    # Verify column distribution
    print("\n7. Column name distribution check...")
    train_cols = Counter()
    test_cols = Counter()
    for item in train:
        if item["ground_truth"] and isinstance(item["ground_truth"][0], dict):
            train_cols.update(item["ground_truth"][0].keys())
    for item in test:
        if item["ground_truth"] and isinstance(item["ground_truth"][0], dict):
            test_cols.update(item["ground_truth"][0].keys())
    
    print(f"\n   Columns in train: {len(train_cols)}")
    print(f"   Columns in test: {len(test_cols)}")
    
    train_only = set(train_cols.keys()) - set(test_cols.keys())
    test_only = set(test_cols.keys()) - set(train_cols.keys())
    
    if train_only:
        print(f"   ⚠️  Columns only in train: {train_only}")
    if test_only:
        print(f"   ⚠️  Columns only in test: {test_only}")
    if not train_only and not test_only:
        print(f"   ✅ All columns appear in both train and test!")


if __name__ == "__main__":
    main()

