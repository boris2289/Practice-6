from __future__ import annotations

import argparse

from app.db.repository import initialize_schema, seed_input_data_from_dataframe
from app.training.data import load_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Seed PostgreSQL input_data from fashion_test.csv.')
    parser.add_argument('--limit', type=int, default=100, help='Number of rows to insert.')
    parser.add_argument('--clear-existing', action='store_true', help='Truncate input_data and predictions before inserting.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initialize_schema()
    _, test_df = load_datasets()
    inserted = seed_input_data_from_dataframe(
        df=test_df,
        limit=args.limit,
        clear_existing=args.clear_existing,
    )
    print(f'Inserted {inserted} rows into input_data.')


if __name__ == '__main__':
    main()
