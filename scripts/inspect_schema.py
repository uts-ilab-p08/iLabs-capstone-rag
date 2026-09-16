"""Explore the given Postgres schema: list tables, columns, row counts, and a
sample row per table. Read-only (SELECT / information_schema only).

Usage:
    uv run scripts/inspect_schema.py [schema_name]

Reads DATABASE_URL (and default DB_SCHEMA) from .env.
"""

from __future__ import annotations

import os
import sys

import psycopg
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

console = Console()

DATABASE_URL = os.environ["DATABASE_URL"]
SCHEMA = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DB_SCHEMA", "public")


def list_tables(cur, schema: str) -> list[str]:
    cur.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = %s
        ORDER BY table_name
        """,
        (schema,),
    )
    return [r[0] for r in cur.fetchall()]


def list_columns(cur, schema: str, table: str) -> list[tuple[str, str, str]]:
    cur.execute(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """,
        (schema, table),
    )
    return cur.fetchall()


def row_count(cur, schema: str, table: str) -> int:
    cur.execute(f'SELECT count(*) FROM "{schema}"."{table}"')
    return cur.fetchone()[0]


def sample_row(cur, schema: str, table: str):
    cur.execute(f'SELECT * FROM "{schema}"."{table}" LIMIT 1')
    cols = [d.name for d in cur.description]
    row = cur.fetchone()
    return cols, row


def main() -> None:
    console.print(f"[bold]Connecting to Supabase Postgres[/bold], schema=[cyan]{SCHEMA}[/cyan]")
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            tables = list_tables(cur, SCHEMA)
            if not tables:
                console.print(f"[red]No tables found in schema '{SCHEMA}'.[/red]")
                cur.execute(
                    "SELECT schema_name FROM information_schema.schemata ORDER BY schema_name"
                )
                schemas = [r[0] for r in cur.fetchall()]
                console.print(f"Available schemas: {schemas}")
                return

            console.print(f"Found [bold]{len(tables)}[/bold] table(s): {tables}\n")

            for table in tables:
                columns = list_columns(cur, SCHEMA, table)
                try:
                    count = row_count(cur, SCHEMA, table)
                except Exception as e:  # noqa: BLE001
                    count = f"error: {e}"

                col_table = Table(title=f"{SCHEMA}.{table}  (rows: {count})")
                col_table.add_column("column")
                col_table.add_column("type")
                col_table.add_column("nullable")
                for name, dtype, nullable in columns:
                    col_table.add_row(name, dtype, nullable)
                console.print(col_table)

                try:
                    cols, row = sample_row(cur, SCHEMA, table)
                    if row:
                        console.print("[dim]sample row:[/dim]")
                        for c, v in zip(cols, row):
                            text = str(v)
                            if len(text) > 200:
                                text = text[:200] + "…"
                            console.print(f"  [dim]{c}[/dim] = {text}")
                    else:
                        console.print("[dim](table is empty)[/dim]")
                except Exception as e:  # noqa: BLE001
                    console.print(f"[red]could not sample: {e}[/red]")
                console.print()


if __name__ == "__main__":
    main()
