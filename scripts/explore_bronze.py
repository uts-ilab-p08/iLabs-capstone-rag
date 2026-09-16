"""Deeper look at bronze: distincts, full JSON blobs, join shape, per-video counts."""

from __future__ import annotations

import json
import os

import psycopg
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()
DATABASE_URL = os.environ["DATABASE_URL"]


def q(cur, sql, params=None):
    cur.execute(sql, params or ())
    cols = [d.name for d in cur.description]
    return cols, cur.fetchall()


def show(title, cols, rows):
    console.rule(title)
    for row in rows:
        console.print(dict(zip(cols, row)))


def main():
    with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        show(*("events.source_status distinct", *q(
            cur, "SELECT source_status, count(*) FROM bronze.events GROUP BY 1"
        )))

        show(*("events per video", *q(
            cur,
            """SELECT v.video_name, v.scene, count(e.event_id) AS n_events
               FROM bronze.videos v LEFT JOIN bronze.events e USING (video_id)
               GROUP BY 1,2 ORDER BY 1""",
        )))

        show(*("distinct event_name (sample 30)", *q(
            cur,
            """SELECT event_name, count(*) FROM bronze.events
               GROUP BY 1 ORDER BY 2 DESC LIMIT 30""",
        )))

        show(*("event_objects fan-out: objects per event", *q(
            cur,
            """SELECT event_id, count(*) FROM bronze.event_objects
               GROUP BY 1 ORDER BY 2 DESC LIMIT 10""",
        )))

        show(*("full label_details for one object", *q(
            cur,
            """SELECT label_details FROM bronze.objects LIMIT 1""",
        )))

        show(*("geometries.spatial_position distinct", *q(
            cur, "SELECT spatial_position, count(*) FROM bronze.geometries GROUP BY 1"
        )))

        show(*("geometries.label distinct", *q(
            cur, "SELECT label, count(*) FROM bronze.geometries GROUP BY 1 ORDER BY 2 DESC"
        )))

        show(*("one full event + its objects + a geometry", *q(
            cur,
            """
            SELECT e.event_id, e.event_name, e.description, e.start_seconds, e.end_seconds,
                   o.object_id, o.label_details->0->>'label' AS top_label
            FROM bronze.events e
            JOIN bronze.event_objects eo ON eo.event_id = e.event_id
            JOIN bronze.objects o ON o.object_id = eo.object_id
            LIMIT 5
            """,
        )))

        # schemas other than bronze, in case there's a silver/gold too
        show(*("all schemas", *q(
            cur,
            """SELECT schema_name FROM information_schema.schemata
               WHERE schema_name NOT LIKE 'pg_%' AND schema_name NOT IN ('information_schema')
               ORDER BY 1""",
        )))


if __name__ == "__main__":
    main()
