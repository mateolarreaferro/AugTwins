from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import List, Union

DB_PATH = Path("seed_data.db")


def init_db(path: Union[str, Path] = DB_PATH) -> None:
    """Create the seeds table and insert dummy data if none exist."""
    path = Path(path)
    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS seeds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL,
                text TEXT NOT NULL
            )"""
        )
        conn.commit()
        cur.execute("SELECT COUNT(*) FROM seeds")
        if cur.fetchone()[0] == 0:
            sample = [
                ("mateo", "I study HCI at Stanford."),
                ("mateo", "I enjoy Radiohead."),
                ("dünya", "I was born in Turkey."),
                ("dünya", "I like to dance."),
            ]
            cur.executemany(
                "INSERT INTO seeds(agent, text) VALUES(?, ?)", sample
            )
            conn.commit()


def load_seed_memories(
    agent: str,
    *,
    path: Union[str, Path] = DB_PATH,
) -> List[str]:
    """Return seed memory texts for *agent* from the database."""
    path = Path(path)
    if not path.exists():
        return []
    agent = agent.lower()
    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT text FROM seeds WHERE agent=? ORDER BY id", (agent,))
        rows = cur.fetchall()
    return [r[0] for r in rows]
