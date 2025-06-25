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
                mode TEXT NOT NULL,
                text TEXT NOT NULL
            )"""
        )
        conn.commit()
        cur.execute("SELECT COUNT(*) FROM seeds")
        if cur.fetchone()[0] == 0:
            sample = [
                ("mateo", "interview", "I study HCI at Stanford."),
                ("mateo", "web", "I enjoy Radiohead."),
                ("dünya", "interview", "I was born in Turkey."),
                ("dünya", "web", "I like to dance."),
            ]
            cur.executemany(
                "INSERT INTO seeds(agent, mode, text) VALUES(?, ?, ?)", sample
            )
            conn.commit()


def load_seed_memories(
    agent: str,
    mode: str = "combined",
    *,
    path: Union[str, Path] = DB_PATH,
) -> List[str]:
    """Return seed memory texts for *agent* and *mode* from the database."""
    path = Path(path)
    if not path.exists():
        return []
    agent = agent.lower()
    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        if mode == "combined":
            cur.execute("SELECT text FROM seeds WHERE agent=? ORDER BY id", (agent,))
        else:
            cur.execute(
                "SELECT text FROM seeds WHERE agent=? AND mode=? ORDER BY id",
                (agent, mode),
            )
        rows = cur.fetchall()
    return [r[0] for r in rows]
