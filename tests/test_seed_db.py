from core import seed_db


def test_seed_db(tmp_path):
    db = tmp_path / "seed.db"
    seed_db.init_db(db)

    memories = seed_db.load_seed_memories("mateo", path=db)

    assert memories
