from core import seed_db


def test_seed_db(tmp_path):
    db = tmp_path / "seed.db"
    seed_db.init_db(db)

    interview = seed_db.load_seed_memories("mateo", "interview", path=db)
    web = seed_db.load_seed_memories("mateo", "web", path=db)
    combined = seed_db.load_seed_memories("mateo", "combined", path=db)

    assert set(combined) == set(interview + web)
    assert interview and web
