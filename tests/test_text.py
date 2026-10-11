from src.text import clean_text


def test_collapses_spaces_and_trims():
    assert clean_text("  المنتج   رائع  ") == "المنتج رائع"


def test_removes_diacritics():
    assert clean_text("جداً") == "جدا"


def test_shortens_elongated_letters():
    raw = "رااااااااااااااااااااائعة"
    assert len(clean_text(raw)) < len(raw)


def test_cleaning_twice_changes_nothing():
    once = clean_text("ممتاز نوعا ما . النظافة والموقع !!")
    assert clean_text(once) == once
