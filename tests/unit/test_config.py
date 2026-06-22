from cipher.config import find_project_root


def test_find_project_root_from_nested_directory(project_fixture, monkeypatch) -> None:
    nested = project_fixture.root / "one/two"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert find_project_root() == project_fixture.root


def test_paths_resolve_against_project_root(project_fixture) -> None:
    expected = project_fixture.root / "data/raw/matches"
    assert project_fixture.resolve(project_fixture.project.paths.matches_root) == expected


def test_extensions_are_normalized(project_fixture) -> None:
    assert project_fixture.project.validation.allowed_extensions == [".jpeg", ".jpg", ".png"]
