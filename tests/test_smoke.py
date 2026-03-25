from asset_allocation_tc import __version__
from asset_allocation_tc.cli import main


def test_version_is_defined() -> None:
    assert __version__ == "0.1.0"


def test_main_runs(capsys) -> None:
    main()
    captured = capsys.readouterr()
    assert "ready" in captured.out.lower()
