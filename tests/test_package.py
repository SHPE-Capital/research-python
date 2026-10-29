from shpe_research import __version__
from shpe_research.cli import main


def test_version_is_set():
    assert __version__


def test_cli_without_command_prints_help(capsys):
    assert main([]) == 0
    assert "shpe" in capsys.readouterr().out
