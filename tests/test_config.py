#!/bin/env python
"""
Tests for extra_swift_lightcones.config: downloading the FLAMINGO shell redshift files into the environment,
and exporting their paths when the environment is activated. The files are served by a local HTTP server.
"""
import functools
import http.server
import os
import sys
import threading

import pytest

from extra_swift_lightcones import config
from extra_swift_lightcones import swift_snapshot_redshift_conversion as nz

FILE_CONTENTS = {
    "L1": "0.0, 0.05\n0.05, 0.1\n",
    "L2p8": "0.0, 0.025\n0.025, 0.05\n0.05, 0.075\n",
}


@pytest.fixture
def file_server(tmp_path, monkeypatch):
    """
    Serve fake shell redshift files and a HTML page from a local HTTP server,
    and point config.SHELL_REDSHIFT_FILES at them.

    Returns a dict of {box: number of requests for its file}.
    """
    served = tmp_path / "served"
    served.mkdir()
    for box, text in FILE_CONTENTS.items():
        (served / f"{box}.txt").write_text(text)
    (served / "login.txt").write_text("<html><body>please log in</body></html>")

    requests = {box: 0 for box in FILE_CONTENTS}

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            box = self.path.strip("/").removesuffix(".txt")
            if box in requests:
                requests[box] += 1
            return super().do_GET()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(served)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    files = {box: (env_name, filename, f"{url}/{box}.txt") for box, (env_name, filename, _) in config.SHELL_REDSHIFT_FILES.items()}
    monkeypatch.setattr(config, "SHELL_REDSHIFT_FILES", files)
    monkeypatch.setattr(nz, "SHELL_REDSHIFT_FILES", files)
    yield requests, url
    server.shutdown()


@pytest.fixture
def fake_venv(tmp_path, monkeypatch):
    """
    A fake virtual environment, with an activate script, used as sys.prefix.
    """
    prefix = tmp_path / "venv"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "activate").write_text("# venv activate script\n")
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    for env_name, _, _ in config.SHELL_REDSHIFT_FILES.values():
        monkeypatch.delenv(env_name, raising=False)
    return prefix


def test_download_shell_redshifts(file_server, fake_venv):
    requests, _ = file_server
    paths = config.download_shell_redshifts(verbose=False)
    # downloaded into the environment
    redshift_dir = fake_venv / "share" / "extra_swift_lightcones" / "redshifts"
    assert paths == {"L1_REDSHIFTS_FILENAME": str(redshift_dir / "L1_shell_redshifts_z3.txt"),
                     "L2P8_REDSHIFTS_FILENAME": str(redshift_dir / "L2p8_shell_redshifts_z5.txt")}
    assert open(paths["L1_REDSHIFTS_FILENAME"]).read() == FILE_CONTENTS["L1"]
    assert open(paths["L2P8_REDSHIFTS_FILENAME"]).read() == FILE_CONTENTS["L2p8"]
    # files already downloaded are not downloaded again, unless forced
    config.download_shell_redshifts(verbose=False)
    assert requests == {"L1": 1, "L2p8": 1}
    config.download_shell_redshifts(force=True, verbose=False)
    assert requests == {"L1": 2, "L2p8": 2}


def test_download_rejects_web_pages(file_server, tmp_path, monkeypatch):
    _, url = file_server
    files = dict(config.SHELL_REDSHIFT_FILES)
    files["L1"] = (files["L1"][0], files["L1"][1], f"{url}/login.txt")
    monkeypatch.setattr(config, "SHELL_REDSHIFT_FILES", files)
    with pytest.raises(OSError, match="did not return a text file"):
        config.download_shell_redshifts(dest_dir=tmp_path / "redshifts", verbose=False)
    # nothing is left behind for a failed download
    assert not os.path.exists(tmp_path / "redshifts" / "L1_shell_redshifts_z3.txt")
    assert not os.path.exists(tmp_path / "redshifts" / "L1_shell_redshifts_z3.txt.part")


def test_download_reports_unreachable_server(tmp_path, monkeypatch):
    files = {box: (env_name, filename, "http://127.0.0.1:9/missing.txt")
             for box, (env_name, filename, _) in config.SHELL_REDSHIFT_FILES.items()}
    monkeypatch.setattr(config, "SHELL_REDSHIFT_FILES", files)
    with pytest.raises(OSError, match="could not download"):
        config.download_shell_redshifts(dest_dir=tmp_path, verbose=False)


def test_configure_exports_paths_in_venv(file_server, fake_venv):
    paths = config.configure(verbose=False)
    activate = (fake_venv / "bin" / "activate").read_text()
    assert activate.startswith("# venv activate script\n")
    for name, path in paths.items():
        assert f'export {name}="{path}"' in activate
    # running again replaces the lines instead of adding them again
    other_dir = fake_venv / "other"
    config.configure(dest_dir=str(other_dir), verbose=False)
    activate = (fake_venv / "bin" / "activate").read_text()
    assert activate.count("L1_REDSHIFTS_FILENAME") == 1
    assert f'export L1_REDSHIFTS_FILENAME="{other_dir / "L1_shell_redshifts_z3.txt"}"' in activate


def test_configure_exports_paths_in_conda(file_server, tmp_path, monkeypatch):
    prefix = tmp_path / "conda_env"
    (prefix / "conda-meta").mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    paths = config.configure(verbose=False)
    activate = (prefix / "etc" / "conda" / "activate.d" / "extra_swift_lightcones.sh").read_text()
    deactivate = (prefix / "etc" / "conda" / "deactivate.d" / "extra_swift_lightcones.sh").read_text()
    for name, path in paths.items():
        assert f'export {name}="{path}"' in activate
        assert f"unset {name}" in deactivate


def test_configure_outside_an_environment(file_server, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "system"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "system"))
    paths = config.configure(dest_dir=str(tmp_path / "redshifts"))
    # no activate script to update, so the exports are printed instead
    out = capsys.readouterr().out
    for name, path in paths.items():
        assert f'export {name}="{path}"' in out


def test_cli_configure(file_server, fake_venv, tmp_path):
    config.cli_configure(["--dest_dir", str(tmp_path / "cli"), "--no_activate"])
    assert (tmp_path / "cli" / "L1_shell_redshifts_z3.txt").read_text() == FILE_CONTENTS["L1"]
    assert "L1_REDSHIFTS_FILENAME" not in (fake_venv / "bin" / "activate").read_text()


def test_shell_redshift_file_lookup(file_server, fake_venv, tmp_path, monkeypatch):
    requests, _ = file_server
    # the environment variable is used first
    env_file = tmp_path / "from_env.txt"
    env_file.write_text("0.0, 0.1\n")
    monkeypatch.setenv("L1_REDSHIFTS_FILENAME", str(env_file))
    assert nz.flamingo_shell_redshift_file("L1") == str(env_file)
    monkeypatch.delenv("L1_REDSHIFTS_FILENAME")

    # otherwise the files are downloaded into the environment when they are first needed, and found there after
    monkeypatch.setattr(os.path, "expanduser", lambda path: path.replace("~", str(tmp_path / "home")))
    monkeypatch.setattr(nz, "__file__", str(tmp_path / "not_a_repo" / "module.py"))
    path = nz.flamingo_shell_redshift_file("L2p8")
    assert path == str(fake_venv / "share" / "extra_swift_lightcones" / "redshifts" / "L2p8_shell_redshifts_z5.txt")
    assert nz.flamingo_shell_redshift_file("L2p8") == path
    assert requests == {"L1": 1, "L2p8": 1}

    with pytest.raises(ValueError, match="box must be one of"):
        nz.flamingo_shell_redshift_file("L5")


def test_shell_redshift_file_not_found(fake_venv, tmp_path, monkeypatch):
    monkeypatch.setattr(os.path, "expanduser", lambda path: path.replace("~", str(tmp_path / "home")))
    monkeypatch.setattr(nz, "__file__", str(tmp_path / "not_a_repo" / "module.py"))
    with pytest.raises(FileNotFoundError, match="extra_swift_lightcones-configure"):
        nz.flamingo_shell_redshift_file("L1", download=False)