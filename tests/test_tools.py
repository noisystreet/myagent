"""Tests for tool functions."""

from src.tools.file_tools import read_file, write_file, edit_file


class TestFileTools:
    def test_write_and_read(self, temp_workspace):
        path = str(temp_workspace / "test.txt")
        r = write_file(path, "hello")
        assert r.success
        assert "Written" in r.data

        r = read_file(path)
        assert r.success
        assert "hello" in r.data

    def test_read_nonexistent(self):
        r = read_file("/nonexistent/path/file.txt")
        assert not r.success
        assert "not found" in r.error

    def test_edit_file(self, temp_workspace):
        path = str(temp_workspace / "edit.txt")
        write_file(path, "hello world")
        r = edit_file(path, "hello", "hi")
        assert r.success

        r = read_file(path)
        assert "hi world" in r.data

    def test_edit_not_found(self, temp_workspace):
        path = str(temp_workspace / "edit.txt")
        write_file(path, "hello")
        r = edit_file(path, "nonexistent", "x")
        assert not r.success


class TestCommandTools:
    def test_run_echo(self):
        from src.tools.command_tools import run_command
        r = run_command("echo hello")
        assert r.success
        assert "hello" in r.data

    def test_run_failure(self):
        from src.tools.command_tools import run_command
        r = run_command("exit 1")
        assert not r.success

    def test_run_timeout(self):
        from src.tools.command_tools import run_command
        r = run_command("sleep 5", timeout=1)
        assert not r.success
        assert "timed out" in r.error.lower()
