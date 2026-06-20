"""Tests for tool functions."""

from src.tools.file_tools import read_file, write_file, edit_file


class TestFileTools:
    def test_write_and_read(self, temp_workspace):
        path = str(temp_workspace / "test.txt")
        r = write_file(path, "hello")
        assert r.success
        assert r.data["bytes_written"] == 5

        r = read_file(path)
        assert r.success
        assert "hello" in r.data["content"]

    def test_read_nonexistent(self):
        r = read_file("/nonexistent/path/file.txt")
        assert not r.success
        assert r.error["type"] == "file_not_found"

    def test_edit_file(self, temp_workspace):
        path = str(temp_workspace / "edit.txt")
        write_file(path, "hello world")
        r = edit_file(path, "hello", "hi")
        assert r.success
        assert r.data["replacements"] == 1

        r = read_file(path)
        assert "hi world" in r.data["content"]

    def test_edit_not_found(self, temp_workspace):
        path = str(temp_workspace / "edit.txt")
        write_file(path, "hello")
        r = edit_file(path, "nonexistent", "x")
        assert not r.success
        assert r.error["type"] == "string_not_found"


class TestCommandTools:
    def test_run_echo(self):
        from src.tools.command_tools import run_command
        r = run_command("echo hello")
        assert r.success
        assert "hello" in r.data["output"]
        assert r.data["return_code"] == 0

    def test_run_failure(self):
        from src.tools.command_tools import run_command
        r = run_command("exit 1")
        assert not r.success
        assert r.error["type"] == "non_zero_exit"
        assert r.data["return_code"] == 1

    def test_run_timeout(self):
        from src.tools.command_tools import run_command
        r = run_command("sleep 5", timeout=1)
        assert not r.success
        assert r.error["type"] == "timeout"


class TestTimeTools:
    def test_get_time_default_local(self):
        from src.tools.time_tools import get_time

        r = get_time()
        assert r.success
        assert r.tool_name == "get_time"
        assert "timezone" in r.data
        assert "iso" in r.data
        assert "datetime" in r.data
        assert "date" in r.data
        assert "time" in r.data
        assert "weekday" in r.data
        assert "timestamp" in r.data

        # Verify timestamp is recent (within last 10 seconds)
        import time as time_mod

        now_ts = int(time_mod.time())
        assert abs(r.data["timestamp"] - now_ts) < 10

    def test_get_time_utc(self):
        from src.tools.time_tools import get_time

        r = get_time(tz="UTC")
        assert r.success
        assert r.data["timezone"] == "UTC"
        # UTC ISO format should end with +00:00
        assert r.data["iso"].endswith("+00:00")

    def test_get_time_explicit_tz(self):
        from src.tools.time_tools import get_time

        r = get_time(tz="Asia/Shanghai")
        assert r.success
        assert r.data["timezone"] == "Asia/Shanghai"

    def test_get_time_invalid_timezone(self):
        from src.tools.time_tools import get_time

        r = get_time(tz="Invalid/Zone")
        assert not r.success
        assert r.error["type"] == "invalid_timezone"
