import json
import logging
import sys

from legalrag.logging import log_extra, request_id_var, setup_logging


def test_logs_go_to_stderr_as_json_with_request_id(capsys):
    setup_logging("INFO")
    token = request_id_var.set("req-1")
    try:
        logging.getLogger("t").info("hello", extra=log_extra(stage="x"))
    finally:
        request_id_var.reset(token)
    out, err = capsys.readouterr()
    assert out == ""  # stdout stays clean for CLI JSON output
    rec = json.loads(err.strip().splitlines()[-1])
    assert rec["msg"] == "hello" and rec["request_id"] == "req-1" and rec["stage"] == "x"
    assert logging.getLogger().handlers[0].stream is sys.stderr
