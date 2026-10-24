"""Tests for :mod:`core.logging_setup` - sinks and secret redaction."""

from __future__ import annotations

import json
from pathlib import Path

from core.logging_setup import (
    SECRET_KEYS,
    collect_secrets,
    make_redaction_patcher,
    setup_logging,
)


def test_secret_values_are_collected_and_short_values_ignored() -> None:
    environ = {
        "ZERODHA_API_KEY": "abcdef123456",
        "ZERODHA_PASSWORD": "short",
        "ZERODHA_TOTP_KEY": "",
    }
    secrets = collect_secrets(environ, ["ZERODHA_API_KEY", "ZERODHA_PASSWORD", "ZERODHA_TOTP_KEY"])
    # Short placeholder-looking values are ignored to avoid masking everything.
    assert secrets == ("abcdef123456",)


def test_extra_keys_are_redacted_by_name() -> None:
    patcher = make_redaction_patcher(())
    record = {
        "message": "connected",
        "extra": {"api_key": "abcdef", "zerodha_api_secret": "xyz", "user_id": "AB1234"},
    }
    patcher(record)
    assert record["extra"]["api_key"] == "***"
    assert record["extra"]["zerodha_api_secret"] == "***"
    assert record["extra"]["user_id"] == "AB1234"


def test_secret_values_inside_the_message_are_masked() -> None:
    patcher = make_redaction_patcher(("supersecrettoken",))
    record = {"message": "GET https://api.kite.trade/quote?api_key=supersecrettoken", "extra": {}}
    patcher(record)
    assert "supersecrettoken" not in record["message"]
    assert "***" in record["message"]


def test_redaction_handles_nested_structures() -> None:
    patcher = make_redaction_patcher(())
    record = {
        "message": "x",
        "extra": {"credentials": {"password": "hunter2", "user_id": "AB1234"}, "list": [{"totp": "123"}]},
    }
    patcher(record)
    assert record["extra"]["credentials"]["password"] == "***"
    assert record["extra"]["credentials"]["user_id"] == "AB1234"
    assert record["extra"]["list"][0]["totp"] == "***"


def test_secret_keys_include_the_expected_names() -> None:
    assert {"api_key", "api_secret", "password", "totp", "access_token"} <= SECRET_KEYS


def test_setup_logging_creates_the_expected_sinks(tmp_path: Path) -> None:
    logger = setup_logging(
        app_name="anjaneeya_test",
        log_dir=tmp_path / "logs",
        level="DEBUG",
        console=False,
        file=True,
        json_sink=True,
        intercept_stdlib=False,
    )
    logger.info("hello from the test")
    logger.bind(api_key="abcdef123456").info("bound secret")  # redacted by key name

    text_log = tmp_path / "logs" / "anjaneeya_test.log"
    json_log = tmp_path / "logs" / "anjaneeya.jsonl"
    assert text_log.exists() and json_log.exists()

    content = text_log.read_text(encoding="utf-8")
    assert "hello from the test" in content
    # A secret bound to the record must never reach the file, by value or key.
    assert "abcdef123456" not in content

    lines = [line for line in json_log.read_text(encoding="utf-8").strip().splitlines() if line]
    parsed = json.loads(lines[0])
    assert parsed["record"]["extra"]["app"] == "anjaneeya_test"

    # The structured sink keeps the (redacted) bound extras, which is exactly
    # what makes this useful for post-hoc analysis.
    bound = next(json.loads(line) for line in lines if "bound secret" in line)
    assert bound["record"]["extra"]["api_key"] == "***"


def test_setup_logging_is_idempotent_and_respects_the_level(tmp_path: Path) -> None:
    first = setup_logging(
        app_name="anjaneeya_test", log_dir=tmp_path / "logs", level="INFO", console=False
    )
    second = setup_logging(
        app_name="anjaneeya_test", log_dir=tmp_path / "logs", level="ERROR", console=False
    )
    second.debug("this must not be written")
    second.error("this must be written")

    content = (tmp_path / "logs" / "anjaneeya_test.log").read_text(encoding="utf-8")
    assert "this must be written" in content
    assert "this must not be written" not in content
    assert first is not None


def test_setup_logging_debug_level_writes_debug_lines(tmp_path: Path) -> None:
    logger = setup_logging(
        app_name="anjaneeya_debug", log_dir=tmp_path / "logs", level="DEBUG", console=False
    )
    logger.debug("debug line")
    content = (tmp_path / "logs" / "anjaneeya_debug.log").read_text(encoding="utf-8")
    assert "debug line" in content


def test_stdlib_interception_does_not_crash(tmp_path: Path) -> None:
    import logging

    setup_logging(
        app_name="anjaneeya_stdlib",
        log_dir=tmp_path / "logs",
        level="INFO",
        console=False,
        intercept_stdlib=True,
    )
    logging.getLogger("kiteconnect").info("a third-party log line")
    logging.getLogger("anjaneeya.test").warning("a stdlib warning")
    content = (tmp_path / "logs" / "anjaneeya_stdlib.log").read_text(encoding="utf-8")
    assert "a stdlib warning" in content
