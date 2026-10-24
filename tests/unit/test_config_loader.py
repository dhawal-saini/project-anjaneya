"""Tests for :mod:`core.config_loader` - validation, hashing and env handling."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from core.config_loader import (
    CONFIG_FILES,
    ConfigError,
    config_file_hash,
    load_config,
    load_yaml_file,
    referenced_env_vars,
    resolve_env_placeholders,
)


def test_shipped_configuration_loads(config) -> None:
    """The configuration shipped in the repository must be valid as-is."""
    summary = config.summary()
    assert summary["project"] == "ANJANEYA"
    assert summary["timezone"] == "Asia/Kolkata"
    assert summary["schema_version"] >= 1
    assert summary["algorithm"] in {"lightgbm", "xgboost"}
    assert summary["expiry_weekday"] in {
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
    }


def test_risk_controls_are_configured_as_documented(config) -> None:
    """Guard the spec's hard requirements so a YAML edit cannot silently break them."""
    assert config.model.data_gates.exploratory_min_days == 20
    # Relaxed 60 -> 45 (Phase 9) so an early EXPLORATORY model can train on the
    # first weeks of accumulated data; the gate still refuses below it.
    assert config.model.data_gates.ml_training_min_days == 45
    assert config.model.data_gates.meaningful_backtest_min_days == 120
    assert config.model.data_gates.seasonality_min_days == 250
    # The embargo must cover the longest label horizon.
    assert config.model.walk_forward.embargo_minutes >= max(config.model.horizons_minutes)
    assert config.model.walk_forward.holdout_days > 0
    # Options are the research target, so the universe must include strikes.
    assert config.option_universe.strikes_offset
    assert config.option_universe.expiries


def test_paths_resolve_against_repo_root(config, isolated_paths) -> None:
    """Relative configured paths must resolve to absolute paths under the data root."""
    assert config.data_root == isolated_paths["data_root"].resolve()
    assert config.db_path == isolated_paths["db_path"].resolve()
    assert config.data_dir("candles") == config.data_root / "candles"
    assert config.log_dir == config.data_root / "logs"


def test_data_dir_rejects_unknown_kind(config) -> None:
    with pytest.raises(ConfigError, match="Unknown data directory"):
        config.data_dir("not_a_directory")


def test_ensure_directories_creates_every_configured_directory(config) -> None:
    created = config.ensure_directories()
    assert created
    for path in created:
        assert path.exists()
    assert config.log_dir.exists()
    assert config.db_path.parent.exists()


# ---------------------------------------------------------------------------
# Environment placeholder handling
# ---------------------------------------------------------------------------
def test_env_placeholder_with_default_uses_default() -> None:
    resolved = resolve_env_placeholders({"root": "${ANJANEYA_DATA_ROOT:-./data}"}, {})
    assert resolved == {"root": "./data"}


def test_env_placeholder_uses_environment_when_set() -> None:
    resolved = resolve_env_placeholders(
        {"root": "${ANJANEYA_DATA_ROOT:-./data}", "nested": ["${X}"]},
        {"ANJANEYA_DATA_ROOT": "/tmp/anj", "X": "1"},
    )
    assert resolved == {"root": "/tmp/anj", "nested": ["1"]}


def test_env_placeholder_treats_empty_value_as_unset() -> None:
    """An empty env var must fall back to the default, not resolve to an empty string."""
    resolved = resolve_env_placeholders(
        {"level": "${ANJANEYA_LOG_LEVEL:-INFO}"}, {"ANJANEYA_LOG_LEVEL": ""}
    )
    assert resolved == {"level": "INFO"}


def test_missing_env_variable_without_default_is_a_hard_error() -> None:
    with pytest.raises(ConfigError, match="MISSING_VAR"):
        resolve_env_placeholders({"x": "${MISSING_VAR}"}, {})


def test_referenced_env_vars_finds_every_placeholder(config_dir: Path) -> None:
    names: set[str] = set()
    for name in CONFIG_FILES:
        names |= referenced_env_vars(load_yaml_file(config_dir / name))
    assert "ANJANEYA_DATA_ROOT" in names
    assert "ANJANEYA_DB_PATH" in names


def test_env_example_documents_every_referenced_variable(config_dir: Path, repo_root: Path) -> None:
    """``.env.example`` must document every placeholder used by the YAML files.

    Without this test the quickstart silently rots: a new ``${VAR}`` in a config
    file would break a fresh checkout with a confusing error.
    """
    names: set[str] = set()
    for name in CONFIG_FILES:
        names |= referenced_env_vars(load_yaml_file(config_dir / name))

    env_example = (repo_root / ".env.example").read_text(encoding="utf-8")
    missing = sorted(name for name in names if name not in env_example)
    assert not missing, f".env.example does not document: {missing}"

    for documented in (
        "ANJANEYA_DATA_ROOT",
        "ANJANEYA_DB_PATH",
        "ANJANEYA_LOG_LEVEL",
        "ANJANEYA_RUN_LIVE",
        "ZERODHA_API_KEY",
        "ZERODHA_API_SECRET",
        "ZERODHA_USER_ID",
        "ZERODHA_PASSWORD",
        "ZERODHA_TOTP_KEY",
        "ZERODHA_REDIRECT_URL",
    ):
        assert documented in env_example


# ---------------------------------------------------------------------------
# Hashing and validation failures
# ---------------------------------------------------------------------------
def test_config_hash_is_stable_and_content_sensitive(config_dir: Path, tmp_path: Path) -> None:
    first = config_file_hash(config_dir)
    assert first == config_file_hash(config_dir)

    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    settings_path = copied / "settings.yaml"
    settings_path.write_text(
        settings_path.read_text(encoding="utf-8") + "\n# touched\n", encoding="utf-8"
    )
    assert config_file_hash(copied) != first


def test_unknown_configuration_key_is_rejected(config_dir: Path, tmp_path: Path) -> None:
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    settings = yaml.safe_load((copied / "settings.yaml").read_text(encoding="utf-8"))
    settings["market"]["sesssion_start_typo"] = "09:15"
    (copied / "settings.yaml").write_text(yaml.safe_dump(settings), encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid ANJANEYA configuration"):
        load_config(copied)


def test_weights_must_sum_to_one(config_dir: Path, tmp_path: Path) -> None:
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    indicators = yaml.safe_load((copied / "indicators.yaml").read_text(encoding="utf-8"))
    indicators["nifty_scores"]["weights"] = {"fast": 0.5, "medium": 0.4, "long": 0.4}
    (copied / "indicators.yaml").write_text(yaml.safe_dump(indicators), encoding="utf-8")

    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_config(copied)


def test_missing_config_file_is_reported_by_name(tmp_path: Path) -> None:
    empty = tmp_path / "config"
    empty.mkdir()
    with pytest.raises(ConfigError, match="settings.yaml"):
        load_config(empty)


def test_missing_config_directory_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Configuration directory not found"):
        load_config(tmp_path / "nope")


def test_empty_yaml_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_yaml_file(path)


def test_non_mapping_yaml_root_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_yaml_file(path)


def test_market_session_bounds_are_validated(config_dir: Path, tmp_path: Path) -> None:
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    settings = yaml.safe_load((copied / "settings.yaml").read_text(encoding="utf-8"))
    settings["market"]["analysis_end"] = "16:00"  # outside the session
    (copied / "settings.yaml").write_text(yaml.safe_dump(settings), encoding="utf-8")
    with pytest.raises(ConfigError, match="analysis_end"):
        load_config(copied)


def test_embargo_shorter_than_horizon_is_rejected(config_dir: Path, tmp_path: Path) -> None:
    """A short embargo is a leakage bug, so it must fail configuration loading."""
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    model = yaml.safe_load((copied / "model.yaml").read_text(encoding="utf-8"))
    model["walk_forward"]["embargo_minutes"] = 1  # horizons go up to 30 minutes
    (copied / "model.yaml").write_text(yaml.safe_dump(model), encoding="utf-8")
    with pytest.raises(ConfigError, match="embargo_minutes"):
        load_config(copied)


def test_duplicate_holiday_is_rejected(config_dir: Path, tmp_path: Path) -> None:
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    calendar = yaml.safe_load((copied / "calendar.yaml").read_text(encoding="utf-8"))
    calendar["holidays"].append({"date": "2026-01-26", "name": "Duplicate Republic Day"})
    (copied / "calendar.yaml").write_text(yaml.safe_dump(calendar), encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate holiday"):
        load_config(copied)


def test_overlapping_time_buckets_are_rejected(config_dir: Path, tmp_path: Path) -> None:
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    calendar = yaml.safe_load((copied / "calendar.yaml").read_text(encoding="utf-8"))
    calendar["time_of_day_buckets"].append(
        {"name": "OVERLAP", "start": "09:20", "end": "09:40"}
    )
    (copied / "calendar.yaml").write_text(yaml.safe_dump(calendar), encoding="utf-8")
    with pytest.raises(ConfigError, match="must not overlap"):
        load_config(copied)


# ---------------------------------------------------------------------------
# The greeks model assumptions (spec 16, Stage 3.3)
# ---------------------------------------------------------------------------
def test_the_greeks_assumptions_are_the_documented_baseline(config) -> None:
    """A stored IV is only comparable across runs if these are pinned, so pin them here."""
    greeks = config.indicators.greeks

    assert greeks.model == "black_scholes"
    assert greeks.risk_free_rate == 0.065
    assert greeks.dividend_yield == 0.0
    assert greeks.price_field == "close"
    assert greeks.year_days == 365
    assert greeks.min_time_to_expiry_minutes == 375.0  # one session
    assert greeks.volatility_floor < greeks.volatility_ceiling


def test_the_greeks_section_defaults_when_the_file_omits_it(config_dir: Path, tmp_path: Path) -> None:
    """A Phase-1 style ``indicators.yaml`` still loads: the section is optional."""
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    indicators = yaml.safe_load((copied / "indicators.yaml").read_text(encoding="utf-8"))
    indicators.pop("greeks")
    (copied / "indicators.yaml").write_text(yaml.safe_dump(indicators), encoding="utf-8")

    loaded = load_config(copied)

    assert loaded.indicators.greeks.risk_free_rate == 0.065
    assert loaded.indicators.greeks.model == "black_scholes"


def test_a_greeks_bracket_that_cannot_be_searched_is_rejected(
    config_dir: Path, tmp_path: Path
) -> None:
    """A ceiling below the floor would make every IV unsolvable - a config error, loudly."""
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    indicators = yaml.safe_load((copied / "indicators.yaml").read_text(encoding="utf-8"))
    indicators["greeks"]["volatility_ceiling"] = 0.001
    (copied / "indicators.yaml").write_text(yaml.safe_dump(indicators), encoding="utf-8")

    with pytest.raises(ConfigError, match="volatility_ceiling"):
        load_config(copied)


def test_an_unknown_greeks_key_is_rejected(config_dir: Path, tmp_path: Path) -> None:
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    indicators = yaml.safe_load((copied / "indicators.yaml").read_text(encoding="utf-8"))
    indicators["greeks"]["volality_floor"] = 0.01  # typo
    (copied / "indicators.yaml").write_text(yaml.safe_dump(indicators), encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid ANJANEYA configuration"):
        load_config(copied)


# ---------------------------------------------------------------------------
# The backtest section (spec 23, Stage 4.3)
# ---------------------------------------------------------------------------
def test_the_backtest_section_is_the_documented_baseline(config) -> None:
    """The shipped signal rule, sizing and exits, so a YAML edit cannot silently change them."""
    backtest = config.model.backtest
    assert backtest.signal.mode == "probability_threshold"
    assert backtest.signal.min_probability == pytest.approx(0.40)
    assert backtest.signal.require_argmax is True
    assert backtest.signal.directions == {"CE": "UP_CONTINUATION", "PE": "DOWN_CONTINUATION"}
    assert backtest.sizing.lots == 1
    assert backtest.sizing.lot_size == 75
    assert backtest.sizing.quantity == 75
    assert backtest.exit.stop_loss_atr == pytest.approx(1.0)
    assert backtest.exit.target_atr == pytest.approx(1.5)
    assert backtest.exit.max_holding_minutes == pytest.approx(30.0)
    assert backtest.exit.flat_at_session_end is True
    assert backtest.exit.intrabar_priority == "stop_first"
    assert backtest.exit.exit_on_opposite_signal is True


def _write_model(copied: Path, mutate) -> Path:
    """Copy ``model.yaml``, apply ``mutate``, write it back, and return the directory."""
    payload = yaml.safe_load((copied / "model.yaml").read_text(encoding="utf-8"))
    mutate(payload)
    (copied / "model.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    return copied


def test_a_signal_class_that_the_labels_never_produce_is_rejected(
    config_dir: Path, tmp_path: Path
) -> None:
    """A direction the labels cannot emit would make every signal dead on arrival."""
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    _write_model(
        copied,
        lambda payload: payload["backtest"]["signal"]["directions"].update({"CE": "MOONSHOT"}),
    )
    with pytest.raises(ConfigError, match="labels.classes does not define"):
        load_config(copied)


def test_an_unknown_option_type_in_the_directions_is_rejected(
    config_dir: Path, tmp_path: Path
) -> None:
    """Only CE and PE exist; anything else is a typo with consequences."""
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    _write_model(
        copied,
        lambda payload: payload["backtest"]["signal"]["directions"].update({"XX": "RANGE"}),
    )
    with pytest.raises(ConfigError, match="unknown option type"):
        load_config(copied)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("backtest", "signal", "min_probability"), 0.0, "min_probability"),
        (("backtest", "signal", "min_probability"), 1.5, "min_probability"),
        (("backtest", "sizing", "lots"), 0, "lots"),
        (("backtest", "sizing", "lot_size"), 0, "lot_size"),
        (("backtest", "exit", "stop_loss_atr"), 0.0, "stop_loss_atr"),
        (("backtest", "exit", "target_atr"), -1.0, "target_atr"),
        (("backtest", "exit", "max_holding_minutes"), 0.0, "max_holding_minutes"),
        (("backtest", "exit", "intrabar_priority"), "either", "intrabar_priority"),
        (("backtest", "signal", "mode"), "always_long", "mode"),
    ],
)
def test_a_backtest_value_that_cannot_be_honoured_is_rejected(
    config_dir: Path, tmp_path: Path, path: tuple[str, ...], value: object, message: str
) -> None:
    """Every knob the engine reads is validated where it is loaded."""
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)

    def mutate(payload: dict) -> None:
        target = payload
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    _write_model(copied, mutate)
    with pytest.raises(ConfigError, match=message):
        load_config(copied)


def test_a_null_exit_turns_that_exit_off_rather_than_defaulting(config_dir: Path, tmp_path: Path) -> None:
    """``stop_loss_atr: null`` means "no stop", which is a decision, not a missing value."""
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    _write_model(
        copied,
        lambda payload: payload["backtest"]["exit"].update(
            {"stop_loss_atr": None, "max_holding_minutes": None}
        ),
    )
    loaded = load_config(copied)
    assert loaded.model.backtest.exit.stop_loss_atr is None
    assert loaded.model.backtest.exit.max_holding_minutes is None
    assert loaded.model.backtest.exit.target_atr == pytest.approx(1.5)


def test_an_unknown_backtest_key_is_rejected(config_dir: Path, tmp_path: Path) -> None:
    """A typo in the backtest block fails loudly rather than being ignored."""
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    _write_model(
        copied, lambda payload: payload["backtest"]["exit"].update({"stop_los_atr": 2.0})
    )
    with pytest.raises(ConfigError, match="Invalid ANJANEYA configuration"):
        load_config(copied)


# ---------------------------------------------------------------------------
# The explainability section (spec 26, Stage 5.1)
# ---------------------------------------------------------------------------
def test_the_explainability_section_is_the_documented_baseline(config) -> None:
    """Spec 26's blocks, in the order the mode renders them, with the presentation rules."""
    block = config.model.explainability
    assert block.spec == "26"
    assert block.probabilities == 3
    assert block.digits == 4
    assert block.rows == 5
    assert block.missing_text == "unknown"
    assert [section.key for section in block.sections] == [
        "nifty",
        "option",
        "price_action",
        "candle",
        "support_resistance",
        "context",
    ]
    assert [section.title for section in block.sections] == [
        "NIFTY",
        "OPTION",
        "PRICE ACTION",
        "CANDLE",
        "SUPPORT / RESISTANCE",
        "CONTEXT",
    ]
    assert block.sections[2].groups == ["levels", "level_events", "structure"]
    assert block.sections[-1].groups == ["context", "relative"]


def test_an_explainability_group_that_does_not_exist_is_rejected(
    config_dir: Path, tmp_path: Path
) -> None:
    """A section naming a group the feature contract does not have would render nothing."""
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    _write_model(
        copied,
        lambda payload: payload["explainability"]["sections"][0].update({"groups": ["niftee"]}),
    )
    with pytest.raises(ConfigError, match="unknown feature group"):
        load_config(copied)


def test_an_explainability_column_that_does_not_exist_is_rejected(
    config_dir: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    _write_model(
        copied,
        lambda payload: payload["explainability"]["sections"][0].update(
            {"columns": ["nifty_regime_typo"]}
        ),
    )
    with pytest.raises(ConfigError, match="unknown feature column"):
        load_config(copied)


def test_a_feature_column_explained_twice_is_rejected(config_dir: Path, tmp_path: Path) -> None:
    """The same number in two blocks is how an explanation contradicts itself."""
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    _write_model(
        copied,
        lambda payload: payload["explainability"]["sections"][1].update(
            {"columns": ["nifty_regime"]}
        ),
    )
    with pytest.raises(ConfigError, match="explained twice"):
        load_config(copied)


def test_the_reserved_identity_and_provenance_groups_cannot_be_sections(
    config_dir: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    _write_model(
        copied,
        lambda payload: payload["explainability"]["sections"].append(
            {"key": "audit", "title": "AUDIT", "groups": ["identity"]}
        ),
    )
    with pytest.raises(ConfigError, match="reserved feature group"):
        load_config(copied)


def test_a_section_that_names_nothing_is_rejected(config_dir: Path, tmp_path: Path) -> None:
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    _write_model(
        copied,
        lambda payload: payload["explainability"]["sections"].append(
            {"key": "empty", "title": "EMPTY"}
        ),
    )
    with pytest.raises(ConfigError, match="neither groups nor columns"):
        load_config(copied)


def test_an_explainability_section_keyed_ai_is_rejected(config_dir: Path, tmp_path: Path) -> None:
    """``ai`` belongs to the model's own block, which is always appended last."""
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    _write_model(
        copied,
        lambda payload: payload["explainability"]["sections"].append(
            {"key": "ai", "title": "AI", "groups": ["option"]}
        ),
    )
    with pytest.raises(ConfigError, match="reserved key 'ai'"):
        load_config(copied)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("explainability", "probabilities"), 0, "probabilities"),
        (("explainability", "digits"), -1, "digits"),
        (("explainability", "rows"), 0, "rows"),
        (("explainability", "sections"), [], "sections"),
    ],
)
def test_an_explainability_value_that_cannot_be_honoured_is_rejected(
    config_dir: Path, tmp_path: Path, path: tuple[str, ...], value: object, message: str
) -> None:
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)

    def mutate(payload: dict) -> None:
        target = payload
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    _write_model(copied, mutate)
    with pytest.raises(ConfigError, match=message):
        load_config(copied)




# ---------------------------------------------------------------------------
# The importance section (spec 27, Stage 5.2)
# ---------------------------------------------------------------------------
def test_the_importance_block_is_the_documented_baseline(config) -> None:
    """Spec 27's measurement knobs and its sources, as the shipped configuration declares them."""
    block = config.model.importance
    assert block.spec == "27"
    assert block.digits == 4
    assert block.repeats == 5
    assert block.metric == "log_loss"
    assert block.confidence == pytest.approx(0.95)
    assert block.sample == "holdout"
    assert block.max_rows == 2000
    assert block.shap == "auto"
    assert block.shap_rows == 500
    assert block.seed is None
    # Spec 27's own list of sources, in order, then the four the list does not name.
    assert [topic.key for topic in block.topics] == [
        "nifty_regime",
        "option_ma",
        "candle_behaviour",
        "support_resistance",
        "structure",
        "price_action",
        "iv",
        "greeks",
        "oi",
        "option_residual",
        "time_of_day",
        "cycle_phase",
        "time_to_expiry",
        "day_of_week",
        "option_contract",
        "option_context",
    ]
    nifty = block.topics[0]
    assert nifty.groups == ["nifty"] and nifty.columns == []
    greeks = block.topics[7]
    assert greeks.columns == ["delta", "gamma", "theta", "vega"] and greeks.groups == []




def test_every_model_feature_is_claimed_by_exactly_one_importance_source(config) -> None:
    """No model input may be left unattributed: the sources cover the trainable contract exactly.

    A column added to the feature contract without a decision about its source fails here rather
    than appearing silently in the ``unmapped`` bucket of a report nobody reads twice.
    """
    from core.constants import FEATURE_COLUMNS
    from core.features.feature_builder import FEATURE_GROUPS
    from core.ml import LABEL_DERIVED_COLUMNS, META_COLUMNS

    trainable = [
        name
        for name in FEATURE_COLUMNS
        if name not in META_COLUMNS and name not in LABEL_DERIVED_COLUMNS
    ]
    claimed: list[str] = []
    for topic in config.model.importance.topics:
        claimed.extend(topic.columns)
        for name in topic.groups:
            claimed.extend(FEATURE_GROUPS[name])
    assert sorted(claimed) == sorted(trainable)
    assert len(claimed) == len(set(claimed)), "a column is claimed by two sources"


def test_an_importance_group_that_does_not_exist_is_rejected(
    config_dir: Path, tmp_path: Path
) -> None:
    """A source naming a group the feature contract does not have would claim nothing real."""
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    _write_model(
        copied,
        lambda payload: payload["importance"]["topics"][0].update({"groups": ["niftee"]}),
    )
    with pytest.raises(ConfigError, match="unknown feature group"):
        load_config(copied)


def test_an_importance_column_that_does_not_exist_is_rejected(
    config_dir: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    _write_model(
        copied,
        lambda payload: payload["importance"]["topics"][6].update({"columns": ["implied_vol"]}),
    )
    with pytest.raises(ConfigError, match="unknown feature column"):
        load_config(copied)


def test_a_column_claimed_by_two_importance_sources_is_rejected(
    config_dir: Path, tmp_path: Path
) -> None:
    """One column, one source: a feature counted twice would be credited twice."""
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    _write_model(
        copied,
        lambda payload: payload["importance"]["topics"][6].update({"columns": ["iv", "volume"]}),
    )
    with pytest.raises(ConfigError, match="claimed by two sources"):
        load_config(copied)


def test_a_reserved_importance_group_is_rejected(config_dir: Path, tmp_path: Path) -> None:
    """Identity is how a row is found and provenance describes its writer - neither is a source."""
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    _write_model(
        copied,
        lambda payload: payload["importance"]["topics"][0].update({"groups": ["provenance"]}),
    )
    with pytest.raises(ConfigError, match="reserved feature group"):
        load_config(copied)


def test_an_importance_topic_keyed_unmapped_is_rejected(config_dir: Path, tmp_path: Path) -> None:
    """``unmapped`` belongs to the features no source claims, so it cannot be configured."""
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    _write_model(
        copied,
        lambda payload: payload["importance"]["topics"].append(
            {"key": "unmapped", "title": "everything else", "columns": ["iv"]}
        ),
    )
    with pytest.raises(ConfigError, match="reserved key 'unmapped'"):
        load_config(copied)


def test_an_importance_topic_that_claims_nothing_is_rejected(
    config_dir: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)
    _write_model(
        copied,
        lambda payload: payload["importance"]["topics"].append(
            {"key": "mystery", "title": "mystery"}
        ),
    )
    with pytest.raises(ConfigError, match="neither groups nor columns"):
        load_config(copied)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("importance", "digits"), -1, "digits"),
        (("importance", "repeats"), 1, "repeats"),
        (("importance", "metric"), "f1", "metric"),
        (("importance", "confidence"), 1.0, "confidence"),
        (("importance", "sample"), "everything", "sample"),
        (("importance", "max_rows"), 0, "max_rows"),
        (("importance", "shap"), "maybe", "shap"),
        (("importance", "shap_rows"), 0, "shap_rows"),
        (("importance", "seed"), -1, "seed"),
        (("importance", "topics"), [], "topics"),
    ],
)
def test_an_importance_value_that_cannot_be_honoured_is_rejected(
    config_dir: Path, tmp_path: Path, path: tuple[str, ...], value: object, message: str
) -> None:
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)

    def mutate(payload: dict) -> None:
        target = payload
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    _write_model(copied, mutate)
    with pytest.raises(ConfigError, match=message):
        load_config(copied)

# ---------------------------------------------------------------------------
# The charts section (spec 28, Stage 5.3)
# ---------------------------------------------------------------------------
def test_the_charts_block_is_the_documented_baseline(config) -> None:
    """Spec 28's presentation knobs, as the shipped configuration declares them."""
    block = config.model.charts
    assert block.spec == "28"
    assert block.digits == 2
    assert block.dpi == 120
    assert block.width == pytest.approx(14.0)
    assert block.height == pytest.approx(7.0)
    assert block.tall_height == pytest.approx(9.0)
    assert block.format == "png"
    assert block.colormap == "RdYlGn"
    assert block.sequential_colormap == "viridis"
    assert block.top_features == 20
    assert block.max_points == 30_000
    assert block.max_markers == 60


def test_the_charts_defaults_and_the_documented_baseline_agree() -> None:
    """A section-less ``model.yaml`` still resolves exactly the shipped presentation."""
    from core.config_loader import ChartsConfig

    defaults = ChartsConfig()
    assert defaults.spec == "28"
    assert defaults.format == "png"
    assert defaults.top_features == 20


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("charts", "digits"), -1, "digits"),
        (("charts", "dpi"), 10, "dpi"),
        (("charts", "width"), 0.0, "width"),
        (("charts", "height"), 0.0, "height"),
        (("charts", "tall_height"), 0.0, "tall_height"),
        (("charts", "format"), "", "format"),
        (("charts", "colormap"), "", "colormap"),
        (("charts", "sequential_colormap"), "", "sequential_colormap"),
        (("charts", "top_features"), 0, "top_features"),
        (("charts", "max_points"), 1, "max_points"),
        (("charts", "max_markers"), 0, "max_markers"),
    ],
)
def test_a_charts_value_that_cannot_be_honoured_is_rejected(
    config_dir: Path, tmp_path: Path, path: tuple[str, ...], value: object, message: str
) -> None:
    copied = tmp_path / "config"
    shutil.copytree(config_dir, copied)

    def mutate(payload: dict) -> None:
        target = payload
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    _write_model(copied, mutate)
    with pytest.raises(ConfigError, match=message):
        load_config(copied)

