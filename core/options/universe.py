"""Option universe construction - spec 17 (Stage 3.4).

What this module decides
------------------------
Spec 17 fixes the *shape* of ANJANEYA's option universe and leaves every number to
configuration:

* which expiries trade (``current_week``, ``next_week``, ``current_month``,
  ``next_month``),
* which strikes sit around at-the-money (the spec's initial
  ``ATM-100 / ATM / ATM+100`` ladder is a config line, not a constant, and the
  documented expansion to ``-100, -50, 0, +50, +100`` is a config edit),
* how the reference price is snapped to a strike (``nearest`` / ``floor`` /
  ``ceil``; ``nearest`` is ``floor(price / interval + 0.5)``, so an exact midpoint
  rounds **up** - a stated rule, not a floating-point accident), and
* whether at-the-money means the **spot** index or the **near futures** contract.

Everything here is a pure function of configuration, one reference price, a
trading day and the market calendar. No instrument tokens, no symbol strings, no
I/O: a contract is identified by ``(expiry, strike, option_type)`` - the same
identity ANJANEYA uses in every table - and the token lookup stays where the
instrument dump lives (``apps.anjaneeya_capture.instruments``). That split is what
lets the engine and the AI app reason about *which strikes were in the universe*
long before (or long after) anything was subscribed.

Spec 17: "Do not assume the best strike beforehand."
----------------------------------------------------
The universe is a **menu**, so :class:`OptionSelection` keeps the strike ladder
and the ATM anchor side by side instead of collapsing them into one number. The
ladder is *not* required to contain the ATM (a strangle is a legitimate
configuration), and :attr:`OptionSelection.includes_atm` states which of the two
cases a configuration is in rather than letting a reader assume the ATM was
selected.

What this module deliberately does not do
-----------------------------------------
It does not know whether a strike is actually *listed*, or what its
``instrument_token`` is. A resolved selection is a *specification*; only the
instrument master can say whether the exchange lists it. Capture reports the
difference through ``OptionUniverse.missing``, and ``--mode universe`` reports it
against what the database holds - neither is invented here.

Importing this module performs no I/O.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from core.market_calendar import NseCalendar
from core.schemas import OptionType

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from core.config_loader import OptionUniverseConfig

__all__ = [
    "ATM_ROUNDINGS",
    "EXPIRY_SELECTORS",
    "OPTION_SIDES",
    "ContractSpec",
    "OptionSelection",
    "UniverseResolutionError",
    "atm_strike_for",
    "build_selection",
    "current_monthly_expiry",
    "next_monthly_expiry",
    "render_symbol",
    "select_contracts",
    "select_expiries",
    "strike_ladder",
    "template_pattern",
]

#: Expiry selectors understood in ``option_universe.yaml``.
EXPIRY_SELECTORS = ("current_week", "next_week", "current_month", "next_month")

#: How a reference price may be snapped to a strike.
ATM_ROUNDINGS = ("nearest", "floor", "ceil")

#: Option sides, in a stable order (so a resolved universe is reproducible).
OPTION_SIDES: tuple[OptionType, OptionType] = (OptionType.CE, OptionType.PE)


class UniverseResolutionError(RuntimeError):
    """Raised when the configured option universe cannot be resolved."""


def _validate_reference_price(price: float) -> float:
    """Return ``price`` as a float, refusing values that cannot place an ATM.

    A ``NaN`` from a failed quote, a zero or a negative number would otherwise
    snap to strike ``0``/``NaN`` and produce a universe that *looks* resolved.
    """
    try:
        value = float(price)
    except (TypeError, ValueError):
        raise ValueError(f"reference price must be a number, got {price!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"reference price must be a positive finite number, got {price!r}")
    return value


def atm_strike_for(price: float, *, interval: float, rounding: str = "nearest") -> float:
    """Snap a reference price to a strike.

    ``nearest`` rounds to the closest strike, ``floor``/``ceil`` pick the strike at
    or below/above the price.

    An exact midpoint (``25025`` on a 50-point ladder) rounds **up**: the rule is
    ``floor(price / interval + 0.5)``, which is deterministic for any price. The
    tie only exists on a half-interval boundary, and it is stated here rather than
    left to a reader's assumption - changing it would move the ATM of a live
    universe by one strike, which is a decision for the config, not a refactor.
    """
    if interval <= 0:
        raise ValueError(f"strike interval must be > 0, got {interval}")
    if rounding == "floor":
        return float(interval * math.floor(price / interval))
    if rounding == "ceil":
        return float(interval * math.ceil(price / interval))
    if rounding != "nearest":
        raise ValueError(
            f"Unknown ATM rounding {rounding!r}; expected one of {', '.join(ATM_ROUNDINGS)}"
        )
    return float(interval * math.floor(price / interval + 0.5))


def strike_ladder(
    atm_strike: float,
    offsets: Sequence[float],
    *,
    interval: float | None = None,
) -> tuple[float, ...]:
    """The ATM plus each configured offset, deduplicated and ascending.

    ``interval`` (the strike spacing) is optional; when given, every offset must
    be a whole number of intervals, because ``ATM+25`` on a 50-point ladder is
    not a strike the exchange lists - the failure would otherwise be a silent
    hole in the universe, discovered only when that strike fails to resolve.
    """
    if not offsets:
        raise ValueError("at least one strike offset is required to build a strike ladder")
    strikes: list[float] = []
    for offset in offsets:
        try:
            value = float(offset)
        except (TypeError, ValueError):
            raise ValueError(f"strike offset {offset!r} is not a number") from None
        if not math.isfinite(value):
            raise ValueError(f"strike offset {offset!r} is not a finite number")
        if interval is not None and interval > 0:
            remainder = math.fmod(abs(value), float(interval))
            if remainder > 1e-9 and abs(float(interval) - remainder) > 1e-9:
                raise ValueError(
                    f"strike offset {offset!r} is not a multiple of the strike interval "
                    f"{interval!r}; it would name a strike the exchange does not list"
                )
        strike = float(atm_strike) + value
        if strike <= 0:
            raise ValueError(
                f"strike offset {offset!r} from ATM {atm_strike!r} gives a non-positive strike"
            )
        strikes.append(strike)
    return tuple(sorted(set(strikes)))


def select_expiries(
    selectors: Sequence[str],
    *,
    day: date,
    calendar: NseCalendar,
) -> list[date]:
    """Resolve configured expiry selectors (``current_week``...) to real dates.

    Expiries that have already passed are dropped - subscribing to an expired
    contract would silently capture nothing.
    """
    resolved: list[date] = []
    for selector in selectors:
        if selector == "current_week":
            candidate = calendar.current_weekly_expiry(day)
        elif selector == "next_week":
            candidate = calendar.next_weekly_expiry(day)
        elif selector == "current_month":
            candidate = current_monthly_expiry(day, calendar=calendar)
        elif selector == "next_month":
            candidate = next_monthly_expiry(day, calendar=calendar)
        else:
            raise UniverseResolutionError(
                f"Unknown expiry selector {selector!r}; expected one of {EXPIRY_SELECTORS}"
            )
        if candidate >= day and candidate not in resolved:
            resolved.append(candidate)
    return sorted(resolved)


def current_monthly_expiry(day: date, *, calendar: NseCalendar) -> date:
    """The monthly contract currently trading: the nearest monthly expiry on/after ``day``.

    When the month's own monthly expiry has already passed (end of the expiry
    month), the next month's contract is the live one. The rule lives on the
    calendar (:meth:`NseCalendar.current_monthly_expiry`) so the universe and the
    status report cannot disagree about it.
    """
    return calendar.current_monthly_expiry(day)


def next_monthly_expiry(day: date, *, calendar: NseCalendar) -> date:
    """The monthly expiry strictly after :func:`current_monthly_expiry`."""
    return _month_expiry_after(current_monthly_expiry(day, calendar=calendar), calendar=calendar)


def _month_expiry_after(expiry: date, *, calendar: NseCalendar) -> date:
    """Monthly expiry of the calendar month following ``expiry``."""
    year, month = (expiry.year + 1, 1) if expiry.month == 12 else (expiry.year, expiry.month + 1)
    return calendar.monthly_expiry(year, month)


def render_symbol(
    template: str,
    *,
    expiry: date,
    strike: float,
    option_type: OptionType | str,
    monthly_codes: Mapping[int, str] | None = None,
) -> str:
    """Render a symbol from a configured template (a *cross-check*, not a lookup).

    Supported placeholders: ``{expiry_short}`` (the configured month code),
    ``{expiry}`` (ISO date), ``{strike}`` and ``{option_type}``.
    """
    kind = option_type.value if isinstance(option_type, OptionType) else str(option_type).upper()
    codes = monthly_codes or {}
    short = codes.get(expiry.month, expiry.strftime("%b").upper())
    strike_text = f"{int(round(strike))}" if float(strike).is_integer() else f"{strike:g}"
    return (
        template.replace("{expiry_short}", short)
        .replace("{expiry}", expiry.isoformat())
        .replace("{strike}", strike_text)
        .replace("{option_type}", kind)
    )


def template_pattern(template: str) -> re.Pattern[str]:
    """Compile a symbol template into a regex, so the *shape* can be checked.

    ``NIFTY{expiry_short}{strike}{option_type}`` becomes
    ``^NIFTY.+?\\d+(?:CE|PE)$`` - enough to detect that NSE changed the format
    without needing to know today's exact encoding.
    """
    escaped = re.escape(template)
    escaped = escaped.replace(r"\{expiry_short\}", ".+?").replace(r"\{expiry\}", ".+?")
    escaped = escaped.replace(r"\{strike\}", r"\d+")
    escaped = escaped.replace(r"\{option_type\}", r"(?:CE|PE)")
    return re.compile(f"^{escaped}$")


@dataclass(frozen=True, slots=True)
class ContractSpec:
    """One option contract named by its attributes - no token, no symbol.

    ``(expiry, strike, option_type)`` is the contract identity every ANJANEYA
    table keys on; the instrument token is an exchange implementation detail that
    this module deliberately does not carry.
    """

    expiry: date
    strike: float
    option_type: OptionType

    @property
    def key(self) -> tuple[date, float, str]:
        """Hashable identity, the same shape the ``instruments`` index uses."""
        return (self.expiry, float(self.strike), self.option_type.value)

    def symbol(self, *, template: str, monthly_codes: Mapping[int, str] | None = None) -> str:
        """The configured template rendered for this contract (a cross-check)."""
        return render_symbol(
            template,
            expiry=self.expiry,
            strike=self.strike,
            option_type=self.option_type,
            monthly_codes=monthly_codes,
        )

    def describe(self) -> dict[str, Any]:
        """JSON-friendly shape."""
        return {
            "expiry": self.expiry.isoformat(),
            "strike": self.strike,
            "option_type": self.option_type.value,
        }


def select_contracts(
    *,
    strikes: Sequence[float],
    expiries: Sequence[date],
    sides: Sequence[OptionType] = OPTION_SIDES,
) -> tuple[ContractSpec, ...]:
    """The full strike x expiry x side menu, in a reproducible order.

    Ascending by ``(expiry, strike, option_type)``: two runs of the same config
    produce identical lists, which is what makes a resolved universe comparable
    across days. An empty input yields an empty menu rather than an exception -
    "nothing selected" is a fact the caller reports (``is_empty``).
    """
    if not strikes or not expiries or not sides:
        return ()
    return tuple(
        ContractSpec(expiry=expiry, strike=float(strike), option_type=side)
        for expiry in sorted(set(expiries))
        for strike in sorted(set(strikes))
        for side in sides
    )


@dataclass(frozen=True, slots=True)
class OptionSelection:
    """What spec 17's configuration selects for one day and one reference price.

    The selection is the *menu*, not the subscription: it names the contracts
    that should exist and is deliberately independent of any instrument dump, so
    it can be recomputed for any past day from the config alone.
    """

    underlying: str
    exchange: str
    reference_label: str
    reference_price: float
    atm_strike: float
    strike_interval: float
    strikes: tuple[float, ...]
    expiries: tuple[date, ...]
    contracts: tuple[ContractSpec, ...]
    created_at: datetime | None = None

    @property
    def is_empty(self) -> bool:
        """True when nothing was selected (no expiry resolved, or no strikes)."""
        return not self.contracts

    @property
    def includes_atm(self) -> bool:
        """True when the strike ladder contains the ATM strike itself."""
        return any(abs(strike - self.atm_strike) < 1e-9 for strike in self.strikes)

    @property
    def offsets(self) -> tuple[float, ...]:
        """The selected strikes expressed as offsets from the ATM, ascending."""
        return tuple(strike - self.atm_strike for strike in self.strikes)

    def describe(self) -> dict[str, Any]:
        """JSON-friendly shape (a log line, a summary or a report row)."""
        return {
            "underlying": self.underlying,
            "exchange": self.exchange,
            "reference_label": self.reference_label,
            "reference_price": self.reference_price,
            "atm_strike": self.atm_strike,
            "strike_interval": self.strike_interval,
            "offsets": list(self.offsets),
            "strikes": list(self.strikes),
            "includes_atm": self.includes_atm,
            "expiries": [expiry.isoformat() for expiry in self.expiries],
            "contracts": len(self.contracts),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


def build_selection(
    *,
    universe_config: OptionUniverseConfig,
    reference_price: float,
    day: date,
    calendar: NseCalendar,
    reference_label: str | None = None,
    created_at: datetime | None = None,
) -> OptionSelection:
    """Turn ``option_universe.yaml`` + a reference price into a contract menu.

    ``reference_label`` records *which* price was handed in (``"spot"`` or
    ``"futures"``) and defaults to the configured ``atm.reference``. Capture passes
    the label of the price it actually fetched, so a run whose price came from the
    fallback path cannot claim to have used the other one - that mismatch is the
    caller's to report, never this function's to hide.
    """
    price = _validate_reference_price(reference_price)
    interval = float(universe_config.instrument.strike_interval)
    atm = atm_strike_for(price, interval=interval, rounding=universe_config.atm.rounding)
    strikes = strike_ladder(atm, universe_config.strikes_offset, interval=interval)
    expiries = select_expiries(universe_config.expiries, day=day, calendar=calendar)
    return OptionSelection(
        underlying=universe_config.underlying,
        exchange=universe_config.exchange.upper(),
        reference_label=reference_label or universe_config.atm.reference,
        reference_price=price,
        atm_strike=atm,
        strike_interval=interval,
        strikes=strikes,
        expiries=tuple(expiries),
        contracts=select_contracts(strikes=strikes, expiries=expiries),
        created_at=created_at,
    )

