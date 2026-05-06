"""Database engine layer.

Provides a SQLAlchemy engine plus a thin connection adapter that mimics
the historic ``sqlite3.Connection`` API used throughout ``app.data.database``.
This lets the rest of the codebase keep using ``connection.execute(sql, params)``
and ``connection.executemany(sql, rows)`` while the underlying database is
PostgreSQL.

Translation rules applied transparently (legacy SQLite-style SQL kept in
callers for readability, rewritten before reaching Postgres):
- ``?`` placeholders -> ``:p0, :p1, ...`` (SQLAlchemy named binds)
- ``INSERT OR IGNORE INTO <t>`` -> ``INSERT INTO <t> ... ON CONFLICT DO NOTHING``
- ``INSERT OR REPLACE INTO <t> (cols) VALUES (...)`` ->
  ``INSERT ... ON CONFLICT (<pk>) DO UPDATE SET col = EXCLUDED.col, ...``

For ``INSERT OR REPLACE`` the conflict target column(s) come from
``REPLACE_CONFLICT_KEYS``.
"""
from __future__ import annotations

import os
import re
import threading
from contextlib import contextmanager
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.cursor import CursorResult


# Conflict targets for INSERT OR REPLACE rewriting (table -> conflict columns).
REPLACE_CONFLICT_KEYS: dict[str, tuple[str, ...]] = {
    "model_runs": ("run_id",),
    "backtest_runs": ("backtest_id",),
    "risk_alerts": ("alert_id",),
    "trade_outcome_runs": ("run_id",),
    "news_events": ("event_id",),
    "paper_trading_signals": ("signal_id",),
    "paper_positions": ("position_id",),
    "real_positions": ("position_id",),
    "qualitative_features": ("feature_id",),
    "fusion_predictions": ("fusion_id",),
}


_engine_lock = threading.Lock()
_engine: Engine | None = None


def _build_database_url() -> str:
    url = os.getenv("DATABASE_URL") or os.getenv("PROFIT_APP_DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is required (PostgreSQL). Example: "
            "postgresql+psycopg://user:pass@host:5432/dbname"
        )
    # Normalize bare postgres:// (Heroku-style) to SA's postgresql+psycopg://
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://") :]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def get_engine() -> Engine:
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        url = _build_database_url()
        engine = create_engine(
            url,
            future=True,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        _engine = engine
        return _engine


def is_postgres() -> bool:
    return get_engine().dialect.name == "postgresql"


# --- SQL translation ---------------------------------------------------------

_INSERT_OR_IGNORE_RE = re.compile(r"\bINSERT\s+OR\s+IGNORE\s+INTO\b", re.IGNORECASE)
_INSERT_OR_REPLACE_RE = re.compile(
    r"\bINSERT\s+OR\s+REPLACE\s+INTO\s+(?P<table>\w+)\s*\((?P<cols>[^)]+)\)",
    re.IGNORECASE,
)
_DELETE_QMARK_RE = re.compile(r"\?")


def _translate_sql_qmark_to_named(sql: str) -> tuple[str, list[str]]:
    """Replace ``?`` with ``:p0, :p1, ...`` and return param-name order."""
    names: list[str] = []
    counter = 0

    def repl(_match: re.Match[str]) -> str:
        nonlocal counter
        name = f"p{counter}"
        counter += 1
        names.append(name)
        return f":{name}"

    return _DELETE_QMARK_RE.sub(repl, sql), names


def _rewrite_insert_or_replace(sql: str) -> str:
    match = _INSERT_OR_REPLACE_RE.search(sql)
    if not match:
        return sql
    table = match.group("table")
    cols_raw = match.group("cols")
    cols = [c.strip() for c in cols_raw.split(",")]
    pk_cols = REPLACE_CONFLICT_KEYS.get(table)
    if not pk_cols:
        raise RuntimeError(
            f"INSERT OR REPLACE for table '{table}' has no conflict key registered "
            f"in REPLACE_CONFLICT_KEYS"
        )
    update_cols = [c for c in cols if c not in pk_cols]
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
    if not set_clause:
        # Only PK columns provided; treat as DO NOTHING.
        suffix = f"ON CONFLICT ({', '.join(pk_cols)}) DO NOTHING"
    else:
        suffix = f"ON CONFLICT ({', '.join(pk_cols)}) DO UPDATE SET {set_clause}"
    new_head = f"INSERT INTO {table} ({cols_raw})"
    rewritten = _INSERT_OR_REPLACE_RE.sub(new_head, sql, count=1)
    return rewritten + " " + suffix


def _rewrite_insert_or_ignore(sql: str) -> str:
    if not _INSERT_OR_IGNORE_RE.search(sql):
        return sql
    rewritten = _INSERT_OR_IGNORE_RE.sub("INSERT INTO", sql, count=1)
    return rewritten + " ON CONFLICT DO NOTHING"


def translate_sql(sql: str, postgres: bool | None = None) -> str:
    """Apply SQL rewrites (idempotent for non-matching SQL).

    The ``postgres`` argument is kept for backward compatibility with callers
    that may still pass it; it is now ignored because PostgreSQL is the only
    supported backend.
    """
    del postgres  # legacy arg, ignored
    sql = _rewrite_insert_or_ignore(sql)
    sql = _rewrite_insert_or_replace(sql)
    return sql


def _coerce_params_to_dict(
    params: Sequence[Any] | Mapping[str, Any] | None, names: list[str]
) -> dict[str, Any] | None:
    if params is None:
        if names:
            raise ValueError("SQL has placeholders but no parameters supplied")
        return None
    if isinstance(params, Mapping):
        return dict(params)
    seq = list(params)
    if len(seq) != len(names):
        raise ValueError(
            f"parameter count mismatch: sql expects {len(names)}, got {len(seq)}"
        )
    return {name: value for name, value in zip(names, seq)}


# --- LegacyConnection adapter -------------------------------------------------


class LegacyConnection:
    """Mimics ``sqlite3.Connection`` API on top of a SQLAlchemy connection.

    Supports ``with conn:`` semantics: commits on clean exit, rolls back on
    exception. ``execute()`` and ``executemany()`` translate SQL on the fly.
    ``read_sql_query`` is exposed for pandas integration.
    """

    def __init__(self, engine: Engine | None = None) -> None:
        self._engine = engine or get_engine()
        self._sa = self._engine.connect()
        self._tx = self._sa.begin()
        self._closed = False
        self._total_changes = 0

    @property
    def total_changes(self) -> int:
        return self._total_changes

    # context manager
    def __enter__(self) -> "LegacyConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                self._tx.commit()
            else:
                self._tx.rollback()
        finally:
            self._sa.close()
            self._closed = True

    # -- core execute APIs --
    def execute(
        self,
        sql: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> CursorResult:
        translated = translate_sql(sql)
        translated, names = _translate_sql_qmark_to_named(translated)
        bind = _coerce_params_to_dict(params, names) or {}
        result = self._sa.execute(text(translated), bind)
        try:
            rc = result.rowcount
            if rc and rc > 0:
                self._total_changes += rc
        except Exception:
            pass
        return result

    def executemany(
        self,
        sql: str,
        seq_of_params: Iterable[Sequence[Any] | Mapping[str, Any]],
    ) -> CursorResult | None:
        translated = translate_sql(sql)
        translated, names = _translate_sql_qmark_to_named(translated)
        rows = [_coerce_params_to_dict(p, names) or {} for p in seq_of_params]
        if not rows:
            return None
        result = self._sa.execute(text(translated), rows)
        try:
            rc = result.rowcount
            if rc and rc > 0:
                self._total_changes += rc
            else:
                self._total_changes += len(rows)
        except Exception:
            self._total_changes += len(rows)
        return result

    def executescript(self, script: str) -> None:
        # Split on semicolons at end of line, ignore empties.
        statements = [s.strip() for s in re.split(r";\s*\n", script) if s.strip()]
        for stmt in statements:
            # Strip trailing ; if any
            stmt = stmt.rstrip(";").strip()
            if not stmt:
                continue
            self._sa.execute(text(stmt))

    # -- pandas integration --
    def read_sql_query(
        self,
        sql: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        translated = translate_sql(sql)
        translated, names = _translate_sql_qmark_to_named(translated)
        bind = _coerce_params_to_dict(params, names)
        return pd.read_sql_query(text(translated), self._sa, params=bind)

    def cursor(self) -> "_LegacyCursor":
        """Return a DBAPI-cursor-like wrapper so pandas / legacy code works."""
        return _LegacyCursor(self)

    # -- convenience for migrations / introspection --
    @property
    def dialect_name(self) -> str:
        return self._engine.dialect.name

    def commit(self) -> None:
        # Manual commit creates a fresh transaction so further writes work.
        self._tx.commit()
        self._tx = self._sa.begin()

    def rollback(self) -> None:
        self._tx.rollback()
        self._tx = self._sa.begin()


@contextmanager
def connection_scope():
    """Helper alternative to ``with LegacyConnection() as conn:`` style."""
    conn = LegacyConnection()
    try:
        yield conn
        conn._tx.commit()
    except Exception:
        conn._tx.rollback()
        raise
    finally:
        conn._sa.close()


class _LegacyCursor:
    """DBAPI-cursor-like wrapper around a LegacyConnection.

    Sufficient for pandas' legacy (non-SQLAlchemy) ``read_sql_query`` path:
    exposes ``execute``, ``fetchall``, ``fetchone``, ``description``, ``close``.
    """

    def __init__(self, connection: LegacyConnection) -> None:
        self._connection = connection
        self._result: CursorResult | None = None
        self._closed = False

    def execute(
        self,
        sql: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> "_LegacyCursor":
        self._result = self._connection.execute(sql, params)
        return self

    def executemany(
        self,
        sql: str,
        seq_of_params: Iterable[Sequence[Any] | Mapping[str, Any]],
    ) -> "_LegacyCursor":
        self._result = self._connection.executemany(sql, seq_of_params)
        return self

    def fetchall(self) -> list[tuple]:
        if self._result is None:
            return []
        return [tuple(row) for row in self._result.fetchall()]

    def fetchone(self) -> tuple | None:
        if self._result is None:
            return None
        row = self._result.fetchone()
        return tuple(row) if row is not None else None

    def fetchmany(self, size: int = 1000) -> list[tuple]:
        if self._result is None:
            return []
        return [tuple(row) for row in self._result.fetchmany(size)]

    @property
    def description(self):
        if self._result is None:
            return None
        # DBAPI description: 7-tuple (name, type_code, ...)
        keys = list(self._result.keys())
        return [(k, None, None, None, None, None, None) for k in keys]

    @property
    def rowcount(self) -> int:
        if self._result is None:
            return -1
        try:
            return self._result.rowcount
        except Exception:
            return -1

    def close(self) -> None:
        self._closed = True
        if self._result is not None:
            try:
                self._result.close()
            except Exception:
                pass

    def __iter__(self):
        if self._result is None:
            return iter([])
        return (tuple(row) for row in self._result)
