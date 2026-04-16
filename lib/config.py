"""Project configuration for targeting different Langfuse projects.

A "project" in this repo is a Langfuse project with its own keys. Each
project has its own SQLite cache and results directory so queries never
mix data across projects.

Env var conventions:
    LANGFUSE_PROJECT=labs_agent                  # which project to target (default)
    LANGFUSE_HOST=https://us.cloud.langfuse.com  # shared across projects
    LANGFUSE_{PROJECT}_PUBLIC_KEY=pk-lf-...      # per-project keys
    LANGFUSE_{PROJECT}_SECRET_KEY=sk-lf-...

To add a new project, set the three env vars above and pass
`LANGFUSE_PROJECT=your_name` when running a query. Nothing in the query
code needs to change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_PROJECT = "labs_agent"
DEFAULT_HOST = "https://us.cloud.langfuse.com"


@dataclass(frozen=True)
class ProjectConfig:
    """Configuration for a single Langfuse project."""

    name: str
    public_key: str | None
    secret_key: str | None
    host: str

    @property
    def env_prefix(self) -> str:
        return f"LANGFUSE_{self.name.upper()}"

    @property
    def cache_path(self) -> str:
        """Per-project SQLite cache path."""
        return f"data/traces_{self.name}.db"

    @property
    def public_key_env(self) -> str:
        return f"{self.env_prefix}_PUBLIC_KEY"

    @property
    def secret_key_env(self) -> str:
        return f"{self.env_prefix}_SECRET_KEY"


def get_project_config(project: str | None = None) -> ProjectConfig:
    """Load config for a project (defaults to $LANGFUSE_PROJECT or 'labs_agent').

    Raises ValueError if either key env var is unset.
    """
    name = project or os.getenv("LANGFUSE_PROJECT", DEFAULT_PROJECT)
    prefix = f"LANGFUSE_{name.upper()}"
    pk_env = f"{prefix}_PUBLIC_KEY"
    sk_env = f"{prefix}_SECRET_KEY"
    pk = os.getenv(pk_env)
    sk = os.getenv(sk_env)
    if not pk or not sk:
        raise ValueError(
            f"Missing credentials for project '{name}'. "
            f"Set {pk_env} and {sk_env} in your .env."
        )
    return ProjectConfig(
        name=name,
        public_key=pk,
        secret_key=sk,
        host=os.getenv("LANGFUSE_HOST", DEFAULT_HOST),
    )
