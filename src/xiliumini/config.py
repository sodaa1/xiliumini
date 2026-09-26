from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from xiliumini.errors import ConfigError


class Settings(BaseSettings):
    """Validated runtime configuration loaded from ``XILIUMINI_*`` values."""

    model_config = SettingsConfigDict(
        env_prefix="XILIUMINI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr = Field(min_length=1)
    model: str = Field(min_length=1)
    models: str | None = None
    base_url: str | None = None
    temperature: float = 0
    timeout_seconds: float = Field(default=60, gt=0)
    analysis_timeout_seconds: float = Field(default=30, gt=0)
    analysis_max_chars: int = Field(default=8000, ge=1)
    data_dir: Path = Path(".xiliumini")

    @field_validator("api_key", "model", mode="before")
    @classmethod
    def required_text_must_not_be_blank(cls, value: Any) -> str:
        raw = value.get_secret_value() if isinstance(value, SecretStr) else value
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("must not be blank")
        return raw.strip()

    @property
    def model_names(self) -> tuple[str, ...]:
        names = [self.model]
        if self.models:
            configured = [part.strip() for part in self.models.split(",") if part.strip()]
            names = configured if self.model in configured else [self.model, *configured]
        return tuple(dict.fromkeys(names))

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        try:
            dynamic_settings_type = cast(Any, cls)
            if env is None:
                return dynamic_settings_type()
            prefix = "XILIUMINI_"
            values: dict[str, Any] = {
                key.removeprefix(prefix).lower(): value
                for key, value in env.items()
                if key.startswith(prefix)
            }
            return dynamic_settings_type(_env_file=None, **values)
        except ValidationError as exc:
            fields = []
            for error in exc.errors():
                field = str(error["loc"][0]).upper()
                fields.append(f"XILIUMINI_{field}")
            names = ", ".join(dict.fromkeys(fields))
            raise ConfigError(f"Invalid or missing configuration: {names}") from None


def load_settings() -> Settings:
    """Load settings from process environment and the local ``.env`` file."""

    return Settings.from_env()
