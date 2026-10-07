from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, model_validator


class AutomationTask(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")
    name: str = Field(min_length=1, max_length=120)
    prompt: str = Field(min_length=1, max_length=20_000)
    trigger_type: Literal["date", "cron", "interval"]
    schedule: dict[str, Any]
    timezone: str = "Asia/Shanghai"
    enabled: bool = True
    policy_profile: Literal["background"] = "background"

    @model_validator(mode="after")
    def validate_schedule(self) -> AutomationTask:
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError:
            raise ValueError("invalid automation timezone") from None
        data = self.schedule
        if self.trigger_type == "cron":
            if (
                set(data) != {"hour", "minute"}
                or any(type(data[key]) is not int for key in ("hour", "minute"))
                or not (0 <= data["hour"] <= 23 and 0 <= data["minute"] <= 59)
            ):
                raise ValueError("cron schedule needs valid hour and minute")
        elif self.trigger_type == "date":
            if set(data) != {"at"} or not isinstance(data["at"], str):
                raise ValueError("date schedule needs an ISO timestamp")
            try:
                at = datetime.fromisoformat(data["at"])
            except ValueError:
                raise ValueError("invalid date timestamp") from None
            if at.tzinfo is None or at.utcoffset() is None:
                raise ValueError("date timestamp needs an explicit timezone")
        elif self.trigger_type == "interval" and (
            set(data) != {"seconds"} or type(data["seconds"]) is not int or data["seconds"] < 1
        ):
            raise ValueError("interval seconds must be positive")
        return self
