from __future__ import annotations

import xiliumini.runtime as runtime_module
from xiliumini.execution.gateway import active_run
from xiliumini.runtime import Runtime


def test_runtime_sets_and_resets_background_context(monkeypatch, tmp_path):
    observed = []

    def capture(*_args, **_kwargs):
        context = active_run.get()
        assert context is not None
        observed.append((context.actor_type, context.policy_profile))
        return iter(())

    monkeypatch.setattr(runtime_module, "stream_agent", capture)
    runtime = Runtime(object(), data_dir=tmp_path, actor_type="automation", policy_profile="background")
    assert list(runtime.stream("work", "11111111-1111-4111-8111-111111111111")) == []
    assert observed == [("automation", "background")]
    assert active_run.get() is None
