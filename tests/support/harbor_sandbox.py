"""A local Harbor environment that a separate process can reach, as a sandbox is.

``usage_environment()`` names its root the way the cookbook's Blaxel environment
names its sandbox, so an agent program can run its tools there without Blaxel.
Needs ``tests/fixtures/offline`` on ``sys.path``.
"""

from __future__ import annotations

from baseline_environment import LocalEnvironment


class SandboxLocalEnvironment(LocalEnvironment):
    def usage_environment(self) -> dict[str, str]:
        return {"SANDBOX_ROOT": str(self._root())}


class ProvisionedLocalEnvironment(LocalEnvironment):
    """A local environment whose sandbox the attempt's resources own, as Blaxel's is.

    Like the cookbook's Blaxel environment it looks its attempt up by
    ``loops_attempt_id``, provisions in ``start()`` keyed by Harbor's session, and
    releases on ``stop(delete=True)``; its commands still run locally.
    """

    def __init__(self, *args, loops_attempt_id: str, **kwargs):
        from loops_cookbook.harbor.attempts import attempt_for

        self._attempt = attempt_for(loops_attempt_id)
        self._handle = None
        super().__init__(*args, **kwargs)

    async def start(self, force_build: bool) -> None:
        from loops_cookbook.resources import ResourceRequest

        self._handle = await self._attempt.resources.provision(
            ResourceRequest(provider="fake", kind="box", logical_key=self.session_id)
        )
        await super().start(force_build)

    async def stop(self, delete: bool) -> None:
        await super().stop(delete)
        if delete and self._handle is not None:
            handle, self._handle = self._handle, None
            await self._attempt.resources.release(handle.intent.resource_id)
