"""Helpers the resource registry test modules share.

Every registry test drives the real registry and journal through ``FakeProvider``, a
second provider that shares nothing with Blaxel, so these are the resource contracts
any provider is held to.
"""

import dataclasses

from support.resources import FAST, request

from loops_cookbook.resources.api import ProviderDeadlines, ResourceScope


async def provision(registry, key="a", **fields):
    return await registry.provision(
        request(key, **fields), scope=ResourceScope.EPISODE, owner_id="attempt-1"
    )


# A create and lookups that give up fast: a cancelled create, and a lookup that
# proves nothing, each wait out their deadline.
QUICK_LOOKUPS = dataclasses.replace(
    FAST, create_seconds=0.2, reconcile_seconds=0.2, confirm_absent_seconds=0.2
)


TIGHT = ProviderDeadlines(0.05, 0.05, 0.05, 0.05, 0.05, 0.05)


def events(registry, resource_id):
    return [r.event for r in registry.journal.records if r.resource_id == resource_id]
