"""Own resources independently of any provider.

The package holds the resource requests, the leases and the contract that every
provider implements.
"""

from loops_cookbook.resources.api import (
    CapacityProvider,
    CreationDisposition,
    ProviderDeadlines,
    ProviderResource,
    ProvisioningFailure,
    ReconcileResult,
    ReconcileStatus,
    ResourceAdapter,
    ResourceHandle,
    ResourceIntent,
    ResourceLease,
    ResourceRequest,
    SandboxCapacity,
    SandboxGone,
    SandboxKind,
    SandboxProvider,
    SandboxResources,
    SandboxSession,
    SandboxSpec,
)

__all__ = [
    "CapacityProvider",
    "CreationDisposition",
    "ProviderDeadlines",
    "ProviderResource",
    "ProvisioningFailure",
    "ReconcileResult",
    "ReconcileStatus",
    "ResourceAdapter",
    "ResourceHandle",
    "ResourceIntent",
    "ResourceLease",
    "ResourceRequest",
    "SandboxCapacity",
    "SandboxGone",
    "SandboxKind",
    "SandboxProvider",
    "SandboxResources",
    "SandboxSession",
    "SandboxSpec",
]
