# Security policy

## Supported versions

Security fixes go into the latest release of the Loops Cookbook. Pre-releases
and older versions can help to reproduce a report, but they get no fixes.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability.

Send the report to [security@baseten.co](mailto:security@baseten.co). Include:

- the affected version and operating system;
- the expected and observed behavior;
- the security impact;
- the smallest safe reproduction;
- any suggested mitigation.

Do not include live API keys, access tokens, training data, prompts, customer
data, or other secrets. Replace them with synthetic values.

The Baseten security team coordinates validation, remediation, disclosure,
and credit with you in private. Allow time for a fix before you publish details
of the report.

## Security boundary

The cookbook handles Baseten, sandbox provider, and Hugging Face credentials.
It runs model-written code in sandboxes and serves an OpenAI-compatible
gateway. These reports are in scope: credential exposure, unintended network
destinations, sandbox escape through the cookbook's own code, prompt or
completion disclosure, unsafe local file changes, and gateway access control.

Report bugs in Baseten Loops, Harbor, Blaxel, or Daytona to those projects.
General support questions, feature requests, and non-security defects belong
in the public issue tracker.
