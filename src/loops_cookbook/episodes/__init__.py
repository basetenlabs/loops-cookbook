"""Write one ``Episode`` class, and the cookbook runs it as a custom rollout runtime.

Name the class in the run's config like this::


    [rollout_runtime]
    factory = "loops_cookbook.episodes:create_runtime"
    [rollout_runtime.config]
    episode = "my_env:AnswerEpisode"
    agent_timeout_seconds = 600
    [rollout_runtime.files]
    tasks = "tasks.jsonl"
"""

from loops_cookbook.episodes.api import (
    Closing,
    Episode,
    Grading,
    GroupMember,
    Model,
    Processes,
    Sampling,
    Setup,
)
from loops_cookbook.episodes.runtime import EpisodeRuntime, create_runtime
from loops_cookbook.episodes.tasks import Task

__all__ = [
    "Closing",
    "Episode",
    "EpisodeRuntime",
    "Grading",
    "GroupMember",
    "Model",
    "Processes",
    "Sampling",
    "Setup",
    "Task",
    "create_runtime",
]
