"""Settings for the property suite.

Every example builds a fixture and runs a real parser, so the suite trades
example count for wall-clock. ``derandomize`` is deliberately off: a
generator that replays the same corpus every run hides exactly the
intermittent failures this suite exists to find.
"""

from hypothesis import HealthCheck, Verbosity, settings

settings.register_profile(
    "docweave",
    max_examples=60,
    deadline=None,
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
        HealthCheck.filter_too_much,
    ],
    verbosity=Verbosity.normal,
)
settings.load_profile("docweave")
