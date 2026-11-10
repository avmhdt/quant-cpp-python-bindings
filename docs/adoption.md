# Adoption

Dated snapshots of public adoption metrics live in `results/adoption/` (stars, outside issues
and pull requests, known dependents, downloads), written by
`python -m tools.qkdev adoption snapshot`. Downloads are reported but not counted as
adoption, because mirrors and CI inflate them.

The first snapshot is taken after the first PyPI release.
