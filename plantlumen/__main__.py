"""Allow ``python -m plantlumen ...`` in addition to the console script."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
