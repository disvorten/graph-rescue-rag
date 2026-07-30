# Contributing

Contributions that improve reproducibility, add a properly licensed baseline,
or identify a protocol mismatch are welcome.

Before opening a pull request:

1. run `python -m unittest discover -s tests -v`;
2. run `python scripts/release_audit.py`;
3. document any new dependency and deterministic seed;
4. avoid committing benchmark records, derived passage text, model caches,
   credentials, absolute local paths, or per-query reader generations;
5. state whether the change affects a frozen article result.

Use the reproduction-report issue form for independent runs and the baseline
request form for proposed comparisons. Aggregate metrics are preferred over
raw examples that may reproduce benchmark text.

By contributing, you agree that your contribution is licensed under the
repository's Apache-2.0 license.
