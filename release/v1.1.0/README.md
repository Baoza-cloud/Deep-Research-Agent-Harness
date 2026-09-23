# v1.1.0 release manifest

The Git tag `v1.1.0` is the canonical code identity. `SHA256SUMS` pins the benchmark datasets and
selected formal experiment artifacts used by the release documentation.

Verify from the repository root:

```bash
shasum -a 256 -c release/v1.1.0/SHA256SUMS
```

The GitHub Release records the exact tag commit SHA and the SHA256 of the built wheel and source
distribution.
