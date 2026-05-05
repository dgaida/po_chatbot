# Versioning

This project uses `mike` to manage multiple versions of the documentation and `dgaida/auto-version-action` for automated code versioning.

## Documentation Versioning with `mike`

We provide documentation for different releases.

- **`latest`**: Always points to the documentation of the `main` branch.  
- **`vX.Y.Z`**: Specific versions that correspond to Git tags.  

### Publishing a New Version

To manually create a new version of the documentation:
```bash
mike deploy --push --update-aliases 1.1.0 latest
mike set-default --push latest
```

## Code Versioning

We follow the Semantic Versioning (SemVer) scheme.
Versions are automatically bumped through Conventional Commits and GitHub Actions.
