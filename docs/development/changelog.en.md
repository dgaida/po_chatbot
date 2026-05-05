# Changelog Workflow

We use automated changelogs based on **Conventional Commits**.

## Commit Conventions

Each commit subject should conform to one of the following types:

- `feat:` New functionality for the user.  
- `fix:` Bug fix for the user.  
- `docs:` Documentation changes.  
- `style:` Changes that do not affect the meaning of the code (white-space, formatting).  
- `refactor:` Code change that neither fixes a bug nor adds a feature.  
- `perf:` Code change that improves performance.  
- `test:` Adding missing tests or correcting existing tests.  
- `chore:` Changes to the build process or auxiliary tools.  

## Automation

With every push of a new tag (e.g., `v0.1.2`):  
1. `git-cliff` is run to update `CHANGELOG.md`.  
2. A new GitHub Release is created with the generated changelog.  
