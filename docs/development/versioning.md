# Versionierung

Dieses Projekt nutzt `mike` zur Verwaltung mehrerer Versionen der Dokumentation und `dgaida/auto-version-action` für automatisierte Versionierung des Codes.

## Dokumentations-Versionierung mit `mike`

Wir halten Dokumentation für verschiedene Releases bereit.

- **`latest`**: Zeigt immer auf die Dokumentation des `main` Branches.  
- **`vX.Y.Z`**: Spezifische Versionen, die mit Git-Tags korrespondieren.  

### Neue Version veröffentlichen

Um eine neue Version der Dokumentation manuell zu erstellen:
```bash
mike deploy --push --update-aliases 1.1.0 latest
mike set-default --push latest
```

## Code-Versionierung

Wir folgen dem Semantic Versioning (SemVer) Schema.
Versionen werden automatisch durch Conventional Commits und GitHub Actions hochgestuft.
