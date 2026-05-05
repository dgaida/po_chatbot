# Changelog Workflow

Wir verwenden automatisierte Changelogs basierend auf **Conventional Commits**.

## Commit-Konventionen

Jeder Commit-Betreff sollte einem der folgenden Typen entsprechen:

- `feat:` Neue Funktionalität für den Benutzer.
- `fix:` Fehlerbehebung für den Benutzer.
- `docs:` Änderungen an der Dokumentation.
- `style:` Änderungen, die die Bedeutung des Codes nicht beeinflussen (White-space, Formatierung).
- `refactor:` Codeänderung, die weder einen Fehler behebt noch eine Funktion hinzufugt.
- `perf:` Codeänderung, die die Performance verbessert.
- `test:` Hinzufügen fehlender Tests oder Korrigieren bestehender Tests.
- `chore:` Änderungen am Build-Prozess oder an Hilfswerkzeugen.

## Automatisierung

Bei jedem Push eines neuen Tags (z.B. `v0.1.2`) wird:
1. `git-cliff` ausgeführt, um die `CHANGELOG.md` zu aktualisieren.
2. Ein neuer GitHub Release mit dem generierten Changelog erstellt.
