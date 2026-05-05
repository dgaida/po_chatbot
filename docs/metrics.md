# Metriken

Diese Seite visualisiert die Qualität und den Zustand der Dokumentation sowie des Codes.

## Qualitätsübersicht

| Metrik | Wert | Status |
|---|---|---|
| **API-Dokumentations-Abdeckung** | ![Interrogate](assets/interrogate.svg) | Automatisch |
| **Gültigkeit der Links** | <span id="link-check">-</span> | In Prüfung |
| **Markdown Linting** | <span id="md-lint">-</span> | In Prüfung |

## Detail-Statistiken

<div id="metrics-dashboard">
    <p>Lade Metriken aus der letzten CI-Ausführung...</p>
</div>

<script>
fetch('../assets/metrics.json')
  .then(response => response.json())
  .then(data => {
    const dashboard = document.getElementById('metrics-dashboard');
    dashboard.innerHTML = `
      <ul>
        <li>Letztes Update: ${data.timestamp}</li>
        <li>Build-Status: ${data.build_status}</li>
        <li>Anzahl der Warnungen: ${data.warnings_count}</li>
        <li>Changelog-Status: ${data.changelog_freshness}</li>
      </ul>
    `;
    document.getElementById('link-check').innerText = data.links_status;
    document.getElementById('md-lint').innerText = data.lint_status;
  })
  .catch(err => {
    document.getElementById('metrics-dashboard').innerHTML = '<p>Metriken momentan nicht verfügbar.</p>';
  });
</script>
