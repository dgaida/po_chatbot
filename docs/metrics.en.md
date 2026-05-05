# Metrics

This page visualizes the quality and state of the documentation and code.

## Quality Overview

| Metric | Value | Status |
|---|---|---|
| **API Documentation Coverage** | ![Interrogate](assets/interrogate.svg) | Automatic |
| **Link Validity** | <span id="link-check">-</span> | Under Review |
| **Markdown Linting** | <span id="md-lint">-</span> | Under Review |

## Detailed Statistics

<div id="metrics-dashboard">
    <p>Loading metrics from the last CI run...</p>
</div>

<script>
fetch('../assets/metrics.json')
  .then(response => response.json())
  .then(data => {
    const dashboard = document.getElementById('metrics-dashboard');
    dashboard.innerHTML = `
      <ul>
        <li>Last Update: ${data.timestamp}</li>
        <li>Build Status: ${data.build_status}</li>
        <li>Warning Count: ${data.warnings_count}</li>
        <li>Changelog Freshness: ${data.changelog_freshness}</li>
      </ul>
    `;
    document.getElementById('link-check').innerText = data.links_status;
    document.getElementById('md-lint').innerText = data.lint_status;
  })
  .catch(err => {
    document.getElementById('metrics-dashboard').innerHTML = '<p>Metrics currently unavailable.</p>';
  });
</script>
