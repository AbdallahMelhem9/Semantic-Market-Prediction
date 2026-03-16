CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

body {
    background: #0d1b2a !important;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #152238 0%, #0d1b2a 100%);
    padding: 14px 28px;
    border-bottom: 1px solid #1e3448;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.main-header h1 {
    color: #D4A843;
    margin: 0;
    font-size: 1.2rem;
    font-weight: 600;
    letter-spacing: -0.01em;
}

.main-header .subtitle {
    color: #7a8fa3;
    font-size: 0.72rem;
    margin-top: 1px;
}

/* Chart cards — Layer 2 */
.chart-card {
    background: #162638;
    border: 1px solid #1e3448;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.15);
}

.chart-card:hover {
    border-color: #2a4a6a;
    box-shadow: 0 2px 12px rgba(0,0,0,0.25);
}

/* Prediction card */
.prediction-card {
    background: #162638;
    border: 1px solid #1e3448;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    min-height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 1px 8px rgba(0,0,0,0.15);
}

.prediction-card .direction {
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}

.prediction-card .meta {
    color: #7a8fa3;
    font-size: 0.7rem;
    margin-top: 6px;
}

/* Sidebar */
.sidebar {
    background: #11202f !important;
    border-right: 1px solid #1e3448;
    padding: 16px !important;
    min-height: 100vh;
}

.sidebar-section {
    margin-bottom: 16px;
}

.sidebar-section label {
    color: #D4A843;
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
    display: block;
}

/* Region toggle */
.region-toggle {
    text-align: center;
    padding: 10px 0 6px 0;
    background: #0d1b2a;
}

.region-toggle .btn {
    font-size: 0.8rem;
    padding: 5px 20px;
    font-weight: 500;
}

/* Dropdowns — force all text visible on dark background */
.sidebar .dash-dropdown * {
    color: #d4dce8 !important;
}

.sidebar .dash-dropdown .Select-control {
    background: #1e3448 !important;
    border-color: #2a4a6a !important;
}

.sidebar .dash-dropdown .Select-menu-outer,
.sidebar .dash-dropdown .Select-menu-outer * {
    background: #1e3448 !important;
    color: #d4dce8 !important;
}

.sidebar .dash-dropdown .Select-option:hover,
.sidebar .dash-dropdown .VirtualizedSelectFocusedOption {
    background: #2a4a6a !important;
}

.sidebar .dash-dropdown .Select-placeholder {
    color: #7a8fa3 !important;
}

.sidebar .dash-dropdown .Select-clear-zone,
.sidebar .dash-dropdown .Select-arrow-zone {
    color: #7a8fa3 !important;
}

/* Tabs */
.nav-tabs {
    border-bottom: 1px solid #1e3448;
    background: #11202f;
    padding: 0 20px;
}

.nav-tabs .nav-link {
    color: #7a8fa3;
    font-size: 0.8rem;
    font-weight: 500;
    border: none;
    padding: 10px 20px;
}

.nav-tabs .nav-link.active {
    color: #D4A843;
    background: transparent;
    border-bottom: 2px solid #D4A843;
}

.nav-tabs .nav-link:hover {
    color: #c4d0dc;
}

/* Disclaimer */
.disclaimer {
    color: #4a5e72;
    font-size: 0.65rem;
    text-align: center;
    padding: 12px;
    border-top: 1px solid #1e3448;
    margin-top: 16px;
}

/* Chat */
.chat-container {
    background: #162638;
    border: 1px solid #1e3448;
    border-radius: 8px;
    padding: 14px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.15);
}

.chat-messages {
    max-height: 280px;
    overflow-y: auto;
    margin-bottom: 8px;
}

.chat-container h5 {
    font-size: 0.85rem !important;
    font-weight: 600;
    margin-bottom: 10px;
    color: #D4A843 !important;
}

/* Rescore button */
#btn-rescore {
    font-size: 0.72rem;
    padding: 4px 10px;
}

/* Article cards — NEWSPAPER STYLE (cream on dark) */
.article-card {
    background: #FDF6EC !important;
    border: 1px solid #E8DCC8 !important;
    border-left: 3px solid #D4A843 !important;
    transition: all 0.2s ease;
}

.article-card:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
    border-color: #D4A843 !important;
}

.article-card:hover .article-detail {
    max-height: 200px !important;
    margin-top: 8px;
}

/* Plotly chart overrides */
.js-plotly-plot .plotly .modebar {
    right: 8px !important;
    top: 4px !important;
}

.js-plotly-plot .plotly .modebar-btn {
    font-size: 14px !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: #0d1b2a;
}

::-webkit-scrollbar-thumb {
    background: #2a4a6a;
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: #3d6a9a;
}

/* Loading spinner */
.dash-spinner {
    margin-top: 4px;
}
"""
