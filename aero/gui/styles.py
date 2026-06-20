"""Qt stylesheets and chart colors for the Aero CFD desktop GUI."""

# Teal-blue dark palette
_BG_DEEP = "#0a0e14"
_BG_MAIN = "#0f1419"
_BG_PANEL = "#151c24"
_BG_ELEVATED = "#1a2332"
_BG_INPUT = "#1e2a3a"
_BORDER = "#2d3a4f"
_BORDER_FOCUS = "#3d7ea6"
_TEXT = "#e2e8f0"
_TEXT_MUTED = "#94a3b8"
_TEXT_DIM = "#64748b"
_ACCENT = "#2dd4bf"
_ACCENT_BRIGHT = "#4de8ff"
_ACCENT_DEEP = "#14b8a6"
_ACCENT_HOVER = "#5eead4"
_RUN = "#14b8a6"
_RUN_HOVER = "#0d9488"

CHART_THEME = {
    "figure_bg": _BG_PANEL,
    "axes_bg": _BG_ELEVATED,
    "text": _TEXT_MUTED,
    "grid": "#2d3a4f",
    "cd": _ACCENT_BRIGHT,
    "cl": "#f472b6",
    "empty": _TEXT_DIM,
    "pressure_cmap": "coolwarm",
}

DARK_STYLESHEET = f"""
QMainWindow {{
    background-color: {_BG_DEEP};
    color: {_TEXT};
}}
QWidget#centralRoot, QSplitter, QScrollArea, QFrame#panelFrame {{
    background-color: {_BG_MAIN};
    color: {_TEXT};
    font-size: 13px;
}}
QStatusBar {{
    background-color: {_BG_PANEL};
    color: {_TEXT_MUTED};
    border-top: 1px solid {_BORDER};
}}
QMenuBar {{
    background-color: {_BG_PANEL};
    color: {_TEXT};
    border-bottom: 1px solid {_BORDER};
}}
QMenuBar::item:selected {{
    background-color: {_BG_ELEVATED};
    color: {_ACCENT_BRIGHT};
}}
QMenu {{
    background-color: {_BG_ELEVATED};
    color: {_TEXT};
    border: 1px solid {_BORDER};
}}
QMenu::item:selected {{
    background-color: {_BG_INPUT};
    color: {_ACCENT_BRIGHT};
}}
QFrame#ribbonBar {{
    background-color: {_BG_PANEL};
    border-bottom: 1px solid {_BORDER};
}}
QFrame#ribbonGroup {{
    background-color: transparent;
    border-right: 1px solid {_BORDER};
}}
QLabel#ribbonGroupTitle {{
    color: {_TEXT_DIM};
    font-size: 11px;
    font-weight: 600;
}}
QLabel#panelTitle {{
    color: {_ACCENT_BRIGHT};
    font-size: 14px;
    font-weight: 700;
    padding: 4px 0;
}}
QLabel#panelSubtitle, QLabel#statusLabel {{
    color: {_TEXT_MUTED};
}}
QFrame#resultsSummary {{
    background: {_BG_ELEVATED};
    border: 1px solid {_BORDER};
    border-radius: 8px;
    padding: 12px;
}}
QLabel#resultsHeadline {{
    color: {_TEXT};
    font-size: 15px;
    font-weight: 700;
    padding-bottom: 4px;
}}
QLabel#resultsMetricName {{
    color: {_TEXT_MUTED};
    font-size: 12px;
}}
QLabel#resultsMetricValue {{
    color: {_ACCENT_BRIGHT};
    font-size: 16px;
    font-weight: 700;
    font-family: "Menlo", "Consolas", monospace;
}}
QGroupBox {{
    border: 1px solid {_BORDER};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 14px;
    background-color: {_BG_ELEVATED};
    font-weight: 600;
    color: {_TEXT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {_ACCENT};
}}
QPushButton {{
    background-color: {_BG_INPUT};
    border: 1px solid {_BORDER};
    border-radius: 4px;
    padding: 6px 12px;
    color: {_TEXT};
}}
QPushButton:hover {{
    background-color: {_BG_ELEVATED};
    border-color: {_BORDER_FOCUS};
    color: {_ACCENT_BRIGHT};
}}
QPushButton:disabled {{
    background-color: {_BG_PANEL};
    color: {_TEXT_DIM};
    border-color: {_BORDER};
}}
QPushButton#primaryButton {{
    background-color: {_ACCENT_DEEP};
    border-color: {_ACCENT};
    color: {_BG_DEEP};
    font-weight: 600;
}}
QPushButton#primaryButton:hover {{
    background-color: {_ACCENT};
}}
QPushButton#runButton {{
    background-color: {_RUN};
    border-color: {_ACCENT};
    color: {_BG_DEEP};
    font-weight: 700;
    min-width: 72px;
}}
QPushButton#runButton:hover {{
    background-color: {_RUN_HOVER};
    color: {_TEXT};
}}
QComboBox, QLineEdit, QListWidget, QPlainTextEdit, QSpinBox {{
    background-color: {_BG_INPUT};
    border: 1px solid {_BORDER};
    border-radius: 4px;
    padding: 5px 8px;
    color: {_TEXT};
    selection-background-color: {_ACCENT_DEEP};
    selection-color: {_BG_DEEP};
}}
QComboBox:focus, QLineEdit:focus, QPlainTextEdit:focus {{
    border-color: {_BORDER_FOCUS};
}}
QComboBox::drop-down {{
    border: none;
    background: {_BG_ELEVATED};
}}
QComboBox QAbstractItemView {{
    background-color: {_BG_ELEVATED};
    color: {_TEXT};
    border: 1px solid {_BORDER};
    selection-background-color: {_ACCENT_DEEP};
    selection-color: {_BG_DEEP};
}}
QPlainTextEdit {{
    font-family: "Menlo", "Consolas", monospace;
    font-size: 12px;
}}
QTabWidget::pane {{
    border: 1px solid {_BORDER};
    background-color: {_BG_ELEVATED};
    border-radius: 4px;
}}
QTabBar::tab {{
    background-color: {_BG_PANEL};
    border: 1px solid {_BORDER};
    padding: 7px 12px;
    margin-right: 2px;
    color: {_TEXT_MUTED};
}}
QTabBar::tab:selected {{
    background-color: {_BG_ELEVATED};
    border-bottom-color: {_BG_ELEVATED};
    color: {_ACCENT_BRIGHT};
}}
QTabBar::tab:hover {{
    color: {_ACCENT};
}}
QListWidget::item:selected {{
    background-color: {_ACCENT_DEEP};
    color: {_BG_DEEP};
}}
QSlider::groove:horizontal {{
    height: 6px;
    background: {_BG_INPUT};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: {_ACCENT_BRIGHT};
}}
QScrollBar:vertical {{
    background: {_BG_PANEL};
    width: 12px;
}}
QScrollBar::handle:vertical {{
    background: {_BORDER};
    border-radius: 6px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {_BORDER_FOCUS};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QSplitter::handle {{
    background-color: {_BORDER};
}}
QFrame#viewportFrame {{
    background-color: {_BG_DEEP};
    border: 1px solid {_BORDER_FOCUS};
    border-radius: 4px;
}}
QFrame#chartFrame {{
    background-color: {_BG_ELEVATED};
    border: 1px solid {_BORDER};
    border-radius: 4px;
}}
QWidget#paramTab, QWidget#paramFormHost {{
    background-color: {_BG_ELEVATED};
    color: {_TEXT};
}}
QScrollArea#paramScroll {{
    background-color: {_BG_ELEVATED};
    border: 1px solid {_BORDER};
    border-radius: 6px;
}}
QScrollArea#paramScroll > QWidget > QWidget {{
    background-color: {_BG_ELEVATED};
}}
QLabel#paramLabel {{
    color: {_TEXT_MUTED};
    font-size: 12px;
    font-weight: 600;
    padding-right: 8px;
}}
QLineEdit#paramField {{
    background-color: {_BG_INPUT};
    border: 1px solid {_BORDER};
    border-radius: 4px;
    padding: 6px 10px;
    color: {_ACCENT_BRIGHT};
    font-family: "Menlo", "Consolas", monospace;
    font-size: 12px;
}}
QLineEdit#paramField:focus {{
    border-color: {_ACCENT};
    background-color: {_BG_PANEL};
}}
QComboBox#paramCombo {{
    background-color: {_BG_INPUT};
    border: 1px solid {_BORDER};
    border-radius: 4px;
    padding: 6px 10px;
    color: {_ACCENT_BRIGHT};
    min-width: 120px;
}}
QComboBox#paramCombo:focus {{
    border-color: {_ACCENT};
}}
QComboBox#paramCombo::drop-down {{
    border: none;
    background: {_BG_ELEVATED};
}}
QComboBox#paramCombo QAbstractItemView {{
    background-color: {_BG_ELEVATED};
    color: {_TEXT};
    selection-background-color: {_ACCENT_DEEP};
}}
QLabel#validationLine {{
    font-size: 12px;
    padding: 2px 0;
}}
QLabel#validationLine[validationStatus="pass"] {{
    color: {_ACCENT};
}}
QLabel#validationLine[validationStatus="warn"] {{
    color: #fbbf24;
}}
QLabel#validationLine[validationStatus="fail"] {{
    color: #f87171;
}}
QLabel#validationLine[validationStatus="n/a"] {{
    color: {_TEXT_MUTED};
}}
QWidget#casesTab {{
    background-color: {_BG_ELEVATED};
}}
QListWidget#caseList {{
    background-color: {_BG_INPUT};
    border: 1px solid {_BORDER};
    border-radius: 6px;
    color: {_TEXT};
    padding: 4px;
}}
QListWidget#caseList::item {{
    padding: 8px 10px;
    border-radius: 4px;
}}
QListWidget#caseList::item:selected {{
    background-color: {_BG_PANEL};
    color: {_ACCENT_BRIGHT};
    border: 1px solid {_BORDER_FOCUS};
}}
QListWidget#caseList::item:hover {{
    background-color: {_BG_ELEVATED};
    color: {_ACCENT};
}}
QLabel {{
    color: {_TEXT};
}}
QFormLayout QLabel {{
    color: {_TEXT_MUTED};
}}
"""

# Default application theme
APP_STYLESHEET = DARK_STYLESHEET

# Backward-compatible alias
LIGHT_STYLESHEET = APP_STYLESHEET

VIEWPORT_STYLESHEET = f"""
QWidget#FlowViewportRoot {{
    background-color: {_BG_DEEP};
    color: {_ACCENT_BRIGHT};
}}
QLabel#viewportStatus {{
    color: {_ACCENT_BRIGHT};
    background-color: rgba(10, 14, 20, 200);
    padding: 4px 8px;
    border-radius: 4px;
}}
QPushButton#viewportButton {{
    background-color: rgba(21, 28, 36, 230);
    border: 1px solid {_BORDER_FOCUS};
    color: {_ACCENT_BRIGHT};
    border-radius: 4px;
    padding: 5px 10px;
}}
QPushButton#viewportButton:hover {{
    background-color: rgba(30, 42, 58, 240);
    border-color: {_ACCENT};
}}
QPushButton#viewportButton:disabled {{
    background-color: rgba(15, 20, 25, 200);
    color: {_TEXT_DIM};
    border-color: {_BORDER};
}}
QSlider::groove:horizontal {{
    background: {_BG_INPUT};
}}
QSlider::handle:horizontal {{
    background: {_ACCENT_BRIGHT};
}}
"""
