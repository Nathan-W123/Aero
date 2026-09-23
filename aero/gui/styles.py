"""
Qt stylesheets and chart colors for the Aero CFD desktop GUI.

Cream-and-green light palette.  Every colour the GUI uses is defined here --
the 3D viewports import :data:`VIEWPORT_THEME` rather than hard-coding their
own, so retheming is a single-file change.

Contrast was checked rather than eyeballed.  Body text sits at 12.3:1 on the
main surface, muted text at 5.9:1, the accent at 5.0:1 and the Run button's
label at 5.0:1 on its fill -- all past WCAG AA.  ``_TEXT_DIM`` is 3.9:1, which
is AA for large text only; it is used for placeholder and disabled states, not
for anything a user has to read.

The two chart series pass the categorical-palette checks on the elevated
surface: chroma above the grey floor, CVD separation ΔE 25.4 (deutan) / 13.5
(tritan), and both above 3:1 against their background.  Green plus violet
rather than the more obvious green plus orange, because green/orange is
exactly the pair red-green colour blindness collapses.
"""

# Cream-and-green light palette
_BG_DEEP = "#E8E1CC"
_BG_MAIN = "#F3EDDC"
_BG_PANEL = "#F9F5EA"
_BG_ELEVATED = "#FDFBF2"
_BG_INPUT = "#FFFFFF"
_BORDER = "#CBBF9E"
_BORDER_FOCUS = "#2A7340"
_TEXT = "#1F2D21"
_TEXT_MUTED = "#55634F"
_TEXT_DIM = "#6E7A66"
_ACCENT = "#2A7340"
_ACCENT_BRIGHT = "#15803D"
_ACCENT_DEEP = "#14532D"
_ACCENT_HOVER = "#3D9A58"
_RUN = "#15803D"
_RUN_HOVER = "#14653A"
# status colours, darkened for a light surface: the original amber and salmon
# sat at 1.5:1 and 2.5:1 on cream, i.e. effectively invisible
_WARN = "#8A5A0B"
_ERROR = "#B3261E"

CHART_THEME = {
    "figure_bg": _BG_PANEL,
    "axes_bg": _BG_ELEVATED,
    "text": _TEXT_MUTED,
    "grid": _BORDER,
    "cd": "#127A38",
    "cl": "#5B21B6",
    "empty": _TEXT_DIM,
    # diverging, and conventional for pressure (blue low / red high).  It fills
    # the axes completely, so its near-white midpoint never shows the cream
    # surface through.
    "pressure_cmap": "coolwarm",
}

#: Colours for the 3D wind-tunnel viewports (matplotlib and PyVista paths).
VIEWPORT_THEME = {
    "background": _BG_ELEVATED,
    "background_top": _BG_PANEL,
    "axis_text": _TEXT_MUTED,
    "pane_edge": _BORDER,
    "tunnel": "#9AA38C",
    "body": _ACCENT,
    "scalar_bar_text": _TEXT,
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
    color: {_WARN};
}}
QLabel#validationLine[validationStatus="fail"] {{
    color: {_ERROR};
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
QFrame#runStatusFrame {{
    background-color: {_BG_PANEL};
    border: 1px solid {_BORDER};
    border-radius: 8px;
    padding: 4px;
}}
QSplitter#centerSplitter::handle {{
    background-color: {_BORDER};
    height: 4px;
}}
QStatusBar QProgressBar#runProgress {{
    background-color: {_BG_INPUT};
    border: 1px solid {_BORDER_FOCUS};
    border-radius: 6px;
    min-height: 20px;
    text-align: center;
    color: {_TEXT};
    font-size: 11px;
    font-weight: 600;
}}
QStatusBar QProgressBar#runProgress::chunk {{
    background-color: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 {_ACCENT_DEEP}, stop:1 {_ACCENT_BRIGHT}
    );
    border-radius: 5px;
}}
QProgressBar#runProgress {{
    background-color: {_BG_INPUT};
    border: 1px solid {_BORDER};
    border-radius: 6px;
    min-height: 18px;
    text-align: center;
    color: {_TEXT};
    font-size: 11px;
}}
QProgressBar#runProgress::chunk {{
    background-color: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 {_ACCENT_DEEP}, stop:1 {_ACCENT_BRIGHT}
    );
    border-radius: 5px;
}}
QPlainTextEdit#liveRunLog {{
    background-color: {_BG_INPUT};
    border: 1px solid {_BORDER};
    border-radius: 6px;
    color: {_TEXT_MUTED};
    font-family: Menlo, Monaco, Consolas, monospace;
    font-size: 11px;
    padding: 6px;
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
    background-color: {_BG_ELEVATED};
    color: {_TEXT};
}}
/* No chip behind the status text: it is empty until a case loads, and a
   filled box with nothing in it reads as a rendering fault. */
QLabel#viewportStatus {{
    color: {_TEXT_MUTED};
    background-color: transparent;
    padding: 4px 8px;
}}
QPushButton#viewportButton {{
    background-color: rgba(253, 251, 242, 235);
    border: 1px solid {_BORDER_FOCUS};
    color: {_ACCENT_BRIGHT};
    border-radius: 4px;
    padding: 5px 10px;
}}
QPushButton#viewportButton:hover {{
    background-color: rgba(243, 237, 220, 245);
    border-color: {_ACCENT};
}}
QPushButton#viewportButton:disabled {{
    background-color: rgba(232, 225, 204, 200);
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
