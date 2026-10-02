"""Export a notebook to PDF via HTML + headless browser printing.

    .venv\Scripts\python.exe export_pdf.py lab01_ols/A1_OLS_Regression.ipynb

Why not "Export to PDF" in VS Code: that path goes through LaTeX, which needs
extra setup for Thai text. Printing the HTML from a browser keeps every glyph
and renders the MathJax equations as they appear in the notebook.
"""

import subprocess
import sys
from pathlib import Path

# Long code lines do not wrap in the default nbconvert stylesheet and get cut
# off at the page margin when printed.
EXTRA_CSS = """
<style>
  div.jp-OutputArea-output pre,
  div.highlight pre,
  pre, code {
    white-space: pre-wrap !important;
    word-break: break-word !important;
  }
  /* A wide DataFrame (11 columns) is scrolled sideways on screen but cut off when printed. */
  table.dataframe { font-size: 8px !important; }
  table.dataframe th, table.dataframe td { padding: 1px 4px !important; }
  div.jp-RenderedHTMLCommon, div.jp-OutputArea-output { overflow-x: visible !important; }
  @page { margin: 15mm; }
</style>
</head>"""

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


def find_browser():
    for path in BROWSERS:
        if Path(path).exists():
            return path
    sys.exit("No Chrome or Edge found - install one, or print the .html by hand.")


def main(notebook):
    nb = Path(notebook).resolve()
    html, pdf = nb.with_suffix(".html"), nb.with_suffix(".pdf")

    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html",
                    "--embed-images", str(nb)], check=True)

    html.write_text(html.read_text(encoding="utf-8").replace("</head>", EXTRA_CSS, 1),
                    encoding="utf-8")

    # virtual-time-budget gives MathJax time to typeset before the page prints.
    subprocess.run([find_browser(), "--headless=new", "--disable-gpu",
                    "--no-pdf-header-footer", "--virtual-time-budget=40000",
                    "--run-all-compositor-stages-before-draw",
                    f"--print-to-pdf={pdf}", html.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"\nwrote {pdf} ({pdf.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lab01_ols/A1_OLS_Regression.ipynb")
