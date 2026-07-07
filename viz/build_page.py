# Assemble the self-contained static page: template.html + app.js + demo data
# -> viz/index.html (full document, open locally) and viz/artifact.html
# (same content without the doctype, for Artifact publishing).
# Run: uv run python viz/build_page.py
import json
from pathlib import Path

VIZ = Path(__file__).resolve().parent
DATA = VIZ / "data"


def main() -> None:
    template = (VIZ / "template.html").read_text(encoding="utf-8")
    app_js = (VIZ / "app.js").read_text(encoding="utf-8")

    index = json.loads((DATA / "index.json").read_text(encoding="utf-8"))
    demos = [
        json.loads((DATA / f"{e['slug']}.json").read_text(encoding="utf-8"))
        for e in index
    ]
    boot = {"mode": "embed", "demos": demos}
    # "</" would close the <script> tag if a vocab string contains it.
    boot_json = json.dumps(boot, ensure_ascii=False, separators=(",", ":"))
    boot_json = boot_json.replace("</", "<\\/")

    page = template.replace("__BOOT__", boot_json).replace("__APP_JS__", app_js)
    (VIZ / "artifact.html").write_text(page, encoding="utf-8")
    (VIZ / "index.html").write_text(
        '<!doctype html>\n<html lang="zh-CN">\n' + page + "\n</html>",
        encoding="utf-8",
    )
    print(f"index.html    {(VIZ / 'index.html').stat().st_size / 1e6:.2f} MB")
    print(f"artifact.html {(VIZ / 'artifact.html').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
