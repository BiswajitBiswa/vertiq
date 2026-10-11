"""vertiq stays independent of any host: no web framework, no ORM, no HTTP client at import time, and
nothing from the product that embeds it. A host passes data and callables in; vertiq never reaches out."""
import ast
import pathlib

PKG = pathlib.Path(__file__).resolve().parents[1] / "vertiq"
FORBIDDEN = ("sqlalchemy", "fastapi", "httpx", "requests", "flask", "django", "aiohttp", "urllib3")


def _imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module.split(".")[0])
    return out


def test_no_host_or_framework_imports():
    bad = {p.name: sorted(_imports(p) & set(FORBIDDEN)) for p in PKG.glob("*.py")}
    assert {k: v for k, v in bad.items() if v} == {}
