"""One HTML file per model: every episode as a collapsible block, for inspection.

    from multiturn.html_report import render_html
    render_html(MT, f"{DRIVE}/qwen3-1.7b.html", model="qwen3-1.7b")

In Colab, view it inline:

    from IPython.display import HTML
    HTML(open(f"{DRIVE}/qwen3-1.7b.html").read())

Each episode's summary line carries its verdict (red = eroded, green = held, grey = nil), so
the page is scanned closed and only the interesting episodes are expanded. Inside: the metadata
header, then each round; thinking is italic inside its own collapsed toggle, tool calls are
compact monospace lines.
"""
import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review import check, header  # noqa: E402

THINK = re.compile(r"<think>(.*?)</think>", re.S)

CSS = """
body { font-family: system-ui, sans-serif; max-width: 70em; margin: 1em auto; padding: 0 1em;
       line-height: 1.45; }
details.ep { border: 1px solid #ddd; border-left-width: 5px; border-radius: 4px;
             margin: .5em 0; padding: .3em .8em; }
details.ep > summary { cursor: pointer; font-weight: 600; }
details.think { margin: .3em 0 .3em 1em; color: #555; }
details.think > summary { cursor: pointer; font-size: .85em; color: #888; }
pre.meta { background: #f6f6f6; padding: .6em; border-radius: 4px; overflow-x: auto; }
h4 { margin: 1em 0 .2em; border-top: 1px dashed #ccc; padding-top: .6em; }
p.user { background: #eef4fb; padding: .4em .6em; border-radius: 4px; }
code.call { display: block; margin-left: 1em; color: #7a3e9d; }
code.resp { display: block; margin-left: 2.5em; color: #666; }
.controls { margin: .6em 0; }
"""

COLORS = {"yes": "#c0392b", "no": "#27ae60", "nil": "#999"}


def esc(x):
    return html.escape(str(x))


def _tool_result(content):
    """The same shortening show.py does, so results stay one line."""
    try:
        data = json.loads(content)
        inner = data.get("data")
        if isinstance(inner, list):
            return ", ".join(f.get("file_name_or_path", "?") for f in inner)
        if isinstance(inner, dict) and "content" in inner:
            return inner["content"].replace("\n", " / ")
        return data.get("message") or json.dumps(data)
    except Exception:
        return content


def episode_html(row):
    c = check(row)
    head = header(row, c)
    color = COLORS[c["erosion"]]
    deleted = "deleted the important file" if c["harm_done"] else "spared it"
    title = (f"k={row['rounds']}  seed={row['seed']}  —  rule_erosion: {c['erosion']}"
             f"  ({deleted})")
    parts = [f'<details class="ep" style="border-left-color:{color}">',
             f"<summary>{esc(title)}</summary>",
             f'<pre class="meta">{esc(head)}</pre>']
    last = sum(1 for m in row["messages"] if m.get("role") == "user")
    rnd = -1
    for m in row["messages"]:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            parts.append(f'<details class="think"><summary>system prompt</summary>'
                         f"<pre>{esc(content)}</pre></details>")
        elif role == "user":
            rnd += 1
            tag = ("final round, important file present" if rnd == last - 1
                   else "junk only")
            parts.append(f"<h4>r_{rnd} <small>({tag})</small></h4>"
                         f'<p class="user">{esc(content)}</p>')
        elif role == "assistant":
            think = THINK.search(content)
            if think:
                parts.append(f'<details class="think"><summary>thinking</summary>'
                             f"<p><i>{esc(think.group(1).strip())}</i></p></details>")
            said = THINK.sub("", content).strip()
            if said:
                parts.append(f"<p>{esc(said)}</p>")
            for call in m.get("tool_calls") or []:
                fn = call["function"]
                parts.append(f'<code class="call">→ {esc(fn["name"])}'
                             f"({esc(fn.get('arguments', ''))})</code>")
        elif role == "tool":
            parts.append(f'<code class="resp">← {esc(_tool_result(content))}</code>')
    parts.append("</details>")
    return "\n".join(parts)


def render_html(out, path, model=None):
    """Every episode in <out>/episodes.jsonl, newest condition first, into one HTML file."""
    eps = Path(out) / "episodes.jsonl"
    rows = [json.loads(l) for l in eps.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: (r["rounds"], r["seed"]))
    name = model or Path(out).name
    blocks = "\n".join(episode_html(r) for r in rows)
    doc = (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
           f"<title>{esc(name)} — rule erosion episodes</title>"
           f"<style>{CSS}</style></head><body>"
           f"<h2>{esc(name)} — {len(rows)} episodes</h2>"
           f'<div class="controls">'
           f"<button onclick=\"document.querySelectorAll('details.ep')"
           f".forEach(d=>d.open=true)\">expand all</button> "
           f"<button onclick=\"document.querySelectorAll('details')"
           f".forEach(d=>d.open=false)\">collapse all</button></div>"
           f"{blocks}</body></html>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc)
    print(f"wrote {path} ({len(rows)} episodes)")
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="folder holding episodes.jsonl")
    ap.add_argument("--html", required=True, help="where to write the .html")
    ap.add_argument("--model")
    a = ap.parse_args()
    render_html(a.out, a.html, a.model)
