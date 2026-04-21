"""
Textual ListItem widgets for Topics, Posts, and Reports.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rich.markup import escape
from textual.app import ComposeResult
from textual.widgets import Label, ListItem


class PostItem(ListItem):
    def __init__(self, post: dict):
        super().__init__()
        self.post = post

    def compose(self) -> ComposeResult:
        score = self.post.get("relevance_score", -1)
        if score < 0:
            badge = "[dim]?[/dim]"
        elif score >= 7:
            badge = f"[bold green]{score:.0f}[/bold green]"
        elif score >= 4:
            badge = f"[yellow]{score:.0f}[/yellow]"
        else:
            badge = f"[red]{score:.0f}[/red]"
        sub = self.post.get("subreddit", "")
        title = escape(self.post.get("title", "")[:70])
        yield Label(f"{badge} [dim]r/{sub}[/dim] {title}")


class TopicItem(ListItem):
    def __init__(self, topic: dict):
        super().__init__()
        self.topic = topic

    def compose(self) -> ComposeResult:
        name = escape(self.topic["name"])
        fetched = self.topic.get("last_fetched")
        suffix = " [dim](fetched)[/dim]" if fetched else " [dim](not fetched)[/dim]"
        yield Label(f"[bold]{name}[/bold]{suffix}")


class ReportItem(ListItem):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def compose(self) -> ComposeResult:
        stat = self.path.stat()
        modified = datetime.fromtimestamp(Path(self.path).stat().st_mtime)
        title = escape(self.path.stem.replace("_", " "))
        yield Label(f"[bold]{title}[/bold] [dim]({stat.st_size // 1024} KB, {modified:%b %d %H:%M})[/dim]")
