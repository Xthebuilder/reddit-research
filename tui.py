"""
Textual TUI for Reddit Research Tool.

Layout:
┌─────────────────────────────────────────────────────────────────┐
│ REDDIT RESEARCH  model: llama3.2          [status]             │
├──────────────────┬──────────────────────────────────────────────┤
│ Topics           │  Posts (topic selected)                      │
│                  │                                              │
│  ● zfs snapshots │  [relevance] [sub]  Title                   │
│  ● ollama setup  │  ...                                         │
│  ...             │                                              │
├──────────────────┴──────────────────────────────────────────────┤
│ Chat                                                            │
│  You: how do I set up zfs auto-snapshots?                       │
│  AI:  Use zfs-auto-snapshot or sanoid. [Source 2] explains...  │
├─────────────────────────────────────────────────────────────────┤
│ Search: [___________________]  Subreddits: [_______________]    │
│ Query:  [___________________]  [Fetch] [Ask] [Delete] [Models]  │
└─────────────────────────────────────────────────────────────────┘
"""
import json
import threading
import shutil
import subprocess
import re
from pathlib import Path
from datetime import datetime
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Static, ListView, ListItem,
    Input, Button, Label, Log, RichLog,
)
from textual.binding import Binding
from textual import work
from rich.text import Text
from rich.markup import escape

import db
import reddit
import brave
import exa_client
import llm
import report
import serper_client
import tavily_client
from config import (
    BRAVE_MAX_RESULTS,
    CONTEXT_POSTS,
    DEFAULT_PERSONA,
    DEFAULT_SUBREDDITS,
    MAX_RESEARCH_ITERATIONS,
    OLLAMA_MODEL,
    RELEVANCE_THRESHOLD,
)


AUTO_SUBREDDIT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "sysadmin": (
        "sysadmin", "admin", "server", "servers", "infrastructure", "network", "dns",
        "backup", "monitoring", "vpn", "proxy", "firewall", "storage", "raid",
    ),
    "linux": (
        "linux", "kernel", "distro", "desktop", "wayland", "x11", "package", "packages",
    ),
    "selfhosted": (
        "self-hosted", "selfhosted", "docker", "compose", "container", "containers", "kubernetes",
        "k8s", "service", "homelab", "nas", "media server", "plex", "jellyfin",
    ),
    "homelab": (
        "homelab", "home lab", "proxmox", "vm", "virtual machine", "backup", "nas", "storage",
    ),
    "LocalLLaMA": (
        "llm", "local llama", "ollama", "lm studio", "langchain", "rag", "ai", "model", "models",
    ),
    "linuxquestions": (
        "help", "how do i", "how to", "error", "fix", "install", "configure", "setup", "troubleshoot",
    ),
    "archlinux": (
        "arch", "pacman", "aur", "arch linux", "archlinux", "endeavouros", "manjaro",
    ),
    "debian": (
        "debian", "apt", "apt-get", "ubuntu", "mint", "pop os", "ubuntu server", "package manager",
    ),
    "commandline": (
        "command line", "cli", "shell", "terminal", "bash", "zsh", "sed", "awk", "grep", "ssh",
    ),
}

AUTO_WEBSITE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "reddit.com": ("reddit", "subreddit", "thread", "discussion"),
    "github.com": ("github", "repo", "repository", "source", "code"),
    "docs.python.org": ("python", "pip", "venv", "asyncio"),
    "developer.mozilla.org": ("javascript", "typescript", "web", "browser", "css", "html"),
    "kubernetes.io": ("kubernetes", "k8s", "kubectl", "helm"),
    "docs.docker.com": ("docker", "compose", "container", "containers"),
    "wiki.archlinux.org": ("linux", "arch", "systemd", "networking", "zfs", "drivers"),
    "serverfault.com": ("sysadmin", "network", "dns", "storage", "raid", "backup"),
    "stackoverflow.com": ("error", "exception", "bug", "fix", "how to"),
    "learn.microsoft.com": ("azure", "microsoft", "windows", "powershell"),
}

DEFAULT_WEB_SITES = ["reddit.com", "github.com", "serverfault.com", "stackoverflow.com"]


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


class ResearchApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #tab-bar {
        height: 3;
        padding: 0 1;
        layout: horizontal;
        background: $surface;
        border-bottom: solid $primary-darken-3;
    }

    .tab-button {
        min-width: 12;
        margin-right: 1;
    }

    .tab-button-active {
        background: $accent;
        color: $text;
        text-style: bold underline;
    }

    #content-area {
        height: 1fr;
        layout: vertical;
        padding: 1;
    }

    .tab-pane {
        height: 1fr;
        border: solid $primary-darken-2;
        padding: 0 1;
    }

    .hidden {
        display: none;
    }

    #topics-pane {
        layout: vertical;
    }

    #data-pane {
        layout: vertical;
    }

    #reports-pane {
        layout: vertical;
    }

    #summary-box {
        height: auto;
        margin-bottom: 1;
    }

    #topic-list {
        height: 1fr;
    }

    #post-list {
        height: 1fr;
        margin-bottom: 1;
    }

    #report-filter {
        margin-bottom: 1;
    }

    #report-actions {
        height: 3;
        margin-bottom: 1;
    }

    #report-list {
        height: 10;
        margin-bottom: 1;
    }

    #report-preview {
        height: 1fr;
        border: solid $secondary-darken-3;
        padding: 0 1;
    }

    #chat-panel {
        height: 14;
        border: solid $accent-darken-2;
        padding: 0 1;
    }

    #bottom-bar {
        height: auto;
        padding: 0 1;
        background: $surface;
        border-top: solid $primary-darken-3;
        layout: vertical;
    }

    #search-row {
        layout: horizontal;
        height: 3;
        margin-bottom: 0;
    }

    #persona-row {
        layout: horizontal;
        height: 3;
        margin-bottom: 0;
    }

    #persona-input {
        width: 1fr;
    }

    #query-row {
        layout: horizontal;
        height: 3;
    }

    #search-input {
        width: 1fr;
        margin-right: 1;
    }

    #subs-input {
        width: 1fr;
    }

    #subs-mode-badge {
        width: 12;
        content-align: center middle;
        margin-left: 1;
        border: solid $primary-darken-2;
        text-style: bold;
    }

    .badge-auto {
        background: $success;
        color: $text;
    }

    .badge-manual {
        background: $warning;
        color: $text;
    }

    #query-input {
        width: 1fr;
        margin-right: 1;
    }

    Button {
        min-width: 10;
        margin-right: 1;
    }

    #status-bar {
        height: 1;
        background: $primary-darken-3;
        color: $text-muted;
        padding: 0 1;
    }

    .panel-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+1", "show_topics", "Topics"),
        Binding("ctrl+2", "show_data", "Data"),
        Binding("ctrl+3", "show_reports", "Reports"),
        Binding("ctrl+f", "focus_search", "Search"),
        Binding("ctrl+a", "focus_query", "Ask"),
        Binding("ctrl+o", "open_selected_report", "Open report"),
        Binding("f5", "refresh_topic", "Refresh"),
        Binding("ctrl+d", "deepen", "Deepen"),
        Binding("delete", "delete_topic", "Delete topic"),
    ]

    def __init__(self):
        super().__init__()
        db.init_db()
        self._current_topic_id: int | None = None
        self._session_id: int | None = None
        self._posts: list[dict] = []
        self._report_paths: list[Path] = []
        self._selected_report_path: Path | None = None
        self._report_filter: str = ""
        self._active_tab: str = "topics"
        self._subs_manually_edited: bool = False
        self._last_auto_subs: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="tab-bar"):
            yield Button("Topics", id="tab-topics", classes="tab-button")
            yield Button("Data", id="tab-data", classes="tab-button")
            yield Button("Reports", id="tab-reports", classes="tab-button")
        with Vertical(id="content-area"):
            with Vertical(id="topics-pane", classes="tab-pane"):
                yield Label("Topics", classes="panel-title")
                yield Static("Select a topic to load its data and reports.", id="summary-box")
                yield ListView(id="topic-list")
            with Vertical(id="data-pane", classes="tab-pane"):
                yield Label("Data", classes="panel-title")
                yield ListView(id="post-list")
                yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True)
            with Vertical(id="reports-pane", classes="tab-pane"):
                yield Label("Reports", classes="panel-title")
                yield Input(placeholder="Filter reports by topic or filename...", id="report-filter")
                with Horizontal(id="report-actions"):
                    yield Button("Open in Editor", id="btn-open-report", variant="primary")
                    yield Button("Export Current", id="btn-export", variant="warning")
                yield ListView(id="report-list")
                yield RichLog(id="report-preview", highlight=True, markup=True, wrap=True)
        with Vertical(id="bottom-bar"):
            with Horizontal(id="search-row"):
                yield Input(placeholder="Topic or research question...", id="search-input")
                yield Input(
                    placeholder=f"Subreddits, comma-separated (default: {','.join(DEFAULT_SUBREDDITS[:3])}...)",
                    id="subs-input",
                )
                yield Static("AUTO", id="subs-mode-badge", classes="badge-auto")
            with Horizontal(id="persona-row"):
                yield Input(
                    placeholder="Persona (e.g. 'You are a senior DevOps engineer...')",
                    id="persona-input",
                    value=DEFAULT_PERSONA,
                )
            with Horizontal(id="query-row"):
                yield Input(placeholder="Ask about the selected topic...", id="query-input")
                yield Button("Fetch", id="btn-fetch", variant="primary")
                yield Button("Ask", id="btn-ask", variant="success")
                yield Button("Deepen", id="btn-deepen", variant="warning")
                yield Button("Delete", id="btn-delete", variant="error")
                yield Button("Models", id="btn-models", variant="default")
        yield Static("Ready", id="status-bar")
        yield Footer()

    def on_mount(self):
        self._set_active_tab("topics")
        self._reload_topics()
        self._refresh_inspector()
        self._update_subreddit_mode_badge()
        ok, msg = llm.check_ollama()
        if ok:
            self._status(f"Ollama OK | model: {OLLAMA_MODEL} | available: {msg}")
        else:
            self._status(f"[red]Ollama unreachable: {msg}[/red]")

    def _on_report_updated(self, topic_name: str, path):
        self.call_from_thread(
            self._status,
            f"Report updated: {path.name}",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _status(self, msg: str):
        self.query_one("#status-bar", Static).update(msg)

    def _set_input_value(self, widget_id: str, value: str):
        self.query_one(widget_id, Input).value = value

    def _update_subreddit_mode_badge(self):
        badge = self.query_one("#subs-mode-badge", Static)
        if self._subs_manually_edited:
            badge.update("MANUAL")
            badge.remove_class("badge-auto")
            badge.add_class("badge-manual")
        else:
            badge.update("AUTO")
            badge.remove_class("badge-manual")
            badge.add_class("badge-auto")

    def _suggest_subreddits_for_query(self, query: str):
        query = query.strip()
        if not query:
            return
        suggested = self._auto_subreddits_for_query(query)
        subs_input = self.query_one("#subs-input", Input)
        current = subs_input.value.strip()

        # Keep auto-suggestions fresh until the user manually edits the field.
        if (not self._subs_manually_edited) or (current == ", ".join(self._last_auto_subs)):
            joined = ", ".join(suggested)
            self._set_input_value("#subs-input", joined)
            self._last_auto_subs = suggested
            self._subs_manually_edited = False
            self._update_subreddit_mode_badge()
            self._status(f"Suggested subreddits: {joined}")

    def _set_active_tab(self, tab: str):
        self._active_tab = tab
        for name in ("topics", "data", "reports"):
            pane = self.query_one(f"#{name}-pane")
            button = self.query_one(f"#tab-{name}", Button)
            if name == tab:
                pane.remove_class("hidden")
                button.add_class("tab-button-active")
            else:
                pane.add_class("hidden")
                button.remove_class("tab-button-active")

    def _auto_subreddits_for_query(self, query: str) -> list[str]:
        normalized = query.lower().strip()
        scored: list[tuple[int, int, str]] = []

        for index, subreddit in enumerate(DEFAULT_SUBREDDITS):
            score = 0
            sub_key = subreddit.lower().replace(" ", "")
            compact_query = re.sub(r"[^a-z0-9]+", "", normalized)
            if sub_key and sub_key in compact_query:
                score += 5

            for keyword in AUTO_SUBREDDIT_KEYWORDS.get(subreddit, ()):
                if keyword in normalized:
                    score += 2

            if score:
                scored.append((score, index, subreddit))

        if not scored:
            # No keyword match — ask LLM to suggest relevant subreddits
            try:
                suggested = llm.suggest_subreddits(query, num=6)
                if suggested:
                    return suggested
            except Exception:
                pass
            return DEFAULT_SUBREDDITS[:4]

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [subreddit for _, _, subreddit in scored[:5]]

    def _auto_sites_for_query(self, query: str) -> list[str]:
        normalized = query.lower().strip()
        scored: list[tuple[int, str]] = []

        for site, keywords in AUTO_WEBSITE_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in normalized:
                    score += 2
            if site.replace(".", "") in normalized.replace(".", ""):
                score += 4
            if score:
                scored.append((score, site))

        if not scored:
            return DEFAULT_WEB_SITES[:]

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [site for _, site in scored[:4]]

    def _plan_research_targets(self, query: str) -> tuple[list[str], list[str]]:
        subreddits = self._auto_subreddits_for_query(query)
        sites = self._auto_sites_for_query(query)
        return subreddits, sites

    def _search_web_for_sites(self, query: str, sites: list[str]) -> list[dict]:
        if not brave.is_configured():
            return []

        per_site = max(3, min(8, BRAVE_MAX_RESULTS // max(1, len(sites))))
        seen_urls: set[str] = set()
        combined: list[dict] = []

        for site in sites:
            scoped_query = f"{query} site:{site}"
            try:
                results = brave.search(scoped_query, count=per_site)
            except Exception:
                continue
            for result in results:
                url = result.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                combined.append(result)
                if len(combined) >= BRAVE_MAX_RESULTS:
                    return combined

        return combined

    def _open_report_in_editor(self, path: Path):
        candidates = ["code", "codium"]
        command = next((candidate for candidate in candidates if shutil.which(candidate)), None)
        if not command:
            self._status("No editor command found (expected code or codium)")
            return
        try:
            subprocess.Popen([command, "--reuse-window", "--goto", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._status(f"Opened report in editor: {path.name}")
        except Exception as e:
            self._status(f"[red]Could not open report in editor: {e}[/red]")

    def _reload_topics(self):
        lv = self.query_one("#topic-list", ListView)
        lv.clear()
        for t in db.list_topics():
            lv.append(TopicItem(t))

    def _reload_reports(self):
        lv = self.query_one("#report-list", ListView)
        lv.clear()
        filter_text = self._report_filter.strip().lower()
        self._report_paths = sorted(
            Path(report.REPORTS_DIR).glob("*.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if filter_text:
            self._report_paths = [
                path
                for path in self._report_paths
                if filter_text in path.stem.lower() or filter_text in str(path).lower()
            ]
        for path in self._report_paths:
            lv.append(ReportItem(path))

    def _preview_report(self, path: Path):
        self._selected_report_path = path
        preview = self.query_one("#report-preview", RichLog)
        preview.clear()
        if not path.exists():
            preview.write("Report file not found.")
            return
        preview.write(Text.from_markup(f"[bold]{escape(path.name)}[/bold]"))
        preview.write(Text.from_markup(f"[dim]{escape(str(path))}[/dim]"))
        preview.write("")
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines[:40]:
            preview.write(line)

    def _refresh_inspector(self):
        topics = db.list_topics()
        total_posts = 0
        total_web = 0
        total_messages = 0
        current_topic_name = "No topic selected"
        current_report = "No report selected"
        current_report_path: Path | None = None

        if self._current_topic_id:
            topic = db.get_topic(self._current_topic_id)
            if topic:
                current_topic_name = topic["name"]
                total_posts = len(db.get_posts(self._current_topic_id, min_relevance=-1))
                total_web = len(db.get_web_results(self._current_topic_id, min_relevance=-1))
                if self._session_id:
                    total_messages = len(db.get_messages(self._session_id, limit=200))
                current_report_path = report.report_path(topic)
                current_report = str(current_report_path)

        summary = (
            f"[bold]Current topic:[/bold] {escape(current_topic_name)}\n"
            f"[bold]Topics saved:[/bold] {len(topics)}\n"
            f"[bold]Posts in topic:[/bold] {total_posts}\n"
            f"[bold]Web results:[/bold] {total_web}\n"
            f"[bold]Chat turns:[/bold] {total_messages}\n"
            f"[bold]Report path:[/bold] {escape(current_report)}\n\n"
            f"[dim]Tip: use Ctrl+F to jump to search, Ctrl+A to ask, F5 to refresh.[/dim]"
        )
        self.query_one("#summary-box", Static).update(summary)
        self._reload_reports()
        if current_report_path and current_report_path.exists():
            self._preview_report(current_report_path)
        elif self._report_paths:
            self._preview_report(self._report_paths[0])
        else:
            self.query_one("#report-preview", RichLog).clear()
            self.query_one("#report-preview", RichLog).write("No reports yet. Press Export after fetching a topic.")

    def _reload_posts(self, topic_id: int):
        self._posts = db.get_posts(topic_id, min_relevance=-1)
        lv = self.query_one("#post-list", ListView)
        lv.clear()
        for p in self._posts:
            lv.append(PostItem(p))

    def _chat_write(self, role: str, text: str):
        log = self.query_one("#chat-log", RichLog)
        if role == "user":
            log.write(Text.from_markup(f"[bold cyan]You:[/bold cyan] {escape(text)}"))
        else:
            log.write(Text.from_markup(f"[bold green]AI:[/bold green] {text}"))
        log.scroll_end(animate=False)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_list_view_selected(self, event: ListView.Selected):
        item = event.item
        if isinstance(item, TopicItem):
            self._current_topic_id = item.topic["id"]
            self._session_id = db.get_or_create_session(self._current_topic_id)
            self._reload_posts(self._current_topic_id)
            self._refresh_inspector()
            # Load topic persona into the persona input
            topic_persona = item.topic.get("persona")
            if topic_persona:
                self._set_input_value("#persona-input", topic_persona)
            # reload chat history
            log = self.query_one("#chat-log", RichLog)
            log.clear()
            for msg in db.get_messages(self._session_id):
                self._chat_write(msg["role"], msg["content"])
            self._status(f"Topic: {item.topic['name']}")

        elif isinstance(item, PostItem):
            post = item.post
            log = self.query_one("#chat-log", RichLog)
            log.write(Text.from_markup(
                f"\n[bold yellow]--- Post Details ---[/bold yellow]\n"
                f"[bold]{escape(post['title'])}[/bold]\n"
                f"r/{post['subreddit']} | Reddit score: {post['score']} | "
                f"Relevance: {post.get('relevance_score', '?')}/10\n"
                f"URL: {post['url']}\n"
            ))
            if post.get("content"):
                log.write(post["content"][:800])
            if post.get("comments"):
                log.write(Text.from_markup("\n[dim]Top comments:[/dim]"))
                for c in post["comments"][:3]:
                    log.write(Text.from_markup(f"  [dim]•[/dim] {escape(c[:300])}"))
            log.scroll_end(animate=False)
        elif isinstance(item, ReportItem):
            self._set_active_tab("reports")
            self._preview_report(item.path)

    def on_list_view_highlighted(self, event: ListView.Highlighted):
        item = event.item
        if isinstance(item, ReportItem):
            self._preview_report(item.path)

    def on_input_changed(self, event: Input.Changed):
        if event.input.id == "report-filter":
            self._report_filter = event.value
            self._reload_reports()
            if self._report_paths:
                self._preview_report(self._report_paths[0])
            else:
                preview = self.query_one("#report-preview", RichLog)
                preview.clear()
                preview.write("No matching reports.")
        elif event.input.id == "search-input":
            self._suggest_subreddits_for_query(event.value)
        elif event.input.id == "subs-input":
            current = event.value.strip()
            auto_value = ", ".join(self._last_auto_subs)
            self._subs_manually_edited = bool(current and current != auto_value)
            self._update_subreddit_mode_badge()

    def on_button_pressed(self, event: Button.Pressed):
        btn_id = event.button.id
        if btn_id == "tab-topics":
            self.action_show_topics()
        elif btn_id == "tab-data":
            self.action_show_data()
        elif btn_id == "tab-reports":
            self.action_show_reports()
        if btn_id == "btn-fetch":
            self.action_fetch()
        elif btn_id == "btn-ask":
            self.action_ask()
        elif btn_id == "btn-deepen":
            self.action_deepen()
        elif btn_id == "btn-open-report":
            self.action_open_selected_report()
        elif btn_id == "btn-export":
            self.action_export()
        elif btn_id == "btn-delete":
            self.action_delete_topic()
        elif btn_id == "btn-models":
            self.action_list_models()

    def on_input_submitted(self, event: Input.Submitted):
        if event.input.id == "search-input":
            self.action_fetch()
        elif event.input.id == "query-input":
            self.action_ask()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_focus_search(self):
        self.query_one("#search-input", Input).focus()

    def action_focus_query(self):
        self.query_one("#query-input", Input).focus()

    def action_show_topics(self):
        self._set_active_tab("topics")

    def action_show_data(self):
        self._set_active_tab("data")

    def action_show_reports(self):
        self._set_active_tab("reports")

    def action_open_selected_report(self):
        if self._selected_report_path and self._selected_report_path.exists():
            self._open_report_in_editor(self._selected_report_path)
        else:
            self._status("Select a report first")

    def action_list_models(self):
        models = llm.list_models()
        if models:
            self._status(f"Ollama models: {', '.join(models)}")
            log = self.query_one("#chat-log", RichLog)
            log.write(Text.from_markup(
                "[bold yellow]Available Ollama models:[/bold yellow]\n" +
                "\n".join(f"  • {m}" for m in models)
            ))
        else:
            self._status("[red]Could not reach Ollama or no models installed[/red]")

    def _fetch_and_process_posts(self, topic_id: int, query: str, subreddits: list[str], tag: str = "", original_topic: str | None = None, seen_urls: set | None = None):
        """Fetch Reddit posts for a query, then judge + embed + summarize. Returns post count."""
        prefix = f"[{tag}] " if tag else ""

        def progress(sub, done, total):
            self.call_from_thread(self._status, f"{prefix}Fetching r/{sub} ({done}/{total})...")

        posts = reddit.fetch_topic(query, subreddits, progress_cb=progress, seen_urls=seen_urls)
        self.call_from_thread(self._status, f"{prefix}Fetched {len(posts)} Reddit posts (deduped) — processing...")

        max_upvotes = max((p.get("score", 0) for p in posts), default=1)

        for i, post in enumerate(posts):
            post_id = db.save_post(topic_id, post)
            llm_score = llm.judge_relevance(post, query, original_topic=original_topic)
            hybrid = llm.blend_scores(llm_score, post.get("score", 0), max_upvotes)
            db.update_relevance(post_id, hybrid)
            self.call_from_thread(
                self._status,
                f"{prefix}[Reddit] {i+1}/{len(posts)}: {post['title'][:35]}... → {hybrid:.1f}/10",
            )
            try:
                embedding = llm.embed_post(post)
                db.update_post_embedding(post_id, embedding)
            except Exception:
                pass
            if hybrid >= RELEVANCE_THRESHOLD:
                try:
                    summary = llm.summarize_post(post)
                    if summary:
                        db.update_post_summary(post_id, summary)
                except Exception:
                    pass

        return len(posts)

    def _fetch_all_web(self, query: str, sites: list[str]) -> list[dict]:
        """Fire all configured search APIs in parallel, deduplicate by URL."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        tasks: list[tuple[str, callable]] = []
        if brave.is_configured():
            tasks.append(("brave", lambda q=query: self._search_web_for_sites(q, sites)))
        if tavily_client.is_configured():
            tasks.append(("tavily", lambda q=query: tavily_client.search(q, count=15)))
        if serper_client.is_configured():
            tasks.append(("serper", lambda q=query: serper_client.search(q, count=15)))
        if exa_client.is_configured():
            tasks.append(("exa", lambda q=query: exa_client.search(q, count=10)))

        if not tasks:
            return []

        api_names = ", ".join(n for n, _ in tasks)
        self.call_from_thread(self._status, f"Searching [{api_names}] in parallel...")

        seen: set[str] = set()
        combined: list[dict] = []

        def _run(name, fn):
            try:
                results = fn()
                for r in results:
                    r.setdefault("source", name)
                return results
            except Exception:
                return []

        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(_run, name, fn): name for name, fn in tasks}
            for fut in as_completed(futures):
                for r in fut.result():
                    url = r.get("url", "")
                    if url and url not in seen:
                        seen.add(url)
                        combined.append(r)

        return combined

    def _fetch_and_process_web(self, topic_id: int, query: str, sites: list[str], tag: str = "", original_topic: str | None = None):
        """Fetch web results from all APIs, then judge + embed + summarize. Returns result count."""
        prefix = f"[{tag}] " if tag else ""

        web_results = self._fetch_all_web(query, sites)
        if not web_results:
            self.call_from_thread(self._status, f"{prefix}No web APIs configured — skipping web search")
            return 0

        web_count = len(web_results)
        self.call_from_thread(self._status, f"{prefix}Fetched {web_count} unique web results — processing...")

        for i, result in enumerate(web_results):
            rid = db.save_web_result(topic_id, result)
            score = llm.judge_web_relevance(result, query, original_topic=original_topic)
            db.update_web_relevance(rid, score)
            self.call_from_thread(
                self._status,
                f"{prefix}[Web] {i+1}/{web_count}: {result['title'][:35]}... → {score:.0f}/10",
            )
            try:
                embedding = llm.embed_web_result(result)
                db.update_web_embedding(rid, embedding)
            except Exception:
                pass
            if score >= RELEVANCE_THRESHOLD:
                try:
                    summary = llm.summarize_web_result(result)
                    if summary:
                        db.update_web_summary(rid, summary)
                except Exception:
                    pass

        return web_count

    @work(thread=True)
    def action_fetch(self):
        query = self.query_one("#search-input", Input).value.strip()
        if not query:
            self.call_from_thread(self._status, "Enter a topic to search")
            return

        persona = self.query_one("#persona-input", Input).value.strip() or None
        subs_raw = self.query_one("#subs-input", Input).value.strip()
        auto_subs, auto_sites = self._plan_research_targets(query)
        subreddits = [s.strip() for s in subs_raw.split(",") if s.strip()] if subs_raw else auto_subs
        if not subs_raw:
            self.call_from_thread(
                self._status,
                f"Auto-selected subreddits: {', '.join(subreddits)}",
            )
            self.call_from_thread(self._set_input_value, "#subs-input", ", ".join(subreddits))
        self._last_auto_subs = subreddits
        self._subs_manually_edited = False
        self.call_from_thread(self._update_subreddit_mode_badge)

        # --- Query expansion + topic decomposition ---
        self.call_from_thread(self._status, "Expanding query and decomposing sub-questions...")
        expanded = llm.expand_query(query)
        sub_questions = llm.decompose_topic(query)
        queries = list(dict.fromkeys(expanded + sub_questions))  # dedupe, preserve order
        self.call_from_thread(
            self._status,
            f"Research plan → {len(queries)} queries × {len(subreddits)} subreddits × {len(auto_sites)} sites",
        )

        topic_id = db.upsert_topic(query, subreddits, persona=persona)
        self._current_topic_id = topic_id
        self._session_id = db.get_or_create_session(topic_id)
        self.call_from_thread(self._reload_topics)

        seen_reddit_urls: set[str] = set()

        # --- Pass 1: Fetch with original + expanded + sub-questions ---
        total_posts = 0
        total_web = 0
        for qi, q in enumerate(queries):
            tag = f"Q{qi+1}/{len(queries)}"
            total_posts += self._fetch_and_process_posts(topic_id, q, subreddits, tag=tag, seen_urls=seen_reddit_urls)
            total_web += self._fetch_and_process_web(topic_id, q, auto_sites, tag=tag, original_topic=query)

        db.mark_topic_fetched(topic_id)
        self.call_from_thread(self._reload_posts, topic_id)
        self.call_from_thread(self._refresh_inspector)

        kept_r = len(db.get_posts(topic_id, min_relevance=RELEVANCE_THRESHOLD))
        kept_w = len(db.get_web_results(topic_id, min_relevance=RELEVANCE_THRESHOLD))
        self.call_from_thread(
            self._status,
            f"Pass 1 done — Reddit: {total_posts} posts ({kept_r} relevant) | "
            f"Web: {total_web} results ({kept_w} relevant)",
        )

        # --- Iterative research: analyze gaps and do follow-up passes ---
        for iteration in range(MAX_RESEARCH_ITERATIONS):
            self.call_from_thread(
                self._status,
                f"Analyzing research gaps (iteration {iteration + 1}/{MAX_RESEARCH_ITERATIONS})...",
            )
            all_posts = db.get_posts(topic_id, min_relevance=RELEVANCE_THRESHOLD)
            all_web = db.get_web_results(topic_id, min_relevance=RELEVANCE_THRESHOLD)

            gap_queries = llm.analyze_gaps(query, all_posts, all_web)
            if not gap_queries:
                self.call_from_thread(self._status, "No significant gaps found — research complete.")
                break

            self.call_from_thread(
                self._status,
                f"Gap analysis found {len(gap_queries)} follow-up queries: "
                + " | ".join(q[:40] for q in gap_queries),
            )

            for gi, gq in enumerate(gap_queries):
                tag = f"Gap{iteration+1}.{gi+1}"
                self._fetch_and_process_posts(topic_id, gq, subreddits, tag=tag)
                self._fetch_and_process_web(topic_id, gq, auto_sites, tag=tag)

            db.mark_topic_fetched(topic_id)
            self.call_from_thread(self._reload_posts, topic_id)
            self.call_from_thread(self._refresh_inspector)

        # --- Final stats ---
        final_r = len(db.get_posts(topic_id, min_relevance=RELEVANCE_THRESHOLD))
        final_w = len(db.get_web_results(topic_id, min_relevance=RELEVANCE_THRESHOLD))
        self.call_from_thread(
            self._status,
            f"Research complete — Reddit: {final_r} relevant posts | Web: {final_w} relevant results",
        )

        try:
            path = report.generate(topic_id)
            self.call_from_thread(self._status, f"Done — report generated: {path.name}")
            self.call_from_thread(self._refresh_inspector)
        except Exception as e:
            self.call_from_thread(self._status, f"[yellow]Report generation skipped: {e}[/yellow]")

    @work(thread=True)
    def action_ask(self):
        question = self.query_one("#query-input", Input).value.strip()
        if not question:
            self.call_from_thread(self._status, "Enter a question first")
            return
        if not self._current_topic_id:
            self.call_from_thread(self._status, "Select or fetch a topic first")
            return

        self.call_from_thread(self.query_one("#query-input", Input).clear)
        topic = db.get_topic(self._current_topic_id)
        persona = topic.get("persona") or self.query_one("#persona-input", Input).value.strip() or None

        # Try semantic retrieval first, fall back to relevance-score ranking
        self.call_from_thread(self._status, "Embedding question for semantic search...")
        posts = []
        web_results = []
        try:
            q_emb = llm.embed(question)
            posts = db.vector_search_posts(self._current_topic_id, q_emb, top_k=CONTEXT_POSTS)
            web_results = db.vector_search_web(self._current_topic_id, q_emb, top_k=CONTEXT_POSTS)
            if posts or web_results:
                self.call_from_thread(
                    self._status,
                    f"Semantic search: {len(posts)} posts + {len(web_results)} web results",
                )
        except Exception:
            pass

        # Fall back to relevance-score ranking if semantic search returned nothing
        if not posts:
            posts = db.get_posts(self._current_topic_id, min_relevance=RELEVANCE_THRESHOLD)
            if not posts:
                posts = db.get_posts(self._current_topic_id)
        if not web_results:
            web_results = db.get_web_results(self._current_topic_id, min_relevance=RELEVANCE_THRESHOLD)
            if not web_results:
                web_results = db.get_web_results(self._current_topic_id)

        history = db.get_messages(self._session_id)

        db.add_message(self._session_id, "user", question)
        self.call_from_thread(self._chat_write, "user", question)
        self.call_from_thread(self._status, "Thinking...")

        buf = []
        log = self.query_one("#chat-log", RichLog)
        first_token = [True]

        def on_token(token):
            buf.append(token)
            if first_token[0]:
                first_token[0] = False
                self.call_from_thread(
                    log.write,
                    Text.from_markup("[bold green]AI:[/bold green] "),
                )

        response = llm.answer(
            question, posts, topic["name"], history,
            on_token=on_token, web_results=web_results,
            persona=persona,
        )

        # Write the full response after streaming
        self.call_from_thread(log.write, Text(response))
        self.call_from_thread(log.scroll_end, False)

        db.add_message(self._session_id, "assistant", response)
        try:
            report.generate(self._current_topic_id)
        except Exception:
            pass
        self.call_from_thread(self._refresh_inspector)
        self.call_from_thread(self._status, "Ready")

    @work(thread=True)
    def action_deepen(self):
        question = self.query_one("#query-input", Input).value.strip()
        if not question:
            self.call_from_thread(self._status, "Enter a follow-up question to deepen research")
            return
        if not self._current_topic_id:
            self.call_from_thread(self._status, "Select or fetch a topic first")
            return

        topic = db.get_topic(self._current_topic_id)
        subreddits, sites = self._plan_research_targets(question)
        self.call_from_thread(
            self._status,
            f"Deepening '{topic['name']}' — expanding query...",
        )

        # Query expansion for deeper research
        queries = llm.expand_query(question)
        self.call_from_thread(
            self._status,
            f"Deepening with {len(queries)} queries across {len(subreddits)} subreddits + {len(sites)} sites",
        )

        for qi, q in enumerate(queries):
            tag = f"Deep Q{qi+1}/{len(queries)}"
            self._fetch_and_process_posts(self._current_topic_id, q, subreddits, tag=tag)
            self._fetch_and_process_web(self._current_topic_id, q, sites, tag=tag)

        db.add_message(self._session_id, "user", f"[deepen] {question}")
        db.mark_topic_fetched(self._current_topic_id)
        self.call_from_thread(self._reload_posts, self._current_topic_id)
        try:
            path = report.generate(self._current_topic_id)
            self.call_from_thread(self._status, f"Deep research complete — report updated: {path.name}")
        except Exception as e:
            self.call_from_thread(self._status, f"Deep research saved, report skipped: {e}")
        self.call_from_thread(self._refresh_inspector)

    def action_export(self):
        if not self._current_topic_id:
            self._status("Select a topic first")
            return
        try:
            path = report.generate(self._current_topic_id)
            self._status(f"Exported: {path}")
            self._refresh_inspector()
            self.action_show_reports()
            self._preview_report(path)
        except Exception as e:
            self._status(f"[red]Export failed: {e}[/red]")

    def action_refresh_topic(self):
        if self._current_topic_id:
            search_input = self.query_one("#search-input", Input)
            topic = db.get_topic(self._current_topic_id)
            search_input.value = topic["name"]
            self.action_fetch()

    def action_delete_topic(self):
        if not self._current_topic_id:
            self._status("No topic selected")
            return
        topic = db.get_topic(self._current_topic_id)
        db.delete_topic(self._current_topic_id)
        self._current_topic_id = None
        self._session_id = None
        self._posts = []
        self.query_one("#post-list", ListView).clear()
        self.query_one("#chat-log", RichLog).clear()
        self._reload_topics()
        self._refresh_inspector()
        self._status(f"Deleted topic: {topic['name']}")


def main():
    app = ResearchApp()
    app.run()


if __name__ == "__main__":
    main()
