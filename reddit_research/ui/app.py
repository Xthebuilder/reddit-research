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
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from rich.markup import escape
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button, Footer, Header, Input, Label, ListView, RichLog, Static,
)

from reddit_research import db, llm, report, researcher
from reddit_research.config import (
    CONTEXT_POSTS,
    DEFAULT_PERSONA,
    MAX_RESEARCH_ITERATIONS,
    OLLAMA_MODEL,
    RELEVANCE_THRESHOLD,
)
from reddit_research.search import reddit
from reddit_research.ui.keywords import DEFAULT_SUBREDDITS
from reddit_research.ui.widgets import PostItem, ReportItem, TopicItem


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
        suggested = researcher.auto_subreddits(query)
        subs_input = self.query_one("#subs-input", Input)
        current = subs_input.value.strip()

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

    def _open_report_in_editor(self, path: Path):
        candidates = ["code", "codium"]
        command = next((c for c in candidates if shutil.which(c)), None)
        if not command:
            self._status("No editor command found (expected code or codium)")
            return
        try:
            subprocess.Popen(
                [command, "--reuse-window", "--goto", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
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
                path for path in self._report_paths
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
            self.query_one("#report-preview", RichLog).write(
                "No reports yet. Press Export after fetching a topic."
            )

    def _reload_posts(self, topic_id: int):
        self._posts = db.get_posts(topic_id, min_relevance=-1)
        lv = self.query_one("#post-list", ListView)
        lv.clear()
        for p in self._posts:
            lv.append(PostItem(p))

    def _chat_write(self, role: str, text: str):
        log_widget = self.query_one("#chat-log", RichLog)
        if role == "user":
            log_widget.write(Text.from_markup(f"[bold cyan]You:[/bold cyan] {escape(text)}"))
        else:
            log_widget.write(Text.from_markup(f"[bold green]AI:[/bold green] {text}"))
        log_widget.scroll_end(animate=False)

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
            topic_persona = item.topic.get("persona")
            if topic_persona:
                self._set_input_value("#persona-input", topic_persona)
            log_widget = self.query_one("#chat-log", RichLog)
            log_widget.clear()
            for msg in db.get_messages(self._session_id):
                self._chat_write(msg["role"], msg["content"])
            self._status(f"Topic: {item.topic['name']}")

        elif isinstance(item, PostItem):
            post = item.post
            log_widget = self.query_one("#chat-log", RichLog)
            log_widget.write(Text.from_markup(
                f"\n[bold yellow]--- Post Details ---[/bold yellow]\n"
                f"[bold]{escape(post['title'])}[/bold]\n"
                f"r/{post['subreddit']} | Reddit score: {post['score']} | "
                f"Relevance: {post.get('relevance_score', '?')}/10\n"
                f"URL: {post['url']}\n"
            ))
            if post.get("content"):
                log_widget.write(post["content"][:800])
            if post.get("comments"):
                log_widget.write(Text.from_markup("\n[dim]Top comments:[/dim]"))
                for c in post["comments"][:3]:
                    log_widget.write(Text.from_markup(f"  [dim]•[/dim] {escape(c[:300])}"))
            log_widget.scroll_end(animate=False)

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
            log_widget = self.query_one("#chat-log", RichLog)
            log_widget.write(Text.from_markup(
                "[bold yellow]Available Ollama models:[/bold yellow]\n" +
                "\n".join(f"  • {m}" for m in models)
            ))
        else:
            self._status("[red]Could not reach Ollama or no models installed[/red]")

    @work(thread=True)
    def action_fetch(self):
        query = self.query_one("#search-input", Input).value.strip()
        if not query:
            self.call_from_thread(self._status, "Enter a topic to search")
            return

        corrected, was_corrected = llm.correct_query(query)
        if was_corrected:
            query = corrected
            self.call_from_thread(self._set_input_value, "#search-input", query)
            self.call_from_thread(self._status, f"Autocorrected query: {query}")

        persona = self.query_one("#persona-input", Input).value.strip() or None
        subs_raw = self.query_one("#subs-input", Input).value.strip()
        auto_subs = researcher.auto_subreddits(query, llm_fallback=True)
        auto_sites = researcher.auto_sites(query)
        subreddits = [s.strip() for s in subs_raw.split(",") if s.strip()] if subs_raw else auto_subs

        self.call_from_thread(self._status, "Validating subreddits...")
        subreddits = reddit.filter_valid_subreddits(subreddits)
        if not subs_raw:
            self.call_from_thread(self._status, f"Auto-selected subreddits: {', '.join(subreddits)}")
        self.call_from_thread(self._set_input_value, "#subs-input", ", ".join(subreddits))
        self._last_auto_subs = subreddits
        self._subs_manually_edited = False
        self.call_from_thread(self._update_subreddit_mode_badge)

        self.call_from_thread(self._status, "Expanding query and decomposing sub-questions...")
        expanded = llm.expand_query(query)
        sub_questions = llm.decompose_topic(query)
        queries = list(dict.fromkeys(expanded + sub_questions))
        self.call_from_thread(
            self._status,
            f"Research plan → {len(queries)} queries × {len(subreddits)} subreddits × {len(auto_sites)} sites",
        )

        topic_id = db.upsert_topic(query, subreddits, persona=persona)
        self._current_topic_id = topic_id
        self._session_id = db.get_or_create_session(topic_id)
        self.call_from_thread(self._reload_topics)

        seen_reddit_urls: set[str] = set()

        def progress(msg: str):
            self.call_from_thread(self._status, msg)

        total_posts = 0
        total_web = 0
        for qi, q in enumerate(queries):
            tag = f"Q{qi+1}/{len(queries)}"
            total_posts += researcher.fetch_and_process_posts(
                topic_id, q, subreddits, progress=progress, tag=tag,
                original_topic=query, seen_urls=seen_reddit_urls,
            )
            total_web += researcher.fetch_and_process_web(
                topic_id, q, auto_sites, progress=progress, tag=tag, original_topic=query,
            )

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
                researcher.fetch_and_process_posts(
                    topic_id, gq, subreddits, progress=progress, tag=tag,
                )
                researcher.fetch_and_process_web(
                    topic_id, gq, auto_sites, progress=progress, tag=tag,
                )

            db.mark_topic_fetched(topic_id)
            self.call_from_thread(self._reload_posts, topic_id)
            self.call_from_thread(self._refresh_inspector)

        final_r = len(db.get_posts(topic_id, min_relevance=RELEVANCE_THRESHOLD))
        final_w = len(db.get_web_results(topic_id, min_relevance=RELEVANCE_THRESHOLD))
        self.call_from_thread(
            self._status,
            f"Research complete — Reddit: {final_r} relevant posts | Web: {final_w} relevant results",
        )

        try:
            path = report.generate(topic_id, question=query)
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

        log_widget = self.query_one("#chat-log", RichLog)
        first_token = [True]

        def on_token(token):
            if first_token[0]:
                first_token[0] = False
                self.call_from_thread(
                    log_widget.write,
                    Text.from_markup("[bold green]AI:[/bold green] "),
                )

        response = llm.answer(
            question, posts, topic["name"], history,
            on_token=on_token, web_results=web_results, persona=persona,
        )

        self.call_from_thread(log_widget.write, Text(response))
        self.call_from_thread(log_widget.scroll_end, False)

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
        subreddits = researcher.auto_subreddits(question)
        sites = researcher.auto_sites(question)
        self.call_from_thread(self._status, f"Deepening '{topic['name']}' — expanding query...")

        queries = llm.expand_query(question)
        self.call_from_thread(
            self._status,
            f"Deepening with {len(queries)} queries across {len(subreddits)} subreddits + {len(sites)} sites",
        )

        def progress(msg: str):
            self.call_from_thread(self._status, msg)

        for qi, q in enumerate(queries):
            tag = f"Deep Q{qi+1}/{len(queries)}"
            researcher.fetch_and_process_posts(
                self._current_topic_id, q, subreddits, progress=progress, tag=tag,
            )
            researcher.fetch_and_process_web(
                self._current_topic_id, q, sites, progress=progress, tag=tag,
            )

        db.add_message(self._session_id, "user", f"[deepen] {question}")
        db.mark_topic_fetched(self._current_topic_id)
        self.call_from_thread(self._reload_posts, self._current_topic_id)
        try:
            path = report.generate(self._current_topic_id, question=question)
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
