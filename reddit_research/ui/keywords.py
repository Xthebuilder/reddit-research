"""
Keyword maps for auto-selecting subreddits and websites based on query content.
Single source of truth — used by both the TUI (ui/app.py) and headless.py.
"""

AUTO_SUBREDDIT_KEYWORDS: dict[str, tuple[str, ...]] = {
    # Tech
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
    # Travel
    "travel": (
        "travel", "trip", "vacation", "holiday", "visit", "country", "countries",
        "tourist", "tourism", "abroad", "international", "passport", "visa",
    ),
    "solotravel": (
        "solo travel", "solo trip", "traveling alone", "first time travel", "new traveler",
        "backpack", "backpacking", "budget travel", "hostel",
    ),
    "digitalnomad": (
        "digital nomad", "remote work abroad", "work while traveling", "nomad",
    ),
    # Finance / Career
    "personalfinance": (
        "money", "finance", "budget", "saving", "investing", "debt", "salary", "income",
        "retirement", "401k", "credit", "loan", "mortgage",
    ),
    "financialindependence": (
        "fire", "financial independence", "retire early", "passive income", "frugal",
    ),
    "careerguidance": (
        "career", "job", "resume", "interview", "salary negotiation", "career change",
        "career advice", "job search", "promotion",
    ),
    "cscareerquestions": (
        "software engineer", "developer", "coding job", "tech job", "swe", "programming career",
    ),
    # Health / Fitness
    "fitness": (
        "fitness", "workout", "gym", "exercise", "weight loss", "muscle", "training",
    ),
    "nutrition": (
        "nutrition", "diet", "eating", "food", "calories", "meal", "supplements",
    ),
    "mentalhealth": (
        "mental health", "anxiety", "depression", "stress", "therapy", "wellbeing",
    ),
    # General
    "AskReddit": (
        "what do people think", "best way to", "opinions on", "recommend", "advice",
        "experience with", "thoughts on",
    ),
    "datascience": (
        "data science", "data analysis", "machine learning", "python", "pandas",
        "data analyst", "data engineer", "analytics",
    ),
    "programming": (
        "programming", "coding", "software", "code", "developer", "development",
        "algorithm", "api", "framework", "library",
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

DEFAULT_SUBREDDITS = list(AUTO_SUBREDDIT_KEYWORDS.keys())
