# dashboards/fusion_tui.py
import json
import time
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from emergence.core.redis_bus import RedisBus

class FusionTUI:
    def __init__(self):
        self.console = Console()
        self.bus = RedisBus()
        self.events = []

    def start(self):
        self.bus.subscribe("system:vitals", self.handle)
        self.bus.subscribe("perception:vision:event", self.handle)
        self.bus.subscribe("collapse:mirror", self.handle)
        self.bus.subscribe("memory:collapse_created", self.handle)
        self.bus.subscribe("introspection:reflect_chain", self.handle)
        self.bus.subscribe("human:event", self.handle)
        self.console.print("[bold magenta]Fusion TUI Listening...[/bold magenta]")
        self.render_loop()

    def handle(self, msg):
        try:
            payload = json.loads(msg)
            event_type = payload.get("event") or payload.get("intent") or payload.get("type")
            summary = payload.get("summary") or "No summary"
            observer = payload.get("observer") or payload.get("source") or "unknown"
            ts = payload.get("timestamp") or payload.get("received_at")
            self.events.append((ts, observer, event_type, summary))
            if len(self.events) > 20:
                self.events.pop(0)
        except Exception as e:
            self.console.print(f"[red]Error:[/red] {e}")

    def render_loop(self):
        with Live(self.render_panel(), refresh_per_second=2, screen=False) as live:
            while True:
                time.sleep(1)
                live.update(self.render_panel())

    def render_panel(self):
        table = Table(title="Fusion Event Stream")
        table.add_column("Time", style="dim", width=20)
        table.add_column("Observer", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta")
        table.add_column("Summary", style="white")

        for ts, obs, typ, summ in reversed(self.events):
            table.add_row(ts[-8:], obs, typ, summ[:50])

        return Panel(table, title="[bold blue]Orion Fusion Monitor[/bold blue]", border_style="bright_blue")

if __name__ == "__main__":
    FusionTUI().start()

