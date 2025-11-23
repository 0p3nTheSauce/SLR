from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
import time

console = Console()

# 1. Basic pretty printing with colors
console.print("\n[bold cyan]Welcome to Rich Library Demo![/bold cyan]\n")

# 2. Creating a table
table = Table(title="Sample Data Table", show_header=True, header_style="bold magenta")
table.add_column("Name", style="cyan", width=12)
table.add_column("Age", justify="right", style="green")
table.add_column("City", style="yellow")

table.add_row("Alice", "28", "New York")
table.add_row("Bob", "35", "London")
table.add_row("Charlie", "42", "Tokyo")

console.print(table)
console.print()

# 3. Panel with formatted text
panel = Panel(
    "[bold yellow]Rich[/bold yellow] makes terminal output beautiful!\n"
    "Features include: [green]colors[/green], [blue]tables[/blue], "
    "[magenta]progress bars[/magenta], and much more!",
    title="Info Box",
    border_style="bright_blue"
)
console.print(panel)
console.print()

# 4. Markdown rendering
markdown_text = """
# Rich Features
- **Beautiful** formatting
- *Easy* to use
- Supports `code snippets`
"""
console.print(Markdown(markdown_text))

# 5. Syntax highlighting
code = '''
def greet(name):
    return f"Hello, {name}!"
'''
syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
console.print(Panel(syntax, title="Code Example", border_style="green"))

# 6. Progress bar
console.print("\n[bold]Processing items...[/bold]")
for i in track(range(20), description="Loading..."):
    time.sleep(0.1)  # Simulate work

console.print("\n[bold green]✓ Demo complete![/bold green]")