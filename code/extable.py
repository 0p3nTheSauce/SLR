from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# Basic table
def basic_table():
    table = Table(title="Basic Table")
    
    table.add_column("Name", style="cyan")
    table.add_column("Age", style="magenta")
    table.add_column("City", style="green")
    
    table.add_row("Alice", "30", "New York")
    table.add_row("Bob", "25", "San Francisco")
    table.add_row("Charlie", "35", "London")
    
    console.print(table)

# Table with custom styling
def styled_table():
    table = Table(
        title="Styled Table",
        caption="Sales data for Q4",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
        box=box.ROUNDED
    )
    
    table.add_column("Product", style="cyan", no_wrap=True)
    table.add_column("Price", justify="right", style="green")
    table.add_column("Stock", justify="center", style="yellow")
    table.add_column("Status", justify="center")
    
    table.add_row("Laptop", "$999", "15", "[green]In Stock[/green]")
    table.add_row("Mouse", "$25", "150", "[green]In Stock[/green]")
    table.add_row("Keyboard", "$75", "0", "[red]Out of Stock[/red]")
    table.add_row("Monitor", "$350", "8", "[yellow]Low Stock[/yellow]")
    
    console.print(table)

# Table with different box styles
def box_styles_demo():
    boxes = [
        (box.SQUARE, "SQUARE"),
        (box.ROUNDED, "ROUNDED"),
        (box.MINIMAL, "MINIMAL"),
        (box.SIMPLE, "SIMPLE"),
        (box.DOUBLE, "DOUBLE")
    ]
    
    for box_style, name in boxes:
        table = Table(title=f"{name} Box Style", box=box_style)
        table.add_column("Column 1")
        table.add_column("Column 2")
        table.add_row("Row 1", "Data")
        table.add_row("Row 2", "Data")
        console.print(table)
        console.print()

# Advanced table with ratios and widths
def advanced_table():
    table = Table(
        title="Advanced Table",
        show_edge=False,
        padding=(0, 1)
    )
    
    # Control column widths with ratio or fixed width
    table.add_column("ID", style="dim", width=5)
    table.add_column("Description", ratio=2)
    table.add_column("Status", ratio=1, justify="center")
    
    table.add_row("001", "Complete the project documentation", "✓ Done")
    table.add_row("002", "Review pull requests and merge changes", "⏳ In Progress")
    table.add_row("003", "Update test coverage for new features", "○ Todo")
    
    console.print(table)

# Run all examples
if __name__ == "__main__":
    console.print("\n[bold]1. Basic Table[/bold]")
    basic_table()
    
    console.print("\n[bold]2. Styled Table[/bold]")
    styled_table()
    
    console.print("\n[bold]3. Box Styles[/bold]")
    box_styles_demo()
    
    console.print("\n[bold]4. Advanced Table[/bold]")
    advanced_table()