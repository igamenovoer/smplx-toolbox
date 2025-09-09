import numpy as np
import pickle
import rich
from rich.console import Console
from rich.table import Table
import smplx

console = Console()

# Load both files
npz_file = r'data/body_models/smplh/SMPLH_MALE.npz'
pkl_file = r'data/body_models/smplh/SMPLH_MALE.pkl'

console.print(f"[bold blue]Loading files:[/bold blue]")
console.print(f"NPZ: {npz_file}")
console.print(f"PKL: {pkl_file}")

try:
    # Load NPZ file
    npz_data = np.load(npz_file, allow_pickle=True)
    console.print(f"[green]✓[/green] NPZ file loaded successfully")
except Exception as e:
    console.print(f"[red]✗[/red] Error loading NPZ file: {e}")
    npz_data = None

try:
    # Load PKL file with different encoding approaches
    try:
        with open(pkl_file, 'rb') as f:
            pkl_data = pickle.load(f, encoding='latin1')
        console.print("[green]✓[/green] PKL file loaded successfully with latin1 encoding")
    except Exception as e1:
        try:
            with open(pkl_file, 'rb') as f:
                pkl_data = pickle.load(f, encoding='bytes')
            console.print("[green]✓[/green] PKL file loaded successfully with bytes encoding")
        except Exception as e2:
            console.print(f"[red]✗[/red] Error loading PKL file with latin1: {e1}")
            console.print(f"[red]✗[/red] Error loading PKL file with bytes: {e2}")
            pkl_data = None
except Exception as e:
    console.print(f"[red]✗[/red] Error loading PKL file: {e}")
    pkl_data = None

if npz_data is not None and pkl_data is not None:
    console.print("\n[bold yellow]Comparison Analysis:[/bold yellow]")
    
    # Get keys from both files
    npz_keys = set(npz_data.keys()) if hasattr(npz_data, 'keys') else set()
    pkl_keys = set(pkl_data.keys()) if hasattr(pkl_data, 'keys') else set()
    
    # Create comparison table
    table = Table(title="Key Comparison")
    table.add_column("Key", style="cyan")
    table.add_column("In NPZ", style="green")
    table.add_column("In PKL", style="magenta")
    table.add_column("NPZ Type", style="yellow")
    table.add_column("PKL Type", style="yellow")
    table.add_column("NPZ Shape", style="blue")
    table.add_column("PKL Shape", style="blue")
    
    all_keys = npz_keys.union(pkl_keys)
    
    for key in sorted(all_keys):
        in_npz = "✓" if key in npz_keys else "✗"
        in_pkl = "✓" if key in pkl_keys else "✗"
        
        npz_type = str(type(npz_data[key]).__name__) if key in npz_keys else "N/A"
        pkl_type = str(type(pkl_data[key]).__name__) if key in pkl_keys else "N/A"
        
        npz_shape = str(npz_data[key].shape) if key in npz_keys and hasattr(npz_data[key], 'shape') else "N/A"
        pkl_shape = str(pkl_data[key].shape) if key in pkl_keys and hasattr(pkl_data[key], 'shape') else "N/A"
        
        table.add_row(key, in_npz, in_pkl, npz_type, pkl_type, npz_shape, pkl_shape)
    
    console.print(table)
    
    # Summary statistics
    console.print(f"\n[bold green]Summary:[/bold green]")
    console.print(f"NPZ keys: {len(npz_keys)}")
    console.print(f"PKL keys: {len(pkl_keys)}")
    console.print(f"Common keys: {len(npz_keys.intersection(pkl_keys))}")
    console.print(f"NPZ only: {len(npz_keys - pkl_keys)}")
    console.print(f"PKL only: {len(pkl_keys - npz_keys)}")
    
    # Check data differences for common keys
    console.print(f"\n[bold yellow]Data Comparison for Common Keys:[/bold yellow]")
    common_keys = npz_keys.intersection(pkl_keys)
    
    for key in sorted(common_keys):
        npz_val = npz_data[key]
        pkl_val = pkl_data[key]
        
        if isinstance(npz_val, np.ndarray) and isinstance(pkl_val, np.ndarray):
            if npz_val.shape == pkl_val.shape:
                if np.allclose(npz_val, pkl_val, atol=1e-10):
                    status = "[green]✓ Identical[/green]"
                else:
                    max_diff = np.max(np.abs(npz_val - pkl_val))
                    status = f"[yellow]≈ Similar (max_diff: {max_diff:.2e})[/yellow]"
            else:
                status = "[red]✗ Different shapes[/red]"
        else:
            status = "[blue]? Non-array data[/blue]"
        
        console.print(f"  {key}: {status}")

elif npz_data is not None:
    console.print("\n[bold yellow]NPZ File Contents:[/bold yellow]")
    for key in npz_data.keys():
        value = npz_data[key]
        console.print(f"  {key}: {type(value)} {getattr(value, 'shape', 'N/A')}")

elif pkl_data is not None:
    console.print("\n[bold yellow]PKL File Contents:[/bold yellow]")
    for key in pkl_data.keys():
        value = pkl_data[key]
        console.print(f"  {key}: {type(value)} {getattr(value, 'shape', 'N/A')}")