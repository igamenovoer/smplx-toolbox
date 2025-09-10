"""
Test script to compare SMPL model series with rich formatting.

This script loads and compares different SMPL model variants (MANO, SMPLH, SMPLX)
and displays the results using rich tables for beautiful console output.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


@dataclass
class ModelInfo:
    """Data class for storing model information."""

    name: str
    filepath: Path
    vertices: int
    faces: int
    joints: int
    shape_params: int
    pose_params: int
    has_hands: bool
    has_face: bool
    file_format: str
    file_size_mb: float


class SMPLModelComparator:
    """Compare SMPL model series with detailed analysis."""

    def __init__(self) -> None:
        """Initialize the comparator."""
        self.m_console: Console | None = None
        self.m_base_path: Path | None = None
        self.m_models: list[ModelInfo] | None = None
        self.m_model_data: dict[str, dict[str, Any]] | None = None

    @classmethod
    def from_directory(cls, base_path: Path) -> 'SMPLModelComparator':
        """Create comparator from model directory."""
        instance = cls()
        instance.set_base_path(base_path)
        instance.set_console(Console())
        return instance

    def set_base_path(self, path: Path) -> None:
        """Set the base path for model files."""
        self.m_base_path = path

    def set_console(self, console: Console) -> None:
        """Set the console for output."""
        self.m_console = console

    @property
    def console(self) -> Console:
        """Get the console."""
        if self.m_console is None:
            self.m_console = Console()
        return self.m_console

    @property
    def base_path(self) -> Path:
        """Get the base path."""
        if self.m_base_path is None:
            self.m_base_path = Path('data/body_models')
        return self.m_base_path

    def load_pickle_model(self, filepath: Path) -> dict[str, Any]:
        """Load a pickle model file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f, encoding='latin1')
        return model_data

    def load_npz_model(self, filepath: Path) -> dict[str, Any]:
        """Load an NPZ model file."""
        npz_data = np.load(filepath, allow_pickle=True)
        model_data = {}
        for key in npz_data.files:
            data = npz_data[key]
            if data.dtype == object:
                model_data[key] = data.item() if data.size == 1 else data
            else:
                model_data[key] = data
        return model_data

    def extract_model_info(self, model_name: str, filepath: Path,
                          model_data: dict[str, Any]) -> ModelInfo:
        """Extract information from a model."""
        # Get file info
        file_format = filepath.suffix[1:]  # Remove the dot
        file_size_mb = filepath.stat().st_size / (1024 * 1024)

        # Extract vertex count
        vertices = 0
        if 'v_template' in model_data:
            vertices = model_data['v_template'].shape[0]

        # Extract face count
        faces = 0
        if 'f' in model_data:
            faces = model_data['f'].shape[0]

        # Extract joint count
        joints = 0
        if 'J' in model_data:
            joints = model_data['J'].shape[0]
        elif 'J_regressor' in model_data:
            joints = model_data['J_regressor'].shape[0]
        elif 'kintree_table' in model_data:
            joints = model_data['kintree_table'].shape[1]

        # Extract shape parameters
        shape_params = 0
        if 'shapedirs' in model_data:
            shape_params = model_data['shapedirs'].shape[-1]

        # Extract pose parameters
        pose_params = 0
        if 'posedirs' in model_data:
            pose_params = model_data['posedirs'].shape[-1]

        # Check for hands
        has_hands = any('hand' in k.lower() for k in model_data)

        # Check for face (SMPLX specific)
        has_face = any(k in model_data for k in ['lmk_faces_idx', 'dynamic_lmk_faces_idx'])

        return ModelInfo(
            name=model_name,
            filepath=filepath,
            vertices=vertices,
            faces=faces,
            joints=joints,
            shape_params=shape_params,
            pose_params=pose_params,
            has_hands=has_hands,
            has_face=has_face,
            file_format=file_format,
            file_size_mb=round(file_size_mb, 2)
        )

    def load_models(self) -> None:
        """Load all models for comparison."""
        models_to_load = [
            ('MANO_Left', self.base_path / 'mano_v1_2/models/MANO_LEFT.pkl', 'pickle'),
            ('MANO_Right', self.base_path / 'mano_v1_2/models/MANO_RIGHT.pkl', 'pickle'),
            ('MANO_SMPLH', self.base_path / 'mano_v1_2/models/SMPLH_male.pkl', 'pickle'),
            ('SMPLH_NPZ', self.base_path / 'smplh/male/model.npz', 'npz'),
            ('SMPLH_PKL', self.base_path / 'smplh/SMPLH_MALE.pkl', 'pickle'),
            ('SMPLX_NPZ', self.base_path / 'smplx/SMPLX_MALE.npz', 'npz'),
            ('SMPLX_PKL', self.base_path / 'smplx/SMPLX_MALE.pkl', 'pickle'),
        ]

        self.m_models = []
        self.m_model_data = {}

        for model_name, filepath, file_type in track(models_to_load,
                                                     description="Loading models..."):
            if not filepath.exists():
                self.console.print(f"[yellow]Warning: {filepath} not found[/yellow]")
                continue

            # Load model data
            if file_type == 'pickle':
                model_data = self.load_pickle_model(filepath)
            else:
                model_data = self.load_npz_model(filepath)

            # Extract model info
            model_info = self.extract_model_info(model_name, filepath, model_data)
            self.m_models.append(model_info)
            self.m_model_data[model_name] = model_data

    def create_comparison_table(self) -> Table:
        """Create a rich comparison table."""
        table = Table(title="SMPL Model Series Comparison", box=box.ROUNDED)

        # Add columns
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Format", style="magenta")
        table.add_column("Size (MB)", justify="right", style="yellow")
        table.add_column("Vertices", justify="right", style="green")
        table.add_column("Faces", justify="right", style="green")
        table.add_column("Joints", justify="right", style="blue")
        table.add_column("Shape", justify="right", style="red")
        table.add_column("Pose", justify="right", style="red")
        table.add_column("Hands", justify="center")
        table.add_column("Face", justify="center")

        # Add rows
        for model in self.m_models:
            table.add_row(
                model.name,
                model.file_format.upper(),
                str(model.file_size_mb),
                f"{model.vertices:,}",
                f"{model.faces:,}",
                str(model.joints),
                str(model.shape_params),
                str(model.pose_params),
                "✓" if model.has_hands else "✗",
                "✓" if model.has_face else "✗"
            )

        return table

    def create_parameter_table(self) -> Table:
        """Create a table showing parameter details."""
        table = Table(title="Model Parameters Detail", box=box.MINIMAL_HEAVY_HEAD)

        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Key Parameters", style="yellow")
        table.add_column("Dimensions", style="green")

        for model_name, model_data in self.m_model_data.items():
            key_params = []
            dimensions = []

            # Check for important parameters
            params_to_check = [
                'v_template', 'f', 'weights', 'shapedirs', 'posedirs',
                'J', 'J_regressor', 'kintree_table'
            ]

            for param in params_to_check:
                if param in model_data:
                    key_params.append(param)
                    data = model_data[param]
                    if hasattr(data, 'shape'):
                        dimensions.append(str(data.shape))
                    else:
                        dimensions.append(str(type(data).__name__))

            # Join parameters for display
            param_str = "\n".join(key_params[:5])  # Show first 5
            dim_str = "\n".join(dimensions[:5])

            table.add_row(model_name, param_str, dim_str)

        return table

    def create_evolution_tree(self) -> Tree:
        """Create a tree showing model evolution."""
        tree = Tree("SMPL Model Evolution")

        # SMPL base
        smpl = tree.add("[bold blue]SMPL[/bold blue] (2015)")
        smpl.add("6,890 vertices")
        smpl.add("23 joints")
        smpl.add("10 shape parameters")

        # MANO branch
        mano = tree.add("[bold green]MANO[/bold green] (2017)")
        mano.add("778 vertices per hand")
        mano.add("16 joints per hand")
        mano.add("45 PCA components")

        # SMPLH branch
        smplh = tree.add("[bold yellow]SMPLH[/bold yellow] (2017)")
        smplh.add("SMPL + MANO")
        smplh.add("6,890 vertices")
        smplh.add("52 joints")

        # SMPLX branch
        smplx = tree.add("[bold red]SMPLX[/bold red] (2019)")
        smplx.add("SMPL + MANO + FLAME")
        smplx.add("10,475 vertices")
        smplx.add("55 joints")
        smplx.add("400 shape parameters")

        return tree

    def create_summary_panel(self) -> Panel:
        """Create a summary panel."""
        summary_text = Text()

        summary_text.append("Key Findings:\n", style="bold underline")
        summary_text.append("• MANO: Specialized hand model with 778 vertices\n")
        summary_text.append("• SMPLH: Body + hands with 6,890 vertices\n")
        summary_text.append("• SMPLX: Body + hands + face with 10,475 vertices\n\n")

        summary_text.append("Shared Components:\n", style="bold underline")
        summary_text.append("• v_template (template vertices)\n")
        summary_text.append("• f (face connectivity)\n")
        summary_text.append("• weights (skinning weights)\n")
        summary_text.append("• posedirs (pose blend shapes)\n")
        summary_text.append("• kintree_table (kinematic tree)\n")

        return Panel(summary_text, title="Summary", border_style="green")

    def display_comparison(self) -> None:
        """Display the full comparison."""
        self.console.clear()

        # Header
        self.console.print(Panel.fit(
            "[bold magenta]SMPL Model Series Analysis[/bold magenta]\n"
            "[dim]Comparing MANO, SMPLH, and SMPLX models[/dim]",
            border_style="blue"
        ))

        # Load models
        self.load_models()

        # Display comparison table
        self.console.print(self.create_comparison_table())
        self.console.print()

        # Display parameter details
        self.console.print(self.create_parameter_table())
        self.console.print()

        # Display evolution tree and summary side by side
        self.console.print(Columns([
            Panel(self.create_evolution_tree(), title="Evolution", border_style="blue"),
            self.create_summary_panel()
        ]))

    def run_tests(self) -> None:
        """Run comparison tests."""
        assert self.m_base_path is not None, "Base path not set"
        assert self.m_base_path.exists(), f"Base path {self.m_base_path} does not exist"

        # Load models
        self.load_models()

        # Basic assertions
        assert len(self.m_models) > 0, "No models loaded"

        # Check MANO models
        mano_models = [m for m in self.m_models if 'MANO' in m.name and 'SMPLH' not in m.name]
        for model in mano_models:
            assert model.vertices == 778, f"MANO {model.name} should have 778 vertices"
            assert model.joints == 16, f"MANO {model.name} should have 16 joints"

        # Check SMPLH models
        smplh_models = [m for m in self.m_models if 'SMPLH' in m.name and 'MANO' not in m.name]
        for model in smplh_models:
            assert model.vertices == 6890, f"SMPLH {model.name} should have 6890 vertices"
            assert model.joints in [24, 52], f"SMPLH {model.name} should have 24 or 52 joints"

        # Check SMPLX models
        smplx_models = [m for m in self.m_models if 'SMPLX' in m.name]
        for model in smplx_models:
            assert model.vertices == 10475, f"SMPLX {model.name} should have 10475 vertices"
            assert model.joints == 55, f"SMPLX {model.name} should have 55 joints"
            assert model.shape_params == 400, f"SMPLX {model.name} should have 400 shape params"

        self.console.print("[green]✓ All tests passed![/green]")


def test_model_comparison() -> None:
    """Test function for pytest."""
    comparator = SMPLModelComparator.from_directory(Path('data/body_models'))
    comparator.run_tests()


def main() -> None:
    """Main function for standalone execution."""
    console = Console()

    try:
        # Create comparator
        comparator = SMPLModelComparator.from_directory(Path('data/body_models'))

        # Display comparison
        comparator.display_comparison()

        # Run tests
        console.print("\n[bold]Running Tests...[/bold]")
        comparator.run_tests()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
