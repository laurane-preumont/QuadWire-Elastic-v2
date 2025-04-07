import ast
import os
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class CodeAnalyzer:
    def __init__(self, root_dir, specific_paths=None):
        """
        Initialize the analyzer with root directory and optional path restrictions
        specific_paths: list of paths/files to analyze (relative to root_dir)
        """
        self.root_dir = Path(root_dir)
        self.specific_paths = specific_paths or []
        self.dependencies = {}
        self.functions = {}
        self.imports = {}

    def is_path_allowed(self, file_path):
        """Check if the file path should be analyzed based on restrictions."""
        if not self.specific_paths:
            return True

        # Convert file_path to relative path from root_dir
        rel_path = file_path.relative_to(self.root_dir)

        # Check if the file or its parent folders match any specific paths
        for allowed_path in self.specific_paths:
            allowed_path = Path(allowed_path)
            if str(rel_path) == str(allowed_path):  # Direct file match
                return True
            if any(parent == allowed_path for parent in rel_path.parents):  # Folder match
                return True
        return False

    def analyze_file(self, file_path):
        """Analyze a single Python file for functions and their dependencies."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        try:
            tree = ast.parse(content)
        except SyntaxError:
            print(f"Syntax error in {file_path}")
            return

        module_name = file_path.stem
        self.functions[module_name] = []
        self.imports[module_name] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self.functions[module_name].append(node.name)
                function_calls = []

                # Analyze function body for calls
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Call):
                        if isinstance(subnode.func, ast.Name):
                            function_calls.append(subnode.func.id)

                self.dependencies[f"{module_name}.{node.name}"] = function_calls

            elif isinstance(node, ast.Import):
                for name in node.names:
                    self.imports[module_name].append(name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self.imports[module_name].append(node.module)

    def analyze_directory(self):
        """Analyze Python files in the directory based on path restrictions."""
        for file_path in self.root_dir.rglob("*.py"):
            if not any(part.startswith('.') for part in file_path.parts):  # Skip hidden directories
                if self.is_path_allowed(file_path):
                    print(f"Analyzing: {file_path.relative_to(self.root_dir)}")
                    self.analyze_file(file_path)

    def generate_dependency_graph(self, output_file='dependencies.png'):
        """Generate a visual dependency graph using networkx."""
        G = nx.DiGraph()

        # Add nodes for all functions
        for module, funcs in self.functions.items():
            for func in funcs:
                G.add_node(f"{module}.{func}")

        # Add edges for dependencies
        for func, calls in self.dependencies.items():
            for call in calls:
                # Try to find the full path of the called function
                target = None
                for module, funcs in self.functions.items():
                    if call in funcs:
                        target = f"{module}.{call}"
                        break

                if target:
                    G.add_edge(func, target)

        # Generate the visualization
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=1, iterations=50)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=2000, font_size=8, font_weight='bold',
                arrows=True, edge_color='gray', arrowsize=20)

        plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
        plt.close()

    def print_summary(self):
        """Print a summary of the code structure."""
        print("\nCode Structure Summary:")
        print("=" * 50)

        for module in self.functions:
            print(f"\nModule: {module}")
            print("-" * 30)
            print("Functions:")
            for func in self.functions[module]:
                print(f"  - {func}")
                if f"{module}.{func}" in self.dependencies:
                    calls = self.dependencies[f"{module}.{func}"]
                    if calls:
                        print("    Calls:", ", ".join(calls))

            print("\nImports:")
            for imp in self.imports[module]:
                print(f"  - {imp}")

# Define the paths you want to analyze
specific_paths = [
    "qw_additive.py",
    "modules",
    "shape"
]

# Initialize the analyzer with your project directory and specific paths
analyzer = CodeAnalyzer(
    "C:/Users/preumont/OneDrive/Documents/GitHub/QuadWire",
    specific_paths=specific_paths
)

# Analyze the specified files
analyzer.analyze_directory()

# Generate visual graph
analyzer.generate_dependency_graph()

# Print summary
analyzer.print_summary()

# %%import os
import os
import ast
import networkx as nx
from typing import Dict, List, Set, Tuple
import numpy as np

class CodeAnalyzer:
    def __init__(self, base_path: str, target_files: List[str], target_folders: List[str]):
        self.base_path = base_path
        self.target_files = target_files
        self.target_folders = target_folders
        self.dependencies = nx.DiGraph()
        self.module_info = {}
        self.failed_files = []

    def get_python_files(self) -> List[str]:
        """Collect all relevant Python files based on specified targets."""
        python_files = []

        # Add specific files
        for file in self.target_files:
            full_path = os.path.join(self.base_path, file)
            if os.path.exists(full_path):
                python_files.append(full_path)

        # Add files from target folders
        for folder in self.target_folders:
            folder_path = os.path.join(self.base_path, folder)
            if os.path.exists(folder_path):
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        if file.endswith('.py'):
                            python_files.append(os.path.join(root, file))

        return python_files

    def analyze_file(self, file_path: str) -> Tuple[Set[str], Dict]:
        """Analyze a single Python file for imports and structure."""
        # Initialize empty info dictionary with default values
        info = {
            'classes': [],
            'functions': [],
            'docstring': None,
            'status': 'success'
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            info['docstring'] = ast.get_docstring(tree) or ''

            imports = set()

            # Helper function to check if a node is inside a class definition
            def is_in_class(node):
                parent = getattr(node, 'parent', None)
                while parent:
                    if isinstance(parent, ast.ClassDef):
                        return True
                    parent = getattr(parent, 'parent', None)
                return False

            # Add parent references to all nodes
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    child.parent = parent

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imports.add(name.name)
                    else:
                        if node.module:
                            imports.add(node.module)

                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                        'docstring': ast.get_docstring(node) or ''
                    }
                    info['classes'].append(class_info)

                elif isinstance(node, ast.FunctionDef) and not is_in_class(node):
                    func_info = {
                        'name': node.name,
                        'docstring': ast.get_docstring(node) or ''
                    }
                    info['functions'].append(func_info)

            return imports, info

        except Exception as e:
            rel_path = os.path.relpath(file_path, self.base_path)
            self.failed_files.append((rel_path, str(e)))
            info['status'] = 'failed'
            return set(), info

    def analyze_codebase(self):
        """Analyze all relevant Python files in the codebase."""
        python_files = self.get_python_files()

        for file_path in python_files:
            rel_path = os.path.relpath(file_path, self.base_path)
            imports, info = self.analyze_file(file_path)

            if info['status'] == 'success':
                self.module_info[rel_path] = info
                self.dependencies.add_node(rel_path)

                for imp in imports:
                    if not imp.startswith('.'):
                        self.dependencies.add_edge(rel_path, imp)

    def generate_readme(self) -> str:
        """Generate a README with the analysis results."""
        readme = "# Code Structure Analysis\n\n"

        # Project Overview
        readme += "## Project Overview\n\n"
        readme += f"This analysis covers {len(self.module_info)} Python files "
        readme += f"from the following locations:\n"
        readme += f"- Files: {', '.join(self.target_files)}\n"
        readme += f"- Folders: {', '.join(self.target_folders)}\n\n"

        # Failed Files
        if self.failed_files:
            readme += "### Analysis Failures\n\n"
            readme += "The following files could not be analyzed due to syntax errors:\n\n"
            for file_path, error in self.failed_files:
                readme += f"- **{file_path}**: {error}\n"
            readme += "\n"

        # Module Details
        readme += "## Module Details\n\n"
        for module, info in self.module_info.items():
            readme += f"### {module}\n\n"

            docstring = info.get('docstring', '')
            if docstring:
                readme += f"{docstring}\n\n"

            classes = info.get('classes', [])
            if classes:
                readme += "#### Classes\n\n"
                for class_info in classes:
                    readme += f"- **{class_info['name']}**\n"
                    if class_info.get('docstring'):
                        readme += f"  - {class_info['docstring']}\n"
                    if class_info.get('methods'):
                        readme += "  - Methods: " + ", ".join(class_info['methods']) + "\n"
                readme += "\n"

            functions = info.get('functions', [])
            if functions:
                readme += "#### Functions\n\n"
                for func_info in functions:
                    readme += f"- **{func_info['name']}**\n"
                    if func_info.get('docstring'):
                        readme += f"  - {func_info['docstring']}\n"
                readme += "\n"

        # Dependencies
        readme += "## Dependencies\n\n"
        readme += "### Internal Dependencies\n\n"
        for source, target in self.dependencies.edges():
            if not target.startswith(('os', 'sys', 'numpy', 'scipy')):
                readme += f"- {source} → {target}\n"

        readme += "\n### External Dependencies\n\n"
        external_deps = set()
        for _, target in self.dependencies.edges():
            if target.startswith(('os', 'sys', 'numpy', 'scipy')):
                external_deps.add(target)
        for dep in sorted(external_deps):
            readme += f"- {dep}\n"

        return readme

def main():
    base_path = "C:/Users/preumont/OneDrive/Documents/GitHub/QuadWire"
    target_files = ["qw_additive.py"]
    target_folders = ["modules", "shape"]

    analyzer = CodeAnalyzer(base_path, target_files, target_folders)
    analyzer.analyze_codebase()

    readme_content = analyzer.generate_readme()

    # Save README
    with open('code_analysis_readme.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)

if __name__ == "__main__":
    main()

