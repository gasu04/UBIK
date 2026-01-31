#!/usr/bin/env python3
"""
Ubik Scripts Documentation Generator

Scans the ubik project for Python scripts and generates a versioned
SCRIPTS.md documentation file in the scripts/ subdirectory.

Usage:
    python generate_scripts_doc.py
"""

import os
import re
import glob
from datetime import datetime
from pathlib import Path


UBIK_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = UBIK_ROOT / "scripts"
OUTPUT_DIR = SCRIPTS_DIR


def get_latest_version() -> str:
    """Find the latest version from existing SCRIPTS_*.md files."""
    pattern = SCRIPTS_DIR / "SCRIPTS_v*.md"
    existing = glob.glob(str(pattern))

    if not existing:
        # Check root SCRIPTS.md for version
        root_scripts = UBIK_ROOT / "SCRIPTS.md"
        if root_scripts.exists():
            with open(root_scripts) as f:
                content = f.read()
                match = re.search(r'\*\*Version:\*\*\s*(\d+)\.(\d+)\.(\d+)', content)
                if match:
                    major, minor, patch = map(int, match.groups())
                    return f"{major}.{minor}.{patch}"
        return "1.0.0"

    versions = []
    for f in existing:
        match = re.search(r'SCRIPTS_v(\d+)\.(\d+)\.(\d+)\.md', f)
        if match:
            versions.append(tuple(map(int, match.groups())))

    if versions:
        versions.sort(reverse=True)
        major, minor, patch = versions[0]
        return f"{major}.{minor}.{patch}"

    return "1.0.0"


def increment_version(version: str) -> str:
    """Increment the patch version."""
    match = re.match(r'(\d+)\.(\d+)\.(\d+)', version)
    if match:
        major, minor, patch = map(int, match.groups())
        return f"{major}.{minor}.{patch + 1}"
    return "1.0.1"


def get_script_description(filepath: Path) -> str:
    """Extract description from a Python script's docstring."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read(2000)  # Read first 2000 chars

            # Look for module docstring
            match = re.search(r'^(?:#!/.*\n)?[\'\"]{3}(.*?)[\'\"]{3}', content, re.DOTALL)
            if match:
                docstring = match.group(1).strip()
                # Get first line or sentence
                first_line = docstring.split('\n')[0].strip()
                if first_line:
                    return first_line

            return "No description available"
    except Exception:
        return "Unable to read file"


def scan_python_scripts() -> dict:
    """Scan the ubik directory for Python scripts."""
    scripts = {
        'somatic/mcp_client': [],
        'somatic/inference': [],
        'somatic/tests': [],
        'tests': [],
        'data': [],
        'other': [],
    }

    for py_file in UBIK_ROOT.rglob("*.py"):
        # Skip this script and __pycache__
        if '__pycache__' in str(py_file):
            continue
        if py_file.name == 'generate_scripts_doc.py':
            continue

        rel_path = py_file.relative_to(UBIK_ROOT)
        rel_str = str(rel_path)
        description = get_script_description(py_file)

        entry = {
            'name': py_file.name,
            'path': rel_str,
            'description': description,
        }

        # Categorize
        if rel_str.startswith('somatic/mcp_client'):
            scripts['somatic/mcp_client'].append(entry)
        elif rel_str.startswith('somatic/inference'):
            scripts['somatic/inference'].append(entry)
        elif rel_str.startswith('somatic/tests'):
            scripts['somatic/tests'].append(entry)
        elif rel_str.startswith('tests/'):
            scripts['tests'].append(entry)
        elif rel_str.startswith('data/'):
            scripts['data'].append(entry)
        else:
            scripts['other'].append(entry)

    return scripts


def generate_markdown(scripts: dict, version: str) -> str:
    """Generate the markdown content."""
    today = datetime.now().strftime('%Y-%m-%d')

    content = f"""# Ubik Project - Python Scripts Documentation

**Version:** {version}
**Generated:** {today}

---

This document lists all Python scripts in the Ubik project with their paths and descriptions.

---

"""

    sections = [
        ('somatic/mcp_client', 'MCP Client Package', 'somatic/mcp_client/'),
        ('somatic/inference', 'Inference Scripts', 'somatic/inference/'),
        ('somatic/tests', 'Test Scripts (somatic)', 'somatic/tests/'),
        ('tests', 'Root Test Scripts', 'tests/'),
        ('data', 'Cache/Data', 'data/'),
        ('other', 'Other Scripts', ''),
    ]

    total_scripts = 0
    category_counts = {}

    for key, title, path_prefix in sections:
        if not scripts[key]:
            continue

        content += f"## {title}"
        if path_prefix:
            content += f" (`{path_prefix}`)"
        content += "\n\n"

        content += "| Script | Path | Description |\n"
        content += "|--------|------|-------------|\n"

        for entry in sorted(scripts[key], key=lambda x: x['name']):
            content += f"| `{entry['name']}` | `{entry['path']}` | {entry['description']} |\n"
            total_scripts += 1

        content += "\n---\n\n"
        category_counts[title] = len(scripts[key])

    # Summary
    content += "## Summary\n\n"
    content += f"**Total Scripts:** {total_scripts}\n\n"
    content += "| Category | Count |\n"
    content += "|----------|-------|\n"

    for title, count in category_counts.items():
        content += f"| {title} | {count} |\n"

    return content


def main():
    """Main entry point."""
    print("=" * 60)
    print("Ubik Scripts Documentation Generator")
    print("=" * 60)

    # Get version
    current_version = get_latest_version()
    new_version = increment_version(current_version)

    print(f"\nCurrent version: {current_version}")
    print(f"New version:     {new_version}")

    # Scan scripts
    print("\nScanning for Python scripts...")
    scripts = scan_python_scripts()

    total = sum(len(v) for v in scripts.values())
    print(f"Found {total} Python scripts")

    # Generate markdown
    print("\nGenerating documentation...")
    markdown = generate_markdown(scripts, new_version)

    # Write output
    output_file = OUTPUT_DIR / f"SCRIPTS_v{new_version}.md"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown)

    print(f"\nGenerated: {output_file}")
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
