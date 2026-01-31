#!/usr/bin/env python3
"""
Ubik RAG Status Report

Check ChromaDB collections and display available embeddings for the DeepSeek model.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)

# Import settings from config module
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings

# Load defaults from settings (no hardcoded IPs)
_settings = get_settings()
DEFAULT_CHROMA_HOST = _settings.hippocampal.host
DEFAULT_CHROMA_PORT = _settings.hippocampal.chroma_port
DEFAULT_MCP_PORT = _settings.hippocampal.mcp_port


class RAGStatusChecker:
    def __init__(self, host: str, chroma_port: int, mcp_port: int):
        self.host = host
        self.chroma_port = chroma_port
        self.mcp_port = mcp_port
        self.chroma_base = f"http://{host}:{chroma_port}/api/v2"
        self.mcp_base = f"http://{host}:{mcp_port}"
        self.collections = {}

    def check_connectivity(self) -> dict:
        """Check connectivity to ChromaDB and MCP server."""
        status = {
            "chromadb": False,
            "mcp_server": False,
            "chromadb_error": None,
            "mcp_error": None,
        }

        # Check ChromaDB
        try:
            url = f"{self.chroma_base}/tenants/default_tenant/databases/default_database/collections"
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            status["chromadb"] = True
        except requests.exceptions.RequestException as e:
            status["chromadb_error"] = str(e)

        # Check MCP server
        try:
            resp = requests.get(f"{self.mcp_base}/health", timeout=5)
            if resp.status_code == 200:
                status["mcp_server"] = True
            else:
                status["mcp_error"] = f"HTTP {resp.status_code}"
        except requests.exceptions.RequestException as e:
            status["mcp_error"] = str(e)

        return status

    def get_collections(self) -> list:
        """Get all ChromaDB collections."""
        try:
            url = f"{self.chroma_base}/tenants/default_tenant/databases/default_database/collections"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            collections = resp.json()
            for col in collections:
                self.collections[col["name"]] = col
            return collections
        except requests.exceptions.RequestException as e:
            print(f"Error fetching collections: {e}")
            return []

    def get_collection_count(self, collection_id: str) -> int:
        """Get the number of embeddings in a collection."""
        try:
            url = f"{self.chroma_base}/tenants/default_tenant/databases/default_database/collections/{collection_id}/count"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return int(resp.text)
        except requests.exceptions.RequestException:
            return -1

    def get_collection_items(self, collection_id: str, limit: int = 100) -> dict:
        """Get items from a collection."""
        try:
            url = f"{self.chroma_base}/tenants/default_tenant/databases/default_database/collections/{collection_id}/get"
            resp = requests.post(
                url,
                json={"limit": limit, "include": ["documents", "metadatas"]},
                timeout=30
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def print_report(self, verbose: bool = False, show_content: bool = False):
        """Print the full RAG status report."""
        print("=" * 70)
        print("UBIK RAG STATUS REPORT")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        # Connectivity
        print("\n## Connectivity\n")
        conn = self.check_connectivity()

        chroma_status = "\033[92mONLINE\033[0m" if conn["chromadb"] else "\033[91mOFFLINE\033[0m"
        mcp_status = "\033[92mONLINE\033[0m" if conn["mcp_server"] else "\033[91mOFFLINE\033[0m"

        print(f"  Hippocampal Node:  {self.host}")
        print(f"  ChromaDB:          {chroma_status} (port {self.chroma_port})")
        if conn["chromadb_error"]:
            print(f"                     Error: {conn['chromadb_error']}")
        print(f"  MCP Server:        {mcp_status} (port {self.mcp_port})")
        if conn["mcp_error"]:
            print(f"                     Error: {conn['mcp_error']}")

        if not conn["chromadb"]:
            print("\n\033[91mCannot retrieve collections - ChromaDB is offline\033[0m")
            return

        # Collections
        print("\n## Collections\n")
        collections = self.get_collections()

        if not collections:
            print("  No collections found.")
            return

        total_embeddings = 0
        collection_stats = []

        for col in collections:
            count = self.get_collection_count(col["id"])
            total_embeddings += max(0, count)
            dimension = col.get("dimension", "unknown")
            desc = col.get("metadata", {}).get("description", "No description")
            collection_stats.append({
                "name": col["name"],
                "id": col["id"],
                "count": count,
                "dimension": dimension,
                "description": desc,
            })

        # Summary table
        print(f"  {'Collection':<20} {'Count':>8} {'Dimension':>10}  Description")
        print(f"  {'-' * 20} {'-' * 8} {'-' * 10}  {'-' * 30}")
        for stat in collection_stats:
            count_str = str(stat["count"]) if stat["count"] >= 0 else "error"
            print(f"  {stat['name']:<20} {count_str:>8} {stat['dimension']:>10}  {stat['description'][:40]}")

        print(f"\n  Total embeddings: {total_embeddings}")

        # Detailed content
        if verbose or show_content:
            for stat in collection_stats:
                print(f"\n## {stat['name']} ({stat['count']} items)\n")

                items = self.get_collection_items(stat["id"])
                if "error" in items:
                    print(f"  Error: {items['error']}")
                    continue

                documents = items.get("documents", [])
                metadatas = items.get("metadatas", [])
                ids = items.get("ids", [])

                if not documents:
                    print("  No documents found.")
                    continue

                # Group by type
                by_type = {}
                for i, doc in enumerate(documents):
                    meta = metadatas[i] if i < len(metadatas) else {}
                    item_type = meta.get("type", "unknown")
                    if item_type not in by_type:
                        by_type[item_type] = []
                    by_type[item_type].append({
                        "id": ids[i] if i < len(ids) else "?",
                        "content": doc,
                        "metadata": meta,
                    })

                for item_type, items_list in by_type.items():
                    print(f"  ### {item_type} ({len(items_list)} items)\n")
                    for item in items_list:
                        meta = item["metadata"]
                        content = item["content"]

                        # Truncate content for display
                        if len(content) > 80 and not show_content:
                            content = content[:77] + "..."

                        # Format metadata
                        meta_parts = []
                        for key in ["category", "emotional_valence", "confidence", "importance", "stability"]:
                            if key in meta:
                                meta_parts.append(f"{key}={meta[key]}")

                        print(f"    - {content}")
                        if meta_parts:
                            print(f"      [{', '.join(meta_parts)}]")
                    print()

        # RAG readiness
        print("\n## RAG Readiness\n")
        if conn["chromadb"] and total_embeddings > 0:
            print("  \033[92m[OK]\033[0m ChromaDB has embeddings available")
        else:
            print("  \033[91m[X]\033[0m No embeddings in ChromaDB")

        if conn["mcp_server"]:
            print("  \033[92m[OK]\033[0m MCP server is running - RAG queries available")
        else:
            print("  \033[93m[!]\033[0m MCP server offline - direct ChromaDB access only")
            print("      Start the Hippocampal MCP server to enable RAG queries")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Check Ubik RAG status and available embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic status check
  python rag_status.py

  # Verbose output with memory details
  python rag_status.py -v

  # Show full content of memories
  python rag_status.py --content

  # Custom host/port
  python rag_status.py --host 192.168.1.100 --chroma-port 8001
"""
    )

    parser.add_argument("--host", type=str, default=DEFAULT_CHROMA_HOST,
                        help=f"Hippocampal node host (default: {DEFAULT_CHROMA_HOST})")
    parser.add_argument("--chroma-port", type=int, default=DEFAULT_CHROMA_PORT,
                        help=f"ChromaDB port (default: {DEFAULT_CHROMA_PORT})")
    parser.add_argument("--mcp-port", type=int, default=DEFAULT_MCP_PORT,
                        help=f"MCP server port (default: {DEFAULT_MCP_PORT})")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed memory contents")
    parser.add_argument("--content", action="store_true",
                        help="Show full content (not truncated)")

    args = parser.parse_args()

    checker = RAGStatusChecker(args.host, args.chroma_port, args.mcp_port)
    checker.print_report(verbose=args.verbose, show_content=args.content)


if __name__ == "__main__":
    main()
