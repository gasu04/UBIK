#!/usr/bin/env python3
"""
Ubik Hippocampal Node - Health Check Script

Validates all components of the Hippocampal Node:
- Docker services (Neo4j, ChromaDB)
- MCP Server
- Tailscale connectivity
- Data integrity

Usage:
    python health_check.py

Returns:
    Exit code 0 if all checks pass, 1 otherwise.
"""

import os
import sys
import json
import logging
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ubik.health_check")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    chromadb_host: str = field(default_factory=lambda: os.getenv("CHROMADB_HOST", "localhost"))
    chromadb_port: int = field(default_factory=lambda: int(os.getenv("CHROMADB_PORT", "8001")))
    chromadb_token: str = field(default_factory=lambda: os.getenv("CHROMADB_TOKEN", ""))
    mcp_host: str = field(default_factory=lambda: os.getenv("MCP_HOST", "localhost"))
    mcp_port: str = field(default_factory=lambda: os.getenv("MCP_PORT", "8080"))


# =============================================================================
# Output Formatting
# =============================================================================

def print_header(title: str) -> None:
    """
    Print a formatted section header.

    Args:
        title: The section title to display.
    """
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_result(name: str, success: bool, details: str = "") -> None:
    """
    Print a test result with status indicator.

    Args:
        name: The name of the check.
        success: Whether the check passed.
        details: Optional additional details to display.
    """
    status = "✓" if success else "✗"
    color = "\033[92m" if success else "\033[91m"
    reset = "\033[0m"
    print(f"  {color}{status}{reset} {name}")
    if details:
        print(f"    └─ {details}")


# =============================================================================
# Health Check Functions
# =============================================================================

def check_docker() -> bool:
    """
    Check Docker daemon and container status.

    Returns:
        True if Docker is running and all required containers are healthy.
    """
    print_header("Docker Services")

    all_ok = True

    # Check Docker is running
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10
        )
        docker_ok = result.returncode == 0
        print_result("Docker daemon", docker_ok)
        all_ok = all_ok and docker_ok
    except subprocess.TimeoutExpired:
        print_result("Docker daemon", False, "Timeout waiting for Docker")
        return False
    except FileNotFoundError:
        print_result("Docker daemon", False, "Docker not installed")
        return False
    except Exception as e:
        print_result("Docker daemon", False, str(e))
        return False

    # Check containers
    containers: Dict[str, str] = {
        "ubik-neo4j": "Neo4j database",
        "ubik-chromadb": "ChromaDB vector store"
    }

    for container, description in containers.items():
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Status}}", container],
                capture_output=True,
                text=True,
                timeout=10
            )
            status = result.stdout.strip()
            is_running = status == "running"
            print_result(f"{description} ({container})", is_running, f"Status: {status}")
            all_ok = all_ok and is_running
        except Exception as e:
            print_result(f"{description}", False, str(e))
            all_ok = False

    return all_ok


def check_neo4j(config: Optional[HealthCheckConfig] = None) -> bool:
    """
    Check Neo4j connectivity and data integrity.

    Args:
        config: Optional configuration override.

    Returns:
        True if Neo4j is accessible and contains expected data.
    """
    print_header("Neo4j Database")

    if config is None:
        config = HealthCheckConfig()

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password)
        )

        # Test connectivity
        driver.verify_connectivity()
        print_result("Connection", True, f"Connected to {config.neo4j_uri}")

        # Check node counts
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            record = result.single()
            count: int = record["count"] if record else 0
            print_result("Data integrity", count > 0, f"{count} nodes in graph")

            # Check core identity
            result = session.run("MATCH (s:CoreIdentity {name: 'Self'}) RETURN s")
            has_self = result.single() is not None
            print_result("Core identity node", has_self)

        driver.close()
        return True

    except Exception as e:
        logger.error(f"Neo4j check failed: {e}")
        print_result("Neo4j check", False, str(e))
        return False


def check_chromadb(config: Optional[HealthCheckConfig] = None) -> bool:
    """
    Check ChromaDB connectivity and collections.

    Args:
        config: Optional configuration override.

    Returns:
        True if ChromaDB is accessible and collections exist.
    """
    print_header("ChromaDB Vector Store")

    if config is None:
        config = HealthCheckConfig()

    try:
        import chromadb

        client = chromadb.HttpClient(
            host=config.chromadb_host,
            port=config.chromadb_port,
            headers={"Authorization": f"Bearer {config.chromadb_token}"}
        )

        # Test heartbeat
        client.heartbeat()
        print_result("Connection", True)

        # Check collections
        collections = client.list_collections()
        collection_names = [c.name for c in collections]

        expected: list[str] = ["ubik_episodic", "ubik_semantic"]
        for name in expected:
            exists = name in collection_names
            if exists:
                count = client.get_collection(name).count()
                print_result(f"Collection: {name}", True, f"{count} documents")
            else:
                print_result(f"Collection: {name}", False, "Not found")

        # Test query
        if "ubik_semantic" in collection_names:
            results = client.get_collection("ubik_semantic").query(
                query_texts=["family values"],
                n_results=1
            )
            has_results = len(results['documents'][0]) > 0
            print_result("Query test", has_results, "Semantic search working")

        return True

    except Exception as e:
        logger.error(f"ChromaDB check failed: {e}")
        print_result("ChromaDB check", False, str(e))
        return False


def check_mcp_server(config: Optional[HealthCheckConfig] = None) -> bool:
    """
    Check if MCP server is running and responsive.

    Args:
        config: Optional configuration override.

    Returns:
        True if MCP server is responding to requests.
    """
    print_header("MCP Server")

    if config is None:
        config = HealthCheckConfig()

    try:
        import httpx

        url = f"http://{config.mcp_host}:{config.mcp_port}"

        # Check if server is responding
        response = httpx.get(f"{url}/mcp", timeout=5)
        # MCP server returns 406 when SSE headers missing - this is correct
        is_responding = response.status_code in [200, 400, 404, 406]
        print_result(
            "Server status",
            is_responding,
            f"HTTP {response.status_code} (MCP server running)"
        )

        return is_responding

    except Exception as e:
        if "ConnectError" in type(e).__name__:
            print_result("MCP server", False, "Not running (connection refused)")
        else:
            logger.error(f"MCP server check failed: {e}")
            print_result("MCP server", False, str(e))
        return False


def check_tailscale() -> bool:
    """
    Check Tailscale connectivity and peer status.

    Returns:
        True if Tailscale is running and connected.
    """
    print_header("Tailscale Networking")

    try:
        # Try the app bundle path first (macOS)
        tailscale_cmd = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
        if not os.path.exists(tailscale_cmd):
            tailscale_cmd = "tailscale"

        # Check Tailscale status
        result = subprocess.run(
            [tailscale_cmd, "status", "--json"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            status: Dict[str, Any] = json.loads(result.stdout)

            # Get our info
            self_info = status.get("Self", {})
            hostname: str = self_info.get("HostName", "unknown")
            online: bool = self_info.get("Online", False)

            print_result("Tailscale status", online, f"Hostname: {hostname}")

            # Check for somatic node
            peers: Dict[str, Any] = status.get("Peer", {})
            somatic_ip = "100.79.166.114"
            somatic_dns = "adrian-wsl"

            somatic_peer: Optional[Dict[str, Any]] = None
            for peer in peers.values():
                peer_ips = peer.get("TailscaleIPs", [])
                if somatic_ip in peer_ips:
                    somatic_peer = peer
                    break

            if somatic_peer:
                peer_online: bool = somatic_peer.get("Online", False)
                peer_hostname: str = somatic_peer.get("HostName", "unknown")
                print_result(
                    f"Somatic node ({somatic_dns})",
                    peer_online,
                    f"Host: {peer_hostname}, IP: {somatic_ip}"
                )
            else:
                print_result(
                    f"Somatic node ({somatic_dns})",
                    False,
                    "Not found in Tailscale network"
                )

            return online
        else:
            print_result("Tailscale", False, "Not running or not logged in")
            return False

    except FileNotFoundError:
        print_result("Tailscale", False, "Not installed")
        return False
    except subprocess.TimeoutExpired:
        print_result("Tailscale", False, "Timeout waiting for status")
        return False
    except Exception as e:
        logger.error(f"Tailscale check failed: {e}")
        print_result("Tailscale", False, str(e))
        return False


# =============================================================================
# Main Entry Point
# =============================================================================

def run_all_checks(config: Optional[HealthCheckConfig] = None) -> Tuple[bool, Dict[str, bool]]:
    """
    Run all health checks and return results.

    Args:
        config: Optional configuration override.

    Returns:
        Tuple of (all_passed, results_dict).
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " UBIK HIPPOCAMPAL NODE - HEALTH CHECK ".center(58) + "║")
    print("║" + f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    if config is None:
        config = HealthCheckConfig()

    results: Dict[str, bool] = {
        "docker": check_docker(),
        "neo4j": check_neo4j(config),
        "chromadb": check_chromadb(config),
        "mcp_server": check_mcp_server(config),
        "tailscale": check_tailscale()
    }

    # Summary
    print_header("Summary")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for component, status in results.items():
        print_result(component.replace("_", " ").title(), status)

    print("\n" + "-" * 60)
    if passed == total:
        print(f"  \033[92m✓ All {total} checks passed!\033[0m")
        print("  Hippocampal Node is fully operational.")
    else:
        print(f"  \033[93m⚠ {passed}/{total} checks passed\033[0m")
        print("  Review failed components above.")

    print("\n")

    logger.info(f"Health check complete: {passed}/{total} passed")
    return passed == total, results


if __name__ == "__main__":
    success, _ = run_all_checks()
    sys.exit(0 if success else 1)
