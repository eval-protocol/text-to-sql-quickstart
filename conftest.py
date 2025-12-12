import os
import sys
import time
import socket
import subprocess
import pytest
import requests
from pathlib import Path

# Add root to path so we can import mcp_server if needed, though we run it as subprocess
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0

# @pytest.fixture(scope="session", autouse=True)
def mcp_server():
    """
    Automatically starts the MCP server for tests if not already running.
    """
    port = 8080
    mcp_url = f"http://127.0.0.1:{port}"
    
    # If already running (e.g. by user), just use it
    if is_port_open(port):
        os.environ["MCP_SERVER_URL"] = mcp_url
        yield
        return

    # Start server
    print(f"Starting MCP server on port {port}...")
    server_script = ROOT_DIR / "mcp_server" / "run_mcp_server.py"
    
    # Ensure we use the same python interpreter
    cmd = [sys.executable, str(server_script)]
    
    # Set env vars for the server
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["DB_PATH"] = str(ROOT_DIR / "data" / "synthetic_openflights.db")
    
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for ready
    start_time = time.time()
    ready = False
    while time.time() - start_time < 10:
        try:
            # The server exposes /mcp, but maybe we can just check if port is open now
            if is_port_open(port):
                ready = True
                break
        except Exception:
            pass
        time.sleep(0.5)
        
    if not ready:
        proc.terminate()
        stdout, stderr = proc.communicate()
        print(f"MCP Server failed to start:\nSTDOUT: {stdout.decode()}\nSTDERR: {stderr.decode()}")
        raise RuntimeError("Failed to start MCP server for tests")

    os.environ["MCP_SERVER_URL"] = mcp_url
    
    yield
    
    # Teardown
    print("Stopping MCP server...")
    proc.terminate()
    proc.wait()
