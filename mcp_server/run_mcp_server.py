import os
import contextlib
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp_server_motherduck import build_application


DB = os.environ.get("DB_PATH", "data/synthetic_openflights.db")
PORT = int(os.environ.get("PORT", "8080"))

server, _ = build_application(db_path=DB, read_only=True)
sess = StreamableHTTPSessionManager(app=server, event_store=None, stateless=True)


async def handler(scope, receive, send):
    await sess.handle_request(scope, receive, send)


@contextlib.asynccontextmanager
async def lifespan(app):
    async with sess.run():
        yield


app = Starlette(routes=[Mount("/mcp", app=handler)], lifespan=lifespan)


if __name__ == "__main__":
    print(f"MCP endpoint → http://0.0.0.0:{PORT}/mcp")
    uvicorn.run(app, host="0.0.0.0", port=PORT)


