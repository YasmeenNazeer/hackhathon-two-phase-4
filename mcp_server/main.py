from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List

from sqlmodel import Session # Import Session

import sys
import os
# Add the parent directory to the path so we can import from backend
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sdk_mock import TaskMCPServer, FunctionDeclaration, CallToolResult, TextContent, McpBaseTool
from mcp_tools import ALL_TOOLS, TOOL_IMPLEMENTATIONS

# Import get_session from the backend database module
from backend.app.database import get_session

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await task_mcp_server.connect()
    yield
    # Shutdown
    await task_mcp_server.cleanup()

app = FastAPI(lifespan=lifespan)
task_mcp_server = TaskMCPServer(session_factory=get_session) # Pass the session_factory here

# Register all tools with the MCP server
for tool_def in ALL_TOOLS:
    task_mcp_server.register_tool(tool_def, TOOL_IMPLEMENTATIONS[tool_def.name])

class MCPRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]

class MCPResponse(BaseModel):
    results: CallToolResult

@app.post("/mcp_call", response_model=MCPResponse)
async def mcp_call_endpoint(
    request_data: MCPRequest,
    session: Session = Depends(get_session) # Get session from dependency
):
    """
    Endpoint for an external agent to call tools hosted by this MCP server.
    """
    try:
        # Pass the session to the call_tool method
        results = await task_mcp_server.call_tool(request_data.tool_name, request_data.arguments, session)
        # Ensure the results conform to expected schema by cleaning any problematic fields
        cleaned_results = clean_result_for_response(results)
        return MCPResponse(results=cleaned_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def clean_result_for_response(results):
    """
    Clean the result to remove any fields that might cause schema validation issues
    """
    if hasattr(results, 'content') and isinstance(results.content, list):
        cleaned_content = []
        for item in results.content:
            if hasattr(item, '__dict__'):
                # Create a new object without the 'title' field if it exists
                item_dict = item.__dict__.copy()
                item_dict.pop('title', None)  # Remove 'title' field if it exists
                # Recreate the object with cleaned data
                cleaned_item = TextContent(**{k: v for k, v in item_dict.items() if k != 'title'})
                cleaned_content.append(cleaned_item)
            else:
                cleaned_content.append(item)
        # Return a new CallToolResult with cleaned content
        return CallToolResult(content=cleaned_content)
    return results

@app.get("/mcp_tools", response_model=List[McpBaseTool])
async def list_mcp_tools():
    """
    Endpoint to list all available tools on this MCP server.
    """
    return await task_mcp_server.list_tools()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
