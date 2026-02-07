from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from pydantic import BaseModel, Field
from sqlmodel import Session # Import Session

# --- Mock for Gemini FunctionDeclaration ---
class FunctionDeclaration(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

# --- Mock for Gemini Content ---
class TextContent(BaseModel):
    text: str
    type: str = "text"

class CallToolResult(BaseModel):
    content: List[TextContent]

# --- Mock for McpBaseTool (from an assumed MCP SDK) ---
class McpBaseTool(BaseModel):
    name: str
    description: Optional[str] = None
    inputSchema: Dict[str, Any] = Field(default_factory=dict)

# --- Mock for MCPServer (abstract base class from assumed SDK) ---
class MCPServer(ABC):
    @abstractmethod
    async def connect(self):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def cleanup(self):
        pass

    @abstractmethod
    async def list_tools(self) -> list[McpBaseTool]:
        pass

    @abstractmethod
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] | None, session: Session) -> CallToolResult:
        pass

# --- A concrete implementation of a mock MCPServer for our task management ---
class TaskMCPServer(MCPServer):
    def __init__(self, name: str = "task_management_mcp_server", session_factory: Optional[Callable[[], Session]] = None):
        self._name = name
        self._tools: Dict[str, McpBaseTool] = {}
        self._tool_implementations: Dict[str, Any] = {}
        self.session_factory = session_factory

    @property
    def name(self) -> str:
        return self._name

    async def connect(self):
        print(f"{self.name} connected.")

    async def cleanup(self):
        print(f"{self.name} cleaned up.")

    def register_tool(self, tool: McpBaseTool, implementation: Any):
        self._tools[tool.name] = tool
        self._tool_implementations[tool.name] = implementation

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] | None, session: Session) -> CallToolResult:
        if tool_name not in self._tools:
            return CallToolResult(content=[TextContent(text=f"Error: Tool '{tool_name}' not found.", type="text")])

        tool_impl = self._tool_implementations[tool_name]
        try:
            # Pass the session to the tool implementation
            result = await tool_impl(session=session, **(arguments or {}))
            return CallToolResult(content=[TextContent(text=str(result), type="text")])
        except Exception as e:
            return CallToolResult(content=[TextContent(text=f"Error calling tool '{tool_name}': {e}", type="text")])

    async def list_tools(self) -> list[McpBaseTool]:
        return list(self._tools.values())