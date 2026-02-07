from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import httpx
from typing import Dict, Any, List, Union, Optional
from sqlmodel import Session, select
from datetime import datetime, timezone
import json
from uuid import uuid4 # Import uuid4 for tool_call_id

import google.generativeai as genai
from app.config import settings

from app.database import get_session
from app.models.chat import Conversation, Message

class MemoryEntry(BaseModel):
    content: str
    timestamp: datetime
    tags: Optional[List[str]] = None
    importance: int = 1  # 1-5 scale of importance

class RememberRequest(BaseModel):
    content: str
    tags: Optional[List[str]] = None
    importance: int = 3  # Default medium importance

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatMessage(BaseModel):
    message: str

class ToolCall(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    output: Optional[str] = None # Added output field

class ChatResponse(BaseModel):
    ai_message: str
    tool_calls: List[ToolCall] = []

class MessageDisplay(BaseModel):
    message: str
    sender: str
    created_at: datetime
    tool_calls: List[ToolCall] = []

MCP_SERVER_URL = "http://localhost:8001"

async def get_mcp_client():
    async with httpx.AsyncClient(base_url=MCP_SERVER_URL) as client:
        yield client

async def get_or_create_conversation(user_id: str, session: Session) -> Conversation:
    conversation = session.exec(
        select(Conversation).where(Conversation.user_id == user_id)
    ).first()
    if not conversation:
        conversation = Conversation(user_id=user_id)
        session.add(conversation)
        session.commit()
        session.refresh(conversation)
    return conversation


def extract_memories_from_messages(messages: List[Message], importance_threshold: int = 3) -> List[MemoryEntry]:
    """Extract important information from message history"""
    memories = []

    for msg in messages:
        # Look for important tags or message types
        if msg.tags and "important" in msg.tags.lower():
            memories.append(MemoryEntry(
                content=msg.content,
                timestamp=msg.created_at,
                tags=msg.tags.split(",") if msg.tags else [],
                importance=importance_threshold
            ))
        elif msg.message_type in ["memory", "reminder", "important"]:
            tags = msg.tags.split(",") if msg.tags else []
            memories.append(MemoryEntry(
                content=msg.content,
                timestamp=msg.created_at,
                tags=tags,
                importance=importance_threshold
            ))

    return memories


def clean_schema_for_gemini(schema):
    """Clean up the schema to remove fields that Gemini doesn't expect"""
    if isinstance(schema, dict):
        cleaned = {}
        for key, value in schema.items():
            if key == "title":  # Skip the title field that causes issues
                continue
            elif isinstance(value, dict):
                if key == "properties":
                    # Clean properties by removing 'title' fields
                    cleaned_props = {}
                    for prop_key, prop_value in value.items():
                        if isinstance(prop_value, dict):
                            cleaned_props[prop_key] = clean_schema_for_gemini(prop_value)
                        else:
                            cleaned_props[prop_key] = prop_value
                    cleaned[key] = cleaned_props
                else:
                    cleaned[key] = clean_schema_for_gemini(value)
            elif isinstance(value, list):
                cleaned[key] = [clean_schema_for_gemini(item) for item in value]
            else:
                cleaned[key] = value
        return cleaned
    return schema


# Configure and initialize Gemini client
genai.configure(api_key=settings.GEMINI_API_KEY)
GEMINI_MODEL = "gemini-1.5-flash"
gemini_client = genai.GenerativeModel(GEMINI_MODEL)

async def get_user_memories(user_id: str, session: Session, limit: int = 10) -> List[MemoryEntry]:
    """Retrieve important memories for the user"""
    try:
        conversation = session.exec(
            select(Conversation).where(Conversation.user_id == user_id)
        ).first()

        if not conversation:
            return []

        # Get recent messages that are marked as important, memories, or reminders
        memory_messages = session.exec(
            select(Message)
            .where(
                (Message.conversation_id == conversation.id) &
                ((Message.message_type.in_(["memory", "reminder", "important"])) |
                 (Message.tags.contains("important") if Message.tags else False))
            )
            .order_by(Message.created_at.desc())
            .limit(limit)
        ).all()

        memories = []
        for msg in memory_messages:
            tags = msg.tags.split(",") if msg.tags else []
            memories.append(MemoryEntry(
                content=msg.content,
                timestamp=msg.created_at,
                tags=tags,
                importance=5 if msg.message_type in ["memory", "important"] else 3
            ))

        return memories
    except Exception as e:
        print(f"Error retrieving user memories: {e}")
        return []


async def gemini_agent_response(
    user_message: str,
    conversation_history: List[Message],
    mcp_client: httpx.AsyncClient,
    user_id: str,
    session: Session
) -> Dict[str, Union[str, List[ToolCall]]]:
    ai_response_content = ""
    tool_calls_executed: List[ToolCall] = [] # To store executed tool calls and their outputs

    # 1. Get user memories to provide context
    user_memories = await get_user_memories(user_id, session)

    # 2. Fetch available tools from MCP server
    try:
        tools_response = await mcp_client.get("/mcp_tools")
        tools_response.raise_for_status()
        mcp_tools = tools_response.json()
    except httpx.RequestError as e:
        print(f"Error fetching tools from MCP server: {e}")
        return {"ai_message": "Sorry, I can't connect to the task management tools right now.", "tool_calls": []}
    except httpx.HTTPStatusError as e:
        print(f"Error fetching tools from MCP server: {e.response.status_code} - {e.response.text}")
        return {"ai_message": "Sorry, there was an issue with the task management tools.", "tool_calls": []}

    # 3. Prepare initial conversation history for Gemini API
    # Add system message with memory context
    system_context = f"You are a helpful task management assistant with memory capabilities. Use the available tools to manage tasks. All task operations require a user_id. The current user_id is '{user_id}'. Use this user_id for all tool calls unless explicitly told otherwise by the user.\n\n"

    if user_memories:
        system_context += f"Important information about the user:\n" + "\n".join([f"- {mem.content}" for mem in user_memories[:5]]) + "\n\n"

    # Prepare tools for Gemini (function declarations)
    gemini_tools = []
    for tool_def in mcp_tools:
        # Clean up the schema to remove fields that Gemini doesn't expect
        cleaned_schema = clean_schema_for_gemini(tool_def["inputSchema"])
        
        function_declaration = {
            "name": tool_def["name"],
            "description": tool_def["description"],
            "parameters": cleaned_schema
        }
        gemini_tools.append({"function_declarations": [function_declaration]})

    # Configure the model with tools
    model_with_tools = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        tools=gemini_tools
    )

    # Prepare conversation history for Gemini
    gemini_history = []

    for msg in conversation_history:
        role = "user" if msg.sender == "user" else "model"
        content = msg.content

        if msg.sender == "agent" and "Tool Calls: [" in msg.content:
            # Handle tool calls in agent responses
            parts = msg.content.split("Tool Calls: [", 1)
            text_content = parts[0].strip()

            # Add the text response part
            if text_content:
                gemini_history.append({"role": "model", "parts": [{"text": text_content}]})

            # Handle the tool calls and their outputs
            json_tool_calls_str = "[" + parts[1]
            try:
                tool_call_data_list = json.loads(json_tool_calls_str)
                for tool_call_data in tool_call_data_list:
                    # Add the function response (result of tool call)
                    if "output" in tool_call_data and tool_call_data["output"]:
                        function_response_part = {
                            "function_response": {
                                "name": tool_call_data["tool_name"],
                                "response": {
                                    "result": tool_call_data["output"]
                                }
                            }
                        }
                        gemini_history.append({"role": "function", "parts": [function_response_part]})
            except json.JSONDecodeError:
                print(f"Could not parse tool calls from message content: {msg.content}")
            except Exception as e:
                print(f"Error preparing tool calls from history for Gemini: {e}")
        else:
            # Regular message without tool calls
            gemini_history.append({"role": role, "parts": [{"text": content}]})

    # Add the current user message
    gemini_history.append({"role": "user", "parts": [{"text": user_message}]})

    # 4. Generate content with the model
    try:
        # Generate response with the full history
        response = await model_with_tools.generate_content_async(gemini_history, request_options={"timeout": 60})

        # Process the response
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]

            # Check for function calls in the response
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.function_call:
                        # This is a function call request
                        func_call = part.function_call
                        tool_name = func_call.name

                        # Convert the function call arguments to a dict
                        tool_args = {}
                        if func_call.args:
                            # Convert protobuf Struct to regular dict
                            tool_args = {key: func_call.args[key] for key in func_call.args}

                        print(f"Gemini requested tool call: {tool_name} with args {tool_args}")

                        # Execute the tool via MCP server
                        tool_output = "Error: Tool execution failed."
                        try:
                            mcp_call_response = await mcp_client.post(
                                f"{MCP_SERVER_URL}/mcp_call",
                                json={"tool_name": tool_name, "arguments": tool_args}
                            )
                            mcp_call_response.raise_for_status()
                            tool_output = mcp_call_response.json()["results"]["content"][0]["text"]
                        except httpx.RequestError as e:
                            tool_output = f"MCP server request failed: {e}"
                        except httpx.HTTPStatusError as e:
                            tool_output = f"MCP server returned error: {e.response.status_code} - {e.response.text}"
                        except Exception as e:
                            tool_output = f"Error during tool execution: {e}"

                        tool_calls_executed.append(ToolCall(
                            tool_name=tool_name,
                            arguments=tool_args,
                            output=tool_output # Store the output
                        ))

                        # Create a new chat session to continue the conversation
                        chat_session = model_with_tools.start_chat(history=gemini_history[:-1])  # Exclude last user message

                        # Send the function result back to Gemini
                        function_response_part = {
                            "function_response": {
                                "name": tool_name,
                                "response": {
                                    "result": tool_output
                                }
                            }
                        }

                        # Get final response after function call
                        final_response = await chat_session.send_message_async([function_response_part], request_options={"timeout": 60})
                        ai_response_content = ''.join([part.text for part in final_response.candidates[0].content.parts if hasattr(part, 'text') and part.text])
                        break
                    else:
                        # This is a regular text response
                        if hasattr(part, 'text') and part.text:
                            ai_response_content += part.text
            else:
                # Fallback to basic response if no content parts
                ai_response_content = str(response.text) if hasattr(response, 'text') else "I received a response but couldn't process it properly."
        else:
            ai_response_content = "No response received from Gemini API."

    except Exception as e:
        print(f"Error with Gemini API: {e}")
        ai_response_content = "I'm having trouble connecting to my AI brain. Please try again later."

    return {"ai_message": ai_response_content, "tool_calls": tool_calls_executed}


@router.post("/{user_id}/chat", response_model=ChatResponse)
async def chat(
    user_id: str,
    chat_message: ChatMessage,
    session: Session = Depends(get_session),
    mcp_client: httpx.AsyncClient = Depends(get_mcp_client)
):
    """
    This endpoint receives a message from the user, processes it using a Gemini AI agent,
    persists the conversation, and returns an AI response along with any triggered tool calls.
    """
    if not chat_message.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if len(chat_message.message) > 2000:
        raise HTTPException(status_code=400, detail="Message too long, max 2000 characters")

    try:
        conversation = await get_or_create_conversation(user_id, session)

        user_msg_db = Message(
            conversation_id=conversation.id,
            sender="user",
            content=chat_message.message,
            created_at=datetime.now(timezone.utc),
            message_type="standard"
        )
        session.add(user_msg_db)
        session.commit()
        session.refresh(user_msg_db)

        # Fetch conversation history for context
        conversation_history_db = session.exec(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at)
        ).all()

        ai_output = await gemini_agent_response(chat_message.message, conversation_history_db, mcp_client, user_id, session)
        ai_message_content = ai_output["ai_message"]
        tool_calls_executed = ai_output["tool_calls"] # Renamed for clarity


        # Persist AI's response and tool calls
        content_to_store = ai_message_content
        if tool_calls_executed:
            content_to_store += "\n\nTool Calls: " + json.dumps([tc.model_dump() for tc in tool_calls_executed])

        ai_msg_db = Message(
            conversation_id=conversation.id,
            sender="agent",
            content=content_to_store,
            created_at=datetime.now(timezone.utc),
            message_type="response"
        )
        session.add(ai_msg_db)
        session.commit()
        session.refresh(ai_msg_db)

        return ChatResponse(ai_message=ai_message_content, tool_calls=tool_calls_executed)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the error and return a generic error response
        print(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="An error occurred processing your request")

@router.get("/{user_id}/chat/history", response_model=List[MessageDisplay])
async def get_chat_history(user_id: str, session: Session = Depends(get_session)):
    """
    Retrieves the conversation history for a given user.
    """
    conversation = session.exec(
        select(Conversation).where(Conversation.user_id == user_id)
    ).first()

    if not conversation:
        return []

    messages = session.exec(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at)
    ).all()

    history_display = []
    for msg in messages:
        tool_calls_parsed: List[ToolCall] = []
        message_content = msg.content

        if "Tool Calls: [" in message_content:
            try:
                parts = message_content.split("Tool Calls: [", 1)
                message_content = parts[0].strip()
                json_tool_calls_str = "[" + parts[1]
                tool_call_data_list = json.loads(json_tool_calls_str)
                for tool_call_data in tool_call_data_list:
                    tool_calls_parsed.append(ToolCall(**tool_call_data))
            except json.JSONDecodeError:
                print(f"Could not parse tool calls from message content: {msg.content}")
            except Exception as e:
                print(f"Error extracting tool calls: {e} from {msg.content}")

        history_display.append(MessageDisplay(
            message=message_content,
            sender=msg.sender,
            created_at=msg.created_at,
            tool_calls=tool_calls_parsed
        ))

    return history_display


@router.post("/{user_id}/remember")
async def remember_information(
    user_id: str,
    remember_request: RememberRequest,
    session: Session = Depends(get_session)
):
    """
    Allows the user to explicitly ask the system to remember specific information.
    """
    if not remember_request.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    if len(remember_request.content) > 2000:
        raise HTTPException(status_code=400, detail="Content too long, max 2000 characters")

    if remember_request.importance < 1 or remember_request.importance > 5:
        raise HTTPException(status_code=400, detail="Importance must be between 1 and 5")

    conversation = await get_or_create_conversation(user_id, session)

    # Create a special message that marks this as an important memory
    memory_message = Message(
        conversation_id=conversation.id,
        sender="system",
        content=remember_request.content,
        created_at=datetime.utcnow(),
        message_type="memory",
        tags=",".join(remember_request.tags) if remember_request.tags else "important"
    )

    try:
        session.add(memory_message)
        session.commit()
        session.refresh(memory_message)
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail="Failed to save memory")

    return {"message": "Information saved to memory successfully", "id": memory_message.id}


@router.get("/{user_id}/memories", response_model=List[MemoryEntry])
async def get_memories(
    user_id: str,
    session: Session = Depends(get_session),
    importance_threshold: int = 2
):
    """
    Retrieve important memories stored for the user.
    """
    conversation = session.exec(
        select(Conversation).where(Conversation.user_id == user_id)
    ).first()

    if not conversation:
        return []

    messages = session.exec(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.desc())  # Most recent first
    ).all()

    # Extract memories based on tags, message type, and importance
    memories = extract_memories_from_messages(messages, importance_threshold)

    return memories


@router.delete("/{user_id}/memory/{message_id}")
async def forget_memory(
    user_id: str,
    message_id: int,
    session: Session = Depends(get_session)
):
    """
    Remove a specific memory entry.
    """
    message = session.get(Message, message_id)

    if not message:
        raise HTTPException(status_code=404, detail="Memory not found")

    if message.conversation.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this memory")

    session.delete(message)
    session.commit()

    return {"message": "Memory removed successfully"}
