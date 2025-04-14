import streamlit as st
import asyncio
import json
import os
from typing import Any, Dict, List
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import httpx

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="MCP SQLite Chat", 
    page_icon="💬", 
    layout="wide"
)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Configuration class to handle environment variables and server config
class Configuration:
    """Manages configuration and environment variables."""
    
    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        
    def load_config(self, file_path: str) -> dict:
        """Load server configuration from JSON file."""
        with open(file_path, "r") as f:
            return json.load(f)
        
    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key."""
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        return self.api_key


class Tool:
    """Represents a tool with its properties."""
    
    def __init__(self, name: str, description: str, input_schema: dict) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        
    def format_for_llm(self) -> str:
        """Format tool information for LLM."""
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = f"- {param_name}: {param_info.get('description', 'No description')}"
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)
                
        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class Server:
    """Manages MCP server connection and tool execution."""
    
    def __init__(self, name: str, config: dict) -> None:
        self.name = name
        self.config = config
        self.session = None
        self.exit_stack = AsyncExitStack()
        
    async def initialize(self) -> None:
        """Initialize the server connection."""
        with st.spinner(f"Initializing server: {self.name}..."):
            server_params = StdioServerParameters(
                command=self.config["command"],
                args=self.config["args"],
                env={**os.environ, **self.config["env"]} if self.config.get("env") else None,
            )
            
            try:
                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                read, write = stdio_transport
                session = await self.exit_stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()
                self.session = session
                st.success(f"Successfully initialized server: {self.name}")
            except Exception as e:
                st.error(f"Error initializing server {self.name}: {e}")
                await self.cleanup()
                raise
            
    async def list_tools(self) -> List[Tool]:
        """List available tools from the server."""
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")
            
        tools_response = await self.session.list_tools()
        tools = []
        
        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))
                    
        return tools
        
    async def execute_tool(self, tool_name: str, arguments: dict) -> Any:
        """Execute a tool."""
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")
            
        try:
            with st.spinner(f"Executing tool: {tool_name}..."):
                result = await self.session.call_tool(tool_name, arguments)
                return result
        except Exception as e:
            st.error(f"Error executing tool: {e}")
            raise
            
    async def cleanup(self) -> None:
        """Clean up server resources."""
        try:
            await self.exit_stack.aclose()
            self.session = None
        except Exception as e:
            st.error(f"Error during cleanup of server {self.name}: {e}")


class OpenAIClient:
    """Handles communication with OpenAI-compatible API."""
    
    def __init__(self, api_key: str, model_name: str = "openai/gpt-4o-mini") -> None:
        self.api_key = api_key
        self.model_name = model_name
        
    async def get_response(self, messages: List[Dict[str, str]], tools=None) -> dict:
        """Get a response from the LLM."""
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        payload = {
            "messages": messages,
            "model": self.model_name,
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1,
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload, timeout=30.0)
                response.raise_for_status()
                return response.json()
        except httpx.RequestError as e:
            st.error(f"Error getting LLM response: {e}")
            if isinstance(e, httpx.HTTPStatusError):
                st.error(f"Status code: {e.response.status_code}")
                st.error(f"Response details: {e.response.text}")
            return {"error": str(e)}


class ChatSession:
    """Manages the chat interaction between user, LLM, and tools."""
    
    def __init__(self, server: Server, openai_client: OpenAIClient) -> None:
        self.server = server
        self.openai_client = openai_client
        
    async def process_tool_calls(self, tool_calls, messages):
        """Process tool calls from the LLM response."""
        if not tool_calls:
            return messages, None
            
        # Only process the first tool call
        tool_call = tool_calls[0]
        function_call = tool_call.get("function", {})
        tool_name = function_call.get("name")
        args_str = function_call.get("arguments", "{}")
        
        try:
            args = json.loads(args_str)
            # Create a descriptive message about the tool being used
            tool_call_desc = f"🛠️ *Using tool: {tool_name}* with parameters: {json.dumps(args, indent=2)}"
            with st.chat_message("assistant"):
                st.write(tool_call_desc)
            
            # Add tool call message to chat history
            st.session_state.messages.append({"role": "assistant", "content": tool_call_desc})
            
            result = await self.server.execute_tool(tool_name, args)
            result_str = str(result)
            
            # Only show tool results if debug mode is enabled
            if st.session_state.get("debug_mode", False):
                with st.chat_message("system"):
                    st.text(f"Tool result:\n{result_str[:1000]}")
                    if len(result_str) > 1000:
                        st.text("... (truncated)")
            
            # Add the assistant's message with the tool call to actual message list for the API
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call.get("id", "call_1"),
                        "type": "function", 
                        "function": {
                            "name": tool_name,
                            "arguments": args_str
                        }
                    }
                ]
            })
            
            # Add the tool response to actual message list for the API
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id", "call_1"),
                "content": result_str
            })
            
            return messages, result_str
            
        except Exception as e:
            st.error(f"Error executing tool {tool_name}: {e}")
            error_msg = f"Error executing tool: {str(e)}"
            
            # Add error message to context
            messages.append({
                "role": "system",
                "content": error_msg
            })
            
            return messages, error_msg
    
    async def get_available_tools(self):
        """Get available tools formatted for OpenAI API."""
        tools = await self.server.list_tools()
        
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            })
        
        return tools, openai_tools
            
    async def process_message(self, user_input, system_prompt=""):
        """Process a new user message and get a response."""
        try:
            # Get available tools
            raw_tools, openai_tools = await self.get_available_tools()
            
            # Prepare system message with tools description
            tools_description = "\n".join([tool.format_for_llm() for tool in raw_tools])
            if not system_prompt:
                system_prompt = (
                    "You are a helpful assistant with access to these tools:\n\n"
                    f"{tools_description}\n\n"
                    "Use these tools to provide helpful responses. If no tool is needed, reply directly.\n\n"
                    "IMPORTANT: Only one tool can be called per user message. Choose the most appropriate tool."
                )
            
            # Prepare messages for the API
            api_messages = [{"role": "system", "content": system_prompt}]
            
            # Add all previous messages (excluding system messages)
            for msg in st.session_state.messages:
                if msg["role"] != "system":
                    api_messages.append(msg)
            
            # Get LLM response
            with st.spinner("Getting response..."):
                response = await self.openai_client.get_response(api_messages, openai_tools)
            
            if "error" in response:
                error_msg = f"Error: {response['error']}"
                st.error(error_msg)
                return error_msg
                
            # Process response
            choice = response["choices"][0]
            message = choice["message"]
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])
            
            # Check if there's more than one tool call
            if tool_calls and len(tool_calls) > 1:
                st.warning("Multiple tool calls detected. Only processing the first one.")
                tool_calls = [tool_calls[0]]
            
            # Process any tool calls
            if tool_calls:
                with st.spinner("Processing tool calls..."):
                    updated_messages, tool_result = await self.process_tool_calls(tool_calls, api_messages.copy())
                    
                    if tool_result:
                        # Get a new response with the tool result
                        with st.spinner("Getting final response..."):
                            response = await self.openai_client.get_response(updated_messages)
                            
                            if "error" in response:
                                error_msg = f"Error: {response['error']}"
                                st.error(error_msg)
                                return error_msg
                                
                            content = response["choices"][0]["message"].get("content", "")
            
            # Display and return the final response
            return content
                    
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            return error_msg


async def init_chat_session():
    """Initialize the chat session and server."""
    try:
        # Load configuration
        config = Configuration()
        server_config = config.load_config("servers_config.json")
        api_key = config.llm_api_key
        
        # Check if we have a valid database path
        db_path = server_config["mcpServers"]["sqlite"]["args"][1]
        if not os.path.exists(db_path):
            st.error(f"Database file not found at: {db_path}")
            st.info("Please check the 'servers_config.json' file and make sure the database path is correct.")
            return None, None
        
        # Create server
        server = Server("sqlite", server_config["mcpServers"]["sqlite"])
        
        # Initialize the server
        await server.initialize()
        
        # Create OpenAI client
        openai_client = OpenAIClient(api_key)
        
        # Create chat session
        chat_session = ChatSession(server, openai_client)
        
        return server, chat_session
        
    except FileNotFoundError:
        st.error("Error: servers_config.json not found.")
        st.info("Please make sure the 'servers_config.json' file exists in the current directory.")
    except KeyError:
        st.error("Error: SQLite server configuration not found in config file.")
        st.info("Please check the format of your 'servers_config.json' file.")
    except Exception as e:
        st.error(f"Error: {e}")
    
    return None, None


async def main():
    st.title("MCP SQLite Chat")
    
    # Sidebar for configuration
    with st.sidebar:
        st.title("Configuration")
        
        # Custom system prompt input (optional)
        system_prompt = st.text_area(
            "System Prompt (optional):", 
            value="You are a helpful assistant with access to a SQLite database. Help the user query and analyze the database content.",
            height=100
        )
        
        # Debug mode toggle
        if "debug_mode" not in st.session_state:
            st.session_state.debug_mode = False
        
        debug_mode = st.checkbox("Debug Mode (Show Tool Results)", value=st.session_state.debug_mode)
        if debug_mode != st.session_state.debug_mode:
            st.session_state.debug_mode = debug_mode
            st.rerun()
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Display introduction
    if not st.session_state.messages:
        st.markdown("""
        ## Welcome to MCP SQLite Chat!
        
        This app allows you to have a conversation with an AI about your SQLite database.
        You can ask questions about the database schema, query data, and more.
        
        ### Example Queries:
        - "What tables are in this database?"
        - "Show me the schema of the Customers table"
        - "How many albums are there in total?"
        - "Show me the artists with the most albums"
        - "List all employees hired before 2010"
        
        ### Tool Usage:
        When the AI needs to access the database, you'll see a message showing which tool is being used.
        The actual database results are processed behind the scenes to keep the interface clean.
        
        *Start chatting below!*
        """)
    
    # Initialize server and chat session
    server, chat_session = await init_chat_session()
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if chat_session:
        user_input = st.chat_input("Ask about your database...")
        
        if user_input:
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Get AI response
            ai_response = await chat_session.process_message(user_input, system_prompt)
            
            # Display AI response
            with st.chat_message("assistant"):
                st.markdown(ai_response)
            
            # Add AI response to chat history
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
        # Cleanup server when done
        if server:
            await server.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
