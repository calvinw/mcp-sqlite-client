import asyncio
import json
import os
from typing import Any, Dict, List
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import httpx


class Configuration:
    """Manages configuration and environment variables."""
    
    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        load_dotenv()  # Load environment variables from .env file
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
        print(f"Initializing server: {self.name}")
        print(f"Command: {self.config['command']}")
        print(f"Args: {self.config['args']}")
        
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
            print(f"Successfully initialized server: {self.name}")
        except Exception as e:
            print(f"Error initializing server {self.name}: {e}")
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
            print(f"Executing tool: {tool_name}")
            result = await self.session.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            print(f"Error executing tool: {e}")
            raise
            
    async def cleanup(self) -> None:
        """Clean up server resources."""
        try:
            await self.exit_stack.aclose()
            self.session = None
            print(f"Cleaned up server: {self.name}")
        except Exception as e:
            print(f"Error during cleanup of server {self.name}: {e}")


class OpenAIClient:
    """Handles communication with OpenAI-compatible API."""
    
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        
    async def get_response(self, messages: List[Dict[str, str]], tools=None) -> dict:
        """Get a response from the LLM."""
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        payload = {
            "messages": messages,
            "model": "openai/gpt-4o-mini",
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
            print(f"Error getting LLM response: {e}")
            if isinstance(e, httpx.HTTPStatusError):
                print(f"Status code: {e.response.status_code}")
                print(f"Response details: {e.response.text}")
            return {"error": str(e)}


class ChatSession:
    """Manages the interaction between user, LLM, and tools."""
    
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
            print(f"Calling tool: {tool_name} with args: {args}")
            
            result = await self.server.execute_tool(tool_name, args)
            result_str = str(result)
            
            # Add the assistant's message with the tool call
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
            
            # Add the tool response
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id", "call_1"),
                "content": result_str
            })
            
            return messages, result_str
            
        except Exception as e:
            print(f"Error executing tool {tool_name}: {e}")
            error_msg = f"Error executing tool: {str(e)}"
            
            # Add error message to context
            messages.append({
                "role": "system",
                "content": error_msg
            })
            
            return messages, error_msg
        
    async def run(self) -> None:
        """Run the chat session."""
        try:
            # Initialize server
            await self.server.initialize()
            
            # Get available tools
            tools = await self.server.list_tools()
            print(f"Available tools: {[tool.name for tool in tools]}")
            
            # Format tools for OpenAI API
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
            
            # Prepare system message
            tools_description = "\n".join([tool.format_for_llm() for tool in tools])
            system_message = (
                "You are a helpful assistant with access to these tools:\n\n"
                f"{tools_description}\n\n"
                "Use these tools to provide helpful responses. If no tool is needed, reply directly.\n\n"
                "IMPORTANT: Only one tool can be called per user message. Choose the most appropriate tool."
            )
            
            messages = [{"role": "system", "content": system_message}]
            
            print("\nWelcome to the MCP SQLite Chat!")
            print("Type 'exit' to quit.")
            
            while True:
                try:
                    # Get user input
                    user_input = input(f"\nUSER:").strip()
                    if user_input.lower() in ["exit", "quit"]:
                        break
                        
                    # Add user message
                    messages.append({"role": "user", "content": user_input})
                    
                    # Get LLM response
                    response = await self.openai_client.get_response(messages, openai_tools)
                    
                    if "error" in response:
                        print(f"Error: {response['error']}")
                        continue
                        
                    # Process response
                    choice = response["choices"][0]
                    message = choice["message"]
                    content = message.get("content", "")
                    tool_calls = message.get("tool_calls", [])
                    
                    # Check if there's more than one tool call
                    if tool_calls and len(tool_calls) > 1:
                        print("Warning: Multiple tool calls detected. Only processing the first one.")
                        tool_calls = [tool_calls[0]]
                    
                    # Process any tool calls
                    if tool_calls:
                        updated_messages, tool_result = await self.process_tool_calls(tool_calls, messages.copy())
                        
                        if tool_result:
                            # Get a new response with the tool result
                            messages = updated_messages
                            response = await self.openai_client.get_response(messages)
                            
                            if "error" in response:
                                print(f"Error: {response['error']}")
                                continue
                                
                            content = response["choices"][0]["message"].get("content", "")
                            messages.append({"role": "assistant", "content": content})
                    else:
                        # No tool calls, just add the assistant response
                        messages.append({"role": "assistant", "content": content})
                    
                    # Print the response
                    print(f"\nAI: {content}")
                    
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"\nError: {e}")
                    
        finally:
            # Clean up
            await self.server.cleanup()


async def main() -> None:
    """Initialize and run the chat session."""
    print("Starting MCP SQLite client...")
    
    try:
        # Load configuration
        config = Configuration()
        server_config = config.load_config("servers_config.json")
        
        # Create server
        server = Server("sqlite", server_config["mcpServers"]["sqlite"])
        
        # Create OpenAI client
        openai_client = OpenAIClient(config.llm_api_key)
        
        # Create and run chat session
        chat_session = ChatSession(server, openai_client)
        await chat_session.run()
        
    except FileNotFoundError:
        print("Error: servers_config.json not found.")
    except KeyError:
        print("Error: SQLite server configuration not found in config file.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
