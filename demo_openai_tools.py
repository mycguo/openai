"""
Streamlit app demonstrating OpenAI built-in tools:
- Web Search (built-in web_search tool)
- File Search (built-in file_search tool)
- Code Interpreter (built-in code_interpreter tool)
- MCP (Model Context Protocol) integration concepts

This app demonstrates how to use OpenAI's built-in tools via Responses API.
The Responses API replaces the deprecated Assistants API.

USAGE:
    streamlit run demo_openai_tools.py

PREREQUISITES:
    - Python 3.8+
    - OpenAI API key set in .env file (OPENAI_API_KEY)
    - Required packages: openai, python-dotenv, streamlit
"""

import os
import json
import time
import tempfile
import warnings
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="OpenAI Built-in Tools Demo",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    api_key = st.secrets["OPENAI_API_KEY"]
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY not found in environment variables!")
        st.stop()
    return OpenAI(api_key=api_key)

client = get_openai_client()


def call_responses_api(input_text, tools, model="gpt-4o", use_assistants_fallback=True):
    """Call OpenAI Responses API with error handling and fallback to Assistants API."""
    
    # First, try adding container parameter to tools if needed
    tools_with_container = []
    for tool in tools:
        if isinstance(tool, dict):
            tool_copy = tool.copy()
            # Add container parameter if it's a built-in tool
            if tool_copy.get("type") in ["web_search", "file_search", "code_interpreter"]:
                if "container" not in tool_copy:
                    tool_copy["container"] = {}
            tools_with_container.append(tool_copy)
        else:
            tools_with_container.append(tool)
    
    # Try Responses API first
    try:
        response = client.responses.create(
            model=model,
            input=input_text,
            tools=tools_with_container
        )
        if hasattr(response, 'output_text'):
            return response.output_text, None
        return str(response), None
    except AttributeError:
        # Try beta Responses API
        try:
            response = client.beta.responses.create(
                model=model,
                input=input_text,
                tools=tools_with_container
            )
            if hasattr(response, 'output_text'):
                return response.output_text, None
            return str(response), None
        except Exception as e:
            if use_assistants_fallback:
                # Fallback to Assistants API
                return call_assistants_api(input_text, tools, model)
            return None, f"Responses API error: {str(e)}"
    except Exception as e:
        error_str = str(e)
        # If container error, try with empty container
        if "container" in error_str.lower():
            try:
                tools_fixed = []
                for tool in tools:
                    if isinstance(tool, dict):
                        tool_fixed = tool.copy()
                        tool_fixed["container"] = {}
                        tools_fixed.append(tool_fixed)
                    else:
                        tools_fixed.append(tool)
                
                response = client.responses.create(
                    model=model,
                    input=input_text,
                    tools=tools_fixed
                )
                if hasattr(response, 'output_text'):
                    return response.output_text, None
                return str(response), None
            except Exception as e2:
                if use_assistants_fallback:
                    return call_assistants_api(input_text, tools, model)
                return None, f"Error with container: {str(e2)}"
        else:
            if use_assistants_fallback:
                return call_assistants_api(input_text, tools, model)
            return None, f"Error: {error_str}"


def call_assistants_api(input_text, tools, model="gpt-4o"):
    """Fallback to Assistants API if Responses API is not available."""
    # Suppress deprecation warnings for Assistants API (used as fallback)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*Assistants API.*")
        
        try:
            # Filter tools - Assistants API only supports: code_interpreter, function, file_search
            # web_search is not supported in Assistants API
            supported_tools = []
            has_web_search = False
            
            for tool in tools:
                if isinstance(tool, dict):
                    tool_type = tool.get("type")
                    if tool_type == "web_search":
                        has_web_search = True
                        # Skip web_search - not supported in Assistants API
                        continue
                    elif tool_type in ["code_interpreter", "file_search", "function"]:
                        supported_tools.append(tool)
                    else:
                        # Try to include it anyway
                        supported_tools.append(tool)
                else:
                    supported_tools.append(tool)
            
            # If web_search was requested but not supported, add a note to instructions
            instructions = "You are a helpful assistant."
            if has_web_search:
                instructions += " Note: Web search is not available in Assistants API. Please provide information based on your training data."
            
            # Create assistant with supported tools only
            assistant = client.beta.assistants.create(
                name="Demo Assistant",
                instructions=instructions,
                model=model,
                tools=supported_tools if supported_tools else None
            )
            
            # Create thread
            thread = client.beta.threads.create()
            
            # Add message
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=input_text
            )
            
            # Run assistant
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id
            )
            
            # Wait for completion
            start_time = time.time()
            while True:
                run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
                if run.status == "completed":
                    break
                elif run.status == "failed":
                    client.beta.assistants.delete(assistant.id)
                    return None, f"Run failed: {run.last_error}"
                if time.time() - start_time > 60:
                    client.beta.assistants.delete(assistant.id)
                    return None, "Run timed out"
                time.sleep(1)
            
            # Get response
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            response_text = ""
            for message in messages.data:
                if message.role == "assistant":
                    for content in message.content:
                        if hasattr(content, 'text'):
                            response_text += content.text.value + "\n"
            
            # Cleanup
            client.beta.assistants.delete(assistant.id)
            
            return response_text.strip(), None
            
        except Exception as e:
            return None, f"Assistants API error: {str(e)}"


def demo_web_search(custom_query=None):
    """Demonstrate web search using Responses API with built-in web_search tool."""
    st.subheader("🌐 Web Search Tool")
    st.markdown("Demonstrates OpenAI's built-in `web_search` tool via Responses API.")
    
    query = custom_query or "What are the latest developments in AI in 2025?"
    
    with st.expander("Query Details", expanded=False):
        st.code(query)
    
    if st.button("🔍 Run Web Search", type="primary", key="web_search_btn"):
        with st.spinner("Searching the web..."):
            response_text, error = call_responses_api(
                query,
                tools=[{"type": "web_search"}]
            )
            
            if error:
                st.error(f"❌ Error: {error}")
                st.info("💡 Tip: The Responses API structure may differ. Check OpenAI's latest documentation.")
            else:
                st.success("✅ Web search completed!")
                st.markdown("### Response:")
                st.markdown(response_text)
                return response_text
    
    return None


def demo_file_search(uploaded_file=None):
    """Demonstrate file search using Responses API with built-in file_search tool."""
    st.subheader("📁 File Search Tool")
    st.markdown("Demonstrates OpenAI's built-in `file_search` tool via Responses API.")
    
    if uploaded_file is None:
        st.info("📤 Please upload a file to search through.")
        uploaded_file = st.file_uploader(
            "Upload a file",
            type=['txt', 'md', 'pdf', 'docx'],
            key="file_search_upload"
        )
    
    if uploaded_file:
        query = st.text_input(
            "Search query",
            value="What are the key points in this document?",
            key="file_search_query"
        )
        
        if st.button("🔍 Search File", type="primary", key="file_search_btn"):
            with st.spinner("Uploading file and searching..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Upload to OpenAI
                    with open(tmp_path, "rb") as f:
                        openai_file = client.files.create(
                            file=f,
                            purpose="file_search"
                        )
                    
                    # Search the file
                    response_text, error = call_responses_api(
                        query,
                        tools=[{"type": "file_search", "file_id": openai_file.id}]
                    )
                    
                    # Cleanup
                    client.files.delete(openai_file.id)
                    os.unlink(tmp_path)
                    
                    if error:
                        st.error(f"❌ Error: {error}")
                        if "Assistants API" in error:
                            st.info("💡 Note: Falling back to Assistants API (deprecated but still functional)")
                    else:
                        st.success("✅ File search completed!")
                        st.markdown("### Response:")
                        st.markdown(response_text)
                        
                        with st.expander("File Info", expanded=False):
                            st.write(f"**File:** {uploaded_file.name}")
                            st.write(f"**Size:** {uploaded_file.size} bytes")
                            st.write(f"**Type:** {uploaded_file.type}")
                        
                        return response_text
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    return None
    
    return None


def demo_file_operations(custom_query=None):
    """Demonstrate file operations using code_interpreter tool."""
    st.subheader("💻 File Operations (Code Interpreter)")
    st.markdown("Demonstrates OpenAI's built-in `code_interpreter` tool for file operations.")
    
    default_query = """Please do the following:
1. Create a file called 'demo_output.txt' with the content 'Hello from OpenAI built-in tools demo!'
2. Read the file back and tell me what it contains"""
    
    query = custom_query or default_query
    
    with st.expander("Query Details", expanded=False):
        st.code(query)
    
    if st.button("🔧 Run File Operations", type="primary", key="file_ops_btn"):
        with st.spinner("Executing code..."):
            response_text, error = call_responses_api(
                query,
                tools=[{"type": "code_interpreter"}]
            )
            
            if error:
                st.error(f"❌ Error: {error}")
                if "Assistants API" in error:
                    st.info("💡 Note: Falling back to Assistants API (deprecated but still functional)")
            else:
                st.success("✅ File operations completed!")
                st.markdown("### Response:")
                st.markdown(response_text)
                
                # Check if file was created
                if os.path.exists("demo_output.txt"):
                    st.info("📄 File created: `demo_output.txt`")
                    with open("demo_output.txt", "r") as f:
                        st.code(f.read(), language="text")
                
                return response_text
    
    return None


def demo_mcp_integration():
    """Demonstrate MCP concepts."""
    st.subheader("🔗 MCP Integration Concepts")
    st.markdown("""
    **MCP (Model Context Protocol)** allows AI models to interact with external tools.
    
    OpenAI's built-in tools (`web_search`, `file_search`, `code_interpreter`) provide
    similar functionality to what MCP servers offer.
    
    **Note:** `web_search` is only available in Responses API, not in Assistants API.
    
    **For full MCP integration with OpenAI:**
    1. Set up an MCP server (e.g., Playwright, filesystem)
    2. Connect to it via stdio or HTTP
    3. Expose its tools via function calling in Chat Completions API
    """)
    
    query = "Explain how OpenAI's built-in tools (web_search, file_search, code_interpreter) relate to MCP (Model Context Protocol)"
    
    if st.button("🔍 Explain MCP Integration", type="primary", key="mcp_btn"):
        with st.spinner("Generating explanation..."):
            response_text, error = call_responses_api(
                query,
                tools=[
                    {"type": "web_search"},
                    {"type": "code_interpreter"}
                ],
                use_assistants_fallback=False  # Don't fallback since web_search isn't supported
            )
            
            if error:
                st.error(f"❌ Error: {error}")
                st.warning("""
                **Note:** The Responses API may not be fully available yet, or `web_search` 
                requires a different API structure. 
                
                **Workaround:** Try using only `code_interpreter` tool, or check OpenAI's 
                latest documentation for Responses API availability.
                """)
                
                # Try with just code_interpreter as fallback
                st.info("🔄 Trying with code_interpreter only...")
                response_text, error2 = call_responses_api(
                    query,
                    tools=[{"type": "code_interpreter"}],
                    use_assistants_fallback=True
                )
                
                if error2:
                    st.error(f"❌ Fallback also failed: {error2}")
                else:
                    st.success("✅ Explanation generated (using code_interpreter only)!")
                    st.markdown("### Response:")
                    st.markdown(response_text)
                    return response_text
            else:
                st.success("✅ Explanation generated!")
                st.markdown("### Response:")
                st.markdown(response_text)
                return response_text
    
    return None


def demo_task_management(custom_query=None):
    """Demonstrate task management using code_interpreter."""
    st.subheader("✅ Task Management (Code Interpreter)")
    st.markdown("Demonstrates creating todo lists using the `code_interpreter` tool.")
    
    default_query = "Create a todo list file called 'todos.md' with 3 tasks: 1) Research AI trends, 2) Write a summary, 3) Review documentation"
    
    query = custom_query or default_query
    
    with st.expander("Query Details", expanded=False):
        st.code(query)
    
    if st.button("📝 Create Todo List", type="primary", key="todo_btn"):
        with st.spinner("Creating todo list..."):
            response_text, error = call_responses_api(
                query,
                tools=[{"type": "code_interpreter"}]
            )
            
            if error:
                st.error(f"❌ Error: {error}")
                if "Assistants API" in error:
                    st.info("💡 Note: Falling back to Assistants API (deprecated but still functional)")
            else:
                st.success("✅ Todo list created!")
                st.markdown("### Response:")
                st.markdown(response_text)
                
                # Check if file was created
                if os.path.exists("todos.md"):
                    st.info("📄 File created: `todos.md`")
                    with open("todos.md", "r") as f:
                        st.markdown(f.read())
                
                return response_text
    
    return None


def demo_combined_workflow(custom_query=None):
    """Demonstrate a combined workflow using multiple built-in tools."""
    st.subheader("🚀 Combined Workflow")
    st.markdown("Demonstrates using multiple built-in tools together.")
    
    default_query = """Perform the following tasks:
1. Search the web for information about Python async programming
2. Create a summary file called 'async_summary.md' with the key findings
3. Create a todo list for learning async programming"""
    
    query = custom_query or default_query
    
    with st.expander("Query Details", expanded=False):
        st.code(query)
    
    if st.button("🎯 Run Combined Workflow", type="primary", key="combined_btn"):
        with st.spinner("Running combined workflow..."):
            response_text, error = call_responses_api(
                query,
                tools=[
                    {"type": "web_search"},
                    {"type": "code_interpreter"}
                ]
            )
            
            if error:
                st.error(f"❌ Error: {error}")
                if "Assistants API" in error:
                    st.info("💡 Note: Falling back to Assistants API (deprecated but still functional)")
            else:
                st.success("✅ Combined workflow completed!")
                st.markdown("### Response:")
                st.markdown(response_text)
                
                # Check if files were created
                files_created = []
                if os.path.exists("async_summary.md"):
                    files_created.append("async_summary.md")
                if os.path.exists("todos.md"):
                    files_created.append("todos.md")
                
                if files_created:
                    st.info(f"📄 Files created: {', '.join(files_created)}")
                    for file in files_created:
                        with st.expander(f"View {file}", expanded=False):
                            with open(file, "r") as f:
                                st.markdown(f.read())
                
                return response_text
    
    return None


# Sidebar
with st.sidebar:
    st.title("🔧 OpenAI Tools Demo")
    st.markdown("---")
    
    st.markdown("### 📚 Available Demos")
    demo_options = {
        "🌐 Web Search": "web_search",
        "📁 File Search": "file_search",
        "💻 File Operations": "file_operations",
        "🔗 MCP Integration": "mcp",
        "✅ Task Management": "task_management",
        "🚀 Combined Workflow": "combined"
    }
    
    selected_demo = st.radio(
        "Select a demo:",
        options=list(demo_options.keys()),
        index=0
    )
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Settings")
    model = st.selectbox(
        "Model",
        ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0
    )
    
    st.markdown("---")
    
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app demonstrates OpenAI's built-in tools:
    - **web_search**: Search the web
    - **file_search**: Search uploaded files
    - **code_interpreter**: Execute Python code
    
    Uses the **Responses API** (replaces deprecated Assistants API).
    """)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Cache"):
        st.cache_resource.clear()
        st.rerun()


# Main content area
st.title("🔧 OpenAI Built-in Tools Demonstration")
st.markdown("""
This app demonstrates how to use OpenAI's built-in tools via the **Responses API**.

**Built-in Tools Available:**
- 🌐 **Web Search** - Search the web for up-to-date information
- 📁 **File Search** - Search through uploaded documents
- 💻 **Code Interpreter** - Execute Python code for file operations

Select a demo from the sidebar to get started!
""")

st.markdown("---")

# Display selected demo
demo_key = demo_options[selected_demo]

if demo_key == "web_search":
    st.markdown("### Custom Query (Optional)")
    custom_query = st.text_area(
        "Enter a custom search query:",
        value="What are the latest developments in AI in 2025?",
        height=100,
        key="custom_web_query"
    )
    demo_web_search(custom_query)

elif demo_key == "file_search":
    uploaded_file = st.file_uploader(
        "Upload a file to search",
        type=['txt', 'md', 'pdf', 'docx'],
        key="main_file_upload"
    )
    demo_file_search(uploaded_file)

elif demo_key == "file_operations":
    st.markdown("### Custom Query (Optional)")
    custom_query = st.text_area(
        "Enter a custom query for file operations:",
        value="""Please do the following:
1. Create a file called 'demo_output.txt' with the content 'Hello from OpenAI built-in tools demo!'
2. Read the file back and tell me what it contains""",
        height=150,
        key="custom_file_query"
    )
    demo_file_operations(custom_query)

elif demo_key == "mcp":
    demo_mcp_integration()

elif demo_key == "task_management":
    st.markdown("### Custom Query (Optional)")
    custom_query = st.text_area(
        "Enter a custom todo list query:",
        value="Create a todo list file called 'todos.md' with 3 tasks: 1) Research AI trends, 2) Write a summary, 3) Review documentation",
        height=100,
        key="custom_todo_query"
    )
    demo_task_management(custom_query)

elif demo_key == "combined":
    st.markdown("### Custom Query (Optional)")
    custom_query = st.text_area(
        "Enter a custom combined workflow query:",
        value="""Perform the following tasks:
1. Search the web for information about Python async programming
2. Create a summary file called 'async_summary.md' with the key findings
3. Create a todo list for learning async programming""",
        height=150,
        key="custom_combined_query"
    )
    demo_combined_workflow(custom_query)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>Built with Streamlit | OpenAI Responses API</small>
</div>
""", unsafe_allow_html=True)
