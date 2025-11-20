import datetime
import os, time, json
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
from azure.ai.agents.models import (
    ListSortOrder,
    McpTool,
    RequiredMcpToolCall,
    RunStepActivityDetails,
    SubmitToolApprovalAction,
    ToolApproval,
    FunctionTool,
    MessageRole,
    ConnectedAgentTool,
    FilePurpose,
)
from azure.ai.agents.models import AzureAISearchTool, AzureAISearchQueryType, MessageRole, ListSortOrder, ToolDefinition, FilePurpose, FileSearchTool
from azure.ai.agents.models import CodeInterpreterTool, FunctionTool, ToolSet
import requests
import streamlit as st
from datetime import datetime, timedelta
import yfinance as yf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

endpoint = os.environ["PROJECT_ENDPOINT"] # Sample : https://<account_name>.services.ai.azure.com/api/projects/<project_name>
model_endpoint = os.environ["MODEL_ENDPOINT"] # Sample : https://<account_name>.services.ai.azure.com
model_api_key= os.environ["MODEL_API_KEY"]
model_deployment_name = os.environ["MODEL_DEPLOYMENT_NAME"] # Sample : gpt-4o-mini

# Get MCP server configuration from environment variables
mcp_server_url = os.environ.get("MCP_SERVER_URL", "https://learn.microsoft.com/api/mcp")
mcp_server_label = os.environ.get("MCP_SERVER_LABEL", "MicrosoftLearn")

# Environment variables
AZURE_SUBSCRIPTION_ID = os.environ["AZURE_SUBSCRIPTION_ID"]
AZURE_RESOURCE_GROUP = os.environ["AZURE_RESOURCE_GROUP"]
# AZURE_DATA_FACTORY_NAME = os.environ["AZURE_DATA_FACTORY_NAME"]

# Create the project client (Foundry project and credentials)
project_client = AIProjectClient(
        endpoint=endpoint,
        credential=DefaultAzureCredential(),
)

client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version="2024-10-21",
)

from azure.monitor.opentelemetry import configure_azure_monitor
# connection_string = project_client.telemetry.get_application_insights_connection_string()
connection_string = os.getenv("APPLICATION_INSIGHTS_CONNECTION_STRING")

if not connection_string:
    print("Application Insights is not enabled. Enable by going to Tracing in your Azure AI Foundry project.")
    exit()

configure_azure_monitor(connection_string=connection_string) #enable telemetry collection

from opentelemetry import trace
tracer = trace.get_tracer(__name__)

def parse_and_display_json_multi(json_input):
    try:
        # Check if input is already a dictionary
        if isinstance(json_input, dict):
            data = json_input
        else:
            # Assume input is a JSON string and parse it
            data = json.loads(json_input)
        
        # Display Summary
        print("=== Construction Management Services Summary ===")
        summary_lines = data['summary'].split('\n')
        for line in summary_lines:
            if line.strip() and not line.startswith('Would you like'):
                print(line.strip())
        print()  # Add spacing
        
        # Display Token Usage
        print("=== Token Usage ===")
        token_usage = data['token_usage']
        print(f"Prompt Tokens: {token_usage['prompt_tokens']}")
        print(f"Completion Tokens: {token_usage['completion_tokens']}")
        print(f"Total Tokens: {token_usage['total_tokens']}")
        print()
        
        # Display Status
        print("=== Status ===")
        print(f"Run Status: {data['status'].capitalize()}")
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    except KeyError as e:
        print(f"Missing key in JSON data: {e}")
    except TypeError as e:
        print(f"Type error: {e}")

def multi_agent(query: str) -> str:
    returntxt = ""

    # Orchestrate the connected agent with the main agent
    agent = project_client.agents.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name="Existing_MultiAgent_Demo",
        instructions="""
        You are a helpful assistant, and use the connected agents to get stock prices, construction RFP Data, 
        Sustainability Paper.

        Question:
        {query}
        Summarize and respond to the user's query using the connected agents as needed.
        """,
        # tools=list(unique_tools.values()), #search_connected_agent.definitions,  # Attach the connected agents
        # tools=[
        #     connected_agent.definitions[0],
        #     search_connected_agent.definitions[0],
        #     mcp_connected_agent.definitions[0],
        # ]
    )

    print(f"Created agent, ID: {agent.id}")
    thread = project_client.agents.threads.create()
    print(f"Created thread, ID: {thread.id}")

    # Create message to thread
    message = project_client.agents.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        # content="What is the stock price of Microsoft?",
        content=query,
    )
    print(f"Created message, ID: {message.id}")
    # Create and process Agent run in thread with tools
    # run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)
    # print(f"Run finished with status: {run.status}")
    
    print("\n" + "="*50)
    print("STREAMING AGENT OUTPUT")
    print("="*50 + "\n")
    
    # Create run and poll with streaming output
    run = project_client.agents.runs.create(thread_id=thread.id, agent_id=agent.id)
    print(f"🟢 Run created: {run.id}")
    
    # Track which messages we've already printed
    printed_messages = set()
    
    # Poll the run status and stream messages as they appear
    while run.status in ["queued", "in_progress", "requires_action"]:
        time.sleep(0.5)  # Shorter polling interval for more responsive output
        run = project_client.agents.runs.get(thread_id=thread.id, run_id=run.id)
        
        # Show status updates
        if run.status == "queued":
            print(f"⏳ Run queued...", end='\r', flush=True)
        elif run.status == "in_progress":
            print(f"▶️  Run in progress...                    ", end='\r', flush=True)
            
            # Fetch and stream any new messages
            messages = project_client.agents.messages.list(thread_id=thread.id)
            for message in reversed(list(messages)):
                if message.id not in printed_messages and message.role == MessageRole.AGENT:
                    printed_messages.add(message.id)
                    print("\n" + "="*50)
                    print("🤖 AGENT RESPONSE:")
                    print("="*50)
                    for content in message.content:
                        if hasattr(content, 'text') and hasattr(content.text, 'value'):
                            # Stream the text character by character for effect
                            for char in content.text.value:
                                print(char, end='', flush=True)
                                time.sleep(0.01)  # Small delay for streaming effect
                            print()  # New line after content
                    print("="*50 + "\n")
        
        elif run.status == "requires_action":
            print(f"\n⚠️  Run requires action")
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            for tool_call in tool_calls:
                print(f"   Tool call needed: {tool_call.name} (ID: {tool_call.id})")
    
    # Final status
    if run.status == "completed":
        print(f"\n✅ Run completed successfully")
    elif run.status == "failed":
        print(f"\n❌ Run failed: {run.last_error}")
    
    print("\n" + "="*50 + "\n")

    print(f"Run completed with status: {run.status}")
    # print(f"Run finished with status: {run.status}")

    if run.status == "failed":
        print(f"Run failed: {run.last_error}")

    # Fetch run steps to get the details of the agent run
    run_steps = project_client.agents.run_steps.list(thread_id=thread.id, run_id=run.id)
    for step in run_steps:
        print(f"Step {step['id']} status: {step['status']}")
        step_details = step.get("step_details", {})
        tool_calls = step_details.get("tool_calls", [])

        if tool_calls:
            print("  Tool calls:")
            for call in tool_calls:
                print(f"    Tool Call ID: {call.get('id')}")
                print(f"    Type: {call.get('type')}")

                connected_agent = call.get("connected_agent", {})
                if connected_agent:
                    print(f"    Connected Input(Name of Agent): {connected_agent.get('name')}")
                    print(f"    Connected Output: {connected_agent.get('output')}")

        print()  # add an extra newline between steps

    messages = project_client.agents.messages.list(thread_id=thread.id)
    for message in messages:
        if message.role == MessageRole.AGENT:
            print(f"Role: {message.role}, Content: {message.content}")
            # returntxt += f"Role: {message.role}, Content: {message.content}\n"
            # returntxt += f"Source: {message.content[0]['text']['value']}\n"
            returntxt += f"Source: {message.content[0].text.value}\n"
    # returntxt = f"{message.content[-1].text.value}"

    

    # Token usage (if provided by SDK)
    token_usage = None
    usage = getattr(run, "usage", None)
    if usage:
        token_usage = {k: getattr(usage, k) for k in ["prompt_tokens", "completion_tokens", "total_tokens"] if hasattr(usage, k)} or None

    # delete agent and thread
    # Cleanup
    
    try:
        print(" Clean up -------------------------------------")
        project_client.agents.delete_agent(agent.id)
        project_client.agents.threads.delete(thread.id)
    except Exception:
        pass

    return {"summary": returntxt, "token_usage": token_usage, "status": run.status}

def main():
    st.set_page_config(
        page_title="AI Agent Streaming Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI Agent Streaming Demo")
    st.markdown("### Multi-Agent System with Real-time Streaming Output")
    # Get MCP server configuration from environment variables
    mcp_server_url = os.environ.get("MCP_SERVER_URL", "https://learn.microsoft.com/api/mcp")
    mcp_server_label = os.environ.get("MCP_SERVER_LABEL", "MicrosoftLearn")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This demo showcases a multi-agent system with streaming output capabilities.
        
        **Features:**
        - Real-time streaming responses
        - Multi-agent collaboration
        - Token usage tracking
        - Azure AI Foundry integration
        """)
        
        st.divider()
        st.markdown("**Example Queries:**")
        st.markdown("- What is Azure AI Foundry?")
        st.markdown("- What is quantum computing?")
        st.markdown("- Explain machine learning")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "token_usage" in message and message["token_usage"]:
                with st.expander("📊 Token Usage"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Prompt", message["token_usage"].get("prompt_tokens", "N/A"))
                    col2.metric("Completion", message["token_usage"].get("completion_tokens", "N/A"))
                    col3.metric("Total", message["token_usage"].get("total_tokens", "N/A"))
            if "duration" in message:
                st.caption(f"⏱️ Response time: {message['duration']}")
    # mcp_tool = McpTool(
    #                 server_label=mcp_server_label,
    #                 server_url=mcp_server_url,
    #                 allowed_tools=[],
    #             )
    # Initialize agent MCP tool
    mcp_tool = McpTool(
        server_label=mcp_server_label,
        server_url=mcp_server_url,
        allowed_tools=[],  # Optional: specify allowed tools
    )
    # Chat input
    if prompt := st.chat_input("Ask me anything...", disabled=st.session_state.processing):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process agent response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            status_placeholder = st.empty()
            
            st.session_state.processing = True
            
            try:
                starttime = datetime.now()
                
                # Create agent and thread
                status_placeholder.info("🔧 Creating agent and thread...")
                
                with tracer.start_as_current_span("Streaming_agent_Demo-tracing"):
                    
                    agent = project_client.agents.create_agent(
                        model=os.environ["MODEL_DEPLOYMENT_NAME"],
                        name="Existing_MultiAgent_Demo",
                        instructions=f"""
                        You are a helpful AI Assistant, Use the tools to get context.

                        Question:
                        {prompt}
                        Summarize and respond to the user's query using the connected agents as needed.
                        """,
                        tools=mcp_tool.definitions,
                    )
                    
                    print(f"\n{'='*60}")
                    print(f"🤖 AGENT CREATED")
                    print(f"{'='*60}")
                    print(f"Agent ID: {agent.id}")
                    print(f"Model: {os.environ['MODEL_DEPLOYMENT_NAME']}")
                    print(f"Tools: {len(mcp_tool.definitions)} MCP tools configured")
                    print(f"{'='*60}\n")
                    
                    thread = project_client.agents.threads.create()
                    print(f"🧵 Thread created: {thread.id}\n")
                    
                    # Create message
                    message = project_client.agents.messages.create(
                        thread_id=thread.id,
                        role=MessageRole.USER,
                        content=prompt,
                    )
                    print(f"💬 User message created: {message.id}")
                    print(f"   Content: {prompt}\n")
                    
                    # Create run and stream output
                    status_placeholder.info("▶️ Running agent...")
                    run = project_client.agents.runs.create(thread_id=thread.id, agent_id=agent.id)
                    
                    print(f"▶️  RUN STARTED")
                    print(f"   Run ID: {run.id}")
                    print(f"   Initial Status: {run.status}\n")
                    
                    printed_messages = set()
                    full_response = ""
                    displayed_length = 0
                    
                    # Poll and stream messages
                    while run.status in ["queued", "in_progress", "requires_action"]:
                        time.sleep(0.3)
                        run = project_client.agents.runs.get(thread_id=thread.id, run_id=run.id)
                        
                        if run.status == "queued":
                            status_placeholder.info("⏳ Agent queued...")
                            print("⏳ Status: queued", end='\r')
                        elif run.status == "in_progress":
                            status_placeholder.info("▶️ Agent processing...")
                            print("▶️  Status: in_progress" + " "*20, end='\r')
                            
                            # Fetch and display new messages
                            messages = project_client.agents.messages.list(thread_id=thread.id)
                            for msg in reversed(list(messages)):
                                if msg.id not in printed_messages and msg.role == MessageRole.AGENT:
                                    printed_messages.add(msg.id)
                                    print(f"\n\n{'='*60}")
                                    print(f"🤖 AGENT RESPONSE (Message ID: {msg.id})")
                                    print(f"{'='*60}")
                                    for content in msg.content:
                                        if hasattr(content, 'text') and hasattr(content.text, 'value'):
                                            full_response = content.text.value
                                            print(full_response)
                                            print(f"{'='*60}\n")
                                            
                                            # Stream character by character for new content
                                            if len(full_response) > displayed_length:
                                                for i in range(displayed_length, len(full_response)):
                                                    displayed_length = i + 1
                                                    message_placeholder.markdown(full_response[:displayed_length] + "▌")
                                                    time.sleep(0.01)
                                                message_placeholder.markdown(full_response)
                        
                        elif run.status == "requires_action":
                            status_placeholder.warning("⚠️ Agent requires action - Approving MCP tool calls...")
                            print(f"\n\n{'='*60}")
                            print(f"⚠️  REQUIRES ACTION - MCP Tool Approval Needed")
                            print(f"{'='*60}")
                            
                            # Debug: Print the entire required_action object
                            print(f"   🔍 DEBUG - Required Action Object:")
                            print(f"      Type: {type(run.required_action)}")
                            tool_calls = run.required_action.submit_tool_approval.tool_calls
                            if not tool_calls:
                                print("No tool calls provided - cancelling run")
                                project_client.runs.cancel(thread_id=thread.id, run_id=run.id)
                                break

                            tool_approvals = []
                            for tool_call in tool_calls:
                                if isinstance(tool_call, RequiredMcpToolCall):
                                    try:
                                        print(f"Approving tool call: {tool_call}")
                                        tool_approvals.append(
                                            ToolApproval(
                                                tool_call_id=tool_call.id,
                                                approve=True,
                                                headers=mcp_tool.headers,
                                            )
                                        )
                                    except Exception as e:
                                        print(f"Error approving tool_call {tool_call.id}: {e}")

                            print(f"tool_approvals: {tool_approvals}")
                            if tool_approvals:
                                project_client.agents.runs.submit_tool_outputs(
                                    thread_id=thread.id, run_id=run.id, tool_approvals=tool_approvals
                                )

                        print(f"Current run status: {run.status}")
                    
                    # Final status
                    if run.status == "completed":
                        status_placeholder.success("✅ Agent completed successfully!")
                        print(f"\n{'='*60}")
                        print(f"✅ RUN COMPLETED")
                        print(f"{'='*60}")
                    elif run.status == "failed":
                        status_placeholder.error(f"❌ Agent failed: {run.last_error}")
                        print(f"\n{'='*60}")
                        print(f"❌ RUN FAILED")
                        print(f"   Error: {run.last_error}")
                        print(f"{'='*60}")
                    
                    # Display run steps for debugging
                    print(f"\n{'='*60}")
                    print(f"📋 RUN STEPS SUMMARY")
                    print(f"{'='*60}")
                    run_steps = project_client.agents.run_steps.list(thread_id=thread.id, run_id=run.id)
                    step_count = 0
                    for step in run_steps:
                        step_count += 1
                        print(f"\n   Step {step_count}:")
                        print(f"      ID: {step.id}")
                        print(f"      Type: {step.type}")
                        print(f"      Status: {step.status}")
                        
                        if hasattr(step, 'step_details'):
                            if hasattr(step.step_details, 'tool_calls') and step.step_details.tool_calls:
                                print(f"      Tool Calls:")
                                for tool_call in step.step_details.tool_calls:
                                    print(f"         - Type: {tool_call.type}")
                                    if hasattr(tool_call, 'mcp_tool'):
                                        print(f"           MCP Tool: {tool_call.mcp_tool.tool_name}")
                                        print(f"           Server: {tool_call.mcp_tool.server_label}")
                                        if hasattr(tool_call.mcp_tool, 'output'):
                                            print(f"           Output: {tool_call.mcp_tool.output}")
                                    if hasattr(tool_call, 'function'):
                                        print(f"           Function: {tool_call.function.name}")
                                        if hasattr(tool_call.function, 'output'):
                                            print(f"           Output: {tool_call.function.output}")
                    print(f"\n{'='*60}\n")
                    
                    # Get final messages if not already captured
                    if not full_response:
                        messages = project_client.agents.messages.list(thread_id=thread.id)
                        for msg in messages:
                            if msg.role == MessageRole.AGENT:
                                for content in msg.content:
                                    if hasattr(content, 'text') and hasattr(content.text, 'value'):
                                        full_response = content.text.value
                                        break
                                break
                    
                    message_placeholder.markdown(full_response)
                    
                    # Get token usage
                    token_usage = None
                    usage = getattr(run, "usage", None)
                    if usage:
                        token_usage = {
                            k: getattr(usage, k) 
                            for k in ["prompt_tokens", "completion_tokens", "total_tokens"] 
                            if hasattr(usage, k)
                        } or None
                        
                        if token_usage:
                            print(f"📊 TOKEN USAGE:")
                            print(f"   Prompt Tokens: {token_usage.get('prompt_tokens', 'N/A')}")
                            print(f"   Completion Tokens: {token_usage.get('completion_tokens', 'N/A')}")
                            print(f"   Total Tokens: {token_usage.get('total_tokens', 'N/A')}\n")
                    
                    endtime = datetime.now()
                    duration = str(endtime - starttime)
                    print(f"⏱️  Total Duration: {duration}\n")
                    
                    # Display token usage
                    if token_usage:
                        with st.expander("📊 Token Usage"):
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Prompt", token_usage.get("prompt_tokens", "N/A"))
                            col2.metric("Completion", token_usage.get("completion_tokens", "N/A"))
                            col3.metric("Total", token_usage.get("total_tokens", "N/A"))
                    
                    st.caption(f"⏱️ Response time: {duration}")
                    
                    # Cleanup
                    try:
                        print(f"🧹 Cleaning up resources...")
                        project_client.agents.delete_agent(agent.id)
                        print(f"   ✅ Agent deleted: {agent.id}")
                        project_client.agents.threads.delete(thread.id)
                        print(f"   ✅ Thread deleted: {thread.id}")
                        print(f"{'='*60}\n")
                    except Exception as cleanup_error:
                        print(f"   ⚠️  Cleanup warning: {cleanup_error}\n")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "token_usage": token_usage,
                        "duration": duration
                    })
                    
                    status_placeholder.empty()
            
            except Exception as e:
                print(f"\n{'='*60}")
                print(f"❌ EXCEPTION OCCURRED")
                print(f"{'='*60}")
                print(f"Exception type: {type(e).__name__}")
                print(f"Exception message: {str(e)}")
                import traceback
                print(f"Full traceback:")
                print(traceback.format_exc())
                print(f"{'='*60}\n")
                st.error(f"❌ Error: {str(e)}")
            
            finally:
                st.session_state.processing = False
                st.rerun()


if __name__ == "__main__":
    main()