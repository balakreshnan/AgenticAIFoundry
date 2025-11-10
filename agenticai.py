import asyncio
from datetime import datetime
import time
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    RedTeam,
    AzureOpenAIModelConfiguration,
    AttackStrategy,
    RiskCategory,
)
import os, json
import pandas as pd
from typing import Any, Callable, Set, Dict, List, Optional
from azure.ai.agents.models import CodeInterpreterTool, FunctionTool, ToolSet
from azure.ai.projects.models import (
    EvaluatorConfiguration,
    EvaluatorIds,
)
from azure.ai.projects.models import (
    Evaluation,
    InputDataset
)
from azure.ai.evaluation import AIAgentConverter, IntentResolutionEvaluator
from azure.ai.evaluation import (
    ToolCallAccuracyEvaluator,
    AzureOpenAIModelConfiguration,
    IntentResolutionEvaluator,
    TaskAdherenceEvaluator,
    ResponseCompletenessEvaluator,
    ContentSafetyEvaluator,
    RelevanceEvaluator,
    CoherenceEvaluator,
    GroundednessEvaluator,
    FluencyEvaluator,
    SimilarityEvaluator,
    ViolenceEvaluator,
    SexualEvaluator,
    SelfHarmEvaluator,
    HateUnfairnessEvaluator,
    RetrievalEvaluator,
    BleuScoreEvaluator, GleuScoreEvaluator, RougeScoreEvaluator, MeteorScoreEvaluator, RougeType,
    ProtectedMaterialEvaluator, IndirectAttackEvaluator, RetrievalEvaluator, GroundednessProEvaluator,
    F1ScoreEvaluator,
    AzureOpenAIPythonGrader
)
from pprint import pprint
# specific to agentic workflows
from azure.ai.evaluation import IntentResolutionEvaluator, TaskAdherenceEvaluator, ToolCallAccuracyEvaluator 
# other quality as well as risk and safety metrics
from azure.ai.evaluation import RelevanceEvaluator, CoherenceEvaluator, CodeVulnerabilityEvaluator, ContentSafetyEvaluator, IndirectAttackEvaluator, FluencyEvaluator
from azure.ai.projects.models import ConnectionType
from pathlib import Path
from opentelemetry import trace
# from config import get_logger
from azure.ai.evaluation.red_team import RedTeam, RiskCategory
from openai import AzureOpenAI
from azure.ai.evaluation import evaluate
from azure.ai.evaluation import GroundednessEvaluator, AzureOpenAIModelConfiguration
from azure.ai.agents.models import ConnectedAgentTool, MessageRole
from azure.ai.agents.models import AzureAISearchTool, AzureAISearchQueryType, MessageRole, ListSortOrder, ToolDefinition, FilePurpose, FileSearchTool
from utils import send_email
from user_logic_apps import AzureLogicAppTool, create_send_email_function
from azure.ai.agents.models import MessageTextContent, ListSortOrder
from azure.ai.agents.models import McpTool, RequiredMcpToolCall, SubmitToolApprovalAction, ToolApproval
from azure.ai.evaluation import AzureOpenAILabelGrader, evaluate, AzureOpenAIStringCheckGrader, AzureOpenAITextSimilarityGrader, AzureOpenAIGrader
from openai.types.graders import StringCheckGrader
import traceback

from dotenv import load_dotenv

load_dotenv()

import logging
# logging.basicConfig(level=logging.DEBUG)

endpoint = os.environ["PROJECT_ENDPOINT"] # Sample : https://<account_name>.services.ai.azure.com/api/projects/<project_name>
model_endpoint = os.environ["MODEL_ENDPOINT"] # Sample : https://<account_name>.services.ai.azure.com
model_api_key= os.environ["MODEL_API_KEY"]
model_deployment_name = os.environ["MODEL_DEPLOYMENT_NAME"] # Sample : gpt-4o-mini
os.environ["AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED"] = "true" 

# NOTE: Avoid a long-lived global client for actions because closing it in one
# function (via a context manager) will break others. Prefer creating a fresh
# AIProjectClient per function call using a context manager.

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

credential = DefaultAzureCredential(logging_enable=True)

def code_interpreter() -> str:
    code_interpreter = CodeInterpreterTool()
    # Use a fresh client instance for this operation
    with AIProjectClient(endpoint=endpoint, credential=credential) as _client:
        agent = _client.agents.create_agent(
            model=os.environ["MODEL_DEPLOYMENT_NAME"],
            name="codeint-agent",
            instructions="You are a helpful agent",
            tools=code_interpreter.definitions,
        )
        print(f"Created agent, ID: {agent.id}")

        thread = _client.agents.threads.create()
        print(f"Created thread, ID: {thread.id}")

        message = _client.agents.messages.create(
            thread_id=thread.id,
            role="user",
            content="What is the weather in Seattle today?",
        )
        print(f"Created message, ID: {message['id']}")

        run = _client.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
        print(f"Run finished with status: {run.status}")

        if run.status == "failed":
            print(f"Run failed: {run.last_error}")

        messages = _client.agents.messages.list(thread_id=thread.id)
        for message in messages:
            print(f"Role: {message.role}, Content: {message.content}")

        _client.agents.delete_agent(agent.id)
        _client.agents.threads.delete(thread.id)
        print("Deleted agent and thread")

def eval()-> str:
    returntxt = ""
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_KEY"),
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.environ.get("AZURE_API_VERSION"),
    )
    ## Using Azure AI Foundry Hub project
    # azure_ai_project = {
    #     "subscription_id": os.environ.get("AZURE_SUBSCRIPTION_ID"),
    #     "resource_group_name": os.environ.get("AZURE_RESOURCE_GROUP"),
    #     "project_name": os.environ.get("AZURE_PROJECT_NAME"),
    # }
    # azure_ai_project_dict = {
    #     "subscription_id": os.environ.get("AZURE_SUBSCRIPTION_ID"),
    #     "resource_group_name": os.environ.get("AZURE_RESOURCE_GROUP"),
    #     "project_name": os.environ.get("AZURE_PROJECT_NAME"),
    #     "azure_credential": DefaultAzureCredential(),
    # }
    credential = DefaultAzureCredential()

    azure_ai_project = os.environ.get("PROJECT_ENDPOINT")
    azure_ai_project_dict = os.environ.get("PROJECT_ENDPOINT")

    # Initializing Groundedness and Groundedness Pro evaluators
    # groundedness_eval = GroundednessEvaluator(model_config)
    # instantiate an evaluator with image and multi-modal support
    content_safety_evaluator = ContentSafetyEvaluator(credential=DefaultAzureCredential(), azure_ai_project=azure_ai_project)
    relevance_evaluator = RelevanceEvaluator(model_config)
    coherence_evaluator = CoherenceEvaluator(model_config)
    groundedness_evaluator = GroundednessEvaluator(model_config)
    fluency_evaluator = FluencyEvaluator(model_config)
    similarity_evaluator = SimilarityEvaluator(model_config)
    retrieval_evaluator = RetrievalEvaluator(model_config)
    groundnesspro_evaluator = GroundednessProEvaluator(azure_ai_project=azure_ai_project_dict, credential=credential)

    f1_evaluator = F1ScoreEvaluator()
    bleu_evaluator = BleuScoreEvaluator()
    gleu_evaluator = GleuScoreEvaluator()
    meteor_evaluator = MeteorScoreEvaluator(alpha=0.8)
    rouge_evaluator = RougeScoreEvaluator(rouge_type=RougeType.ROUGE_4)
    indirect_attack_eval = IndirectAttackEvaluator(azure_ai_project=azure_ai_project_dict, credential=credential)
    protected_material_eval = ProtectedMaterialEvaluator(azure_ai_project=azure_ai_project_dict, credential=credential)
    hate_unfairness_eval = HateUnfairnessEvaluator(azure_ai_project=azure_ai_project_dict, credential=credential)
    #  Evaluation criteria: Determine if the response column contains texts that are "too short", "just right", or "too long" and pass if it is "just right"
    label_grader = AzureOpenAILabelGrader(
        model_config=model_config,
        input=[{"content": "{{item.response}}", "role": "user"},
            {"content":"Any text including space that's more than 600 characters are too long, less than 500 characters are too short; 500 to 600 characters are just right.", "role":"user", "type": "message"}],
        labels=["too short", "just right", "too long"],
        passing_labels=["just right"],
        model="gpt-4o",
        name="label",
    )
    # Evaluation criteria: Pass if the query column contains "What is"
    string_grader = AzureOpenAIStringCheckGrader(
        model_config=model_config,
        input="{{item.query}}",
        name="starts with what is",
        operation="like", # "eq" for equal, "ne" for not equal, "like" for contain, "ilike" for case insensitive contain
        reference="What is",
    )
    # Evaluation criteria: Pass if response column and ground_truth column similarity score >= 0.5 using "fuzzy_match"
    sim_grader = AzureOpenAITextSimilarityGrader(
        model_config=model_config,
        evaluation_metric="fuzzy_match", # support evaluation metrics including: "fuzzy_match", "bleu", "gleu", "meteor", "rouge_1", "rouge_2", "rouge_3", "rouge_4", "rouge_5", "rouge_l", "cosine",
        input="{{item.response}}",
        name="similarity",
        pass_threshold=0.5,
        reference="{{item.ground_truth}}",
    )
    # Define an string check grader config directly using the OAI SDK
    # Evaluation criteria: Pass if query column contains "Northwind"
    # oai_string_check_grader = StringCheckGrader(
    #     input="{{item.query}}",
    #     name="contains hello",
    #     operation="like",
    #     reference="Northwind",
    #     type="string_check"
    # )

    # Evaluation criteria: Pass if the query column contains "What is"
    string_grader = AzureOpenAIStringCheckGrader(
        model_config=model_config,
        input="{{item.query}}",
        name="starts with what is",
        operation="like", # "eq" for equal, "ne" for not equal, "like" for contains, "ilike" for case-insensitive contains
        reference="What is",
    )
    # Plug that into the general grader
    general_grader = AzureOpenAIGrader(
        model_config=model_config,
        grader_config=string_grader
    )

    python_similarity_grader = AzureOpenAIPythonGrader(
        model_config=model_config,
        name="custom_similarity",
        image_tag="2025-05-08",
        pass_threshold=0.3,
        source="""
        def grade(sample, item) -> float:
        \"\"\"
        Custom similarity grader using word overlap.
        Note: All data is in the 'item' parameter.
        \"\"\"
        # Extract from item, not sample!
        response = item.get("response", "") if isinstance(item, dict) else ""
        ground_truth = item.get("ground_truth", "") if isinstance(item, dict) else ""
        
        # Simple word overlap similarity
        response_words = set(response.lower().split())
        truth_words = set(ground_truth.lower().split())
        
        if not truth_words:
        return 0.0
        
        overlap = response_words.intersection(truth_words)
        similarity = len(overlap) / len(truth_words)
        
        return min(1.0, similarity)
    """,
    )

     
    # 
    try:
        print()
        result = evaluate(
            data="datarfp.jsonl", # provide your data here
            evaluators={
                "content_safety": content_safety_evaluator,
                "coherence": coherence_evaluator,
                "relevance": relevance_evaluator,
                "groundedness": groundedness_evaluator,
                "fluency": fluency_evaluator,
            #    "similarity": similarity_evaluator,
                "f1": f1_evaluator,
                "bleu": bleu_evaluator,
                "gleu": gleu_evaluator,
                "meteor": meteor_evaluator,
                "rouge": rouge_evaluator,
                "indirect_attack": indirect_attack_eval,
                "protected_material": protected_material_eval,
                "hate_unfairness": hate_unfairness_eval,
                "retrieval": retrieval_evaluator,
                "groundnesspro": groundnesspro_evaluator,
                "similarity": similarity_evaluator,
                "label": label_grader,
                "string": string_grader,
                "sim_grader": sim_grader,
                # "general": general_grader,
                # "custom_similarity": python_similarity_grader,
            },        
            evaluator_config={
                "content_safety": {"query": "${data.query}", "response": "${data.response}"},
                "coherence": {"response": "${data.response}", "query": "${data.query}"},
                "relevance": {"response": "${data.response}", "context": "${data.context}", "query": "${data.query}"},
                "groundedness": {
                    "response": "${data.response}",
                    "context": "${data.context}",
                    "query": "${data.query}",
                },
                "fluency": {"response": "${data.response}", "context": "${data.context}", "query": "${data.query}"},
                "f1": {"response": "${data.response}", "ground_truth": "${data.ground_truth}"},
                "bleu": {"response": "${data.response}", "ground_truth": "${data.ground_truth}"},
                "gleu": {"response": "${data.response}", "ground_truth": "${data.ground_truth}"},
                "meteor": {"response": "${data.response}", "ground_truth": "${data.ground_truth}"},
                "rouge": {"response": "${data.response}", "ground_truth": "${data.ground_truth}"},
                "indirect_attack": {"query": "${data.query}", "response": "${data.response}"},
                "protected_material": {"query": "${data.query}", "response": "${data.response}"},
                "hate_unfairness": {"query": "${data.query}", "response": "${data.response}"},
                "retrieval": {"query": "${data.query}", "context": "${data.context}"},
                "groundnesspro": {"query": "${data.query}", "context" : "${data.context}", "response": "${data.response}"},
                "similarity": {"query": "${data.query}", "response": "${data.response}", "ground_truth": "${data.ground_truth}"},
                #"label": {"query": "${data.query}", "response": "${data.response}", "ground_truth": "${data.ground_truth}"},
                #"general": {"query": "${data.query}", "response": "${data.response}", "ground_truth": "${data.ground_truth}"},
                #"custom": {"query": "${data.query}", "response": "${data.response}", "ground_truth": "${data.ground_truth}"},
                # "simgrader": {"query": "${data.query}", "response": "${data.response}", "ground_truth": "${data.ground_truth}"},
            },
            # Optionally provide your Azure AI Foundry project information to track your evaluation results in your project portal
            azure_ai_project = os.environ["PROJECT_ENDPOINT"],
            # Optionally provide an output path to dump a json of metric summary, row level data and metric and Azure AI project URL
            output_path="./myevalresults.json"
        )
        #returntxt = f"Completed Evaluation: {result.studio_url}" 
        print("Evalutions completed .....")
    except Exception as e:
        print('Exception: ', e)
        traceback.print_exc()   # Full stack trace
        pass  
    returntxt = f"Completed Evaluation\n"
    return returntxt

# A simple example application callback function that always returns a fixed response
def simple_callback(query: str) -> str:
    return "I'm an AI assistant that follows ethical guidelines. I cannot provide harmful content."

# Function to process user input (simple echo bot for demonstration)
def process_message_reasoning(query: str) -> str:
    returntxt = ""
    excelsheetinfo = ""

    client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-12-01-preview",
    )

    system_prompt = (
        """You are a Red Team Agent AI assistant, 
        Format the output to be professional and easy to read — suitable for presentation to IT operations or service management leadership. """
    )

    txtmessages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
        ]
    model_name_reasoning = "o4-mini"
    # model_name_reasoning = "o3"

    response = client.chat.completions.create(
        model=model_name_reasoning,
        #reasoning={"effort": "high"},
        reasoning_effort="high",
        messages=txtmessages,
        # temperature=0.7,
        max_completion_tokens=4000
    )
    print("Response received from Azure OpenAI:", response)
    # returntxt = response.output_text.strip()
    returntxt = response.choices[0].message.content
    return f"{returntxt}"

async def advanced_callback(messages: Dict, stream: bool = False, session_state: Any = None, context: Optional[Dict] =None) -> dict:
    """A more complex callback that processes conversation history"""
    # Extract the latest message from the conversation history
    messages_list = [{"role": chat_message.role,"content": chat_message.content} for chat_message in messages]
    latest_message = messages_list[-1]["content"]
    
    # In a real application, you might process the entire conversation history
    # Here, we're just simulating different responses based on the latest message
    response = "I'm an AI assistant that follows safety guidelines. I cannot provide harmful content."
    
    # Format the response to follow the openAI chat protocol format
    formatted_response = {
        "content": response,
        "role": "assistant"
    }
    
    return {"messages": [formatted_response]}

# A simple example application callback function that always returns a fixed response
def aoai_callback(query: str) -> str:
    returntxt = ""
    excelsheetinfo = ""

    client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-12-01-preview",
    )
    system_prompt = (
        """You are a Red Team Agent AI assistant, Process the user query and provide a detailed response.
        """
    )
    txtmessages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
        ]
    model_name = "gpt-4.1"
    # model_name_reasoning = "o3"

    response = client.chat.completions.create(
        model=model_name,
        #reasoning={"effort": "high"},
        # reasoning_effort="high",
        messages=txtmessages,
        temperature=0.7,
        max_tokens=4000
    )
    print("Response received from Azure OpenAI:", response)
    # returntxt = response.output_text.strip()
    returntxt = response.choices[0].message.content
    return returntxt

async def redteam() -> str:
    returntxt = ""
                
    ## Using Azure AI Foundry Hub project
    azure_ai_project = {
        "subscription_id": os.environ.get("AZURE_SUBSCRIPTION_ID"),
        "resource_group_name": os.environ.get("AZURE_RESOURCE_GROUP"),
        "project_name": os.environ.get("AZURE_PROJECT_NAME"),
    }
    ## Using Azure AI Foundry project, example: AZURE_AI_PROJECT=https://your-account.services.ai.azure.com/api/projects/your-project
    azure_ai_project = os.environ.get("AZURE_AI_PROJECT")

    # Instantiate your AI Red Teaming Agent
    # red_team_agent = RedTeam(
    #     azure_ai_project=azure_ai_project, # required
    #     credential=DefaultAzureCredential() # required
    # )
    # Specifying risk categories and number of attack objectives per risk categories you want the AI Red Teaming Agent to cover
    red_team_agent = RedTeam(
        azure_ai_project=azure_ai_project, # required
        credential=DefaultAzureCredential(), # required
        risk_categories=[ # optional, defaults to all four risk categories
            RiskCategory.Violence,
            RiskCategory.HateUnfairness,
            RiskCategory.Sexual,
            RiskCategory.SelfHarm,
        ], 
        num_objectives=5, # optional, defaults to 10
        # num_turns=5
    )
    # Runs a red teaming scan on the simple callback target
    red_team_result = await red_team_agent.scan(target=simple_callback)


    # Define a model configuration to test
    azure_oai_model_config = {
        "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT_REDTEAM"),
        "azure_deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
        "api_key": os.environ.get("AZURE_OPENAI_KEY"),
    }
    # # Run the red team scan called "Intermediary-Model-Target-Scan"
    result = await red_team_agent.scan(
        target=azure_oai_model_config, scan_name="Intermediary-Model-Target-Scan", attack_strategies=[AttackStrategy.Flip]
    )
    returntxt += str(result)

    # Create the RedTeam instance with all of the risk categories with 5 attack objectives generated for each category
    model_red_team = RedTeam(
        azure_ai_project=azure_ai_project,
        credential=DefaultAzureCredential(),
        risk_categories=[RiskCategory.Violence, RiskCategory.HateUnfairness, RiskCategory.Sexual, RiskCategory.SelfHarm],
        num_objectives=2,
    )

    # Run the red team scan with multiple attack strategies
    advanced_result = await model_red_team.scan(
        target=advanced_callback,
        scan_name="Advanced-Callback-Scan",
        attack_strategies=[
            AttackStrategy.EASY,  # Group of easy complexity attacks
            AttackStrategy.MODERATE,  # Group of moderate complexity attacks
            # AttackStrategy.CHARACTER_SPACE,  # Add character spaces
            #AttackStrategy.ROT13,  # Use ROT13 encoding
            #AttackStrategy.UnicodeConfusable,  # Use confusable Unicode characters
            #AttackStrategy.CharSwap,  # Swap characters in prompts
            #AttackStrategy.Morse,  # Encode prompts in Morse code
            #AttackStrategy.Leetspeak,  # Use Leetspeak
            #AttackStrategy.Url,  # Use URLs in prompts
            #AttackStrategy.Binary,  # Encode prompts in binary
            # AttackStrategy.Compose([AttackStrategy.BASE64, AttackStrategy.ROT13]),  # Use two strategies in one 
        ],
        output_path="./Advanced-Callback-Scan.json",
    )

    returntxt += str(advanced_result)

    #returntxt += f"Red Team scan completed with status: {red_team_agent.ai_studio_url}\n"
        
    return returntxt

# async def redteam() -> str:
#     returntxt = ""


#     return returntxt

# Define some custom python function
def fetch_weather(location: str) -> str:
    """
    Fetches the weather information for the specified location.

    :param location (str): The location to fetch weather for.
    :return: Weather information as a JSON string.
    :rtype: str
    """
    # In a real-world scenario, you'd integrate with a weather API.
    # Here, we'll mock the response.
    mock_weather_data = {"Seattle": "Sunny, 25°C", "London": "Cloudy, 18°C", "Tokyo": "Rainy, 22°C"}
    weather = mock_weather_data.get(location, "Weather data not available for this location.")
    weather_json = json.dumps({"weather": weather})
    return weather_json

def agent_eval() -> str:
    returntxt = ""
    user_functions: Set[Callable[..., Any]] = {
        fetch_weather,
    }

    # Adding Tools to be used by Agent 
    functions = FunctionTool(user_functions)

    toolset = ToolSet()
    toolset.add(functions)


    project_client = AIProjectClient(
        endpoint=endpoint,
        credential=DefaultAzureCredential(),
    )

    AGENT_NAME = "Agentic Eval Assistant"

    # Add Tools to be used by Agent
    functions = FunctionTool(user_functions)

    toolset = ToolSet()
    toolset.add(functions)

    # To enable tool calls executed automatically
    project_client.agents.enable_auto_function_calls(tools=toolset)
    agent = project_client.agents.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name=AGENT_NAME,
        instructions="You are a helpful assistant",
        toolset=toolset,
    )

    print(f"Created agent, ID: {agent.id}")
    # https://github.com/Azure-Samples/azureai-samples/blob/main/scenarios/evaluate/Supported_Evaluation_Metrics/Agent_Evaluation/Evaluate_Azure_AI_Agent_Quality.ipynb
    # Create a thread for communication
    thread = project_client.agents.threads.create()
    print(f"Created thread, ID: {thread.id}")
    # Create message to thread

    MESSAGE = "Can you email me weather info for Seattle ?"

    # Add a message to the thread
    message = project_client.agents.messages.create(
        thread_id=thread.id,
        role="user",  # Role of the message sender
        content=MESSAGE,  # Message content
    )
    print(f"Created message, ID: {message['id']}")
    # Create and process an agent run
    run = project_client.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
    print(f"Run finished with status: {run.status}")
    
    # Check if the run failed
    if run.status == "failed":
        print(f"Run failed: {run.last_error}")
    
    # Fetch and log all messages
    messages = project_client.agents.messages.list(thread_id=thread.id)
    for message in messages:
        print(f"Role: {message.role}, Content: {message.content}")
    # Initialize the converter that will be backed by the project.
    converter = AIAgentConverter(project_client)

    thread_id = thread.id
    run_id = run.id
    file_name = "evaluation_input_data.jsonl"

    # Get a single agent run data
    evaluation_data_single_run = converter.convert(thread_id=thread_id, run_id=run_id)

    # Run this to save thread data to a JSONL file for evaluation
    # Save the agent thread data to a JSONL file
    # evaluation_data = converter.prepare_evaluation_data(thread_ids=thread_id, filename=<>)
    # print(json.dumps(evaluation_data, indent=4))
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ["AZURE_API_VERSION"],
        azure_deployment=os.environ["MODEL_DEPLOYMENT_NAME"],
    )
    # Needed to use content safety evaluators
    # azure_ai_project = {
    #     "subscription_id": os.environ["AZURE_SUBSCRIPTION_ID"],
    #     "project_name": os.environ["AZURE_PROJECT_NAME"],
    #     "resource_group_name": os.environ["AZURE_RESOURCE_GROUP"],
    # }

    azure_ai_project = os.environ["PROJECT_ENDPOINT"]

    intent_resolution = IntentResolutionEvaluator(model_config=model_config)
    tool_call_accuracy = ToolCallAccuracyEvaluator(model_config=model_config)
    task_adherence = TaskAdherenceEvaluator(model_config=model_config)
    #response_completeness_evaluator = CompletenessEvaluator(model_config=model_config, azure_ai_project=azure_ai_project)
    
    response = evaluate(
        data=file_name,
        evaluators={
            "tool_call_accuracy": tool_call_accuracy,
            "intent_resolution": intent_resolution,
            "task_adherence": task_adherence,
            #"response_completeness": response_completeness_evaluator,
        },
        azure_ai_project=os.environ["PROJECT_ENDPOINT"],
    )
    pprint(f'AI Foundary URL: {response.get("studio_url")}')
    # average scores across all runs
    pprint(response["metrics"])
    returntxt = str(response["metrics"])

    # Delete the agent when done
    project_client.agents.delete_agent(agent.id)
    project_client.agents.threads.delete(thread.id)
    print("Deleted agent and thread")

    return returntxt

def ai_search_agent(query: str) -> str:
    returntxt = ""

    # Retrieve the endpoint from environment variables
    project_endpoint = os.environ["PROJECT_ENDPOINT"]
    # https://learn.microsoft.com/en-us/azure/ai-services/agents/how-to/tools/azure-ai-search-samples?pivots=python

    # Initialize the AIProjectClient
    project_client = AIProjectClient(
        endpoint=project_endpoint,
        credential=DefaultAzureCredential(exclude_interactive_browser_credential=False),
        # api_version="latest",
    )
    # Define the Azure AI Search connection ID and index name
    azure_ai_conn_id = "vecdb"
    index_name = "constructionrfpdocs1"

    # Initialize the Azure AI Search tool
    ai_search = AzureAISearchTool(
        index_connection_id=azure_ai_conn_id,
        index_name=index_name,
        query_type=AzureAISearchQueryType.SIMPLE,  # Use SIMPLE query type
        top_k=5,  # Retrieve the top 3 results
        filter="",  # Optional filter for search results
    )
    # Define the model deployment name
    model_deployment_name = os.environ["MODEL_DEPLOYMENT_NAME"]

    # Create an agent with the Azure AI Search tool
    agent = project_client.agents.create_agent(
        model=model_deployment_name,
        name="AISearch-agent",
        instructions="You are a helpful agent",
        tools=ai_search.definitions,
        tool_resources=ai_search.resources,
    )
    print(f"Created agent, ID: {agent.id}")
    # Create a thread for communication
    thread = project_client.agents.threads.create()
    print(f"Created thread, ID: {thread.id}")

    # Send a message to the thread
    message = project_client.agents.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content=query,
    )
    print(f"Created message, ID: {message['id']}")

    # Create and process a run with the specified thread and agent
    run = project_client.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
    print(f"Run finished with status: {run.status}")

    # Check if the run failed
    if run.status == "failed":
        print(f"Run failed: {run.last_error}")

    # Fetch and log all messages in the thread
    messages = project_client.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
    print(str(messages))
    # for message in messages.data:
    #     print(f"Role: {message.role}, Content: {message.content}")
    #     returntxt += f"Role: {message.role}, Content: {message.content}\n"
    for page in messages.by_page():
        for item in page:
            # print(item)
            #returntxt += f"Role: {item.role}, Content: {item.content[0]['text']['value']}\n"
            returntxt += f"Source: {item.content[0]['text']['value']}\n"

    # Delete the agent
    project_client.agents.delete_agent(agent.id)
    project_client.agents.threads.delete(thread.id)
    print("Deleted agent and thread")

    return returntxt

def connected_agent(query: str) -> str:
    returntxt = ""
    #https://learn.microsoft.com/en-us/azure/ai-services/agents/how-to/connected-agents?pivots=python

    project_client = AIProjectClient(
        endpoint=os.environ["PROJECT_ENDPOINT"],
        credential=DefaultAzureCredential(),
        # api_version="latest",
    )
    stock_price_agent = project_client.agents.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name="stockpricebot",
        instructions="Your job is to get the stock price of a company. If you don't know the realtime stock price, return the last known stock price.",
        #tools=... # tools to help the agent get stock prices
    )
    connected_agent_name = "stockpricebot"
    connected_agent = ConnectedAgentTool(
        id=stock_price_agent.id, name=connected_agent_name, description="Gets the stock price of a company"
    )

    #create AI Search tool
    # Define the Azure AI Search connection ID and index name
    azure_ai_conn_id = "vecdb"
    index_name = "constructionrfpdocs1"

    # Initialize the Azure AI Search tool
    ai_search = AzureAISearchTool(
        index_connection_id=azure_ai_conn_id,
        index_name=index_name,
        query_type=AzureAISearchQueryType.SIMPLE,  # Use SIMPLE query type
        top_k=5,  # Retrieve the top 3 results
        filter="",  # Optional filter for search results
    )
    # Define the model deployment name
    model_deployment_name = os.environ["MODEL_DEPLOYMENT_NAME"]

    # Create an agent with the Azure AI Search tool
    rfp_agent = project_client.agents.create_agent(
        model=model_deployment_name,
        name="AISearchagent",
        instructions="You are a helpful agent",
        tools=ai_search.definitions,
        tool_resources=ai_search.resources,
    )
    search_connected_agent_name = "AISearchagent"
    search_connected_agent = ConnectedAgentTool(
        id=rfp_agent.id, name=search_connected_agent_name, description="Gets the construction proposals from the RFP documents"
    )

    # create a custom skill to send emails

    # Define user functions
    user_functions = {send_email}
    # Initialize the FunctionTool with user-defined functions
    # functions = FunctionTool(functions=user_functions)
    # Create an agent
    functions = FunctionTool(functions=user_functions)
    toolset = ToolSet()
    toolset.add(functions)
    # Emailagent = project_client.agents.create_agent(
    #     model=os.environ["MODEL_DEPLOYMENT_NAME"],
    #     name="SendEmailagent",
    #     instructions="You are a specialized agent for sending emails.",
    #     # tools=functions.definitions,
    #     toolset=toolset,  # Attach the FunctionTool to the agent
    # )
    # sendemail_connected_agent_name = "SendEmailagent"
    # sendemail_connected_agent = ConnectedAgentTool(
    #     id=Emailagent.id, name=sendemail_connected_agent_name, description="Get the content from other agents and send an email"
    # )

    # Extract subscription and resource group from the project scope
    subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
    resource_group = os.environ["AZURE_RESOURCE_GROUP"]

    # Logic App details
    logic_app_name = "agenttoolemail"
    trigger_name = "When_a_HTTP_request_is_received"

    # Create and initialize AzureLogicAppTool utility
    logic_app_tool = AzureLogicAppTool(subscription_id, resource_group)
    logic_app_tool.register_logic_app(logic_app_name, trigger_name)
    print(f"Registered logic app '{logic_app_name}' with trigger '{trigger_name}'.")

    # Create the specialized "send_email_via_logic_app" function for your agent tools
    send_email_func = create_send_email_function(logic_app_tool, logic_app_name)

    # Prepare the function tools for the agent
    functions_to_use: Set = {
        # fetch_current_datetime,
        send_email_func,  # This references the AzureLogicAppTool instance via closure
    }
    # [END register_logic_app]

    # Create an agent
    functions = FunctionTool(functions=functions_to_use)
    toolset = ToolSet()
    toolset.add(functions)

    Emailagent = project_client.agents.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name="SendEmailAgent",
        instructions="You are a specialized agent for sending emails.",
        toolset=toolset,
    )
    sendemail_connected_agent_name = "SendEmailagent"
    sendemail_connected_agent = ConnectedAgentTool(
        id=Emailagent.id, name=sendemail_connected_agent_name, description="Get the content from other agents and send an email"
    )

    # File search agent
    # Define the path to the file to be uploaded
    file_path = "./papers/ssrn-4072178.pdf"

    # Upload the file
    file = project_client.agents.files.upload_and_poll(file_path=file_path, purpose=FilePurpose.AGENTS)
    print(f"Uploaded file, file ID: {file.id}")

    # Create a vector store with the uploaded file
    # vector_store = project_client.agents.create_vector_store_and_poll(file_ids=[file.id], name="suspaperstore")
    vector_store = project_client.agents.vector_stores.create_and_poll(file_ids=[file.id], name="suspaperstore")
    print(f"Created vector store, vector store ID: {vector_store.id}")
    # Create a file search tool
    file_search = FileSearchTool(vector_store_ids=[vector_store.id])

    # Create an agent with the file search tool
    Sustainablityagent = project_client.agents.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],  # Model deployment name
        name="Sustainabilitypaperagent",  # Name of the agent
        instructions="You are a helpful agent and can search information from uploaded files",  # Instructions for the agent
        tools=file_search.definitions,  # Tools available to the agent
        tool_resources=file_search.resources,  # Resources for the tools
    )
    # print(f"Created agent, ID: {agent.id}")
    sustaibilityconnectedagentname = "Sustainabilitypaperagent"
    sustainability_connected_agent = ConnectedAgentTool(
        id=Sustainablityagent.id, name=sustaibilityconnectedagentname, description="Summarize the content of the uploaded files and answer questions about it"
    )


    # all_tools = connected_agent.definitions + search_connected_agent.definitions + sendemail_connected_agent.definitions
    # all_tools = connected_agent.definitions + search_connected_agent.definitions
    # all_tools = search_connected_agent.definitions + sendemail_connected_agent.definitions

    # # Deduplicate by tool name (or another unique property) to avoid ValueError
    # unique_tools = {}
    # for tool in all_tools:
    #     unique_tools[getattr(tool, "name", id(tool))] = tool
        


    # Orchestrate the connected agent with the main agent
    agent = project_client.agents.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name="ConnectedMultiagent",
        instructions="""
        You are a helpful assistant, and use the connected agents to get stock prices, construction RFP Data, 
        Send Email and Sustainability Paper.
        """,
        # tools=list(unique_tools.values()), #search_connected_agent.definitions,  # Attach the connected agents
        tools=[
            connected_agent.definitions[0],
            # search_connected_agent.definitions[0],
            # sendemail_connected_agent.definitions[0],
            sustainability_connected_agent.definitions[0],
        ]
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
    run = project_client.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
    # Poll the run status until it is completed or requires action
    while run.status in ["queued", "in_progress", "requires_action"]:
        time.sleep(1)
        run = project_client.agents.runs.get(thread_id=thread.id, run_id=run.id)
        print(f"Run status: {run.status}")

        if run.status == "requires_action":
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            for tool_call in tool_calls:
                print(f"Tool call: {tool_call.name}, ID: {tool_call.id}")
            #     if tool_call.name == "fetch_weather":
            #         output = fetch_weather("New York")
            #         tool_outputs.append({"tool_call_id": tool_call.id, "output": output})
            # project_client.agents.runs.submit_tool_outputs(thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs)

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

    # Delete the Agent when done
    project_client.agents.delete_agent(agent.id)    
    project_client.agents.threads.delete(thread.id)
    # print("Deleted agent")
    # Delete the connected Agent when done
    project_client.agents.delete_agent(stock_price_agent.id)
    project_client.agents.delete_agent(rfp_agent.id)
    project_client.agents.delete_agent(Emailagent.id)
    project_client.agents.delete_agent(Sustainablityagent.id)
    print("Deleted connected agent")
    # # Cleanup resources
    # project_client.agents.files.delete(file_id=file.id)
    # print("Deleted file")
    # project_client.agents.vector_stores.delete(vector_store.id)
    # print("Deleted vector store")
    # print(" # start to delete threads for this agent")
    # # List all threads for this agent
    # try:
    #     threads = list(project_client.agents.threads.list())
    # except Exception as e:
    #     print(f"Error listing threads for agent {agent.id}: {e}")
    #     threads = []

    # for thread in threads:
    #     print(f"  Deleting thread: {thread.id}")
    #     try:
    #         project_client.agents.threads.delete(thread.id)
    #         print(f"  Deleted thread {thread.id}")
    #     except Exception as e:
    #         print(f"  Error deleting thread {thread.id}: {e}")
    # print("# deleted all threads for this agent")
    # Print the Agent's response message with optional citation
    # Fetch and log all messages
    

    return returntxt

def existing_connected_agent(query: str) -> str:
    returntxt = ""

    project = AIProjectClient(
    credential=DefaultAzureCredential(),
    endpoint=endpoint)

    agent = project.agents.get_agent("asst_CdFYKwzfsO6xIscJEYV30vpE")

    thread = project.agents.threads.create()
    print(f"Created thread, ID: {thread.id}")

    message = project.agents.messages.create(
        thread_id=thread.id,
        role="user",
        content=query
    )

    run = project.agents.runs.create_and_process(
        thread_id=thread.id,
        agent_id=agent.id)

    if run.status == "failed":
        print(f"Run failed: {run.last_error}")
        returntxt = f"Run failed: {run.last_error}\n"
    else:
        messages = project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
        for message in messages:
            if message.role == MessageRole.AGENT:
                print(f"Role: {message.role}, Content: {message.content}")
                # returntxt += f"Role: {message.role}, Content: {message.content}\n"
                # returntxt += f"Source: {message.content[0]['text']['value']}\n"
                returntxt += f"Source: {message.content[0].text.value}\n"

    return returntxt

def delete_agent():
    # Authenticate and create the project client
    project_client = AIProjectClient(
        endpoint=os.environ["PROJECT_ENDPOINT"],
        credential=DefaultAzureCredential(),
        # api_version="latest",
    )

    # List all agents
    agents = list(project_client.agents.list_agents())

    for agent in agents:
        print(f"Processing agent: {agent.id} ({agent.name})")

        # List all threads for this agent
        try:
            threads = list(project_client.agents.threads.list())
        except Exception as e:
            print(f"Error listing threads for agent {agent.id}: {e}")
            threads = []

        for thread in threads:
            print(f"  Deleting thread: {thread.id}")
            try:
                project_client.agents.threads.delete(thread.id)
                print(f"  Deleted thread {thread.id}")
            except Exception as e:
                print(f"  Error deleting thread {thread.id}: {e}")

        # Delete the agent
        try:
            project_client.agents.delete_agent(agent.id)
            print(f"Deleted agent {agent.id}")
        except Exception as e:
            print(f"Error deleting agent {agent.id}: {e}")

def load_existing_agent(query: str) -> str:
    returntxt = ""

    project = AIProjectClient(
        credential=DefaultAzureCredential(),
        endpoint=os.environ["PROJECT_ENDPOINT"]
    )

    agent = project.agents.get_agent("asst_GipO6VepHiB41MJdGvlAIXay")

    # thread = project.agents.threads.get("thread_ndWUUGkSrgC0eMME7yNaVwfR")
    print(f"Created agent, ID: {agent.id}")
    thread = project.agents.threads.create()
    print(f"Created thread, ID: {thread.id}")

    message = project.agents.messages.create(
        thread_id=thread.id,
        role="user",
        content=query
    )

    run = project.agents.runs.create_and_process(
        thread_id=thread.id,
        agent_id=agent.id)
    
    

    if run.status == "failed":
        print(f"Run failed: {run.last_error}")
    else:
        messages = project.agents.messages.list(thread_id=thread.id)
        for message in messages:
            if message.role == MessageRole.AGENT:
                # print(f"{message.role}: {message.text_messages[-1].text.value}")
                returntxt += f"Source: {message.content[0].text.value}\n"
            print(f"Messages inside thread {message}\n\n")

        # Fetch run steps to get the details of the agent run
        run_steps = project.agents.run_steps.list(thread_id=thread.id, run_id=run.id)
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
        messages = project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
        for message in messages:
            if message.role == MessageRole.AGENT and message.url_citation_annotations:
                placeholder_annotations = {
                    annotation.text: f" [see {annotation.url_citation.title}] ({annotation.url_citation.url})"
                    for annotation in message.url_citation_annotations
                }
                for message_text in message.text_messages:
                    message_str = message_text.text.value
                    for k, v in placeholder_annotations.items():
                        message_str = message_str.replace(k, v)
                    print(f"{message.role}: {message_str}")
            else:
                for message_text in message.text_messages:
                    print(f"{message.role}: {message_text.text.value}")
        print()  # add an extra newline between steps
    # Delete the agent when done
    project.agents.threads.delete(thread.id)

    return returntxt

def msft_learn_mcp_agent(query: str) -> str:
    returntxt = ""

    # Retrieve the endpoint from environment variables
    project_endpoint = os.environ["PROJECT_ENDPOINT_WEST"]
    # https://learn.microsoft.com/en-us/azure/ai-services/agents/how-to/tools/azure-ai-search-samples?pivots=python

    # Initialize the AIProjectClient
    project_client = AIProjectClient(
        endpoint=project_endpoint,
        credential=DefaultAzureCredential(exclude_interactive_browser_credential=False),
        # api_version="latest",
    )

    with project_client:
        agent = project_client.agents.create_agent(
            model=os.environ["MODEL_DEPLOYMENT_NAME"], 
            name="msftlearnmcp-agent", 
            instructions="You are a helpful assistant. Use the tools provided to answer the user's questions. Be sure to cite your sources.",
            tools= [
                {
                    "type": "mcp",
                    "server_label": "MicrosoftLearn",
                    "server_url": "https://learn.microsoft.com/api/mcp",
                    "require_approval": "never"
                }
            ],
            tool_resources=None
        )
        print(f"Created agent, agent ID: {agent.id}")
        thread = project_client.agents.threads.create()
        print(f"Created thread, thread ID: {thread.id}")

        message = project_client.agents.messages.create(
            thread_id=thread.id, role="user", content="<a question for your MCP server>",
        )
        print(f"Created message, message ID: {message.id}")

        run = project_client.agents.runs.create(thread_id=thread.id, agent_id=agent.id)
        
        # Poll the run as long as run status is queued or in progress
        while run.status in ["queued", "in_progress", "requires_action"]:
            # Wait for a second
            time.sleep(1)
            run = project_client.agents.runs.get(thread_id=thread.id, run_id=run.id)
            print(f"Run status: {run.status}")

        if run.status == "failed":
            print(f"Run error: {run.last_error}")

        run_steps = project_client.agents.run_steps.list(thread_id=thread.id, run_id=run.id)
        for step in run_steps:
            print(f"Run step: {step.id}, status: {step.status}, type: {step.type}")
            if step.type == "tool_calls":
                print(f"Tool call details:")
                for tool_call in step.step_details.tool_calls:
                    print(json.dumps(tool_call.as_dict(), indent=2))

        messages = project_client.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
        for data_point in messages:
            last_message_content = data_point.content[-1]
            if isinstance(last_message_content, MessageTextContent):
                print(f"{data_point.role}: {last_message_content.text.value}")
                returntxt += f"{data_point.role}: {last_message_content.text.value}\n"
    project_client.agents.delete_agent(agent.id)
    print(f"Deleted agent, agent ID: {agent.id}")

    return returntxt

def hf_mcp_agent(query: str) -> str:
    returntxt = ""

    # Retrieve the endpoint from environment variables
    project_endpoint = os.environ["PROJECT_ENDPOINT_WEST"]
    # https://learn.microsoft.com/en-us/azure/ai-services/agents/how-to/tools/azure-ai-search-samples?pivots=python
    # https://github.com/Azure/azure-sdk-for-python/blob/glharper/agents-mcp-tool/sdk/ai/azure-ai-agents/samples/agents_tools/sample_agents_mcp.py
    # Initialize the AIProjectClient
    project_client = AIProjectClient(
        endpoint=project_endpoint,
        credential=DefaultAzureCredential(exclude_interactive_browser_credential=False),
        # api_version="latest",
    )
    # [START create_agent_with_mcp]
    # Initialize agent MCP tool
    mcp_tool = McpTool(
        server_label="huggingface",
        server_url="https://hf.co/mcp",
        allowed_tools=[],  # Optional: specify allowed tools
    )

    # You can also add or remove allowed tools dynamically
    # search_api_code = "search_azure_rest_api_code"
    # mcp_tool.add_allowed_tool(search_api_code)
    # print(f"Allowed tools: {mcp_tool.allowed_tools}")

    with project_client:
        agent = project_client.agents.create_agent(
            model=os.environ["MODEL_DEPLOYMENT_NAME"], 
            name="msftlearnmcp-agent", 
            instructions="You are a helpful assistant. Use the tools provided to answer the user's questions. Be sure to cite your sources.",
            # tools= [
            #     {
            #         "type": "mcp",
            #         "server_label": "huggingface",
            #         "server_url": "https://hf.co/mcp",
            #         "require_approval": "never"
            #     }
            # ],
            tools=mcp_tool.definitions,
            tool_resources=None
        )
        print(f"Created agent, agent ID: {agent.id}")
        print(f"MCP Server: {mcp_tool.server_label} at {mcp_tool.server_url}")
        thread = project_client.agents.threads.create()
        print(f"Created thread, thread ID: {thread.id}")

        message = project_client.agents.messages.create(
            thread_id=thread.id, role="user", content=query,
        )
        print(f"Created message, message ID: {message.id}")

        # Create and process agent run in thread with MCP tools
        # mcp_tool.update_headers("SuperSecret", "123456")
        mcp_tool.update_require_approval("never")
        print(mcp_tool)

        run = project_client.agents.runs.create(thread_id=thread.id, agent_id=agent.id)
        print(f"Created run, ID: {run.id}")
        
        while run.status in ["queued", "in_progress", "requires_action"]:
            time.sleep(1)
            run = project_client.agents.runs.get(thread_id=thread.id, run_id=run.id)

            if run.status == "requires_action" and isinstance(run.required_action, SubmitToolApprovalAction):
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

        # run_steps = project_client.agents.run_steps.list(thread_id=thread.id, run_id=run.id)
        # for step in run_steps:
        #     print(f"Run step: {step.id}, status: {step.status}, type: {step.type}")
        #     if step.type == "tool_calls":
        #         print(f"Tool call details:")
        #         for tool_call in step.step_details.tool_calls:
        #             print(json.dumps(tool_call.as_dict(), indent=2))

        # messages = project_client.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
        # for data_point in messages:
        #     last_message_content = data_point.content[-1]
        #     if isinstance(last_message_content, MessageTextContent):
        #         print(f"{data_point.role}: {last_message_content.text.value}")
        #         returntxt += f"{data_point.role}: {last_message_content.text.value}\n"
        print(f"Run completed with status: {run.status}")
        if run.status == "failed":
            print(f"Run failed: {run.last_error}")

        # Display run steps and tool calls
        run_steps = project_client.agents.run_steps.list(thread_id=thread.id, run_id=run.id)

        # Loop through each step
        for step in run_steps:
            print(f"Step {step['id']} status: {step['status']}")

            # Check if there are tool calls in the step details
            step_details = step.get("step_details", {})
            tool_calls = step_details.get("tool_calls", [])

            if tool_calls:
                print("  MCP Tool calls:")
                for call in tool_calls:
                    print(f"    Tool Call ID: {call.get('id')}")
                    print(f"    Type: {call.get('type')}")

                    # Handle MCP tool calls
                    if call.get("type") == "mcp":
                        print(call)
            print()  # add an extra newline between steps

        # Fetch and log all messages
        messages = project_client.agents.messages.list(thread_id=thread.id)
        print("\nConversation:")
        print("-" * 50)
        for msg in messages:
            if msg.text_messages:
                last_text = msg.text_messages[-1]
                print(f"{msg.role.upper()}: {last_text.text.value}")
                print("-" * 50)

        # Example of dynamic tool management
        print(f"\nDemonstrating dynamic tool management:")
        print(f"Current allowed tools: {mcp_tool.allowed_tools}")
        project_client.agents.delete_agent(agent.id)
        project_client.agents.threads.delete(thread.id)
        print(f"Deleted agent, agent ID: {agent.id}")

    return returntxt

def aiactionplat_agent(query: str) -> str:
    returntxt = ""

       # File search agent
    # Define the path to the file to be uploaded
    file_path = "./data/Americas-AI-Action-Plan.pdf"

    # Upload the file using a fresh client instance
    with AIProjectClient(endpoint=endpoint, credential=DefaultAzureCredential()) as _client:
        file = _client.agents.files.upload_and_poll(file_path=file_path, purpose=FilePurpose.AGENTS)
        print(f"Uploaded file, file ID: {file.id}")

        # Create a vector store with the uploaded file
        vector_store = _client.agents.vector_stores.create_and_poll(file_ids=[file.id], name="aiactionplanstore")
        print(f"Created vector store, vector store ID: {vector_store.id}")
        # Create a file search tool
        file_search = FileSearchTool(vector_store_ids=[vector_store.id])

        # Create an agent with the file search tool
        agent = _client.agents.create_agent(
            model=os.environ["MODEL_DEPLOYMENT_NAME"],
            name="AIActionPlan-agent",
            instructions="You are a helpful agent and can search information from uploaded files, which has new Americas AI Action plan from whitehouse.",
            tools=file_search.definitions,
            tool_resources=file_search.resources,
        )
        print(f"Created agent, ID: {agent.id}")
        # Create a thread
        thread = _client.agents.threads.create()
        print(f"Created thread, ID: {thread.id}")

        # Send a message to the thread
        _client.agents.messages.create(
            thread_id=thread.id,
            role="user",
            content=query,
        )
        print("Created message for thread")
        # Create and process an agent run in the thread
        run = _client.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
        print(f"Run finished with status: {run.status}")

        if run.status == "failed":
            print(f"Run failed: {run.last_error}")

        # Fetch and log all messages from the thread
        messages = _client.agents.messages.list(thread_id=thread.id)
        file_name = os.path.split(file_path)[-1]
        for msg in messages:
            if msg.text_messages:
                last_text = msg.text_messages[-1].text.value
                for annotation in msg.text_messages[-1].text.annotations:
                    citation = (
                        file_name if annotation.file_citation.file_id == file.id else annotation.file_citation.file_id
                    )
                    last_text = last_text.replace(annotation.text, f" [{citation}]")
                
                returntxt += last_text
                print(f"{msg.role}: {last_text}")

        # Cleanup resources (delete after reading messages)
        _client.agents.vector_stores.delete(vector_store.id)
        print("Deleted vector store")
        _client.agents.files.delete(file_id=file.id)
        print("Deleted file")
        _client.agents.delete_agent(agent.id)
        print("Deleted agent")

    return returntxt

def main():
    with tracer.start_as_current_span("manualazureaifoundryagent-tracing"):
        print("Running code interpreter example...")
        #code_interpreter()
        
        print("Running evaluation example...")
        # https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/evaluation/azure-ai-evaluation/samples/evaluation_samples_evaluate_fdp.py
        # https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/ai/azure-ai-agents/samples/agents_tools/sample_agents_logic_apps.py
        evalrs = eval()
        print(evalrs)
        
        print("Running red teaming example...")
        # redteamrs = asyncio.run(redteam())
        # print(redteamrs)
        
        print("Running agent evaluation example...")
        # agentevalrs = agent_eval()
        # print(agentevalrs)

        print("Running connected agent example...")
        starttime = datetime.now()
        # connected_agent_result = connected_agent("Show me details on Construction management services experience we have done before?")
        # connected_agent_result = connected_agent("What is the stock price of Microsoft")
        # connected_agent_result = connected_agent("Show me details on Construction management services experience we have done before and email Bala at babal@microsoft.com")
        # connected_agent_result = connected_agent("Summarize sustainability framework for learning factory from file uploaded?")
        # print('Final Output Answer: ', connected_agent_result)
        # endtime = datetime.now()
        # print(f"Connected agent example completed in {endtime - starttime} seconds")

        print("Running AI Search agent example...")
        # ai_search_result = ai_search_agent("Show me details on Construction management services experience we have done before?")
        # print(ai_search_result)

        print("Calling existing agent example...")
        starttime = datetime.now()
        # exsitingagentrs = load_existing_agent("Show me details on Construction management services experience we have done before and email Bala at babal@microsoft.com with subject as construction manager")
        # exsitingagentrs = load_existing_agent("Summarize sustainability framework for learning factory from file uploaded and get me microsoft stock price")
        #print(exsitingagentrs)
        #endtime = datetime.now()
        # print(f"Delete agent example completed in {endtime - starttime} seconds")

        print("Deleteing agents...")
        # starttime = datetime.now()
        # delete_agent()
        # endtime = datetime.now()
        # print(f"Delete agent example completed in {endtime - starttime} seconds")

        # https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/ai/azure-ai-agents/samples/agents_tools/sample_agents_azure_ai_search.py
        # above is the github for details.


        # https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/ai/azure-ai-agents/samples/agents_multiagent/sample_agents_multi_agent_team.py
        # multiagent

        # msft learn
        # starttime = datetime.now()
        # msft_learn_result = msft_learn_mcp_agent("Show me top 5 AI courses?")
        # print(msft_learn_result)
        # endtime = datetime.now()
        # print(f"MSFT Learn MCP agent example completed in {endtime - starttime} seconds")

        #hf_mcp_agent
        # https://devblogs.microsoft.com/foundry/announcing-model-context-protocol-support-preview-in-azure-ai-foundry-agent-service/
        # starttime = datetime.now()
        # hf_mcp_result = hf_mcp_agent("Show me top 5 latest models with details?")
        # print(hf_mcp_result)
        # endtime = datetime.now()
        # print(f"Hugging Face MCP agent example completed in {endtime - starttime} seconds")

if __name__ == "__main__":
    main()