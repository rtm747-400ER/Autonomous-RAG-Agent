from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import all_tools 

def get_agent_executor(llm):
    """
    Creates an autonomous Tool-Calling Agent.
    The agent receives retrieved documents in its prompt and decides 
    autonomously whether to use the web search tool.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent and autonomous research assistant. 
        
        You have been provided with the following context retrieved from the user's local PDF documents:
        <document_context>
        {retrieved_documents}
        </document_context>

        Your objective is to answer the user's question comprehensively. Follow this logic:
        1. Analyze the provided <document_context>. If it contains enough information to answer the user's question, do so directly without using any tools.
        2. If the <document_context> is empty, irrelevant, or lacks sufficient detail, you MUST autonomously use the `web_search_tool` to find the missing information.
        3. If you use the web search, seamlessly integrate that information into your final answer, and briefly mention that you had to search the web to find the complete answer."""),
                
        # Memory placeholder so the agent remembers past questions
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        
        # The user's current question
        ("human", "{input}"),
        
        # Scratchpad where the agent "thinks" and records tool outputs
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, all_tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent, 
        tools=all_tools, 
        verbose=True, 
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )

    return agent_executor