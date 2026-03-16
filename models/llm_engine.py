"""
models/llm_engine.py

Agentic loop with tool use. Routes between:
    - Tool Use: transaction data questions
    - RAG: card terms / fees / benefits questions
"""

import os
import cohere
from dotenv import load_dotenv
from models.wallet_engine import WalletEngine
from models.rag_engine import RAGEngine

load_dotenv()

TOOL_DEFINITIONS = [
    {
        "name": "get_transactions",
        "description": "Retrieve filtered transactions by source, category, date, or month.",
        "parameter_definitions": {
            "source":    {"description": "Card or wallet name.", "type": "str", "required": False},
            "category":  {"description": "Spending category.", "type": "str", "required": False},
            "month":     {"description": "Month name e.g. January.", "type": "str", "required": False},
            "date_from": {"description": "Start date YYYY-MM-DD.", "type": "str", "required": False},
            "date_to":   {"description": "End date YYYY-MM-DD.", "type": "str", "required": False},
            "limit":     {"description": "Max rows to return.", "type": "int", "required": False},
        },
    },
    {
        "name": "get_spending_summary",
        "description": "Spending totals by category. Use for how much did I spend questions.",
        "parameter_definitions": {
            "source": {"description": "Filter by payment source.", "type": "str", "required": False},
            "month":  {"description": "Filter by month name.", "type": "str", "required": False},
            "period": {"description": "last_30_days or last_7_days.", "type": "str", "required": False},
        },
    },
    {
        "name": "get_monthly_comparison",
        "description": "Compare spending across months. Use for trend questions.",
        "parameter_definitions": {
            "source":        {"description": "Filter by source.", "type": "str", "required": False},
            "category":      {"description": "Filter by category.", "type": "str", "required": False},
            "last_n_months": {"description": "Months to compare.", "type": "int", "required": False},
        },
    },
    {
        "name": "get_top_merchants",
        "description": "Top merchants by spend. Use for where am I spending most questions.",
        "parameter_definitions": {
            "source": {"description": "Filter by source.", "type": "str", "required": False},
            "month":  {"description": "Filter by month.", "type": "str", "required": False},
            "limit":  {"description": "Number of merchants.", "type": "int", "required": False},
        },
    },
    {
        "name": "get_source_breakdown",
        "description": "Spending split across all cards and digital wallets.",
        "parameter_definitions": {
            "month": {"description": "Filter by month.", "type": "str", "required": False},
        },
    },
    {
        "name": "detect_anomalies",
        "description": "Flag unusually large transactions. Use for suspicious charges questions.",
        "parameter_definitions": {
            "source": {"description": "Filter by source.", "type": "str", "required": False},
        },
    },
]


def dispatch_tool(tool_name: str, tool_args: dict, engine: WalletEngine) -> str:
    tool_map = {
        "get_transactions":       engine.get_transactions,
        "get_spending_summary":   engine.get_spending_summary,
        "get_monthly_comparison": engine.get_monthly_comparison,
        "get_top_merchants":      engine.get_top_merchants,
        "get_source_breakdown":   engine.get_source_breakdown,
        "detect_anomalies":       engine.detect_anomalies,
    }
    if tool_name not in tool_map:
        return f"Unknown tool: {tool_name}"
    try:
        return tool_map[tool_name](**tool_args)
    except Exception as e:
        return f"Error in {tool_name}: {str(e)}"


class WalletLLM:

    MAX_TOOL_ITERATIONS = 5

    def __init__(self, engine: WalletEngine, rag: RAGEngine):
        self.engine = engine
        self.rag    = rag
        self.client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        self._build_system_prompt()

    def _build_system_prompt(self):
        ctx = self.engine.get_data_context()
        self.system_prompt = f"""You are a smart financial assistant for a multi-source digital wallet. You have access to transaction history across all payment sources.

DATA CONTEXT:
- Payment sources: {', '.join(ctx['sources'])}
- Categories: {', '.join(ctx['categories'])}
- Transaction history: {ctx['date_from']} to {ctx['date_to']}
- Total transactions: {ctx['total_transactions']}

Always use tools to retrieve real data before answering financial questions.
Format currency as $ with 2 decimal places.
Be concise but insightful. Highlight trends and anomalies when relevant."""

    def chat(
        self,
        user_message: str,
        chat_history: list,
    ) -> tuple:
        """
        Returns: (answer, updated_history, tool_calls_made, route_used)
        route_used: 'tool_use' or 'rag'
        """
        if self.rag.is_document_question(user_message):
            answer, chunks = self.rag.answer(user_message)
            sources        = [c["source"] for c in chunks]
            tool_calls     = [("RAG retrieval", {"query": user_message},
                               f"Retrieved from: {sources}")]
            chat_history.append({"role": "USER",    "message": user_message})
            chat_history.append({"role": "CHATBOT", "message": answer})
            return answer, chat_history, tool_calls, "rag"

        tool_calls_made = []
        current_message = user_message
        iterations      = 0

        while iterations < self.MAX_TOOL_ITERATIONS:
            iterations += 1

            response = self.client.chat(
                model="command-r-08-2024",
                message=current_message,
                preamble=self.system_prompt,
                chat_history=chat_history,
                tools=TOOL_DEFINITIONS,
                temperature=0.3,
            )

            if response.finish_reason == "COMPLETE":
                final_answer = response.text
                chat_history.append({"role": "USER",    "message": user_message})
                chat_history.append({"role": "CHATBOT", "message": final_answer})
                return final_answer, chat_history, tool_calls_made, "tool_use"

            elif response.finish_reason == "TOOL_CALL":
                tool_results = []
                for tool_call in response.tool_calls:
                    result = dispatch_tool(
                        tool_call.name,
                        tool_call.parameters or {},
                        self.engine
                    )
                    tool_calls_made.append((
                        tool_call.name,
                        tool_call.parameters or {},
                        result
                    ))
                    tool_results.append(
                        cohere.ToolResult(
                            call=tool_call,
                            outputs=[{"result": result}]
                        )
                    )

                response2 = self.client.chat(
                    model="command-r-08-2024",
                    message="",
                    preamble=self.system_prompt,
                    chat_history=chat_history,
                    tools=TOOL_DEFINITIONS,
                    tool_results=tool_results,
                    temperature=0.3,
                )

                if response2.finish_reason == "COMPLETE":
                    final_answer = response2.text
                    chat_history.append({"role": "USER",    "message": user_message})
                    chat_history.append({"role": "CHATBOT", "message": final_answer})
                    return final_answer, chat_history, tool_calls_made, "tool_use"

                current_message = ""

        return (
            "Reached maximum tool iterations. Please try a simpler question.",
            chat_history,
            tool_calls_made,
            "tool_use",
        )
