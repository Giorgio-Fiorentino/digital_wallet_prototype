"""
models/llm_engine.py

Agentic loop with tool use. Routes between:
    - Tool Use: transaction data questions
    - RAG: card terms / fees / benefits questions

Updated for Cohere v5 (ClientV2 API).
"""

import os
import json
import cohere
from dotenv import load_dotenv
from models.wallet_engine import WalletEngine
from models.rag_engine import RAGEngine

load_dotenv()

# Cohere v5 ClientV2 tool format (OpenAI-compatible)
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_transactions",
            "description": "Retrieve filtered transactions by source, category, date, or month.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source":    {"type": "string",  "description": "Card or wallet name."},
                    "category":  {"type": "string",  "description": "Spending category."},
                    "month":     {"type": "string",  "description": "Month name e.g. January."},
                    "date_from": {"type": "string",  "description": "Start date YYYY-MM-DD."},
                    "date_to":   {"type": "string",  "description": "End date YYYY-MM-DD."},
                    "limit":     {"type": "integer", "description": "Max rows to return."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_spending_summary",
            "description": "Spending totals by category. Use for how much did I spend questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Filter by payment source."},
                    "month":  {"type": "string", "description": "Filter by month name."},
                    "period": {"type": "string", "description": "last_30_days or last_7_days."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_monthly_comparison",
            "description": "Compare spending across months. Use for trend questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source":        {"type": "string",  "description": "Filter by source."},
                    "category":      {"type": "string",  "description": "Filter by category."},
                    "last_n_months": {"type": "integer", "description": "Months to compare."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_merchants",
            "description": "Top merchants by spend. Use for where am I spending most questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string",  "description": "Filter by source."},
                    "month":  {"type": "string",  "description": "Filter by month."},
                    "limit":  {"type": "integer", "description": "Number of merchants."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_source_breakdown",
            "description": "Spending split across all cards and digital wallets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "month": {"type": "string", "description": "Filter by month."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": "Flag unusually large transactions. Use for suspicious charges questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Filter by source."},
                },
                "required": [],
            },
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


def _extract_text(content) -> str:
    """Extract plain text from a Cohere v5 content block list or string."""
    if isinstance(content, str):
        return content
    text_parts = []
    for block in content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
    return "".join(text_parts)


class WalletLLM:

    MAX_TOOL_ITERATIONS = 5

    def __init__(self, engine: WalletEngine, rag: RAGEngine):
        self.engine = engine
        self.rag    = rag
        self.client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
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
        chat_history uses v2 format: {"role": "user"/"assistant", "content": "..."}
        """
        if self.rag.is_document_question(user_message):
            answer, chunks = self.rag.answer(user_message)
            sources    = [c["source"] for c in chunks]
            tool_calls = [("RAG retrieval", {"query": user_message},
                           f"Retrieved from: {sources}")]
            chat_history.append({"role": "user",      "content": user_message})
            chat_history.append({"role": "assistant",  "content": answer})
            return answer, chat_history, tool_calls, "rag"

        # Build messages: system + history + current user turn
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": user_message})

        tool_calls_made = []
        iterations      = 0

        while iterations < self.MAX_TOOL_ITERATIONS:
            iterations += 1

            response = self.client.chat(
                model="command-r-08-2024",
                messages=messages,
                tools=TOOL_DEFINITIONS,
                temperature=0.3,
            )

            if response.finish_reason == "STOP":
                final_answer = _extract_text(response.message.content)
                chat_history.append({"role": "user",     "content": user_message})
                chat_history.append({"role": "assistant", "content": final_answer})
                return final_answer, chat_history, tool_calls_made, "tool_use"

            elif response.finish_reason == "TOOL_USE":
                # Append the assistant's tool-call message
                messages.append({"role": "assistant", "content": response.message.content})

                # Execute each tool and append results
                for tool_call in response.message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        tool_args = {}
                    result = dispatch_tool(tool_name, tool_args, self.engine)
                    tool_calls_made.append((tool_name, tool_args, result))
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tool_call.id,
                        "content":      result,
                    })
                # Loop again to get the final answer

            else:
                break

        return (
            "Reached maximum tool iterations. Please try a simpler question.",
            chat_history,
            tool_calls_made,
            "tool_use",
        )
