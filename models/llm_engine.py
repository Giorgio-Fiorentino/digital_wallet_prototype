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


HEALTH_SCORE_SYSTEM_PROMPT = (
    "You are a financial health AI. Given spending signals, return ONLY valid JSON "
    "with exactly these keys:\n"
    '{"grade": "A+" | "A" | "B+" | "B" | "C" | "D", '
    '"score": <integer 0-100>, '
    '"summary": "<one sentence headline>", '
    '"breakdown": {'
    '"fraud_rate": "<short explanation>", '
    '"anomaly_count": "<short explanation>", '
    '"category_diversity": "<short explanation>", '
    '"monthly_variance": "<short explanation>", '
    '"top_category_share": "<short explanation>"}}\n'
    "Return nothing but the JSON object. No markdown, no code fences."
)

STRATEGY_SYSTEM_PROMPT = (
    "You are a financial strategy AI. Given a topic about a user's spending, "
    "return ONLY valid JSON with exactly these six keys:\n"
    '{"urgency": "low" | "medium" | "high", '
    '"root_cause": "<one sentence diagnosis>", '
    '"narrative": "<2-3 sentences of practical, direct advice written to the user — concrete, specific, and actionable. Mention realistic targets or behaviours the user should adopt>", '
    '"actions": ["<step 1>", "<step 2>", "<step 3>"], '
    '"talking_points": ["<phrase 1>", "<phrase 2>"], '
    '"risk_warning": "<one sentence risk note>"}\n'
    "Return nothing but the JSON object. No markdown, no code fences."
)


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


def _compute_confidence(iterations: int, tool_calls_made: list) -> str:
    """
    Heuristic confidence from agentic loop behaviour.
    Debug entries (names starting with '_') are excluded from the real-call count.
    """
    real_calls = [t for t in tool_calls_made if not t[0].startswith("_")]
    if not real_calls:
        return "low"
    if iterations == 1 and len(real_calls) == 1:
        return "high"
    if iterations <= 2:
        return "medium"
    return "low"


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

    def compute_health_score(self) -> dict:
        """
        Returns a health score dict with keys: grade, score, summary, breakdown.
        Returns {"error": "unavailable"} on any failure.
        """
        inputs = self.engine.get_health_inputs()
        user_prompt = (
            f"Financial signals:\n"
            f"- Fraud rate: {inputs['fraud_rate']:.2%}\n"
            f"- Anomaly count: {inputs['anomaly_count']}\n"
            f"- Category diversity: {inputs['category_diversity']} categories\n"
            f"- Monthly spend std dev: ${inputs['monthly_variance']:,.2f}\n"
            f"- Top category share: {inputs['top_category_share']:.1%}\n\n"
            "Compute the financial health grade and score."
        )
        try:
            response = self.client.chat(
                model="command-r-08-2024",
                messages=[
                    {"role": "system", "content": HEALTH_SCORE_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.1,
            )
            raw = _extract_text(response.message.content).strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw.strip())
        except Exception:
            return {"error": "unavailable"}

        required = {"grade", "score", "summary", "breakdown"}
        if not required.issubset(data.keys()):
            return {"error": "unavailable"}
        valid_grades = {"A+", "A", "B+", "B", "C", "D"}
        if data.get("grade") not in valid_grades:
            return {"error": "unavailable"}
        return data

    def generate_strategy(self, topic: str) -> dict:
        """
        Returns structured strategy dict with keys:
          urgency, root_cause, actions, talking_points, risk_warning.
        Returns {"error": "<reason>"} on failure.
        """
        ctx    = self.engine.get_data_context()
        health = self.engine.get_health_inputs()
        user_prompt = (
            f"Topic: {topic}\n\n"
            f"User financial context:\n"
            f"- Total transactions: {ctx['total_transactions']}\n"
            f"- Date range: {ctx['date_from']} to {ctx['date_to']}\n"
            f"- Payment sources: {', '.join(ctx['sources'])}\n"
            f"- Categories: {', '.join(ctx['categories'])}\n"
            f"- Fraud rate: {health['fraud_rate']:.1%}\n"
            f"- Anomaly count: {health['anomaly_count']}\n"
            f"- Monthly spend std dev: ${health['monthly_variance']:,.2f}\n"
            f"- Top category share: {health['top_category_share']:.1%}"
        )
        try:
            response = self.client.chat(
                model="command-r-08-2024",
                messages=[
                    {"role": "system", "content": STRATEGY_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.1,
            )
            raw = _extract_text(response.message.content).strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw.strip())
        except Exception as e:
            msg = str(e)
            # Truncate verbose API error objects to just the key message
            if "message" in msg and len(msg) > 120:
                try:
                    import re
                    match = re.search(r"'message':\s*'([^']+)'", msg)
                    msg = match.group(1) if match else msg[:120]
                except Exception:
                    msg = msg[:120]
            return {"error": msg}

        required = {"urgency", "root_cause", "actions", "talking_points", "risk_warning"}
        if not required.issubset(data.keys()):
            return {"error": "Incomplete strategy response — missing required fields"}
        if data["urgency"] not in ("low", "medium", "high"):
            data["urgency"] = "medium"
        if not isinstance(data["actions"], list) or not data["actions"]:
            return {"error": "Invalid actions field"}
        if not isinstance(data["talking_points"], list) or not data["talking_points"]:
            return {"error": "Invalid talking_points field"}
        if not data.get("root_cause") or not data.get("risk_warning"):
            return {"error": "Missing root_cause or risk_warning"}
        return data

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
            return answer, chat_history, tool_calls, "rag", "high"

        # Build messages: system + history + current user turn
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": user_message})

        tool_calls_made  = []
        iterations       = 0
        tool_iterations  = 0   # counts only TOOL_CALL rounds, not the final synthesis

        while iterations < self.MAX_TOOL_ITERATIONS:
            iterations += 1

            try:
                response = self.client.chat(
                    model="command-r-08-2024",
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    temperature=0.3,
                )
            except Exception as e:
                err = f"API call failed: {type(e).__name__}: {e}"
                tool_calls_made.append(("ERROR", {}, err))
                return err, chat_history, tool_calls_made, "tool_use"

            fr = response.finish_reason
            tool_calls_made.append(("_debug_finish_reason", {"iteration": iterations}, str(fr)))

            if fr == "COMPLETE":
                content = response.message.content
                final_answer = _extract_text(content) if content else ""
                chat_history.append({"role": "user",     "content": user_message})
                chat_history.append({"role": "assistant", "content": final_answer})
                confidence = _compute_confidence(tool_iterations, tool_calls_made)
                return final_answer, chat_history, tool_calls_made, "tool_use", confidence

            elif fr == "TOOL_CALL":
                tool_iterations += 1
                # Append the full assistant message (contains tool_calls, content may be None)
                messages.append(response.message)

                # Execute each tool and append results
                for tool_call in response.message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments or "{}")
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
                tool_calls_made.append(("_debug_unexpected_reason", {"value": fr}, "broke out of loop"))
                break

        return (
            "Reached maximum tool iterations. Please try a simpler question.",
            chat_history,
            tool_calls_made,
            "tool_use",
            "low",
        )
