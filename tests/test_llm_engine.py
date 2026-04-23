from models.llm_engine import _compute_confidence


class TestComputeConfidence:
    def test_high_single_tool_first_iteration(self):
        calls = [("get_spending_summary", {}, "result")]
        assert _compute_confidence(1, calls) == "high"

    def test_medium_two_iterations(self):
        calls = [
            ("get_spending_summary", {}, "r1"),
            ("get_top_merchants", {}, "r2"),
        ]
        assert _compute_confidence(2, calls) == "medium"

    def test_medium_multiple_tools_first_iteration(self):
        calls = [
            ("get_spending_summary", {}, "r1"),
            ("get_top_merchants", {}, "r2"),
        ]
        assert _compute_confidence(1, calls) == "medium"

    def test_low_no_real_tools_called(self):
        calls = [
            ("_debug_finish_reason", {"iteration": 1}, "COMPLETE"),
        ]
        assert _compute_confidence(1, calls) == "low"

    def test_low_max_iterations(self):
        calls = [("get_spending_summary", {}, "r")] * 5
        assert _compute_confidence(5, calls) == "low"

    def test_debug_entries_ignored(self):
        calls = [
            ("_debug_finish_reason", {}, "COMPLETE"),
            ("get_spending_summary", {}, "result"),
        ]
        assert _compute_confidence(1, calls) == "high"
