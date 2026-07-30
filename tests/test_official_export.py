import json
import tempfile
import unittest
from pathlib import Path

from graph_rescue.io import write_jsonl
from graph_rescue.official_export import export_official_predictions


class OfficialExportTests(unittest.TestCase):
    def test_hotpot_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queries = root / "queries.jsonl"
            results = root / "results.jsonl"
            output = root / "predictions.json"
            write_jsonl(
                queries,
                [
                    {
                        "id": "q1",
                        "question": "Question?",
                        "answers": ["Answer"],
                        "supporting_passage_ids": ["p1"],
                    }
                ],
            )
            write_jsonl(
                results,
                [
                    {
                        "query_id": "q1",
                        "policy": "mrv_gated",
                        "predictions": {"reader": "Answer"},
                        "reader_evidence": {
                            "reader": {
                                "supporting_facts": [["Title", 0]],
                                "evidence_triples": [],
                            }
                        },
                    }
                ],
            )
            export_official_predictions(
                dataset="hotpot",
                query_results_path=results,
                queries_path=queries,
                output_path=output,
                policy="mrv_gated",
                reader="reader",
            )
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["answer"]["q1"], "Answer")
            self.assertEqual(value["sp"]["q1"], [["Title", 0]])


if __name__ == "__main__":
    unittest.main()
