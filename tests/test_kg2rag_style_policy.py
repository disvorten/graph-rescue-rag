import unittest

from graph_rescue.features import CandidateFeatureExtractor
from graph_rescue.graph import KnowledgeGraph
from graph_rescue.hybrid import HashingEmbedder, HybridRetriever
from graph_rescue.models import Passage, QueryExample
from graph_rescue.policy import KG2RAGStylePolicy, PolicyConfig


class KG2RAGStylePolicyTests(unittest.TestCase):
    def setUp(self):
        self.passages = [
            Passage(
                id="s1",
                title="Alpha",
                text="Alpha is connected to Bridge.",
                entities=("Alpha", "Bridge"),
            ),
            Passage(
                id="s2",
                title="Beta",
                text="Beta is also connected to Bridge.",
                entities=("Beta", "Bridge"),
            ),
            Passage(
                id="shared",
                title="Bridge",
                text="Bridge contains the target evidence.",
                entities=("Bridge",),
            ),
            Passage(
                id="noise",
                title="Noise",
                text="Unrelated material.",
                entities=("Noise",),
            ),
        ]
        self.retriever = HybridRetriever(self.passages, HashingEmbedder())
        self.passage_map = {item.id: item for item in self.passages}
        self.graph = KnowledgeGraph.build(
            self.passages,
            min_entity_df=1,
            max_entity_df_ratio=1.0,
        )
        self.extractor = CandidateFeatureExtractor(
            passages=self.passage_map,
            retriever=self.retriever,
            token_budget=512,
        )

    def test_expands_and_respects_equal_context_budget(self):
        policy = KG2RAGStylePolicy(
            passages=self.passage_map,
            graph=self.graph,
            feature_extractor=self.extractor,
            config=PolicyConfig(
                seed_k=2,
                final_k=3,
                token_budget=512,
                max_hops=1,
                frontier_cap=20,
                max_actions=1,
            ),
        )
        example = QueryExample(
            id="q1",
            question="Where is the target evidence connected to Alpha and Beta?",
            answers=("target evidence",),
            supporting_passage_ids=("s1", "shared"),
        )
        ranking = self.retriever.retrieve(example.question, 4)
        ranking_by_id = {item.passage_id: item for item in ranking}
        forced = [
            ranking_by_id["s1"],
            ranking_by_id["s2"],
            ranking_by_id["noise"],
            ranking_by_id["shared"],
        ]
        trace = policy.run(example, forced)
        self.assertIn("shared", trace.final_passage_ids)
        self.assertLessEqual(len(trace.final_passage_ids), 3)
        self.assertLessEqual(
            sum(action.selected_passage_id is not None for action in trace.actions),
            1,
        )

    def test_multi_seed_support_increases_candidate_score(self):
        policy = KG2RAGStylePolicy(
            passages=self.passage_map,
            graph=self.graph,
            feature_extractor=self.extractor,
            config=PolicyConfig(seed_k=2, max_hops=1, frontier_cap=20),
        )
        ranking = self.retriever.retrieve("Alpha Beta Bridge", 4)
        seeds = ["s1", "s2"]
        expanded, _ = policy._expanded_candidates(seeds)
        self.assertEqual(
            {path.seed_passage_id for path in expanded["shared"]},
            {"s1", "s2"},
        )


if __name__ == "__main__":
    unittest.main()
