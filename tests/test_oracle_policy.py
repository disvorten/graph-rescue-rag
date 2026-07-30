import unittest

from graph_rescue.features import CandidateFeatureExtractor
from graph_rescue.graph import KnowledgeGraph
from graph_rescue.hybrid import HybridRetriever, HashingEmbedder
from graph_rescue.models import Passage, QueryExample
from graph_rescue.policy import PolicyConfig, RescuePolicy


class OraclePolicyTests(unittest.TestCase):
    def test_oracle_adds_only_reachable_missing_support(self):
        passages = [
            Passage(
                id="seed",
                title="Alpha",
                text="Alpha points to Beta.",
                entities=("Alpha", "Beta"),
            ),
            Passage(
                id="support",
                title="Beta",
                text="Beta contains the answer.",
                entities=("Beta",),
            ),
            Passage(
                id="noise",
                title="Gamma",
                text="Unrelated.",
                entities=("Gamma",),
            ),
        ]
        retriever = HybridRetriever(passages, HashingEmbedder())
        graph = KnowledgeGraph.build(
            passages, min_entity_df=1, max_entity_df_ratio=1.0
        )
        passage_map = {item.id: item for item in passages}
        extractor = CandidateFeatureExtractor(
            passages=passage_map,
            retriever=retriever,
            token_budget=512,
        )
        policy = RescuePolicy(
            name="oracle_upper_bound",
            passages=passage_map,
            graph=graph,
            feature_extractor=extractor,
            config=PolicyConfig(seed_k=1, final_k=2, max_actions=1),
            selector="oracle",
        )
        example = QueryExample(
            id="q1",
            question="What does Alpha point to?",
            answers=("answer",),
            supporting_passage_ids=("seed", "support"),
        )
        ranking = retriever.retrieve("Alpha", 3)
        trace = policy.run(example, ranking)
        self.assertIn("support", trace.final_passage_ids)
        self.assertNotIn("noise", [
            action.selected_passage_id for action in trace.actions
        ])


if __name__ == "__main__":
    unittest.main()
