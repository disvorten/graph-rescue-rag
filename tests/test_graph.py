import unittest

from graph_rescue.graph import KnowledgeGraph
from graph_rescue.models import Passage


class KnowledgeGraphTests(unittest.TestCase):
    def test_explicit_link_produces_candidate_path(self):
        passages = [
            Passage(id="a", title="A", text="alpha", links=("b",)),
            Passage(id="b", title="B", text="beta"),
            Passage(id="c", title="C", text="gamma"),
        ]
        graph = KnowledgeGraph.build(
            passages, min_entity_df=1, max_entity_df_ratio=1.0
        )
        paths, reads = graph.candidate_paths(["a"], max_hops=2, cap=10)
        self.assertGreater(reads, 0)
        self.assertEqual([item.target_passage_id for item in paths], ["b"])
        self.assertEqual(paths[0].hop_count, 1)

    def test_excluded_passages_are_not_returned(self):
        passages = [
            Passage(id="a", title="A", text="alpha", links=("b",)),
            Passage(id="b", title="B", text="beta"),
        ]
        graph = KnowledgeGraph.build(
            passages, min_entity_df=1, max_entity_df_ratio=1.0
        )
        paths, _ = graph.candidate_paths(
            ["a"], excluded_passage_ids=["b"], max_hops=2, cap=10
        )
        self.assertEqual(paths, [])

    def test_corruption_is_deterministic_and_tracks_changes(self):
        passages = [
            Passage(id="p1", title="One", text="A", links=("p2",)),
            Passage(id="p2", title="Two", text="B", links=("p3",)),
            Passage(id="p3", title="Three", text="C"),
        ]
        first = KnowledgeGraph.build(
            passages, min_entity_df=1, max_entity_df_ratio=1.0
        )
        second = KnowledgeGraph.build(
            passages, min_entity_df=1, max_entity_df_ratio=1.0
        )
        first.corrupt(edge_dropout_rate=0.25, false_edge_ratio=0.5, seed=7)
        second.corrupt(edge_dropout_rate=0.25, false_edge_ratio=0.5, seed=7)
        self.assertEqual(first.stats, second.stats)
        self.assertEqual(first.adjacency, second.adjacency)

    def test_title_links_mode_removes_non_title_surface_entities(self):
        passages = [
            Passage(
                id="p1",
                title="Alpha",
                text="Alpha discusses Shared Topic",
                entities=("alpha", "shared topic"),
            ),
            Passage(
                id="p2",
                title="Beta",
                text="Beta discusses Shared Topic",
                entities=("beta", "shared topic"),
            ),
        ]
        full = KnowledgeGraph.build(
            passages,
            min_entity_df=1,
            max_entity_df_ratio=1.0,
            entity_mode="provided",
        )
        titles = KnowledgeGraph.build(
            passages,
            min_entity_df=1,
            max_entity_df_ratio=1.0,
            entity_mode="title_links",
        )
        self.assertGreater(full.stats.edges, titles.stats.edges)


if __name__ == "__main__":
    unittest.main()
