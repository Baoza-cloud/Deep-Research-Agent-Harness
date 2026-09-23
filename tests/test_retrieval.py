# ruff: noqa: F403, F405

from tests._support import *


class MemoryTests(unittest.TestCase):
    def test_deduplication_and_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = SharedMemory(Path(temp_dir) / "memory.sqlite")
            first_id, first_outcome = memory.add("公司允许员工远程办公。", "policy")
            second_id, second_outcome = memory.add("公司允许员工远程办公。", "policy")
            self.assertEqual(first_outcome, "inserted")
            self.assertEqual(first_id, second_id)
            self.assertEqual(second_outcome, "duplicate")
            self.assertIn("policy", memory.build_context("远程办公"))
            memory.close()


class WebBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_tavily_normalization_without_network(self):
        client = FakeTavilyClient()
        backend = TavilySearchBackend(client=client, include_domains=["example.com"])
        rows = await backend.search("test query", limit=3)
        self.assertEqual(rows[0]["source_type"], "web")
        self.assertEqual(rows[0]["content"], "short snippet")
        self.assertEqual(rows[0]["metadata"]["raw_content_chars"], 31)
        self.assertEqual(client.kwargs["max_results"], 3)
        self.assertFalse(client.kwargs["include_answer"])
        self.assertEqual(client.kwargs["include_domains"], ["example.com"])

    async def test_tavily_can_prefer_bounded_raw_content(self):
        client = FakeTavilyClient()
        backend = TavilySearchBackend(
            client=client,
            prefer_raw_content=True,
            max_content_chars=10,
        )
        rows = await backend.search("test query")
        # The constructor enforces a defensive minimum bound.
        self.assertEqual(rows[0]["content"], "full normalized article content")

    async def test_tavily_can_exclude_url_fragments(self):
        client = FakeTavilyClient()
        backend = TavilySearchBackend(
            client=client,
            exclude_url_fragments=["/article"],
        )
        rows = await backend.search("test query")
        self.assertEqual(rows, [])

    async def test_tavily_enforces_include_domains_client_side(self):
        client = FakeTavilyClient()
        backend = TavilySearchBackend(
            client=client,
            include_domains=["docs.example.com"],
        )
        rows = await backend.search("test query")
        self.assertEqual(rows, [])

    async def test_composite_rrf_deduplicates_urls(self):
        local = FixedBackend(
            [{"content": "local", "url": "https://example.com/article#local", "source": "local"}]
        )
        web = FixedBackend(
            [
                {
                    "content": "longer web content",
                    "url": "https://example.com/article",
                    "source": "web",
                }
            ]
        )
        backend = CompositeSearchBackend({"local": local, "web": web})
        rows = await backend.search("query")
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]["retrieval_backends"]), {"local", "web"})
        self.assertEqual(rows[0]["content"], "longer web content")

    async def test_web_evaluation_uses_domains_not_fixed_rank(self):
        sample = {
            "id": "fixture",
            "question": "q",
            "required_domains": ["docs.example.com"],
            "required_terms": ["async", "search"],
            "min_results": 1,
        }
        rows = [
            {
                "title": "Search docs",
                "content": "An async client",
                "url": "https://docs.example.com/reference",
            }
        ]
        result = evaluate_sample(sample, rows, 0.1)
        self.assertTrue(result["passed"])
        self.assertEqual(result["domain_recall"], 1.0)
