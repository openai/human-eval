import os
import tempfile
import unittest
from unittest import mock

from human_eval.data import read_problems, stream_jsonl, write_jsonl


class JsonlRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def test_plain_round_trip(self):
        rows = [{"task_id": "a", "n": 1}, {"task_id": "b", "n": 2}]
        write_jsonl(self.path("x.jsonl"), rows)
        self.assertEqual(list(stream_jsonl(self.path("x.jsonl"))), rows)

    def test_gzip_round_trip(self):
        rows = [{"task_id": "a"}, {"task_id": "b"}]
        write_jsonl(self.path("x.jsonl.gz"), rows)
        self.assertEqual(list(stream_jsonl(self.path("x.jsonl.gz"))), rows)

    def test_append_adds_rows(self):
        write_jsonl(self.path("x.jsonl"), [{"i": 1}])
        write_jsonl(self.path("x.jsonl"), [{"i": 2}], append=True)
        self.assertEqual(list(stream_jsonl(self.path("x.jsonl"))), [{"i": 1}, {"i": 2}])

    def test_blank_lines_are_skipped(self):
        with open(self.path("x.jsonl"), "w") as fp:
            fp.write('{"i": 1}\n\n   \n{"i": 2}\n')
        self.assertEqual(list(stream_jsonl(self.path("x.jsonl"))), [{"i": 1}, {"i": 2}])

    def test_read_problems_keys_by_task_id(self):
        write_jsonl(self.path("p.jsonl"), [{"task_id": "t/0"}, {"task_id": "t/1"}])
        self.assertEqual(sorted(read_problems(self.path("p.jsonl"))), ["t/0", "t/1"])

    def test_home_directory_is_expanded_when_reading(self):
        # write_jsonl expands "~"; stream_jsonl must read back the same path.
        for name in ("x.jsonl", "x.jsonl.gz"):
            with self.subTest(name=name):
                env = {"HOME": self.tmp.name, "USERPROFILE": self.tmp.name}
                with mock.patch.dict(os.environ, env):
                    write_jsonl("~/" + name, [{"i": 1}])
                    self.assertTrue(os.path.exists(self.path(name)))
                    self.assertEqual(list(stream_jsonl("~/" + name)), [{"i": 1}])


if __name__ == "__main__":
    unittest.main()
