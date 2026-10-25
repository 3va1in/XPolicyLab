"""Check module discovery without requiring a model checkpoint."""
import importlib
import importlib.util
from pathlib import Path
import unittest


class RenamedLayoutTests(unittest.TestCase):
    def test_runtime_resolves_bundled_statistics_after_rename(self):
        runtime = importlib.import_module('go3_nano_preview.runtime_paths')
        package = importlib.import_module('go3_nano_preview')
        policy = Path(__file__).resolve().parents[1]
        self.assertEqual(Path(runtime.__file__).parent, policy / 'go3_nano_preview')
        self.assertEqual(Path(package.__file__).parent, Path(runtime.__file__).parent)
        self.assertEqual(Path(runtime.robotwin_stats_path()), policy / 'assets/stats/robotwin.json')
        self.assertTrue(Path(runtime.robotwin_stats_path()).is_file())

    def test_retained_service_filename_is_discoverable(self):
        spec = importlib.util.find_spec('go3_nano_preview.serve_GO-3')
        self.assertIsNotNone(spec)
        self.assertEqual(Path(spec.origin).name, 'serve_GO-3.py')
        self.assertTrue(Path(spec.origin).is_file())

    def test_training_launcher_resolves_from_renamed_directory(self):
        launcher = importlib.import_module('go3_nano_preview.sft.launch')
        self.assertEqual(Path(launcher.__file__).parents[1].name, 'go3_nano_preview')


if __name__ == '__main__':
    unittest.main()
