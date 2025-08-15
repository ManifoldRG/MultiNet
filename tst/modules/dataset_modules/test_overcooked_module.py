import unittest
import os
import sys
from pathlib import Path
import shutil
import pickle
import csv
import base64

# Add project root to sys.path
ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / 'src' / 'third_party' / 'overcooked_ai' / 'src'))

from src.v1.centralized_processor import OvercookedAIProcessor, ProcessResult

class TestOvercookedAIProcessor(unittest.TestCase):

    def setUp(self):
        """Set up test data and directories."""
        self.test_input_dir = Path("test_data_temp")
        self.test_output_dir = Path("test_output_temp")
        self.overcooked_input_dir = self.test_input_dir / "overcooked_ai"
        self.overcooked_output_dir = self.test_output_dir / "overcooked_ai"

        # Create directories
        self.overcooked_input_dir.mkdir(parents=True, exist_ok=True)
        self.overcooked_output_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy data
        self.dummy_data = [{'state': '{"players": [{"position": [1, 1], "orientation": [0, -1], "held_object": null}], "objects": [], "bonus_orders": [], "all_orders": [], "timestep": 0}', 'layout': '["X", " ", "X"], ["X", "P", "X"], ["X", " ", "X"]'}]
        self.dummy_pickle_path = self.overcooked_input_dir / 'dummy_data.pickle'
        with open(self.dummy_pickle_path, 'wb') as picklefile:
            pickle.dump(self.dummy_data, picklefile)

    def tearDown(self):
        """Clean up test data and directories."""
        if self.test_input_dir.exists():
            shutil.rmtree(self.test_input_dir)
        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)

    def test_process_success(self):
        """Test successful processing of an Overcooked AI dataset."""
        processor = OvercookedAIProcessor("overcooked_ai", self.test_input_dir, self.test_output_dir)
        result = processor.process()

        self.assertTrue(result.success)
        self.assertIsInstance(result, ProcessResult)
        self.assertEqual(result.name, "overcooked_ai")
        self.assertGreater(result.files_processed, 0)
        self.assertIsNone(result.error)

        # Check for output files
        output_prefix = self.overcooked_output_dir / "test" / "dummy_data"
        processed_csv_path = output_prefix.with_suffix('.csv')
        processed_pickle_path = output_prefix.with_suffix('.pickle')
        self.assertTrue(processed_csv_path.exists())
        self.assertTrue(processed_pickle_path.exists())

        # Verify content of processed csv
        with open(processed_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            processed_data = list(reader)
            self.assertEqual(len(processed_data), 1)
            # Check if state is a valid base64 string
            try:
                base64.b64decode(processed_data[0]['state'])
            except Exception as e:
                self.fail(f"State is not a valid base64 string: {e}")

    def test_process_failure_no_input_file(self):
        """Test processing failure when no input file is found."""
        # Remove the dummy file
        os.remove(self.dummy_pickle_path)

        processor = OvercookedAIProcessor("overcooked_ai", self.test_input_dir, self.test_output_dir)
        result = processor.process()

        self.assertFalse(result.success)
        self.assertIsInstance(result, ProcessResult)
        self.assertEqual(result.name, "overcooked_ai")
        self.assertIsNotNone(result.error)
        self.assertIn("No pickle or CSV file found", result.error)

if __name__ == '__main__':
    unittest.main()
