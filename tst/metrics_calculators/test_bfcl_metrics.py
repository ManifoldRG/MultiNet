import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT_DIR)
import pytest
import numpy as np

from src.eval_harness.scoring.bfcl_metrics import BFCLMetricsCalculator


class TestBFCLMetricsCalculator:
    """Test suite for BFCL metrics calculator."""
    
    @pytest.fixture
    def calculator(self):
        """Create a BFCL metrics calculator instance."""
        return BFCLMetricsCalculator()
    
    def test_perfect_single_turn_match(self, calculator):
        """Test perfect match on single-turn conversation."""
        predictions = [
            {
                'conversation_id': 'conv_001',
                'predictions': [
                    {
                        'raw_output': 'get_weather(city="London")',
                        'extracted_calls': ['get_weather(city="London")']
                    }
                ],
                'num_turns': 1
            }
        ]
        
        ground_truths = [
            {
                'conversation_id': 'conv_001',
                'ground_truth': [
                    ['get_weather(city="London")']
                ],
                'num_turns': 1
            }
        ]
        
        metrics = calculator.calculate_metrics(predictions, ground_truths)
        
        # Should have perfect accuracy
        assert metrics['exact_match_accuracy'] == 1.0
        assert metrics['total_samples'] == 1
        assert metrics['total_invalid_turns'] == 0
        assert metrics['total_invalid_conversations'] == 0
        assert metrics['turn_level_accuracy']['turn_1_accuracy'] == 1.0
    
    def test_perfect_multi_turn_match(self, calculator):
        """Test perfect match on multi-turn conversation."""
        predictions = [
            {
                'conversation_id': 'conv_002',
                'predictions': [
                    {
                        'raw_output': 'the function call is get_weather(city="London")!',
                        'extracted_calls': ['get_weather(city="London")']
                    },
                    {
                        'raw_output': 'call set_reminder(time="tomorrow", message="bring umbrella")',
                        'extracted_calls': ['set_reminder(time="tomorrow", message="bring umbrella")']
                    }
                ],
                'num_turns': 2
            }
        ]
        
        ground_truths = [
            {
                'conversation_id': 'conv_002',
                'ground_truth': [
                    ['get_weather(city="London")'],
                    ['set_reminder(time="tomorrow", message="bring umbrella")']
                ],
                'num_turns': 2
            }
        ]
        
        metrics = calculator.calculate_metrics(predictions, ground_truths)
        
        assert metrics['exact_match_accuracy'] == 1.0
        assert metrics['turn_level_accuracy']['turn_1_accuracy'] == 1.0
        assert metrics['turn_level_accuracy']['turn_2_accuracy'] == 1.0
        assert metrics['avg_turn_of_first_failure'] == 3.0  # Never fails (num_turns + 1)
    
    def test_partial_match_first_turn_correct(self, calculator):
        """Test case where first turn is correct but second fails."""
        predictions = [
            {
                'conversation_id': 'conv_003',
                'predictions': [
                    {
                        'raw_output': 'get_weather(city="London")',
                        'extracted_calls': ['get_weather(city="London")']
                    },
                    {
                        'raw_output': 'set_reminder(time="today")',
                        'extracted_calls': ['set_reminder(time="today")']
                    }
                ],
                'num_turns': 2
            }
        ]
        
        ground_truths = [
            {
                'conversation_id': 'conv_003',
                'ground_truth': [
                    ['get_weather(city="London")'],
                    ['set_reminder(time="tomorrow")']  # Different from prediction
                ],
                'num_turns': 2
            }
        ]
        
        metrics = calculator.calculate_metrics(predictions, ground_truths)
        
        assert metrics['exact_match_accuracy'] == 0.0  # Conversation didn't fully match
        assert metrics['turn_level_accuracy']['turn_1_accuracy'] == 1.0
        assert metrics['turn_level_accuracy']['turn_2_accuracy'] == 0.0
        assert metrics['avg_turn_of_first_failure'] == 2.0  # Failed on turn 2
    
    def test_multiple_function_calls_per_turn(self, calculator):
        """Test turn with multiple function calls."""
        predictions = [
            {
                'conversation_id': 'conv_004',
                'predictions': [
                    {
                        'raw_output': 'get_weather(city="London") get_time()',
                        'extracted_calls': ['get_weather(city="London")', 'get_time()']
                    }
                ],
                'num_turns': 1
            }
        ]
        
        ground_truths = [
            {
                'conversation_id': 'conv_004',
                'ground_truth': [
                    ['get_weather(city="London")', 'get_time()']
                ],
                'num_turns': 1
            }
        ]
        
        metrics = calculator.calculate_metrics(predictions, ground_truths)
        
        assert metrics['exact_match_accuracy'] == 1.0
        assert metrics['total_predicted_functions'] == 2
        assert metrics['total_ground_truth_functions'] == 2
    
    def test_empty_prediction_invalid_turn(self, calculator):
        """Test detection of invalid turn (empty prediction when expected)."""
        predictions = [
            {
                'conversation_id': 'conv_005',
                'predictions': [
                    {
                        'raw_output': '',
                        'extracted_calls': []  # No calls extracted
                    }
                ],
                'num_turns': 1
            }
        ]
        
        ground_truths = [
            {
                'conversation_id': 'conv_005',
                'ground_truth': [
                    ['get_weather(city="London")']  # Expected a call
                ],
                'num_turns': 1
            }
        ]
        
        metrics = calculator.calculate_metrics(predictions, ground_truths)
        
        assert metrics['total_invalid_turns'] == 1
        assert metrics['total_invalid_conversations'] == 1
        assert metrics['exact_match_accuracy'] == 0.0
    
    def test_raw_output_with_no_extractable_functions(self, calculator):
        """Test case where raw output has text but no extractable function calls."""
        predictions = [
            {
                'conversation_id': 'conv_005b',
                'predictions': [
                    {
                        'raw_output': 'I cannot help with that request. Please try again.',
                        'extracted_calls': []  # Model failed to extract function calls
                    }
                ],
                'num_turns': 1
            }
        ]
        
        ground_truths = [
            {
                'conversation_id': 'conv_005b',
                'ground_truth': [
                    ['get_weather(city="London")']  # Expected a call
                ],
                'num_turns': 1
            }
        ]
        
        metrics = calculator.calculate_metrics(predictions, ground_truths)
        
        # Should be treated as invalid turn (no function calls when expected)
        assert metrics['total_invalid_turns'] == 1
        assert metrics['total_invalid_conversations'] == 1
        assert metrics['exact_match_accuracy'] == 0.0
        # But raw output should still be used for similarity calculation
        assert metrics['avg_similarity_score'] >= 0.0  # Some similarity score calculated
    
    def test_unexpected_function_calls(self, calculator):
        """Test case where model outputs function calls when none were expected (abstention)."""
        predictions = [
            {
                'conversation_id': 'conv_005c',
                'predictions': [
                    {
                        'raw_output': 'get_weather(city="London")',
                        'extracted_calls': ['get_weather(city="London")']
                    }
                ],
                'num_turns': 1
            }
        ]
        
        ground_truths = [
            {
                'conversation_id': 'conv_005c',
                'ground_truth': [
                    []  # No function calls expected (abstention case)
                ],
                'num_turns': 1
            }
        ]
        
        metrics = calculator.calculate_metrics(predictions, ground_truths)
        
        # Should not match (predicted calls when none expected)
        assert metrics['exact_match_accuracy'] == 0.0
        assert metrics['turn_level_accuracy']['turn_1_accuracy'] == 0.0
        # Invalid turn is only for empty predictions when calls expected, not the reverse
        assert metrics['total_invalid_turns'] == 0
    
    def test_recall_until_failure_partial_match(self, calculator):
        """Test recall until failure when some functions match."""
        predictions = [
            {
                'conversation_id': 'conv_006',
                'predictions': [
                    {
                        'raw_output': 'func1() func2() func3_wrong()',
                        'extracted_calls': ['func1()', 'func2()', 'func3_wrong()']
                    }
                ],
                'num_turns': 1
            }
        ]
        
        ground_truths = [
            {
                'conversation_id': 'conv_006',
                'ground_truth': [
                    ['func1()', 'func2()', 'func3()']  # Third function different
                ],
                'num_turns': 1
            }
        ]
        
        metrics = calculator.calculate_metrics(predictions, ground_truths)
        
        # Should match 2 out of 3 functions before failure
        assert metrics['turn_level_recalls_until_failure']['turn_1_recall_until_failure'] == pytest.approx(2.0 / 3.0)
    
    def test_backward_compatibility_string_predictions(self, calculator):
        """Test backward compatibility with string predictions (legacy format)."""
        predictions = [
            {
                'conversation_id': 'conv_007',
                'predictions': [
                    'get_weather(city="London")',  # String instead of dict
                ],
                'num_turns': 1
            }
        ]
        
        ground_truths = [
            {
                'conversation_id': 'conv_007',
                'ground_truth': [
                    ['get_weather(city="London")']
                ],
                'num_turns': 1
            }
        ]
        
        # Should handle string predictions gracefully
        metrics = calculator.calculate_metrics(predictions, ground_truths)
        
        # With backward compatibility, extracted_calls is empty for strings
        assert metrics['total_samples'] == 1
        assert 'exact_match_accuracy' in metrics
    
    def test_mixed_string_and_dict_predictions(self, calculator):
        """Test handling mix of structured and legacy predictions."""
        predictions = [
            {
                'conversation_id': 'conv_008',
                'predictions': [
                    {
                        'raw_output': 'func1()',
                        'extracted_calls': ['func1()']
                    },
                    'func2()',  # String format (legacy)
                ],
                'num_turns': 2
            }
        ]
        
        ground_truths = [
            {
                'conversation_id': 'conv_008',
                'ground_truth': [
                    ['func1()'],
                    ['func2()']
                ],
                'num_turns': 2
            }
        ]
        
        # Should handle mixed formats
        metrics = calculator.calculate_metrics(predictions, ground_truths)
        assert metrics['total_samples'] == 1
    
    def test_multiple_conversations(self, calculator):
        """Test metrics across multiple conversations."""
        predictions = [
            {
                'conversation_id': 'conv_009',
                'predictions': [
                    {
                        'raw_output': 'func1()',
                        'extracted_calls': ['func1()']
                    }
                ],
                'num_turns': 1
            },
            {
                'conversation_id': 'conv_010',
                'predictions': [
                    {
                        'raw_output': 'func2()',
                        'extracted_calls': ['func2()']
                    }
                ],
                'num_turns': 1
            }
        ]
        
        ground_truths = [
            {
                'conversation_id': 'conv_009',
                'ground_truth': [['func1()']],
                'num_turns': 1
            },
            {
                'conversation_id': 'conv_010',
                'ground_truth': [['func2()']],
                'num_turns': 1
            }
        ]
        
        metrics = calculator.calculate_metrics(predictions, ground_truths)
        
        assert metrics['total_samples'] == 2
        assert metrics['exact_match_accuracy'] == 1.0
    
    def test_similarity_metrics_calculated(self, calculator):
        """Test that similarity metrics are calculated."""
        predictions = [
            {
                'conversation_id': 'conv_011',
                'predictions': [
                    {
                        'raw_output': 'I will call get_weather with city London',
                        'extracted_calls': ['get_weather(city="London")']
                    }
                ],
                'num_turns': 1
            }
        ]
        
        ground_truths = [
            {
                'conversation_id': 'conv_011',
                'ground_truth': [
                    ['get_weather(city="London")']
                ],
                'num_turns': 1
            }
        ]
        
        metrics = calculator.calculate_metrics(predictions, ground_truths)
        
        # Similarity metrics should be present
        assert 'avg_similarity_score' in metrics
        assert 'max_similarity_score' in metrics
        assert 'min_similarity_score' in metrics
        assert 'similarity_std' in metrics
        assert 'high_similarity_percentage' in metrics
        assert 0 <= metrics['avg_similarity_score'] <= 1
    
    def test_turn_level_accuracy_varying_lengths(self, calculator):
        """Test turn-level accuracy with conversations of different lengths."""
        predictions = [
            {
                'conversation_id': 'conv_012',
                'predictions': [
                    {'raw_output': 'func1()', 'extracted_calls': ['func1()']},
                    {'raw_output': 'func2()', 'extracted_calls': ['func2()']},
                ],
                'num_turns': 2
            },
            {
                'conversation_id': 'conv_013',
                'predictions': [
                    {'raw_output': 'func1()', 'extracted_calls': ['func1()']},
                ],
                'num_turns': 1
            }
        ]
        
        ground_truths = [
            {
                'conversation_id': 'conv_012',
                'ground_truth': [['func1()'], ['func2()']],
                'num_turns': 2
            },
            {
                'conversation_id': 'conv_013',
                'ground_truth': [['func1()']],
                'num_turns': 1
            }
        ]
        
        metrics = calculator.calculate_metrics(predictions, ground_truths)
        
        # Turn 1 should be averaged across both conversations
        assert metrics['turn_level_accuracy']['turn_1_accuracy'] == 1.0
        # Turn 2 only exists in first conversation
        assert metrics['turn_level_accuracy']['turn_2_accuracy'] == 1.0
    
    def test_high_similarity_threshold(self, calculator):
        """Test high similarity percentage calculation."""
        # Create prediction with very similar raw output
        predictions = [
            {
                'conversation_id': 'conv_014',
                'predictions': [
                    {
                        'raw_output': 'and get_weather(city="London")',
                        'extracted_calls': ['get_weather(city="London")']
                    }
                ],
                'num_turns': 1
            }
        ]
        
        ground_truths = [
            {
                'conversation_id': 'conv_014',
                'ground_truth': [['get_weather(city="London")']],
                'num_turns': 1
            }
        ]
        
        metrics = calculator.calculate_metrics(predictions, ground_truths)
        
        # Identical text should have high similarity
        assert metrics['avg_similarity_score'] > 0.9
        assert metrics['high_similarity_threshold'] == 0.8
    
    def test_empty_conversations_list(self, calculator):
        """Test handling of empty predictions/ground truths."""
        predictions = []
        ground_truths = []
        
        metrics = calculator.calculate_metrics(predictions, ground_truths)
        
        assert metrics['total_samples'] == 0
        assert metrics['exact_match_accuracy'] == 0.0
    
    def test_function_count_metrics(self, calculator):
        """Test function counting across conversations."""
        predictions = [
            {
                'conversation_id': 'conv_015',
                'predictions': [
                    {
                        'raw_output': 'the function calls are func1() func2()',
                        'extracted_calls': ['func1()', 'func2()']
                    },
                    {
                        'raw_output': 'the function call is func3()',
                        'extracted_calls': ['func3()']
                    }
                ],
                'num_turns': 2
            }
        ]
        
        ground_truths = [
            {
                'conversation_id': 'conv_015',
                'ground_truth': [
                    ['func1()', 'func2()'],
                    ['func3()']
                ],
                'num_turns': 2
            }
        ]
        
        metrics = calculator.calculate_metrics(predictions, ground_truths)
        
        assert metrics['total_predicted_functions'] == 3
        assert metrics['total_ground_truth_functions'] == 3
        assert metrics['avg_predicted_functions_per_sample'] == 3.0
        assert metrics['avg_ground_truth_functions_per_sample'] == 3.0


def test_metrics_calculator_initialization():
    """Test that calculator initializes correctly."""
    calculator = BFCLMetricsCalculator()
    assert calculator is not None
    assert calculator.similarity_model is not None


def test_metrics_calculator_custom_similarity_model():
    """Test initialization with custom similarity model."""
    calculator = BFCLMetricsCalculator(similarity_model_name="all-MiniLM-L6-v2")
    assert calculator is not None


class TestBatchedMultiTurnEvaluation:
    """Test suite for batched multi-turn evaluation with varying conversation lengths."""
    
    @pytest.fixture
    def calculator(self):
        """Create a BFCL metrics calculator instance."""
        return BFCLMetricsCalculator()
    
    def test_batch_with_uniform_lengths(self, calculator):
        """Test batch with all conversations having same length."""
        # Batch of 3 conversations, all with 4 turns
        predictions = [
            {
                'conversation_id': 'conv_batch_1',
                'predictions': [
                    {'raw_output': 'func1()', 'extracted_calls': ['func1()']},
                    {'raw_output': 'func2()', 'extracted_calls': ['func2()']},
                    {'raw_output': 'func3()', 'extracted_calls': ['func3()']},
                    {'raw_output': 'func4()', 'extracted_calls': ['func4()']},
                ],
                'num_turns': 4
            },
            {
                'conversation_id': 'conv_batch_2',
                'predictions': [
                    {'raw_output': 'func1()', 'extracted_calls': ['func1()']},
                    {'raw_output': 'func2()', 'extracted_calls': ['func2()']},
                    {'raw_output': 'func3()', 'extracted_calls': ['func3()']},
                    {'raw_output': 'func4()', 'extracted_calls': ['func4()']},
                ],
                'num_turns': 4
            },
            {
                'conversation_id': 'conv_batch_3',
                'predictions': [
                    {'raw_output': 'func1()', 'extracted_calls': ['func1()']},
                    {'raw_output': 'func2()', 'extracted_calls': ['func2()']},
                    {'raw_output': 'func3()', 'extracted_calls': ['func3()']},
                    {'raw_output': 'func4()', 'extracted_calls': ['func4()']},
                ],
                'num_turns': 4
            }
        ]
        
        ground_truths = [
            {
                'conversation_id': 'conv_batch_1',
                'ground_truth': [['func1()'], ['func2()'], ['func3()'], ['func4()']],
                'num_turns': 4
            },
            {
                'conversation_id': 'conv_batch_2',
                'ground_truth': [['func1()'], ['func2()'], ['func3()'], ['func4()']],
                'num_turns': 4
            },
            {
                'conversation_id': 'conv_batch_3',
                'ground_truth': [['func1()'], ['func2()'], ['func3()'], ['func4()']],
                'num_turns': 4
            }
        ]
        
        metrics = calculator.calculate_metrics(predictions, ground_truths)
        
        assert metrics['total_samples'] == 3
        assert metrics['exact_match_accuracy'] == 1.0
        assert metrics['turn_level_accuracy']['turn_1_accuracy'] == 1.0
        assert metrics['turn_level_accuracy']['turn_4_accuracy'] == 1.0
    
    def test_batch_with_varying_lengths_4_4_5(self, calculator):
        """Test batch with conversation lengths [4, 4, 5] - the canonical example."""
        predictions = [
            {
                'conversation_id': 'conv_len_4_a',
                'predictions': [
                    {'raw_output': 'func1()', 'extracted_calls': ['func1()']},
                    {'raw_output': 'func2()', 'extracted_calls': ['func2()']},
                    {'raw_output': 'func3()', 'extracted_calls': ['func3()']},
                    {'raw_output': 'func4()', 'extracted_calls': ['func4()']},
                ],
                'num_turns': 4
            },
            {
                'conversation_id': 'conv_len_4_b',
                'predictions': [
                    {'raw_output': 'func1()', 'extracted_calls': ['func1()']},
                    {'raw_output': 'func2()', 'extracted_calls': ['func2()']},
                    {'raw_output': 'func3()', 'extracted_calls': ['func3()']},
                    {'raw_output': 'func4()', 'extracted_calls': ['func4()']},
                ],
                'num_turns': 4
            },
            {
                'conversation_id': 'conv_len_5',
                'predictions': [
                    {'raw_output': 'func1()', 'extracted_calls': ['func1()']},
                    {'raw_output': 'func2()', 'extracted_calls': ['func2()']},
                    {'raw_output': 'func3()', 'extracted_calls': ['func3()']},
                    {'raw_output': 'func4()', 'extracted_calls': ['func4()']},
                    {'raw_output': 'func5()', 'extracted_calls': ['func5()']},
                ],
                'num_turns': 5
            }
        ]
        
        ground_truths = [
            {
                'conversation_id': 'conv_len_4_a',
                'ground_truth': [['func1()'], ['func2()'], ['func3()'], ['func4()']],
                'num_turns': 4
            },
            {
                'conversation_id': 'conv_len_4_b',
                'ground_truth': [['func1()'], ['func2()'], ['func3()'], ['func4()']],
                'num_turns': 4
            },
            {
                'conversation_id': 'conv_len_5',
                'ground_truth': [['func1()'], ['func2()'], ['func3()'], ['func4()'], ['func5()']],
                'num_turns': 5
            }
        ]
        
        metrics = calculator.calculate_metrics(predictions, ground_truths)
        
        # All conversations should match perfectly
        assert metrics['total_samples'] == 3
        assert metrics['exact_match_accuracy'] == 1.0
        
        # Turn-level accuracy should be 1.0 for all turns
        # Turn 5 only exists in one conversation but should still be 1.0
        assert metrics['turn_level_accuracy']['turn_1_accuracy'] == 1.0
        assert metrics['turn_level_accuracy']['turn_4_accuracy'] == 1.0
        assert metrics['turn_level_accuracy']['turn_5_accuracy'] == 1.0
    
    def test_batch_with_varying_lengths_partial_failure(self, calculator):
        """Test batch with varying lengths where one conversation fails later turns."""
        predictions = [
            {
                'conversation_id': 'conv_short_perfect',
                'predictions': [
                    {'raw_output': 'func1()', 'extracted_calls': ['func1()']},
                    {'raw_output': 'func2()', 'extracted_calls': ['func2()']},
                ],
                'num_turns': 2
            },
            {
                'conversation_id': 'conv_long_fail_at_turn_3',
                'predictions': [
                    {'raw_output': 'func1()', 'extracted_calls': ['func1()']},
                    {'raw_output': 'func2()', 'extracted_calls': ['func2()']},
                    {'raw_output': 'func_wrong()', 'extracted_calls': ['func_wrong()']},
                    {'raw_output': 'func4()', 'extracted_calls': ['func4()']},
                ],
                'num_turns': 4
            },
            {
                'conversation_id': 'conv_medium_perfect',
                'predictions': [
                    {'raw_output': 'func1()', 'extracted_calls': ['func1()']},
                    {'raw_output': 'func2()', 'extracted_calls': ['func2()']},
                    {'raw_output': 'func3()', 'extracted_calls': ['func3()']},
                ],
                'num_turns': 3
            }
        ]
        
        ground_truths = [
            {
                'conversation_id': 'conv_short_perfect',
                'ground_truth': [['func1()'], ['func2()']],
                'num_turns': 2
            },
            {
                'conversation_id': 'conv_long_fail_at_turn_3',
                'ground_truth': [['func1()'], ['func2()'], ['func3()'], ['func4()']],
                'num_turns': 4
            },
            {
                'conversation_id': 'conv_medium_perfect',
                'ground_truth': [['func1()'], ['func2()'], ['func3()']],
                'num_turns': 3
            }
        ]
        
        metrics = calculator.calculate_metrics(predictions, ground_truths)
        
        # Overall accuracy should be 2/3 (two perfect conversations)
        assert metrics['total_samples'] == 3
        assert metrics['exact_match_accuracy'] == pytest.approx(2.0 / 3.0)
        
        # Turn 1 and 2 should be perfect (all 3 conversations)
        assert metrics['turn_level_accuracy']['turn_1_accuracy'] == 1.0
        assert metrics['turn_level_accuracy']['turn_2_accuracy'] == 1.0
        
        # Turn 3 should be 1/2 (only conv_medium_perfect and conv_long have it, but conv_long fails)
        assert metrics['turn_level_accuracy']['turn_3_accuracy'] == 0.5
        
        # Turn 4 should be 1.0 (only conv_long has it, and it gets it right)
        assert metrics['turn_level_accuracy']['turn_4_accuracy'] == 1.0
    
    def test_batch_single_conversation(self, calculator):
        """Test batch with only one conversation (edge case)."""
        predictions = [
            {
                'conversation_id': 'single_conv',
                'predictions': [
                    {'raw_output': 'func1()', 'extracted_calls': ['func1()']},
                    {'raw_output': 'func2()', 'extracted_calls': ['func2()']},
                ],
                'num_turns': 2
            }
        ]
        
        ground_truths = [
            {
                'conversation_id': 'single_conv',
                'ground_truth': [['func1()'], ['func2()']],
                'num_turns': 2
            }
        ]
        
        metrics = calculator.calculate_metrics(predictions, ground_truths)
        
        assert metrics['total_samples'] == 1
        assert metrics['exact_match_accuracy'] == 1.0
    
    def test_batch_with_extreme_length_variation(self, calculator):
        """Test batch with extreme length variation [1, 5, 10]."""
        predictions = [
            {
                'conversation_id': 'very_short',
                'predictions': [
                    {'raw_output': 'func1()', 'extracted_calls': ['func1()']},
                ],
                'num_turns': 1
            },
            {
                'conversation_id': 'medium',
                'predictions': [
                    {'raw_output': f'func{i}()', 'extracted_calls': [f'func{i}()']}
                    for i in range(1, 6)
                ],
                'num_turns': 5
            },
            {
                'conversation_id': 'very_long',
                'predictions': [
                    {'raw_output': f'func{i}()', 'extracted_calls': [f'func{i}()']}
                    for i in range(1, 11)
                ],
                'num_turns': 10
            }
        ]
        
        ground_truths = [
            {
                'conversation_id': 'very_short',
                'ground_truth': [['func1()']],
                'num_turns': 1
            },
            {
                'conversation_id': 'medium',
                'ground_truth': [[f'func{i}()'] for i in range(1, 6)],
                'num_turns': 5
            },
            {
                'conversation_id': 'very_long',
                'ground_truth': [[f'func{i}()'] for i in range(1, 11)],
                'num_turns': 10
            }
        ]
        
        metrics = calculator.calculate_metrics(predictions, ground_truths)
        
        assert metrics['total_samples'] == 3
        assert metrics['exact_match_accuracy'] == 1.0
        
        # Turn 1 should have 100% accuracy (all 3 conversations have it)
        assert metrics['turn_level_accuracy']['turn_1_accuracy'] == 1.0
        
        # Turn 10 should have 100% accuracy (only 1 conversation has it, and it's correct)
        assert metrics['turn_level_accuracy']['turn_10_accuracy'] == 1.0
    
    def test_batch_with_empty_predictions_in_some_turns(self, calculator):
        """Test batch where some conversations have empty predictions in certain turns."""
        predictions = [
            {
                'conversation_id': 'conv_with_empty',
                'predictions': [
                    {'raw_output': 'func1()', 'extracted_calls': ['func1()']},
                    {'raw_output': '', 'extracted_calls': []},  # Empty on turn 2
                ],
                'num_turns': 2
            },
            {
                'conversation_id': 'conv_perfect',
                'predictions': [
                    {'raw_output': 'func1()', 'extracted_calls': ['func1()']},
                    {'raw_output': 'func2()', 'extracted_calls': ['func2()']},
                ],
                'num_turns': 2
            }
        ]
        
        ground_truths = [
            {
                'conversation_id': 'conv_with_empty',
                'ground_truth': [['func1()'], ['func2()']],
                'num_turns': 2
            },
            {
                'conversation_id': 'conv_perfect',
                'ground_truth': [['func1()'], ['func2()']],
                'num_turns': 2
            }
        ]
        
        metrics = calculator.calculate_metrics(predictions, ground_truths)
        
        assert metrics['total_samples'] == 2
        assert metrics['total_invalid_turns'] == 1  # One empty prediction when expected
        assert metrics['exact_match_accuracy'] == 0.5  # Only one perfect conversation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

