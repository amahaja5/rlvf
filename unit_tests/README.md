# Unit Tests for RL Voice Leading

Comprehensive test suite for the reinforcement learning voice leading library.

## Test Structure

```
unit_tests/
├── test_voice_leading_rules.py  # Voice leading rule checking
├── test_state_space_def.py      # Chord and voicing analysis
├── test_midi_conversion.py      # MIDI generation and conversion
├── test_models.py                # Q-learning model classes
└── MIDI_conv_tests.py           # Legacy MIDI tests
```

## Running Tests

### Run all tests:
```bash
cd /home/user/rlvf
pytest
```

### Run specific test file:
```bash
pytest unit_tests/test_voice_leading_rules.py
```

### Run with verbose output:
```bash
pytest -v
```

### Run with coverage report:
```bash
pytest --cov=src --cov=models --cov-report=html
```

### Run specific test class or function:
```bash
pytest unit_tests/test_voice_leading_rules.py::TestIllegalLeaps
pytest unit_tests/test_models.py::TestQlearner::test_initialization
```

## Test Categories

### Voice Leading Rules (`test_voice_leading_rules.py`)
- **TestIllegalLeaps**: Detects augmented intervals, 7ths, >octave leaps
- **TestVoiceCrossing**: Checks for parts crossing each other
- **TestParallelFifthsOctaves**: Forbidden parallel motion detection
- **TestDirectFifthsOctaves**: Direct/hidden fifths and octaves
- **TestSeventhChords**: Seventh chord approach and resolution
- **TestVoiceLeadingRewardFunction**: Integrated reward calculation

### State Space (`test_state_space_def.py`)
- **TestDetermineChordFromVoicing**: Chord identification from pitches
- **TestDetermineInversion**: Inversion detection (root, 1st, 2nd, 3rd)
- **TestIsComplete**: Chord completeness verification
- **TestDoubledLeadingTone**: Leading tone doubling detection
- **TestNoteNamesToNumbers**: Note name to MIDI conversion

### MIDI Conversion (`test_midi_conversion.py`)
- **TestGetFreeFilename**: Unique filename generation
- **TestStateSeqToMIDI**: Voicing sequence to MIDI conversion
- **TestMelodyToMIDI**: Melody to MIDI conversion
- **TestOnePartNoteList**: Single voice note generation

### Models (`test_models.py`)
- **TestQlearner**: Base Q-learning class
- **TestVoicingModel**: Voicing model specific tests
- **TestHarmonizationModel**: Melody harmonization tests
- **TestFreeModel**: Free composition model tests
- **TestModelSaveLoad**: Model persistence

## Test Coverage

Current test coverage includes:
- ✅ All voice leading rule functions
- ✅ Chord and voicing analysis functions
- ✅ MIDI file generation
- ✅ Model initialization and basic operations
- ✅ Edge cases and error conditions

## Writing New Tests

When adding new tests, follow these conventions:

1. **Use descriptive test names**: `test_should_detect_parallel_octaves()`
2. **One assertion per concept**: Focus each test on a single behavior
3. **Use fixtures for setup**: Reuse test data with pytest fixtures
4. **Test edge cases**: Include boundary conditions and error cases
5. **Add docstrings**: Explain what each test validates

Example:
```python
class TestNewFeature:
    """Test new feature functionality."""

    def test_basic_behavior(self):
        """Should handle basic use case correctly."""
        result = new_feature(input_data)
        assert result == expected_output

    def test_edge_case(self):
        """Should handle empty input gracefully."""
        result = new_feature([])
        assert result is not None
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines. To integrate:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest --cov=src --cov=models --cov-report=xml
```

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the project root:
```bash
cd /home/user/rlvf
pytest
```

### Missing Dependencies
Install test dependencies:
```bash
pip install pytest pytest-cov
```

### MIDI File Tests Failing
Some tests require writing MIDI files. Ensure you have write permissions:
```bash
chmod +w unit_tests/test_results/
```

## Contributing

When contributing new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass: `pytest`
3. Check coverage: `pytest --cov=src --cov=models`
4. Aim for >80% coverage on new code
