
## Voicing Model: 
1. Harmonic progression is provided, algorithm need only provide the voicings
2. We consider only harmonic progressions in the key of C major
3. We restrict the range of each part

## Free Model 
1. Algorithm provides the voicings as well as a chord progression
2. We consider only harmonic progressions in the key of C major
3. We restrict the range of each part

## Melody Harmonizaton Model 
1. Algorithm is provided a melody. It picks chords (with proper voice leading) that fits the given input. 

## To Do: 
1. Rule-breaking functions
    
    a. Voice crossing

    b. Leaps --> augmented intervals, 7ths, leaps larger than an octave

    c. Parallel motion (parallel 5ths and octaves); have to check all 6 pairs!

    d. Direct 5ths: outer parts move in the same direction into a P5 or P8 with a leap in the soprano part

    e. Illegal common tones

    f. Illegal leading tone resolutions

    g. Illegal 7th approaches/resolutions

2. Create "training set" of chord progressions and melodies 

3. Function to convert algorithm output to MIDI

All other rules (spacing, ranges) are taken care of by the state space definition

### Data Structures
A triad chord is a set of three notes (chord tones). There are seven triad chords in the key of C major. Each of these chords has several legal voicings.  

A "voicing" is a list of four MIDI pitches `[b,t,a,s]` sorted from lowest to highest. They denote the pitches taken by the bass, tenor, alto, and soprano parts. The voicing for a given chord must contain at least one copy of each note in the triad chord, as well as one additional note that is a double of one of the chord tones. Note that there are always 4 pitches, but two of them may be identical. 

We define a dictionary `state_indices` that assigns an index to each voicing (as defined above). This allows us to use the state indices in our Q-learning algorithm.

We define a state dictionary that contains all legal states given our problem constraints. They are organized by chord number. Thus, `state_dict[chord_num]` contains a list of legal voicing indices for the given chord.

### Algorithm
The learning agent is defined in the `QLearningAgent` class. The typical Q-learning update is given by:

```
Q(state,action) = Q(state,action) + self.alpha*(reward+gamma*max(Q(next_state, actions))- Q(state,action))
```

In this problem, there are no actions. Another way of looking at it is the action is just what state you transition to next. 

The other main difference between this problem and those covered in class is that we have rewards associated with *state pairs* instead of state action pairs. A state by itself has no associated reward.

We can reformulate the problem as follows:

* We consider our "actions" to be the next state that agent transitions to. Thus, if there are `n` states, there are also `n` actions.
* Our Q-value table is `nxn`
* 

### Listening

The trained models can generate MIDI files which can be converted to audio for listening and evaluation.

## Installation

### Prerequisites
- Python 3.7+
- FluidSynth (for MIDI to audio conversion)
  - Ubuntu/Debian: `sudo apt-get install fluidsynth`
  - macOS: `brew install fluid-synth`
  - Windows: Download from [FluidSynth website](http://www.fluidsynth.org/)

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd rlvf
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import pretty_midi; print('Installation successful!')"
```

## Project Structure

```
rlvf/
├── src/                    # Core source code
│   ├── main.py            # Training and evaluation entry point
│   ├── voice_leading_rules.py  # Rule violation checking
│   ├── state_space_def.py # Legal voicing definitions
│   ├── MIDI_conversion.py # MIDI/audio conversion utilities
│   ├── constants.py       # Configuration constants
│   └── chord_constants.py # Chord definitions
├── models/                # Q-learning agent implementations
│   └── models.py         # VoicingModel, FreeModel, HarmonizationModel
├── examples/              # Jupyter notebooks for analysis
├── data/                  # Training data (melodies, chord progressions)
├── output/savedmodels/    # Saved model checkpoints
├── results/               # Generated outputs and evaluations
└── unit_tests/           # Unit tests

```

## Usage

### Training Models

Train all three models (this will take several hours):
```bash
cd src
python main.py
```

To train individual models, modify the `if __name__ == "__main__"` block in `src/main.py` to call only the desired training functions.

### Model Configuration

Edit these parameters in `src/main.py`:
- `n_epochs`: Number of training epochs (default: 10000)
- `num_runs`: Number of evaluation runs (default: 50)
- `CHECKPOINT`: Checkpoint interval for saving models (default: 500)

### Evaluating Models

The evaluation functions compare model performance against random baselines across multiple metrics:
- Voice leading rewards
- Harmonic progression rewards
- Rule violations (parallel 5ths, voice crossing, etc.)

### Generating Music

See the Jupyter notebooks in `examples/` for:
- Data preprocessing (`jsb_data_preprocessing.ipynb`)
- Reward function visualization (`reward_function_plotting.ipynb`)
- Rule weighting analysis (`rules_weighting.ipynb`)

## Voice Leading Rules

The reward function penalizes violations of classical voice leading rules:
- **Illegal leaps**: Augmented intervals, 7ths, leaps > octave
- **Voice crossing**: Parts crossing each other
- **Parallel 5ths/octaves**: Forbidden parallel motion
- **Direct 5ths/octaves**: Outer voices moving in parallel to perfect intervals
- **Leading tone resolution**: Improper resolution of the leading tone
- **7th chord handling**: Incorrect approach/resolution of 7th chords

Weights for each rule are defined in `src/constants.py` and were determined empirically.

## Data Format

### Voicings
A voicing is represented as a list of 4 MIDI pitches: `[bass, tenor, alto, soprano]`
- Sorted from lowest to highest
- Contains all chord tones with one doubled note

### Input Data
- Chord progressions: `data/jsb_maj_chord_progs.yaml`
- Melodies: `data/jsb_maj_melodies.yaml`
- Original Bach voicings: `data/jsb_maj_orig_voicings.yaml`

## Contributing

When contributing, please:
1. Follow PEP 8 style guidelines
2. Add docstrings to new functions
3. Write tests for new functionality
4. Update tests as needed
5. Keep imports explicit (avoid `from module import *`)

## Testing

The project includes comprehensive unit tests for all core functionality.

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest unit_tests/test_voice_leading_rules.py

# Run with coverage report
pytest --cov=src --cov=models --cov-report=html
```

### Test Coverage

Tests cover:
- Voice leading rule checking (illegal leaps, parallel 5ths, voice crossing, etc.)
- Chord and voicing analysis (chord identification, inversions, completeness)
- MIDI file generation and conversion
- Q-learning model classes (initialization, training, action selection)

See `unit_tests/README.md` for detailed testing documentation.

## License

See LICENSE file for details.
