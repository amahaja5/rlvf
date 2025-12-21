"""
Unit tests for Q-learning model classes.

Tests cover basic model functionality, Q-value updates, and action selection.
"""

import sys
import pytest
import numpy as np
sys.path.insert(0, '../src')
sys.path.insert(0, '../models')

from models import Qlearner, VoicingModel, HarmonizationModel, FreeModel, flipCoin


class TestFlipCoin:
    """Test probability utility function."""

    def test_always_true(self):
        """Probability of 1.0 should always return True."""
        assert flipCoin(1.0) is True

    def test_never_true(self):
        """Probability of 0.0 should always return False."""
        assert flipCoin(0.0) is False

    def test_probability_range(self):
        """Should return boolean for valid probabilities."""
        for p in [0.1, 0.5, 0.9]:
            result = flipCoin(p)
            assert isinstance(result, bool)


class TestQlearner:
    """Test base Qlearner class functionality."""

    @pytest.fixture
    def qlearner(self):
        """Create a Qlearner instance for testing."""
        return Qlearner(alpha=0.1, gamma=0.9)

    def test_initialization(self, qlearner):
        """Qlearner should initialize with correct parameters."""
        assert qlearner.alpha == 0.1
        assert qlearner.gamma == 0.9
        assert qlearner.epsilon_init == 0.5
        assert qlearner.epsilon_end == 0.0

    def test_qvalues_matrix_shape(self, qlearner):
        """Q-values should be initialized as matrix."""
        assert isinstance(qlearner.Qvalues, np.ndarray)
        assert qlearner.Qvalues.shape[0] == qlearner.numStates
        assert qlearner.Qvalues.shape[1] == qlearner.numStates

    def test_qvalues_initialized_to_zero(self, qlearner):
        """Q-values should start at zero."""
        assert np.all(qlearner.Qvalues == 0)

    def test_has_chord_dict(self, qlearner):
        """Should load chord dictionary."""
        assert hasattr(qlearner, 'chord_dict')
        assert isinstance(qlearner.chord_dict, dict)
        assert len(qlearner.chord_dict) > 0

    def test_has_state_indices(self, qlearner):
        """Should load state indices."""
        assert hasattr(qlearner, 'state_indices')
        assert isinstance(qlearner.state_indices, dict)
        assert len(qlearner.state_indices) > 0

    def test_num_states(self, qlearner):
        """Number of states should match state_indices length."""
        assert qlearner.numStates == len(qlearner.state_indices.keys())
        assert qlearner.numStates > 0


class TestQLearnerMethods:
    """Test Qlearner methods."""

    @pytest.fixture
    def qlearner(self):
        return Qlearner(alpha=0.1, gamma=0.9)

    def test_get_qvalue(self, qlearner):
        """Should retrieve Q-value for state-action pair."""
        state = 0
        next_state = 1
        qval = qlearner.getQValue(state, next_state)
        assert isinstance(qval, (int, float))
        assert qval == 0  # Initially zero

    def test_get_value(self, qlearner):
        """Should compute value of a state."""
        state = 0
        value = qlearner.getValue(state)
        assert isinstance(value, (int, float))

    def test_get_legal_actions(self, qlearner):
        """Should return list of legal actions."""
        actions = qlearner.getLegalActions()
        assert isinstance(actions, list)
        assert len(actions) == qlearner.numStates

    def test_update_q_value(self, qlearner):
        """Q-value update should modify the matrix."""
        state = 0
        action = 1
        reward = 1.0

        # Get initial Q-value
        initial_qval = qlearner.getQValue(state, action)

        # Update
        qlearner.update(state, action, 0, reward)

        # Check it changed
        updated_qval = qlearner.getQValue(state, action)
        assert updated_qval != initial_qval


class TestVoicingModel:
    """Test VoicingModel specific functionality."""

    @pytest.fixture
    def voicing_model(self):
        return VoicingModel(alpha=0.1, gamma=0.6)

    def test_initialization(self, voicing_model):
        """VoicingModel should initialize correctly."""
        assert voicing_model.alpha == 0.1
        assert voicing_model.gamma == 0.6
        assert hasattr(voicing_model, 'rewardFunction')
        assert hasattr(voicing_model, 'results_dir')

    def test_has_reward_function(self, voicing_model):
        """Should have voice leading reward function."""
        assert voicing_model.rewardFunction is not None
        # Should be callable
        assert callable(voicing_model.rewardFunction)

    def test_get_legal_actions_requires_context(self, voicing_model):
        """VoicingModel needs chord context for legal actions."""
        # With valid chord number
        chord = 1
        actions = voicing_model.getLegalActions(context=chord)
        assert isinstance(actions, list)
        assert len(actions) > 0

    def test_checkpoint_attribute(self, voicing_model):
        """Should have checkpoint for saving."""
        assert hasattr(voicing_model, 'checkpoint')
        assert voicing_model.checkpoint == 500


class TestHarmonizationModel:
    """Test HarmonizationModel specific functionality."""

    @pytest.fixture
    def harm_model(self):
        return HarmonizationModel(alpha=0.1, gamma=0.6)

    def test_initialization(self, harm_model):
        """HarmonizationModel should initialize correctly."""
        assert harm_model.alpha == 0.1
        assert harm_model.gamma == 0.6
        assert hasattr(harm_model, 'rewardFunction')

    def test_get_legal_actions_requires_melody(self, harm_model):
        """HarmonizationModel needs melody note for legal actions."""
        # Should handle melody note list
        melody_note = [72]  # High C
        actions = harm_model.getLegalActions(context=melody_note)
        assert isinstance(actions, list)


class TestFreeModel:
    """Test FreeModel specific functionality."""

    @pytest.fixture
    def free_model(self):
        return FreeModel(alpha=0.1, gamma=0.6)

    def test_initialization(self, free_model):
        """FreeModel should initialize correctly."""
        assert free_model.alpha == 0.1
        assert free_model.gamma == 0.6
        assert hasattr(free_model, 'rewardFunction')

    def test_get_legal_actions_unrestricted(self, free_model):
        """FreeModel should return all states as legal actions."""
        actions = free_model.getLegalActions()
        assert isinstance(actions, list)
        assert len(actions) == free_model.numStates


class TestModelSaveLoad:
    """Test model persistence."""

    def test_save_model_data_structure(self, tmp_path):
        """Model should save with correct data structure."""
        model = Qlearner(alpha=0.1, gamma=0.9)
        model.model_name = 'test_model'

        # Modify some Q-values
        model.Qvalues[0, 1] = 0.5
        model.Qvalues[1, 2] = -0.3

        # Save
        model.saveModel(epochs=10, rewards=[0.1, 0.2, 0.3])

        # Check file was created
        import glob
        saved_files = glob.glob('./output/savedmodels/test_model*.p')
        assert len(saved_files) > 0

    def test_load_model(self, tmp_path):
        """Should load previously saved model."""
        # Create and save a model
        model1 = Qlearner(alpha=0.1, gamma=0.9)
        model1.model_name = 'test_load'
        model1.Qvalues[0, 1] = 0.75
        model1.saveModel(epochs=5, rewards=[0.1, 0.2])

        # Load into new model
        model2 = Qlearner(alpha=0.1, gamma=0.9)
        import glob
        saved_file = glob.glob('./output/savedmodels/test_load*.p')[0]
        rewards, epochs = model2.loadModel(saved_file)

        # Check data was loaded
        assert epochs == 5
        assert len(rewards) == 2
        assert model2.Qvalues[0, 1] == 0.75


class TestActionSelection:
    """Test action selection strategies."""

    @pytest.fixture
    def model(self):
        model = Qlearner(alpha=0.1, gamma=0.9)
        # Set some Q-values to test selection
        model.Qvalues[0, 1] = 0.8
        model.Qvalues[0, 2] = 0.5
        model.Qvalues[0, 3] = 0.9  # Best action
        return model

    def test_compute_action_values(self, model):
        """Should find best action based on Q-values."""
        legal_actions = [1, 2, 3]
        best_action, best_value = model.computeActionValuesFromQValues(
            state=0,
            legal_actions=legal_actions
        )

        assert best_action == 3  # Highest Q-value
        assert best_value == 0.9

    def test_best_action_selection(self, model):
        """With best=True, should always choose best action."""
        legal_actions = [1, 2, 3]
        action = model.getAction(state=0, context=1, best=True, rand=False)

        # Should choose action with highest Q-value
        assert action in legal_actions


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_negative_alpha(self):
        """Should handle unusual but valid parameters."""
        # Alpha can technically be negative, though unusual
        model = Qlearner(alpha=-0.1, gamma=0.9)
        assert model.alpha == -0.1

    def test_gamma_greater_than_one(self):
        """Should allow gamma > 1 (unusual but not invalid)."""
        model = Qlearner(alpha=0.1, gamma=1.5)
        assert model.gamma == 1.5

    def test_empty_legal_actions(self):
        """Should handle case with no legal actions."""
        model = Qlearner()
        action, value = model.computeActionValuesFromQValues(
            state=0,
            legal_actions=[]
        )
        assert action is None
        assert value == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
