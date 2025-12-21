"""
Q-Learning models for voice leading tasks.

This module contains the Q-learning agent implementations for:
- VoicingModel: Given chord progression, generate voice leading
- FreeModel: Generate both progression and voice leading
- HarmonizationModel: Harmonize a given melody
"""

from .models import Qlearner, VoicingModel, FreeModel, HarmonizationModel

__all__ = ['Qlearner', 'VoicingModel', 'FreeModel', 'HarmonizationModel']
