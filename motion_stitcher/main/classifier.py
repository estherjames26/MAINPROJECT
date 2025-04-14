"""
Dance style classifier for motion data.
"""
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Tuple, Any, Optional
from features import MotionFeatureExtractor

class DanceStyleClassifier:
    """Classifies dance motion clips based on style and beat characteristics."""
    
    def __init__(self, model_path=None):
        """Initialize the classifier.
        
        Args:
            model_path: Path to a pre-trained model file (optional)
        """
        self.model = None
        self.feature_extractor = MotionFeatureExtractor()
        self.styles = ['ballet', 'break', 'house', 'jazz', 'lock', 'pop']
        
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print(f"Loaded dance style classifier model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
    
    def train(self, motion_samples, style_labels, beat_labels=None, beat_frames_list=None):
        """Train the style classifier using motion samples and their labels.
        
        Args:
            motion_samples: List of motion data samples
            style_labels: Corresponding style labels for each sample
            beat_labels: Optional beat labels (e.g., 'strong', 'weak')
            beat_frames_list: Optional list of beat frames for each sample
            
        Returns:
            Training accuracy
        """
        if len(motion_samples) == 0 or len(motion_samples) != len(style_labels):
            print("Error: Invalid training data")
            return 0.0
        
        # Extract features from motion samples
        X = []
        y = []
        
        print(f"Training classifier with {len(motion_samples)} samples")
        
        for i, (motion, style) in enumerate(zip(motion_samples, style_labels)):
            # Skip invalid samples
            if motion is None or not isinstance(motion, np.ndarray) or motion.size == 0:
                continue
                
            # Extract motion features
            motion_features = self._extract_features(motion)
            
            # Add beat-related features if available
            if beat_labels and i < len(beat_labels):
                # Encode beat label as a one-hot feature
                beat_feature = self._encode_beat_label(beat_labels[i])
                motion_features.extend(beat_feature)
                
            # Add beat alignment features if beat frames are provided
            if beat_frames_list and i < len(beat_frames_list) and beat_frames_list[i] is not None:
                beat_align_features = self._extract_beat_alignment_features(motion, beat_frames_list[i])
                motion_features.extend(beat_align_features)
            
            X.append(motion_features)
            y.append(style)
        
        if len(X) == 0:
            print("Error: No valid samples after feature extraction")
            return 0.0
        
        # Train a Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.model.fit(X, y)
        
        # Calculate training accuracy
        accuracy = self.model.score(X, y)
        print(f"Training accuracy: {accuracy:.2f}")
        
        return accuracy
    
    def classify(self, motion_data, beat_frames=None):
        """Classify a motion clip into a dance style.
        
        Args:
            motion_data: Motion data to classify
            beat_frames: Optional beat frames for this motion
            
        Returns:
            Predicted style and confidence
        """
        if self.model is None:
            print("Error: No trained model available")
            return None, 0.0
        
        if motion_data is None or not isinstance(motion_data, np.ndarray) or motion_data.size == 0:
            print("Error: Invalid motion data")
            return None, 0.0
        
        # Extract features
        features = self._extract_features(motion_data)
        
        # Add beat alignment features if beat frames are provided
        if beat_frames is not None:
            beat_align_features = self._extract_beat_alignment_features(motion_data, beat_frames)
            features.extend(beat_align_features)
        
        # Reshape for prediction
        X = np.array(features).reshape(1, -1)
        
        # Make prediction
        predicted_style = self.model.predict(X)[0]
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        return predicted_style, confidence
    
    def save_model(self, model_path):
        """Save the trained model to a file.
        
        Args:
            model_path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        if self.model is None:
            print("Error: No trained model to save")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save the model
            joblib.dump(self.model, model_path)
            print(f"Saved classifier model to {model_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def _extract_features(self, motion_data):
        """Extract features from motion data for classification.
        
        Args:
            motion_data: Motion data [frames, joints*3]
            
        Returns:
            List of features
        """
        # Extract style features
        style_feats = self.feature_extractor.extract_motion_style_features(motion_data)
        
        # Extract beat-related features
        beat_feats = self.feature_extractor.extract_motion_beat_features(motion_data)
        
        # Create feature vector
        features = [
            # Speed and acceleration features
            style_feats['average_speed'],
            style_feats['speed_variance'],
            
            # Movement area (space used by the dancer)
            style_feats['movement_area'],
            
            # Beat-related features
            beat_feats['motion_regularity'],
            beat_feats['energy_level'],
            beat_feats['average_interval']
        ]
        
        # Add selected joint variance features (focusing on most important joints)
        joint_vars = style_feats['joint_variances']
        if len(joint_vars) >= 24:  # Assuming SMPL model with 24 joints
            # Include variance of key joints: root, spine, arms, legs
            key_joints = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18]
            for idx in key_joints:
                if idx < len(joint_vars):
                    features.append(joint_vars[idx])
        
        return features
    
    def _encode_beat_label(self, beat_label):
        """Encode a beat label as a feature.
        
        Args:
            beat_label: Beat label (e.g., 'strong', 'weak')
            
        Returns:
            One-hot encoded feature
        """
        if beat_label == 'strong':
            return [1.0, 0.0, 0.0]
        elif beat_label == 'medium':
            return [0.0, 1.0, 0.0]
        else:  # weak or unknown
            return [0.0, 0.0, 1.0]
    
    def _extract_beat_alignment_features(self, motion_data, beat_frames, fps=30):
        """Extract features related to how motion aligns with beats.
        
        Args:
            motion_data: Motion data
            beat_frames: Beat frame indices
            fps: Frames per second
            
        Returns:
            Beat alignment features
        """
        if len(beat_frames) == 0:
            return [0.0, 0.0, 0.0]
        
        # Calculate velocity of motion
        velocity = np.diff(motion_data, axis=0)
        if len(velocity) == 0:
            return [0.0, 0.0, 0.0]
            
        # Pad velocity to match motion_data length
        velocity = np.vstack([velocity, velocity[-1:]])
        
        # Calculate motion energy at each frame
        motion_energy = np.sum(np.square(velocity), axis=1)
        
        # Normalize motion energy
        if np.max(motion_energy) > 0:
            motion_energy = motion_energy / np.max(motion_energy)
        
        # Calculate energy at beat frames vs. non-beat frames
        beat_indices = []
        for bf in beat_frames:
            if 0 <= bf < len(motion_energy):
                beat_indices.append(bf)
        
        if not beat_indices:
            return [0.0, 0.0, 0.0]
        
        # Beat energy vs. non-beat energy ratio
        beat_energy = np.mean(motion_energy[beat_indices])
        
        # Energy around beats (within a window)
        beat_window_energy = []
        window_size = 2  # frames before/after beat
        
        for bf in beat_indices:
            start = max(0, bf - window_size)
            end = min(len(motion_energy), bf + window_size + 1)
            window_energy = np.mean(motion_energy[start:end])
            beat_window_energy.append(window_energy)
        
        avg_beat_window_energy = np.mean(beat_window_energy) if beat_window_energy else 0.0
        
        # Calculate off-beat energy (frames furthest from beats)
        all_frames = set(range(len(motion_energy)))
        off_beat_frames = list(all_frames - set(beat_indices))
        off_beat_energy = np.mean(motion_energy[off_beat_frames]) if off_beat_frames else 0.0
        
        # Create features
        on_off_ratio = beat_energy / (off_beat_energy + 1e-8)  # Avoid division by zero
        
        return [
            beat_energy,
            on_off_ratio,
            avg_beat_window_energy
        ]
    
    def predict_best_style_for_audio(self, audio_features):
        """Predict the best dance style for given audio features.
        
        Args:
            audio_features: Dictionary of audio features
            
        Returns:
            Recommended style and confidence score
        """
        if self.model is None:
            # Fallback to rule-based recommendation
            return self._rule_based_style_recommendation(audio_features)
        
        # Extract relevant audio properties
        tempo = audio_features.get('tempo', 120)
        
        # Map tempo to suitable dance styles
        if tempo < 80:
            suitable_styles = ['ballet', 'jazz']
        elif 80 <= tempo < 100:
            suitable_styles = ['pop', 'lock', 'jazz']
        elif 100 <= tempo < 120:
            suitable_styles = ['pop', 'lock', 'house']
        else:  # fast tempo
            suitable_styles = ['break', 'house']
        
        # Find style with highest confidence based on audio characteristics
        best_style = None
        best_confidence = 0.0
        
        for style in suitable_styles:
            # Simple mapping of audio features to confidence scores for each style
            confidence = self._calculate_style_confidence(style, audio_features)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_style = style
        
        if best_style is None:
            # Fallback if no suitable style found
            best_style = 'pop'  # Most versatile style
            best_confidence = 0.5
        
        return best_style, best_confidence
    
    def _rule_based_style_recommendation(self, audio_features):
        """Rule-based recommendation of dance style based on audio features."""
        tempo = audio_features.get('tempo', 120)
        beat_pattern = audio_features.get('beat_pattern', {})
        
        # Extract regularity and pattern length
        is_regular = beat_pattern.get('is_regular', False)
        pattern_length = beat_pattern.get('pattern_length', 4)
        
        # Simple rules for style recommendation
        if tempo < 85:
            if is_regular:
                return 'ballet', 0.7
            else:
                return 'jazz', 0.7
        elif 85 <= tempo < 100:
            if pattern_length <= 4:
                return 'pop', 0.8
            else:
                return 'lock', 0.7
        elif 100 <= tempo < 130:
            if is_regular:
                return 'house', 0.8
            else:
                return 'lock', 0.7
        else:  # Fast tempo
            if is_regular:
                return 'house', 0.8
            else:
                return 'break', 0.8
    
    def _calculate_style_confidence(self, style, audio_features):
        """Calculate a confidence score for a style based on audio features."""
        tempo = audio_features.get('tempo', 120)
        beat_pattern = audio_features.get('beat_pattern', {})
        is_regular = beat_pattern.get('is_regular', False)
        regularity_score = beat_pattern.get('regularity_score', 0.5)
        
        if style == 'ballet':
            # Ballet prefers slower, regular tempo
            tempo_match = 1.0 - min(1.0, abs(tempo - 70) / 50)
            regularity_match = regularity_score
            return 0.7 * tempo_match + 0.3 * regularity_match
            
        elif style == 'jazz':
            # Jazz works with varied tempos, less regular beats
            tempo_match = 1.0 - min(1.0, abs(tempo - 85) / 40)
            regularity_match = 1.0 - regularity_score
            return 0.6 * tempo_match + 0.4 * regularity_match
            
        elif style == 'pop':
            # Pop works well with medium tempo, regular beats
            tempo_match = 1.0 - min(1.0, abs(tempo - 100) / 30)
            regularity_match = regularity_score
            return 0.5 * tempo_match + 0.5 * regularity_match
            
        elif style == 'lock':
            # Locking prefers medium-fast tempo with accented beats
            tempo_match = 1.0 - min(1.0, abs(tempo - 110) / 30)
            return tempo_match
            
        elif style == 'house':
            # House works with faster tempo, very regular beats
            tempo_match = 1.0 - min(1.0, abs(tempo - 125) / 25)
            regularity_match = regularity_score
            return 0.6 * tempo_match + 0.4 * regularity_match
            
        elif style == 'break':
            # Breaking works with faster tempo, strong beats
            tempo_match = 1.0 - min(1.0, abs(tempo - 110) / 40)
            return tempo_match
            
        return 0.5  # Default confidence for unknown styles
