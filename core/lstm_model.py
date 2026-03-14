"""
Enhanced LSTM Model with Attention Mechanism
=============================================
Improved violence detection model with:
- Bidirectional LSTM layers
- Self-attention mechanism
- Temporal Convolutional layers
- Multi-class classification support
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from typing import Tuple, List, Optional, Dict
import logging
import json
import h5py
from pathlib import Path

logger = logging.getLogger(__name__)


class AttentionLayer(layers.Layer):
    """
    Self-attention layer for temporal sequences.
    Learns to focus on important timesteps in the sequence.
    """

    def __init__(self, units: int = 64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Score function
        score = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)

        # Context vector
        context = inputs * attention_weights
        context = tf.reduce_sum(context, axis=1)

        return context, attention_weights

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'units': self.units})
        return config


class TemporalBlock(layers.Layer):
    """
    Temporal Convolutional Block for capturing local patterns.
    """

    def __init__(self, filters: int, kernel_size: int = 3, dilation_rate: int = 1, **kwargs):
        super(TemporalBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        self.conv1 = layers.Conv1D(
            self.filters,
            self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding='causal',
            activation='relu'
        )
        self.conv2 = layers.Conv1D(
            self.filters,
            self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding='causal',
            activation='relu'
        )
        self.dropout = layers.Dropout(0.2)
        self.norm = layers.LayerNormalization()

        # Residual connection
        if input_shape[-1] != self.filters:
            self.residual = layers.Conv1D(self.filters, 1, padding='same')
        else:
            self.residual = lambda x: x

        super(TemporalBlock, self).build(input_shape)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.dropout(x, training=training)
        x = self.conv2(x)
        x = self.dropout(x, training=training)

        # Residual connection
        residual = self.residual(inputs)
        x = self.norm(x + residual)

        return x

    def get_config(self):
        config = super(TemporalBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate
        })
        return config


def create_enhanced_lstm_model(
    sequence_length: int = 20,
    num_features: int = 132,
    num_classes: int = 2,
    lstm_units: int = 64,
    dropout_rate: float = 0.3,
    use_attention: bool = True,
    use_bidirectional: bool = True,
    use_tcn: bool = True
) -> Model:
    """
    Create an enhanced LSTM model for violence detection.

    Args:
        sequence_length: Number of timesteps in input sequence
        num_features: Number of features per timestep (132 for pose landmarks)
        num_classes: Number of output classes
        lstm_units: Number of units in LSTM layers
        dropout_rate: Dropout rate for regularization
        use_attention: Whether to use attention mechanism
        use_bidirectional: Whether to use bidirectional LSTM
        use_tcn: Whether to use Temporal Convolutional layers

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(sequence_length, num_features), name='input_sequence')
    x = inputs

    # Optional: Temporal Convolutional preprocessing
    if use_tcn:
        x = TemporalBlock(64, kernel_size=3, dilation_rate=1)(x)
        x = TemporalBlock(64, kernel_size=3, dilation_rate=2)(x)

    # LSTM layers
    lstm_layer = layers.LSTM(lstm_units, return_sequences=True)
    if use_bidirectional:
        x = layers.Bidirectional(lstm_layer, name='bidirectional_lstm_1')(x)
    else:
        x = lstm_layer(x)
    x = layers.Dropout(dropout_rate)(x)

    # Second LSTM layer
    lstm_layer_2 = layers.LSTM(lstm_units, return_sequences=True)
    if use_bidirectional:
        x = layers.Bidirectional(lstm_layer_2, name='bidirectional_lstm_2')(x)
    else:
        x = lstm_layer_2(x)
    x = layers.Dropout(dropout_rate)(x)

    # Attention mechanism or final LSTM
    if use_attention:
        context, attention_weights = AttentionLayer(lstm_units, name='attention')(x)
        x = context
    else:
        x = layers.LSTM(lstm_units, return_sequences=False)(x)
        x = layers.Dropout(dropout_rate)(x)

    # Dense layers
    x = layers.Dense(64, activation='relu', name='dense_1')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation='relu', name='dense_2')(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='enhanced_violence_lstm')

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_simple_lstm_model(
    sequence_length: int = 20,
    num_features: int = 132,
    num_classes: int = 2,
    lstm_units: int = 50,
    dropout_rate: float = 0.2
) -> Model:
    """
    Create a simple LSTM model (similar to original but improved).

    Args:
        sequence_length: Number of timesteps
        num_features: Features per timestep
        num_classes: Number of output classes
        lstm_units: LSTM units per layer
        dropout_rate: Dropout rate

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, num_features)),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.LSTM(lstm_units, return_sequences=False),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ], name='simple_violence_lstm')

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


class ViolenceClassifier:
    """
    High-level violence classifier that handles model loading,
    inference, and temporal smoothing.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        sequence_length: int = 20,
        num_features: int = 132,
        num_classes: int = 2,
        smoothing_window: int = 5,
        threshold: float = 0.6
    ):
        """
        Initialize violence classifier.

        Args:
            model_path: Path to pre-trained model (None to create new)
            sequence_length: Sequence length for LSTM
            num_features: Number of input features
            num_classes: Number of classes
            smoothing_window: Window size for prediction smoothing
            threshold: Violence detection threshold
        """
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.num_classes = num_classes
        self.smoothing_window = smoothing_window
        self.threshold = threshold

        self.model = None
        self.prediction_history: Dict[int, List[np.ndarray]] = {}
        self.class_names = ['neutral', 'violent']

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Load model from file.

        Args:
            model_path: Path to model file (.h5 or SavedModel)
        """
        try:
            # Register custom objects
            custom_objects = {
                'Orthogonal': tf.keras.initializers.Orthogonal,
                'AttentionLayer': AttentionLayer,
                'TemporalBlock': TemporalBlock,
                'keras': tf.keras,   # needed for models with Lambda layers
            }

            model_path = Path(model_path)

            if model_path.suffix == '.h5':
                # Load from H5 file with compatibility handling
                self.model = self._load_h5_model(str(model_path), custom_objects)
            else:
                # Load SavedModel
                self.model = keras.models.load_model(str(model_path), custom_objects=custom_objects)

            logger.info(f"Model loaded successfully from {model_path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_h5_model(self, model_path: str, custom_objects: dict) -> Model:
        """
        Load H5 model with compatibility handling for older models.
        Strategy:
          1. Try direct keras load (works for enhanced model saved in Keras 2 format)
          2. Rebuild architecture from weight shapes + load weights by name
             (handles models saved with DTypePolicy / batch_shape Keras 3 issues)
        """
        # Strategy 1: direct load
        try:
            return keras.models.load_model(model_path, custom_objects=custom_objects)
        except Exception:
            pass

        # Strategy 2: infer architecture from weights, load weights by name
        return self._rebuild_from_weights(model_path)

    def _rebuild_from_weights(self, model_path: str) -> Model:
        """
        Reconstruct the model by reading weight shapes from the H5 file,
        building a matching Sequential architecture, and loading weights by name.

        This handles the case where the H5 was saved with a different Keras
        version that uses DTypePolicy / batch_shape / batch_input_shape keys.
        """
        with h5py.File(model_path, 'r') as f:
            wg = f['model_weights']

            # Collect (layer_name → weight_shapes) for named layers
            layer_shapes: Dict[str, List[tuple]] = {}
            for name in wg.keys():
                g = wg[name]
                if 'weight_names' in g.attrs:
                    wnames = g.attrs['weight_names']
                    layer_shapes[name] = [tuple(wg[name][wn].shape) for wn in wnames]

        lstm_names   = sorted(k for k in layer_shapes if k.startswith('lstm'))
        dense_names  = sorted(k for k in layer_shapes if k.startswith('dense'))
        has_bn       = any(k.startswith('batch_normalization') for k in layer_shapes)
        has_bidir    = any(k.startswith('bidirectional') for k in layer_shapes)

        if not lstm_names or not dense_names:
            raise ValueError("Cannot infer architecture: no lstm/dense layers found in weights")

        # Infer input_dim and units from LSTM kernel shapes (kernel: [input_dim, 4*units])
        def lstm_dims(name):
            kernel_shape = layer_shapes[name][0]   # (input_dim, 4*units)
            return kernel_shape[0], kernel_shape[1] // 4

        input_dim, _ = lstm_dims(lstm_names[0])
        num_outputs  = layer_shapes[dense_names[-1]][0][-1]
        activation   = 'sigmoid' if num_outputs == 1 else 'softmax'

        # Build architecture
        inp = keras.Input(shape=(self.sequence_length, input_dim))
        x = inp
        for i, lname in enumerate(lstm_names):
            _, units = lstm_dims(lname)
            return_seq = i < len(lstm_names) - 1
            lstm_layer = layers.LSTM(units, return_sequences=return_seq, name=lname)
            if has_bidir:
                x = layers.Bidirectional(lstm_layer, name=f'bidirectional_{i}')(x)
            else:
                x = lstm_layer(x)
            if has_bn:
                bn_name = 'batch_normalization' if i == 0 else f'batch_normalization_{i}'
                x = layers.BatchNormalization(name=bn_name)(x)
            x = layers.Dropout(0.3, name=f'dropout_{i}')(x)

        for i, dname in enumerate(dense_names[:-1]):
            units = layer_shapes[dname][0][-1]
            x = layers.Dense(units, activation='relu', name=dname)(x)
            x = layers.Dropout(0.3, name=f'dropout_d{i}')(x)

        x = layers.Dense(num_outputs, activation=activation, name=dense_names[-1])(x)

        model = keras.Model(inputs=inp, outputs=x)
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
        logger.info(f"Rebuilt model from weights: input={input_dim} outputs={num_outputs}")
        return model

    def predict(
        self,
        sequence: np.ndarray,
        person_id: int = 0,
        apply_smoothing: bool = True
    ) -> Tuple[str, float, np.ndarray]:
        """
        Predict violence class for a sequence.

        Args:
            sequence: Input sequence of shape (sequence_length, num_features)
            person_id: ID of the person (for tracking history)
            apply_smoothing: Whether to apply temporal smoothing

        Returns:
            Tuple of (class_name, confidence, raw_probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        # Ensure correct shape
        if sequence.ndim == 2:
            sequence = np.expand_dims(sequence, axis=0)

        # Get prediction
        prediction = self.model.predict(sequence, verbose=0)

        # Apply smoothing
        if apply_smoothing:
            if person_id not in self.prediction_history:
                self.prediction_history[person_id] = []

            self.prediction_history[person_id].append(prediction[0])

            # Keep only recent predictions
            if len(self.prediction_history[person_id]) > self.smoothing_window:
                self.prediction_history[person_id].pop(0)

            # Average predictions
            smoothed = np.mean(self.prediction_history[person_id], axis=0)
        else:
            smoothed = prediction[0]

        # Get class and confidence
        class_idx = np.argmax(smoothed)
        confidence = smoothed[class_idx]
        class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"class_{class_idx}"

        return class_name, float(confidence), smoothed

    def is_violent(self, sequence: np.ndarray, person_id: int = 0) -> Tuple[bool, float]:
        """
        Check if sequence represents violent behavior.

        Args:
            sequence: Input sequence
            person_id: Person ID for tracking

        Returns:
            Tuple of (is_violent, confidence)
        """
        class_name, confidence, probs = self.predict(sequence, person_id)

        # Check if violent class probability exceeds threshold
        violence_prob = probs[1] if len(probs) > 1 else probs[0]
        is_violent = violence_prob > self.threshold

        return is_violent, float(violence_prob)

    def reset_history(self, person_id: Optional[int] = None):
        """
        Reset prediction history.

        Args:
            person_id: Specific person to reset (None for all)
        """
        if person_id is not None:
            self.prediction_history.pop(person_id, None)
        else:
            self.prediction_history.clear()

    def set_class_names(self, names: List[str]):
        """Set custom class names."""
        self.class_names = names


def get_training_callbacks(
    checkpoint_dir: str,
    early_stopping_patience: int = 15,
    reduce_lr_patience: int = 5
) -> List:
    """
    Get callbacks for model training.

    Args:
        checkpoint_dir: Directory to save checkpoints
        early_stopping_patience: Patience for early stopping
        reduce_lr_patience: Patience for learning rate reduction

    Returns:
        List of Keras callbacks
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            str(checkpoint_dir / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=str(checkpoint_dir / 'logs'),
            histogram_freq=1
        )
    ]

    return callbacks
