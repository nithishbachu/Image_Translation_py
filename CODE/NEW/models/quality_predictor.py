import tensorflow as tf
import numpy as np

class QualityMetricsPredictor:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        """Build and compile the neural network model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        return model

    def train(self, X_train, y_train, epochs=100, batch_size=32):
        """Train the model with experimental data"""
        self.model.fit(
            X_train, 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size,
            validation_split=0.2
        )

    def predict(self, approx_metrics):
        """Predict true quality metrics from approximated ones"""
        return self.model.predict(np.array(approx_metrics))
    
if __name__ == '__main__':
    # Import numpy to create training data
    import numpy as np

    # Sample training data (100 samples of 3 features)
    X_train = np.random.rand(100, 3)
    y_train = np.random.rand(100, 3)

    # Initialize and train the model
    predictor = QualityMetricsPredictor()
    predictor.train(X_train, y_train, epochs=10, batch_size=16)

    # Predict on a new example
    test_input = [[20.0, 0.85, 50.0]]  # Example: approximated PSNR, SSIM, FID
    prediction = predictor.predict(test_input)

    print("Predicted Quality Metrics:", prediction)
