package phase1

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// TestMLPBasicFunctionality tests basic MLP operations
func TestMLPBasicFunctionality(t *testing.T) {
	t.Run("Creation", func(t *testing.T) {
		mlp, err := NewMLP(2, []int{3}, 1, []ActivationFunction{Sigmoid, Sigmoid}, 0.1)
		if err != nil {
			t.Fatalf("Failed to create MLP: %v", err)
		}

		if mlp.InputSize != 2 {
			t.Errorf("Expected input size 2, got %d", mlp.InputSize)
		}

		if len(mlp.Layers) != 2 {
			t.Errorf("Expected 2 layers (1 hidden + 1 output), got %d", len(mlp.Layers))
		}

		// Check hidden layer structure
		hiddenLayer := mlp.Layers[0]
		if len(hiddenLayer.Weights) != 3 {
			t.Errorf("Expected 3 neurons in hidden layer, got %d", len(hiddenLayer.Weights))
		}
		if len(hiddenLayer.Weights[0]) != 2 {
			t.Errorf("Expected 2 weights per hidden neuron, got %d", len(hiddenLayer.Weights[0]))
		}

		// Check output layer structure
		outputLayer := mlp.Layers[1]
		if len(outputLayer.Weights) != 1 {
			t.Errorf("Expected 1 neuron in output layer, got %d", len(outputLayer.Weights))
		}
		if len(outputLayer.Weights[0]) != 3 {
			t.Errorf("Expected 3 weights per output neuron, got %d", len(outputLayer.Weights[0]))
		}
	})

	t.Run("ForwardPass", func(t *testing.T) {
		mlp, err := NewMLP(2, []int{3}, 1, []ActivationFunction{Sigmoid, Sigmoid}, 0.1)
		if err != nil {
			t.Fatalf("Failed to create MLP: %v", err)
		}

		input := []float64{0.5, 0.8}
		output, err := mlp.Forward(input)
		if err != nil {
			t.Fatalf("Forward pass failed: %v", err)
		}

		if len(output) != 1 {
			t.Errorf("Expected 1 output, got %d", len(output))
		}

		// Output should be between 0 and 1 for sigmoid activation
		if output[0] < 0 || output[0] > 1 {
			t.Errorf("Sigmoid output should be in [0,1], got %f", output[0])
		}
	})

	t.Run("InputValidation", func(t *testing.T) {
		mlp, _ := NewMLP(2, []int{3}, 1, []ActivationFunction{Sigmoid, Sigmoid}, 0.1)

		// Test wrong input size
		_, err := mlp.Forward([]float64{0.5}) // Only 1 input instead of 2
		if err == nil {
			t.Error("Expected error for wrong input size")
		}

		// Test wrong target size in training
		err = mlp.Train([]float64{0.5, 0.8}, []float64{0.1, 0.2}) // 2 targets instead of 1
		if err == nil {
			t.Error("Expected error for wrong target size")
		}
	})

	t.Run("ConfigurationValidation", func(t *testing.T) {
		// Test mismatched activations
		_, err := NewMLP(2, []int{3}, 1, []ActivationFunction{Sigmoid}, 0.1) // Too few activations
		if err == nil {
			t.Error("Expected error for mismatched activations")
		}
	})
}

// TestActivationFunctions tests all activation functions and their derivatives
func TestActivationFunctions(t *testing.T) {
	mlp := &MLP{} // Just for accessing methods

	testCases := []struct {
		name       string
		activation ActivationFunction
		input      float64
		minOutput  float64
		maxOutput  float64
	}{
		{"Sigmoid", Sigmoid, 0.0, 0.4, 0.6}, // sigmoid(0) ≈ 0.5
		{"Sigmoid_Positive", Sigmoid, 2.0, 0.8, 1.0},
		{"Sigmoid_Negative", Sigmoid, -2.0, 0.0, 0.2},
		{"Tanh", Tanh, 0.0, -0.1, 0.1},         // tanh(0) = 0
		{"Tanh_Positive", Tanh, 1.0, 0.7, 0.8}, // tanh(1) ≈ 0.76
		{"Tanh_Negative", Tanh, -1.0, -0.8, -0.7},
		{"ReLU", ReLU, 0.0, 0.0, 0.0},           // ReLU(0) = 0
		{"ReLU_Positive", ReLU, 2.5, 2.5, 2.5},  // ReLU(2.5) = 2.5
		{"ReLU_Negative", ReLU, -1.0, 0.0, 0.0}, // ReLU(-1) = 0
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			output := mlp.activate(tc.input, tc.activation)
			if output < tc.minOutput || output > tc.maxOutput {
				t.Errorf("Activation %v(%f) = %f, expected in range [%f, %f]",
					tc.activation, tc.input, output, tc.minOutput, tc.maxOutput)
			}

			// Test derivative
			derivative := mlp.activateDerivative(tc.input, tc.activation)
			if math.IsNaN(derivative) || math.IsInf(derivative, 0) {
				t.Errorf("Derivative of %v(%f) is invalid: %f", tc.activation, tc.input, derivative)
			}

			// Derivatives should be non-negative for these functions
			if tc.activation == Sigmoid || tc.activation == ReLU {
				if derivative < 0 {
					t.Errorf("Derivative of %v should be non-negative, got %f", tc.activation, derivative)
				}
			}
		})
	}
}

// TestMLPXORProblem tests the network's ability to learn the XOR function
// This is the key test that demonstrates MLP's advantage over perceptron
func TestMLPXORProblem(t *testing.T) {
	t.Run("XORLearning", func(t *testing.T) {
		// XOR problem: non-linearly separable
		// Input: (0,0) -> 0, (0,1) -> 1, (1,0) -> 1, (1,1) -> 0

		// Create MLP: 2 inputs -> 4 hidden neurons -> 1 output
		// Using 4 hidden neurons ensures sufficient capacity for XOR
		mlp, err := NewMLP(2, []int{4}, 1, []ActivationFunction{Sigmoid, Sigmoid}, 0.5)
		if err != nil {
			t.Fatalf("Failed to create MLP: %v", err)
		}

		// XOR training data
		inputs := [][]float64{
			{0, 0}, // XOR(0,0) = 0
			{0, 1}, // XOR(0,1) = 1
			{1, 0}, // XOR(1,0) = 1
			{1, 1}, // XOR(1,1) = 0
		}
		targets := [][]float64{
			{0}, // Expected output for (0,0)
			{1}, // Expected output for (0,1)
			{1}, // Expected output for (1,0)
			{0}, // Expected output for (1,1)
		}

		// Training phase
		maxEpochs := 5000
		targetAccuracy := 0.95

		// Set seed for reproducible results
		rand.Seed(42)

		var finalAccuracy float64
		for epoch := 0; epoch < maxEpochs; epoch++ {
			// Shuffle training data each epoch
			for i := len(inputs) - 1; i > 0; i-- {
				j := rand.Intn(i + 1)
				inputs[i], inputs[j] = inputs[j], inputs[i]
				targets[i], targets[j] = targets[j], targets[i]
			}

			// Train on all examples
			for i := range inputs {
				err := mlp.Train(inputs[i], targets[i])
				if err != nil {
					t.Fatalf("Training failed at epoch %d: %v", epoch, err)
				}
			}

			// Check accuracy every 100 epochs
			if epoch%100 == 0 || epoch == maxEpochs-1 {
				accuracy := calculateXORAccuracy(t, mlp, inputs, targets)
				finalAccuracy = accuracy

				if accuracy >= targetAccuracy {
					t.Logf("XOR learning converged at epoch %d with %.2f%% accuracy", epoch, accuracy*100)
					break
				}
			}
		}

		// Verify final accuracy meets requirement
		if finalAccuracy < targetAccuracy {
			t.Errorf("XOR learning failed: achieved %.2f%% accuracy, required %.2f%%",
				finalAccuracy*100, targetAccuracy*100)
		}

		// Detailed output verification
		t.Logf("Final XOR results:")
		for i, input := range inputs {
			output, _ := mlp.Predict(input)
			expected := targets[i][0]
			t.Logf("XOR(%v) = %.4f (expected: %.1f)", input, output[0], expected)
		}
	})

	t.Run("XORGeneralization", func(t *testing.T) {
		// Test network's ability to generalize XOR pattern
		mlp, _ := NewMLP(2, []int{4}, 1, []ActivationFunction{Sigmoid, Sigmoid}, 0.3)

		// Train with slightly different XOR data representation
		inputs := [][]float64{
			{0.1, 0.1}, // ~(0,0) -> 0
			{0.1, 0.9}, // ~(0,1) -> 1
			{0.9, 0.1}, // ~(1,0) -> 1
			{0.9, 0.9}, // ~(1,1) -> 0
		}
		targets := [][]float64{
			{0.1}, {0.9}, {0.9}, {0.1},
		}

		// Train for fewer epochs to test generalization
		for epoch := 0; epoch < 2000; epoch++ {
			for i := range inputs {
				mlp.Train(inputs[i], targets[i])
			}
		}

		// Test on exact XOR values
		testInputs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
		testTargets := [][]float64{{0}, {1}, {1}, {0}}

		accuracy := calculateXORAccuracy(t, mlp, testInputs, testTargets)
		if accuracy < 0.5 { // Lower threshold for generalization test
			t.Errorf("XOR generalization failed: %.2f%% accuracy", accuracy*100)
		}
	})
}

// calculateXORAccuracy computes accuracy for XOR problem
func calculateXORAccuracy(t *testing.T, mlp *MLP, inputs [][]float64, targets [][]float64) float64 {
	correct := 0
	total := len(inputs)

	for i := range inputs {
		output, err := mlp.Predict(inputs[i])
		if err != nil {
			t.Fatalf("Prediction failed: %v", err)
		}

		// Convert continuous output to binary decision
		predicted := 0.0
		if output[0] > 0.5 {
			predicted = 1.0
		}

		if math.Abs(predicted-targets[i][0]) < 0.1 {
			correct++
		}
	}

	return float64(correct) / float64(total)
}

// TestLearningDynamics tests the learning process characteristics
func TestLearningDynamics(t *testing.T) {
	t.Run("ErrorDecreasing", func(t *testing.T) {
		mlp, _ := NewMLP(2, []int{3}, 1, []ActivationFunction{Sigmoid, Sigmoid}, 0.1)

		// Simple training data
		inputs := [][]float64{{0, 0}, {1, 1}}
		targets := [][]float64{{0}, {1}}

		// Measure initial error
		initialError, _ := mlp.CalculateError(inputs, targets)

		// Train for several epochs
		for epoch := 0; epoch < 100; epoch++ {
			for i := range inputs {
				mlp.Train(inputs[i], targets[i])
			}
		}

		// Measure final error
		finalError, _ := mlp.CalculateError(inputs, targets)

		if finalError >= initialError {
			t.Errorf("Error should decrease during training: initial=%.4f, final=%.4f",
				initialError, finalError)
		}
	})

	t.Run("LearningRateEffect", func(t *testing.T) {
		// Test different learning rates
		learningRates := []float64{0.01, 0.1, 1.0}

		for _, lr := range learningRates {
			mlp, _ := NewMLP(2, []int{2}, 1, []ActivationFunction{Sigmoid, Sigmoid}, lr)

			// Simple pattern
			input := []float64{0.5, 0.5}
			target := []float64{0.8}

			// Measure change after one training step
			initialOutput, _ := mlp.Predict(input)
			mlp.Train(input, target)
			finalOutput, _ := mlp.Predict(input)

			change := math.Abs(finalOutput[0] - initialOutput[0])

			// Higher learning rate should cause larger changes (within reason)
			if lr > 0.1 && change < 0.001 {
				t.Errorf("Learning rate %.2f seems too small, change=%.6f", lr, change)
			}
		}
	})
}

// TestMLPArchitectures tests different network architectures
func TestMLPArchitectures(t *testing.T) {
	architectures := []struct {
		name        string
		hiddenSizes []int
		activations []ActivationFunction
	}{
		{"SingleHidden", []int{3}, []ActivationFunction{Sigmoid, Sigmoid}},
		{"DoubleHidden", []int{4, 3}, []ActivationFunction{Sigmoid, Sigmoid, Sigmoid}},
		{"ReLUHidden", []int{4}, []ActivationFunction{ReLU, Sigmoid}},
		{"TanhHidden", []int{3}, []ActivationFunction{Tanh, Sigmoid}},
		{"MixedActivation", []int{4, 3}, []ActivationFunction{ReLU, Tanh, Sigmoid}},
	}

	for _, arch := range architectures {
		t.Run(arch.name, func(t *testing.T) {
			mlp, err := NewMLP(2, arch.hiddenSizes, 1, arch.activations, 0.1)
			if err != nil {
				t.Fatalf("Failed to create %s architecture: %v", arch.name, err)
			}

			// Test forward pass
			output, err := mlp.Forward([]float64{0.5, 0.7})
			if err != nil {
				t.Fatalf("Forward pass failed for %s: %v", arch.name, err)
			}

			if len(output) != 1 {
				t.Errorf("Expected 1 output for %s, got %d", arch.name, len(output))
			}

			// Test training step
			err = mlp.Train([]float64{0.5, 0.7}, []float64{0.3})
			if err != nil {
				t.Fatalf("Training failed for %s: %v", arch.name, err)
			}
		})
	}
}

// TestNumericalStability tests numerical precision and stability
func TestNumericalStability(t *testing.T) {
	t.Run("ExtremeLearningRates", func(t *testing.T) {
		extremeRates := []float64{1e-6, 1e6}

		for _, lr := range extremeRates {
			mlp, _ := NewMLP(2, []int{2}, 1, []ActivationFunction{Sigmoid, Sigmoid}, lr)

			// Should not crash with extreme learning rates
			err := mlp.Train([]float64{0.5, 0.5}, []float64{0.5})
			if err != nil {
				t.Errorf("Training crashed with learning rate %.0e: %v", lr, err)
			}
		}
	})

	t.Run("ExtremeInputs", func(t *testing.T) {
		mlp, _ := NewMLP(2, []int{2}, 1, []ActivationFunction{Sigmoid, Sigmoid}, 0.1)

		extremeInputs := [][]float64{
			{1e6, 1e6},   // Very large
			{-1e6, -1e6}, // Very negative
			{0, 0},       // Zero
		}

		for _, input := range extremeInputs {
			output, err := mlp.Forward(input)
			if err != nil {
				t.Errorf("Forward pass failed with extreme input %v: %v", input, err)
				continue
			}

			// Check for NaN or infinity
			for _, val := range output {
				if math.IsNaN(val) || math.IsInf(val, 0) {
					t.Errorf("Invalid output %f for input %v", val, input)
				}
			}
		}
	})
}

// TestMLPModelPersistence tests JSON serialization/deserialization
func TestMLPModelPersistence(t *testing.T) {
	t.Run("JSONSerialization", func(t *testing.T) {
		// Create and train a small network
		mlp, _ := NewMLP(2, []int{3}, 1, []ActivationFunction{Sigmoid, Sigmoid}, 0.1)

		// Train it slightly so it has learned weights
		for i := 0; i < 10; i++ {
			mlp.Train([]float64{0, 1}, []float64{1})
		}

		// Test prediction before serialization
		originalOutput, _ := mlp.Predict([]float64{0.5, 0.5})

		// Serialize to JSON
		jsonData, err := mlp.ToJSON()
		if err != nil {
			t.Fatalf("Failed to serialize MLP: %v", err)
		}

		// Create new MLP and deserialize
		newMLP := &MLP{}
		err = newMLP.FromJSON(jsonData)
		if err != nil {
			t.Fatalf("Failed to deserialize MLP: %v", err)
		}

		// Test prediction after deserialization
		newOutput, err := newMLP.Predict([]float64{0.5, 0.5})
		if err != nil {
			t.Fatalf("Prediction failed after deserialization: %v", err)
		}

		// Outputs should be identical
		if math.Abs(originalOutput[0]-newOutput[0]) > 1e-10 {
			t.Errorf("Deserialized MLP produces different output: original=%.10f, new=%.10f",
				originalOutput[0], newOutput[0])
		}
	})
}

// TestBatchTraining tests batch training functionality
func TestBatchTraining(t *testing.T) {
	t.Run("BatchProcessing", func(t *testing.T) {
		mlp, _ := NewMLP(2, []int{3}, 1, []ActivationFunction{Sigmoid, Sigmoid}, 0.1)

		// Create batch data
		inputBatch := [][]float64{
			{0, 0}, {0, 1}, {1, 0}, {1, 1},
		}
		targetBatch := [][]float64{
			{0}, {1}, {1}, {0}, // XOR pattern
		}

		// Test batch training
		err := mlp.TrainBatch(inputBatch, targetBatch)
		if err != nil {
			t.Fatalf("Batch training failed: %v", err)
		}

		// Test mismatched batch sizes
		invalidTargetBatch := [][]float64{{0}, {1}} // Wrong size
		err = mlp.TrainBatch(inputBatch, invalidTargetBatch)
		if err == nil {
			t.Error("Expected error for mismatched batch sizes")
		}
	})
}

// BenchmarkMLP benchmarks MLP performance
func BenchmarkMLP(b *testing.B) {
	mlp, _ := NewMLP(10, []int{20, 10}, 5, []ActivationFunction{ReLU, ReLU, Sigmoid}, 0.01)
	input := make([]float64, 10)
	target := make([]float64, 5)

	// Initialize with random values
	rand.Seed(time.Now().UnixNano())
	for i := range input {
		input[i] = rand.Float64()
	}
	for i := range target {
		target[i] = rand.Float64()
	}

	b.ResetTimer()

	b.Run("Forward", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			mlp.Forward(input)
		}
	})

	b.Run("Training", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			mlp.Train(input, target)
		}
	})
}
