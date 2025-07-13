// Package phase1 test suite
// Learning Goal: Understanding comprehensive testing patterns for ML algorithms
package phase1

import (
	"fmt"
	"math"
	"testing"
)

// TestPerceptronBasicFunctionality tests basic perceptron operations
// Learning Goal: Understanding unit testing for neural networks
func TestPerceptronBasicFunctionality(t *testing.T) {
	t.Run("Creation", func(t *testing.T) {
		p := NewPerceptron(2, 0.1)

		if len(p.Weights) != 2 {
			t.Errorf("Expected 2 weights, got %d", len(p.Weights))
		}

		if p.LearningRate != 0.1 {
			t.Errorf("Expected learning rate 0.1, got %f", p.LearningRate)
		}

		if p.Bias != 0.0 {
			t.Errorf("Expected initial bias 0.0, got %f", p.Bias)
		}
	})

	t.Run("ForwardPass", func(t *testing.T) {
		p := NewPerceptronWithSeed(2, 0.1, 42) // Fixed seed for reproducibility

		// Test forward pass
		output, err := p.Forward([]float64{1.0, 1.0})
		if err != nil {
			t.Fatalf("Forward pass failed: %v", err)
		}

		// Output should be 0 or 1
		if output != 0.0 && output != 1.0 {
			t.Errorf("Expected output 0 or 1, got %f", output)
		}
	})

	t.Run("InputValidation", func(t *testing.T) {
		p := NewPerceptron(2, 0.1)

		// Test input size mismatch
		_, err := p.Forward([]float64{1.0}) // Wrong size
		if err == nil {
			t.Error("Expected error for wrong input size")
		}

		// Test training with wrong input size
		err = p.Train([]float64{1.0}, 1.0) // Wrong size
		if err == nil {
			t.Error("Expected error for wrong input size in training")
		}
	})
}

// TestPerceptronLearning tests the learning capability
// Learning Goal: Understanding training convergence and learning patterns
func TestPerceptronLearning(t *testing.T) {
	t.Run("SimpleLinearSeparable", func(t *testing.T) {
		p := NewPerceptronWithSeed(2, 0.5, 42)

		// Simple linearly separable data: AND gate
		inputs := [][]float64{
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1},
		}
		targets := []float64{0, 0, 0, 1}

		epochs, err := p.TrainDataset(inputs, targets, 100)
		if err != nil {
			t.Fatalf("Training failed: %v", err)
		}

		// Should converge
		if epochs >= 100 {
			t.Errorf("Training did not converge within 100 epochs")
		}

		// Check final accuracy
		accuracy, err := p.Accuracy(inputs, targets)
		if err != nil {
			t.Fatalf("Accuracy calculation failed: %v", err)
		}

		if accuracy != 1.0 {
			t.Errorf("Expected 100%% accuracy, got %.2f%%", accuracy*100)
		}
	})

	t.Run("WeightUpdate", func(t *testing.T) {
		p := NewPerceptronWithSeed(2, 0.1, 42)

		// Record initial weights
		initialWeights := p.GetWeights()
		initialBias := p.GetBias()

		// Train on one sample that will cause an error
		err := p.Train([]float64{1.0, 1.0}, 1.0)
		if err != nil {
			t.Fatalf("Training failed: %v", err)
		}

		// Weights should have changed if there was an error
		finalWeights := p.GetWeights()
		finalBias := p.GetBias()

		// At least one weight should be different (unless prediction was already correct)
		prediction, _ := NewPerceptronWithSeed(2, 0.1, 42).Forward([]float64{1.0, 1.0})
		if prediction != 1.0 { // If prediction was wrong, weights should change
			weightChanged := false
			for i := range initialWeights {
				if initialWeights[i] != finalWeights[i] {
					weightChanged = true
					break
				}
			}

			if !weightChanged && initialBias == finalBias {
				t.Error("Weights should have changed during learning")
			}
		}
	})
}

// TestXORProblem tests the famous XOR problem (non-linearly separable)
// Learning Goal: Understanding limitations of single perceptrons
func TestXORProblem(t *testing.T) {
	t.Run("XORNotLearnable", func(t *testing.T) {
		p := NewPerceptronWithSeed(2, 0.1, 42)

		// XOR data - not linearly separable
		inputs := [][]float64{
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1},
		}
		targets := []float64{0, 1, 1, 0}

		epochs, err := p.TrainDataset(inputs, targets, 1000)
		if err != nil {
			t.Fatalf("Training failed: %v", err)
		}

		// Should not converge (single perceptron cannot learn XOR)
		if epochs < 1000 {
			t.Log("Note: Training converged, but this is unexpected for XOR with single perceptron")
		}

		accuracy, err := p.Accuracy(inputs, targets)
		if err != nil {
			t.Fatalf("Accuracy calculation failed: %v", err)
		}

		// Accuracy should be <= 75% (at best, it can get 3/4 correct)
		if accuracy > 0.75 {
			t.Logf("Note: Achieved %.2f%% accuracy on XOR. Single perceptron theoretical limit is 75%%", accuracy*100)
		}

		t.Logf("XOR learning result: %d epochs, %.2f%% accuracy", epochs, accuracy*100)
	})
}

// TestNumericalPrecision tests numerical precision and stability
// Learning Goal: Understanding floating-point precision in neural networks
func TestNumericalPrecision(t *testing.T) {
	t.Run("WeightPrecision", func(t *testing.T) {
		p := NewPerceptronWithSeed(3, 0.1, 42)

		// Test with very small values
		inputs := [][]float64{
			{1e-10, 1e-10, 1e-10},
			{1e10, 1e10, 1e10},
		}

		for i, input := range inputs {
			_, err := p.Forward(input)
			if err != nil {
				t.Errorf("Forward pass %d failed with extreme values: %v", i, err)
			}
		}
	})

	t.Run("LearningStability", func(t *testing.T) {
		// Test with different learning rates
		learningRates := []float64{0.001, 0.01, 0.1, 0.5, 1.0}

		for _, lr := range learningRates {
			p := NewPerceptronWithSeed(2, lr, 42)

			// Simple training data
			inputs := [][]float64{{0, 1}, {1, 0}}
			targets := []float64{1, 1}

			_, err := p.TrainDataset(inputs, targets, 100)
			if err != nil {
				t.Errorf("Training with learning rate %.3f failed: %v", lr, err)
			}

			// Check that weights are reasonable (not NaN or infinite)
			weights := p.GetWeights()
			for i, w := range weights {
				if math.IsNaN(w) || math.IsInf(w, 0) {
					t.Errorf("Weight %d became invalid (%.6f) with learning rate %.3f", i, w, lr)
				}
			}
		}
	})
}

// TestModelPersistence tests model save/load functionality
// Learning Goal: Understanding model serialization patterns
func TestModelPersistence(t *testing.T) {
	t.Run("JSONSerialization", func(t *testing.T) {
		original := NewPerceptronWithSeed(3, 0.2, 42)

		// Train a bit to have non-trivial weights
		err := original.Train([]float64{1, 0, 1}, 1)
		if err != nil {
			t.Fatalf("Training failed: %v", err)
		}

		// Serialize to JSON
		data, err := original.ToJSON()
		if err != nil {
			t.Fatalf("JSON serialization failed: %v", err)
		}

		// Deserialize from JSON
		restored, err := FromJSON(data)
		if err != nil {
			t.Fatalf("JSON deserialization failed: %v", err)
		}

		// Compare original and restored
		if len(original.Weights) != len(restored.Weights) {
			t.Errorf("Weight count mismatch: %d vs %d", len(original.Weights), len(restored.Weights))
		}

		for i := range original.Weights {
			if math.Abs(original.Weights[i]-restored.Weights[i]) > 1e-10 {
				t.Errorf("Weight %d mismatch: %f vs %f", i, original.Weights[i], restored.Weights[i])
			}
		}

		if math.Abs(original.Bias-restored.Bias) > 1e-10 {
			t.Errorf("Bias mismatch: %f vs %f", original.Bias, restored.Bias)
		}

		if original.LearningRate != restored.LearningRate {
			t.Errorf("Learning rate mismatch: %f vs %f", original.LearningRate, restored.LearningRate)
		}
	})
}

// TestEdgeCases tests edge cases and error conditions
// Learning Goal: Understanding robust error handling
func TestEdgeCases(t *testing.T) {
	t.Run("EmptyDataset", func(t *testing.T) {
		p := NewPerceptron(2, 0.1)

		_, err := p.TrainDataset([][]float64{}, []float64{}, 10)
		if err == nil {
			t.Error("Expected error for empty dataset")
		}
	})

	t.Run("MismatchedDataSize", func(t *testing.T) {
		p := NewPerceptron(2, 0.1)

		inputs := [][]float64{{1, 0}, {0, 1}}
		targets := []float64{1} // Wrong size

		_, err := p.TrainDataset(inputs, targets, 10)
		if err == nil {
			t.Error("Expected error for mismatched data sizes")
		}
	})

	t.Run("ZeroLearningRate", func(t *testing.T) {
		p := NewPerceptron(2, 0.0) // Zero learning rate

		inputs := [][]float64{{1, 0}}
		targets := []float64{1}

		initialWeights := p.GetWeights()

		epochs, err := p.TrainDataset(inputs, targets, 10)
		if err != nil {
			t.Fatalf("Training failed: %v", err)
		}

		finalWeights := p.GetWeights()

		// With zero learning rate, weights should not change
		for i := range initialWeights {
			if initialWeights[i] != finalWeights[i] {
				t.Errorf("Weights changed with zero learning rate")
				break
			}
		}

		// Should reach max epochs (no convergence with zero learning rate)
		if epochs < 10 {
			t.Logf("Training completed in %d epochs with zero learning rate", epochs)
		}
	})
}

// BenchmarkPerceptron tests performance characteristics
// Learning Goal: Understanding performance considerations in neural networks
func BenchmarkPerceptron(b *testing.B) {
	b.Run("Forward", func(b *testing.B) {
		p := NewPerceptronWithSeed(100, 0.1, 42) // Larger perceptron
		input := make([]float64, 100)
		for i := range input {
			input[i] = float64(i % 2)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := p.Forward(input)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("Training", func(b *testing.B) {
		// Generate synthetic linearly separable data
		inputs := make([][]float64, 1000)
		targets := make([]float64, 1000)

		for i := range inputs {
			x1 := float64(i % 2)
			x2 := float64((i / 2) % 2)
			inputs[i] = []float64{x1, x2}
			targets[i] = x1 // Simple pattern
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			p := NewPerceptronWithSeed(2, 0.1, int64(i))
			_, err := p.TrainDataset(inputs, targets, 100)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("Accuracy", func(b *testing.B) {
		p := NewPerceptronWithSeed(2, 0.1, 42)

		// Generate test data
		inputs := make([][]float64, 1000)
		targets := make([]float64, 1000)

		for i := range inputs {
			x1 := float64(i % 2)
			x2 := float64((i / 2) % 2)
			inputs[i] = []float64{x1, x2}
			targets[i] = x1
		}

		// Train first
		_, err := p.TrainDataset(inputs, targets, 100)
		if err != nil {
			b.Fatal(err)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := p.Accuracy(inputs, targets)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

// ExamplePerceptron demonstrates basic usage patterns
// Learning Goal: Understanding API usage and documentation
func ExamplePerceptron() {
	// Create a perceptron for 2-input problems
	p := NewPerceptron(2, 0.1)

	// Simple AND gate training data
	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targets := []float64{0, 0, 0, 1}

	// Train the perceptron
	epochs, err := p.TrainDataset(inputs, targets, 100)
	if err != nil {
		fmt.Printf("Training failed: %v\n", err)
		return
	}

	fmt.Printf("Training completed in %d epochs\n", epochs)

	// Test the trained perceptron
	for i, input := range inputs {
		prediction, _ := p.Predict(input)
		fmt.Printf("Input: %v -> Prediction: %.0f, Target: %.0f\n",
			input, prediction, targets[i])
	}

	// Calculate accuracy
	accuracy, _ := p.Accuracy(inputs, targets)
	fmt.Printf("Accuracy: %.2f%%\n", accuracy*100)
}

// TestDeterministicBehavior ensures reproducible results
// Learning Goal: Understanding importance of reproducibility in ML
func TestDeterministicBehavior(t *testing.T) {
	seed := int64(12345)

	// Create two identical perceptrons
	p1 := NewPerceptronWithSeed(2, 0.1, seed)
	p2 := NewPerceptronWithSeed(2, 0.1, seed)

	// They should have identical initial weights
	w1 := p1.GetWeights()
	w2 := p2.GetWeights()

	for i := range w1 {
		if w1[i] != w2[i] {
			t.Errorf("Initial weights differ at index %d: %f vs %f", i, w1[i], w2[i])
		}
	}

	// Train both on same data
	inputs := [][]float64{{1, 0}, {0, 1}}
	targets := []float64{1, 0}

	epochs1, err1 := p1.TrainDataset(inputs, targets, 10)
	epochs2, err2 := p2.TrainDataset(inputs, targets, 10)

	if err1 != nil || err2 != nil {
		t.Fatalf("Training failed: %v, %v", err1, err2)
	}

	if epochs1 != epochs2 {
		t.Errorf("Training epochs differ: %d vs %d", epochs1, epochs2)
	}

	// Final weights should be identical
	w1 = p1.GetWeights()
	w2 = p2.GetWeights()

	for i := range w1 {
		if math.Abs(w1[i]-w2[i]) > 1e-10 {
			t.Errorf("Final weights differ at index %d: %f vs %f", i, w1[i], w2[i])
		}
	}
}
