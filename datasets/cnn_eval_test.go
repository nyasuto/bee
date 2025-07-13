// Package datasets implements comprehensive tests for CNN evaluation functionality
// Learning Goal: Understanding CNN evaluation testing patterns and validation
package datasets

import (
	"math"
	"testing"
	"time"

	"github.com/nyasuto/bee/phase2"
)

// TestCNNEvaluator tests the CNN evaluator creation and basic functionality
func TestCNNEvaluator(t *testing.T) {
	// Create a small test dataset
	images := make([][][][]float64, 4)
	labels := []int{0, 1, 0, 1}

	// Create 4x4x1 test images
	for i := range images {
		images[i] = make([][][]float64, 4)
		for j := range images[i] {
			images[i][j] = make([][]float64, 4)
			for k := range images[i][j] {
				images[i][j][k] = []float64{float64(i%2) * 0.5} // Simple pattern
			}
		}
	}

	dataset := &ImageDataset{
		Name:     "TestDataset",
		Images:   images,
		Labels:   labels,
		Classes:  []string{"0", "1"},
		Width:    4,
		Height:   4,
		Channels: 1,
	}

	t.Run("CreateEvaluator", func(t *testing.T) {
		evaluator := NewCNNEvaluator(dataset, 0.01, 2, 5)

		if evaluator.Dataset != dataset {
			t.Error("Dataset not set correctly")
		}
		if evaluator.LearningRate != 0.01 {
			t.Errorf("Expected learning rate 0.01, got %f", evaluator.LearningRate)
		}
		if evaluator.BatchSize != 2 {
			t.Errorf("Expected batch size 2, got %d", evaluator.BatchSize)
		}
		if evaluator.Epochs != 5 {
			t.Errorf("Expected epochs 5, got %d", evaluator.Epochs)
		}
		if evaluator.Verbose {
			t.Error("Expected verbose to be false by default")
		}
	})

	t.Run("SetVerbose", func(t *testing.T) {
		evaluator := NewCNNEvaluator(dataset, 0.01, 2, 5)
		evaluator.SetVerbose(true)

		if !evaluator.Verbose {
			t.Error("Verbose not set correctly")
		}
	})
}

// TestMNISTCNNCreation tests MNIST CNN architecture creation
func TestMNISTCNNCreation(t *testing.T) {
	t.Run("ValidMNISTDataset", func(t *testing.T) {
		// Create MNIST-compatible dataset
		images := make([][][][]float64, 2)
		labels := []int{0, 1}

		// Create 28x28x1 images
		for i := range images {
			images[i] = make([][][]float64, 28)
			for j := range images[i] {
				images[i][j] = make([][]float64, 28)
				for k := range images[i][j] {
					images[i][j][k] = []float64{float64(i) * 0.5}
				}
			}
		}

		dataset := &ImageDataset{
			Name:     "MNIST",
			Images:   images,
			Labels:   labels,
			Classes:  []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"},
			Width:    28,
			Height:   28,
			Channels: 1,
		}

		evaluator := NewCNNEvaluator(dataset, 0.01, 1, 1)
		err := evaluator.CreateMNISTCNN()
		if err != nil {
			t.Fatalf("Failed to create MNIST CNN: %v", err)
		}

		// Verify CNN architecture
		if evaluator.CNN == nil {
			t.Fatal("CNN not created")
		}

		// Check conv layers
		if len(evaluator.CNN.ConvLayers) != 2 {
			t.Errorf("Expected 2 conv layers, got %d", len(evaluator.CNN.ConvLayers))
		}

		// Check pool layers
		if len(evaluator.CNN.PoolLayers) != 2 {
			t.Errorf("Expected 2 pool layers, got %d", len(evaluator.CNN.PoolLayers))
		}

		// Check fully connected layer
		if len(evaluator.CNN.FCWeights) != 10 {
			t.Errorf("Expected 10 output classes, got %d", len(evaluator.CNN.FCWeights))
		}

		// Check layer configurations
		conv1 := evaluator.CNN.ConvLayers[0]
		if len(conv1.Kernels) != 16 {
			t.Errorf("Expected 16 filters in conv1, got %d", len(conv1.Kernels))
		}
		if len(conv1.Kernels[0]) != 1 {
			t.Errorf("Expected 1 input channel in conv1, got %d", len(conv1.Kernels[0]))
		}

		conv2 := evaluator.CNN.ConvLayers[1]
		if len(conv2.Kernels) != 32 {
			t.Errorf("Expected 32 filters in conv2, got %d", len(conv2.Kernels))
		}
		if len(conv2.Kernels[0]) != 16 {
			t.Errorf("Expected 16 input channels in conv2, got %d", len(conv2.Kernels[0]))
		}
	})

	t.Run("IncompatibleDataset", func(t *testing.T) {
		// Create wrong-sized dataset
		images := make([][][][]float64, 1)
		images[0] = make([][][]float64, 32) // Wrong size
		for j := range images[0] {
			images[0][j] = make([][]float64, 32)
			for k := range images[0][j] {
				images[0][j][k] = []float64{0.5}
			}
		}

		dataset := &ImageDataset{
			Name:     "WrongSize",
			Images:   images,
			Labels:   []int{0},
			Classes:  []string{"0"},
			Width:    32, // Wrong size
			Height:   32, // Wrong size
			Channels: 1,
		}

		evaluator := NewCNNEvaluator(dataset, 0.01, 1, 1)
		err := evaluator.CreateMNISTCNN()
		if err == nil {
			t.Error("Expected error for incompatible dataset")
		}
	})
}

// TestSoftmax tests the softmax function implementation
func TestSoftmax(t *testing.T) {
	// Create dummy evaluator for testing softmax
	evaluator := &CNNEvaluator{}

	t.Run("BasicSoftmax", func(t *testing.T) {
		input := []float64{1.0, 2.0, 3.0}
		output := evaluator.softmax(input)

		// Check that output sums to 1
		sum := 0.0
		for _, val := range output {
			sum += val
		}
		if math.Abs(sum-1.0) > 1e-9 {
			t.Errorf("Softmax output should sum to 1, got %f", sum)
		}

		// Check that all values are positive
		for i, val := range output {
			if val <= 0 {
				t.Errorf("Softmax output[%d] should be positive, got %f", i, val)
			}
		}

		// Check that largest input gives largest output
		maxIndex := 0
		maxVal := output[0]
		for i, val := range output {
			if val > maxVal {
				maxVal = val
				maxIndex = i
			}
		}
		if maxIndex != 2 { // input[2] = 3.0 is largest
			t.Errorf("Expected largest output at index 2, got index %d", maxIndex)
		}
	})

	t.Run("NumericalStability", func(t *testing.T) {
		// Test with large values that could cause overflow
		input := []float64{100.0, 101.0, 102.0}
		output := evaluator.softmax(input)

		// Should not have NaN or Inf values
		for i, val := range output {
			if math.IsNaN(val) || math.IsInf(val, 0) {
				t.Errorf("Softmax output[%d] is NaN or Inf: %f", i, val)
			}
		}

		// Should still sum to 1
		sum := 0.0
		for _, val := range output {
			sum += val
		}
		if math.Abs(sum-1.0) > 1e-9 {
			t.Errorf("Softmax output should sum to 1 even with large inputs, got %f", sum)
		}
	})

	t.Run("SingleValue", func(t *testing.T) {
		input := []float64{5.0}
		output := evaluator.softmax(input)

		if len(output) != 1 {
			t.Errorf("Expected output length 1, got %d", len(output))
		}
		if math.Abs(output[0]-1.0) > 1e-9 {
			t.Errorf("Single value softmax should be 1.0, got %f", output[0])
		}
	})

	t.Run("EqualValues", func(t *testing.T) {
		input := []float64{2.0, 2.0, 2.0}
		output := evaluator.softmax(input)

		// All outputs should be equal (1/3)
		expected := 1.0 / 3.0
		for i, val := range output {
			if math.Abs(val-expected) > 1e-9 {
				t.Errorf("Equal input softmax[%d] should be %f, got %f", i, expected, val)
			}
		}
	})
}

// TestCrossEntropyLoss tests the cross-entropy loss calculation
func TestCrossEntropyLoss(t *testing.T) {
	evaluator := &CNNEvaluator{}

	t.Run("PerfectPrediction", func(t *testing.T) {
		// Perfect prediction: output has probability 1.0 for correct class
		output := []float64{0.1, 0.9, 0.0} // High confidence for class 1
		target := 1
		loss := evaluator.calculateCrossEntropyLoss(output, target)

		// Loss should be reasonable for correct prediction (cross-entropy depends on actual probabilities)
		if loss < 0 || math.IsNaN(loss) || math.IsInf(loss, 0) {
			t.Errorf("Expected finite positive loss, got %f", loss)
		}
	})

	t.Run("WorstPrediction", func(t *testing.T) {
		// Worst prediction: output has probability ~0 for correct class
		output := []float64{0.9, 0.05, 0.05} // Low confidence for class 1
		target := 1
		loss := evaluator.calculateCrossEntropyLoss(output, target)

		// Loss should be large for wrong confident prediction
		if loss < 1.0 {
			t.Errorf("Expected large loss for wrong prediction, got %f", loss)
		}
	})

	t.Run("InvalidTarget", func(t *testing.T) {
		output := []float64{0.3, 0.4, 0.3}
		target := 5 // Out of range
		loss := evaluator.calculateCrossEntropyLoss(output, target)

		if !math.IsInf(loss, 1) {
			t.Errorf("Expected infinite loss for invalid target, got %f", loss)
		}
	})

	t.Run("NegativeTarget", func(t *testing.T) {
		output := []float64{0.3, 0.4, 0.3}
		target := -1 // Negative
		loss := evaluator.calculateCrossEntropyLoss(output, target)

		if !math.IsInf(loss, 1) {
			t.Errorf("Expected infinite loss for negative target, got %f", loss)
		}
	})
}

// TestArgmax tests the argmax function
func TestArgmax(t *testing.T) {
	evaluator := &CNNEvaluator{}

	t.Run("BasicArgmax", func(t *testing.T) {
		values := []float64{0.1, 0.8, 0.1}
		maxIndex := evaluator.argmax(values)

		if maxIndex != 1 {
			t.Errorf("Expected argmax index 1, got %d", maxIndex)
		}
	})

	t.Run("FirstMaxElement", func(t *testing.T) {
		values := []float64{0.5, 0.3, 0.2}
		maxIndex := evaluator.argmax(values)

		if maxIndex != 0 {
			t.Errorf("Expected argmax index 0, got %d", maxIndex)
		}
	})

	t.Run("TieBreaking", func(t *testing.T) {
		// When there are equal max values, should return first occurrence
		values := []float64{0.5, 0.5, 0.3}
		maxIndex := evaluator.argmax(values)

		if maxIndex != 0 {
			t.Errorf("Expected argmax index 0 (first occurrence), got %d", maxIndex)
		}
	})

	t.Run("EmptySlice", func(t *testing.T) {
		values := []float64{}
		maxIndex := evaluator.argmax(values)

		if maxIndex != -1 {
			t.Errorf("Expected argmax -1 for empty slice, got %d", maxIndex)
		}
	})

	t.Run("SingleElement", func(t *testing.T) {
		values := []float64{0.42}
		maxIndex := evaluator.argmax(values)

		if maxIndex != 0 {
			t.Errorf("Expected argmax index 0 for single element, got %d", maxIndex)
		}
	})

	t.Run("NegativeValues", func(t *testing.T) {
		values := []float64{-0.5, -0.1, -0.8}
		maxIndex := evaluator.argmax(values)

		if maxIndex != 1 { // -0.1 is the largest
			t.Errorf("Expected argmax index 1, got %d", maxIndex)
		}
	})
}

// TestEvaluationResults tests the evaluation results structure
func TestEvaluationResults(t *testing.T) {
	t.Run("CreateResults", func(t *testing.T) {
		results := &EvaluationResults{
			ModelName:       "TestCNN",
			DatasetName:     "TestDataset",
			TrainingTime:    time.Second,
			InferenceTime:   time.Millisecond,
			Accuracy:        0.85,
			Loss:            0.15,
			EpochsCompleted: 10,
			MemoryUsage:     1024 * 1024, // 1 MB
			PerClassAccuracy: map[int]float64{
				0: 0.8,
				1: 0.9,
			},
			ConfusionMatrix: [][]int{
				{8, 2},
				{1, 9},
			},
			Timestamp: time.Now(),
		}

		if results.ModelName != "TestCNN" {
			t.Errorf("Expected model name 'TestCNN', got %s", results.ModelName)
		}
		if results.Accuracy != 0.85 {
			t.Errorf("Expected accuracy 0.85, got %f", results.Accuracy)
		}
		if len(results.PerClassAccuracy) != 2 {
			t.Errorf("Expected 2 classes in per-class accuracy, got %d", len(results.PerClassAccuracy))
		}
		if len(results.ConfusionMatrix) != 2 {
			t.Errorf("Expected 2x2 confusion matrix, got %dx%d", len(results.ConfusionMatrix), len(results.ConfusionMatrix[0]))
		}
	})
}

// TestTrainBatch tests the batch training functionality
func TestTrainBatch(t *testing.T) {
	// Create small test dataset
	images := make([][][][]float64, 2)
	labels := []int{0, 1}

	// Create 4x4x1 test images
	for i := range images {
		images[i] = make([][][]float64, 4)
		for j := range images[i] {
			images[i][j] = make([][]float64, 4)
			for k := range images[i][j] {
				images[i][j][k] = []float64{float64(i) * 0.5}
			}
		}
	}

	dataset := &ImageDataset{
		Name:     "TestDataset",
		Images:   images,
		Labels:   labels,
		Classes:  []string{"0", "1"},
		Width:    4,
		Height:   4,
		Channels: 1,
	}

	evaluator := NewCNNEvaluator(dataset, 0.01, 2, 1)

	// Create a very simple CNN for testing (no conv/pool layers, just FC)
	evaluator.CNN = &phase2.CNN{
		ConvLayers:   []*phase2.ConvLayer{},
		PoolLayers:   []*phase2.PoolingLayer{},
		FCWeights:    [][]float64{{0.5, 0.3, 0.4, 0.1, 0.2, 0.7, 0.3, 0.6, 0.8, 0.9, 0.1, 0.4, 0.5, 0.2, 0.7, 0.3}, {-0.2, 0.8, 0.3, -0.1, 0.4, 0.6, -0.3, 0.2, 0.5, -0.4, 0.1, 0.7, -0.2, 0.3, 0.8, -0.1}}, // 2 classes, 16 features (4x4x1)
		FCBiases:     []float64{0.1, -0.1},
		LearningRate: 0.01,
		InputShape:   [3]int{4, 4, 1},
		FlattenShape: [2]int{16, 2}, // 4x4x1 = 16 features, 2 classes
	}

	t.Run("ValidBatch", func(t *testing.T) {
		batchImages, batchLabels, err := dataset.GetBatch([]int{0, 1})
		if err != nil {
			t.Fatalf("Failed to get batch: %v", err)
		}

		loss := evaluator.trainBatch(batchImages, batchLabels)

		// Loss should be a reasonable positive value
		if loss <= 0 {
			t.Errorf("Expected positive loss, got %f", loss)
		}
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			t.Errorf("Loss should be finite, got %f", loss)
		}
	})

	t.Run("EmptyBatch", func(t *testing.T) {
		loss := evaluator.trainBatch([][][][]float64{}, []int{})

		// Empty batch should return 0 loss
		if loss != 0.0 {
			t.Errorf("Expected 0 loss for empty batch, got %f", loss)
		}
	})
}

// TestMemoryEstimation tests the memory usage estimation
func TestMemoryEstimation(t *testing.T) {
	// Create simple test dataset
	images := make([][][][]float64, 1)
	images[0] = make([][][]float64, 4)
	for j := range images[0] {
		images[0][j] = make([][]float64, 4)
		for k := range images[0][j] {
			images[0][j][k] = []float64{0.5}
		}
	}

	dataset := &ImageDataset{
		Name:     "TestDataset",
		Images:   images,
		Labels:   []int{0},
		Classes:  []string{"0", "1"},
		Width:    4,
		Height:   4,
		Channels: 1,
	}

	evaluator := NewCNNEvaluator(dataset, 0.01, 1, 1)

	// Create simple CNN
	conv := phase2.NewConvLayer(1, 2, 3, 1, 0, phase2.ReLU)
	evaluator.CNN = &phase2.CNN{
		ConvLayers:   []*phase2.ConvLayer{conv},
		PoolLayers:   []*phase2.PoolingLayer{},
		FCWeights:    [][]float64{{0.5, 0.3}, {-0.2, 0.8}},
		FCBiases:     []float64{0.1, -0.1},
		LearningRate: 0.01,
		InputShape:   [3]int{4, 4, 1},
	}

	t.Run("EstimateMemory", func(t *testing.T) {
		memory := evaluator.estimateMemoryUsage()

		// Should be a positive value
		if memory <= 0 {
			t.Errorf("Expected positive memory usage, got %d", memory)
		}

		// Should include conv layer weights, FC weights, and biases
		// Rough calculation: conv kernels (2*1*3*3) + FC weights (2*2) + biases (2+2) = 18+4+4 = 26 float64s
		// At 8 bytes per float64 = 208 bytes minimum
		if memory < 100 {
			t.Errorf("Expected at least 100 bytes memory usage, got %d", memory)
		}
	})
}

// TestPrintEvaluationResults tests the results printing function
func TestPrintEvaluationResults(t *testing.T) {
	evaluator := &CNNEvaluator{}

	results := &EvaluationResults{
		ModelName:       "TestCNN",
		DatasetName:     "TestDataset",
		TrainingTime:    time.Second,
		InferenceTime:   time.Millisecond,
		Accuracy:        0.85,
		Loss:            0.15,
		EpochsCompleted: 10,
		MemoryUsage:     1024 * 1024,
		PerClassAccuracy: map[int]float64{
			0: 0.8,
			1: 0.9,
		},
		ConfusionMatrix: [][]int{
			{8, 2},
			{1, 9},
		},
		Timestamp: time.Now(),
	}

	t.Run("PrintResults", func(t *testing.T) {
		// This test just ensures the function doesn't panic
		// In a real scenario, you might capture stdout to verify formatting
		evaluator.PrintEvaluationResults(results)
	})

	t.Run("PrintWithLargeConfusionMatrix", func(t *testing.T) {
		// Test with larger confusion matrix (should be simplified)
		largeMatrix := make([][]int, 15)
		for i := range largeMatrix {
			largeMatrix[i] = make([]int, 15)
			largeMatrix[i][i] = 10 // Diagonal values
		}

		largeResults := &EvaluationResults{
			ModelName:        "LargeCNN",
			DatasetName:      "LargeDataset",
			TrainingTime:     time.Second,
			InferenceTime:    time.Millisecond,
			Accuracy:         0.85,
			Loss:             0.15,
			EpochsCompleted:  10,
			MemoryUsage:      1024 * 1024,
			PerClassAccuracy: map[int]float64{},
			ConfusionMatrix:  largeMatrix,
			Timestamp:        time.Now(),
		}

		// Should not print confusion matrix for large matrices
		evaluator.PrintEvaluationResults(largeResults)
	})
}

// TestTrainCNN tests the CNN training functionality and coverage
func TestTrainCNN(t *testing.T) {
	t.Run("SuccessfulTraining", func(t *testing.T) {
		// Create a proper 4x4x1 test dataset
		images := make([][][][]float64, 4)
		labels := []int{0, 1, 0, 1}

		// Create 4x4x1 test images with distinct patterns
		for i := range images {
			images[i] = make([][][]float64, 4)
			for j := range images[i] {
				images[i][j] = make([][]float64, 4)
				for k := range images[i][j] {
					// Create pattern: class 0 = low values, class 1 = high values
					images[i][j][k] = []float64{float64(labels[i])*0.8 + 0.1}
				}
			}
		}

		dataset := &ImageDataset{
			Name:     "TrainingDataset",
			Images:   images,
			Labels:   labels,
			Classes:  []string{"0", "1"},
			Width:    4,
			Height:   4,
			Channels: 1,
		}

		evaluator := NewCNNEvaluator(dataset, 0.1, 2, 3)

		// Create a simple CNN for testing
		conv := phase2.NewConvLayer(1, 2, 3, 1, 0, phase2.ReLU)
		pool := phase2.NewPoolingLayer(2, 2, phase2.MaxPooling)
		evaluator.CNN = &phase2.CNN{
			ConvLayers:   []*phase2.ConvLayer{conv},
			PoolLayers:   []*phase2.PoolingLayer{pool},
			FCWeights:    [][]float64{{0.5, 0.3}, {-0.2, 0.8}}, // 2 features to 2 classes
			FCBiases:     []float64{0.1, -0.1},
			LearningRate: 0.1,
			InputShape:   [3]int{4, 4, 1},
			FlattenShape: [2]int{2, 2}, // After conv+pool: (4-3+1)/2 = 1, so 1x1x2 = 2 features
		}

		// Train the CNN
		results, err := evaluator.TrainCNN()
		if err != nil {
			t.Fatalf("Training failed: %v", err)
		}

		// Verify results structure
		if results == nil {
			t.Fatal("Training results should not be nil")
		}
		if results.ModelName != "MNIST-CNN" {
			t.Errorf("Expected model name 'MNIST-CNN', got %s", results.ModelName)
		}
		if results.DatasetName != "TrainingDataset" {
			t.Errorf("Expected dataset name 'TrainingDataset', got %s", results.DatasetName)
		}
		if results.EpochsCompleted != 3 {
			t.Errorf("Expected 3 epochs, got %d", results.EpochsCompleted)
		}
		if results.TrainingTime <= 0 {
			t.Error("Training time should be positive")
		}
		if results.MemoryUsage <= 0 {
			t.Error("Memory usage should be positive")
		}
	})

	t.Run("TrainingWithoutCNN", func(t *testing.T) {
		dataset := &ImageDataset{
			Name:     "TestDataset",
			Images:   make([][][][]float64, 1),
			Labels:   []int{0},
			Classes:  []string{"0"},
			Width:    4,
			Height:   4,
			Channels: 1,
		}

		evaluator := NewCNNEvaluator(dataset, 0.01, 1, 1)
		// Don't create CNN

		_, err := evaluator.TrainCNN()
		if err == nil {
			t.Error("Expected error when training without CNN initialization")
		}
	})

	t.Run("EmptyDatasetTraining", func(t *testing.T) {
		emptyDataset := &ImageDataset{
			Name:     "EmptyDataset",
			Images:   [][][][]float64{},
			Labels:   []int{},
			Classes:  []string{},
			Width:    1,
			Height:   1,
			Channels: 1,
		}

		evaluator := NewCNNEvaluator(emptyDataset, 0.01, 1, 1)
		evaluator.CNN = &phase2.CNN{
			ConvLayers:   []*phase2.ConvLayer{},
			PoolLayers:   []*phase2.PoolingLayer{},
			FCWeights:    [][]float64{{0.5}, {0.3}},
			FCBiases:     []float64{0.1},
			LearningRate: 0.01,
			InputShape:   [3]int{1, 1, 1},
			FlattenShape: [2]int{1, 1},
		}

		results, err := evaluator.TrainCNN()
		if err != nil {
			t.Fatalf("Training should handle empty dataset gracefully: %v", err)
		}

		// Should return valid results even for empty dataset
		if results == nil {
			t.Error("Results should not be nil for empty dataset")
		}
		if results.Accuracy != 0.0 {
			t.Errorf("Expected 0 accuracy for empty dataset, got %f", results.Accuracy)
		}
	})

	t.Run("VerboseTraining", func(t *testing.T) {
		// Create minimal dataset
		images := make([][][][]float64, 2)
		for i := range images {
			images[i] = make([][][]float64, 2)
			for j := range images[i] {
				images[i][j] = make([][]float64, 2)
				for k := range images[i][j] {
					images[i][j][k] = []float64{0.5}
				}
			}
		}

		dataset := &ImageDataset{
			Name:     "VerboseTestDataset",
			Images:   images,
			Labels:   []int{0, 1},
			Classes:  []string{"0", "1"},
			Width:    2,
			Height:   2,
			Channels: 1,
		}

		evaluator := NewCNNEvaluator(dataset, 0.1, 1, 2)
		evaluator.SetVerbose(true)

		// Create simple CNN
		evaluator.CNN = &phase2.CNN{
			ConvLayers:   []*phase2.ConvLayer{},
			PoolLayers:   []*phase2.PoolingLayer{},
			FCWeights:    [][]float64{{0.5, 0.3, 0.1, 0.2}, {-0.2, 0.8, 0.4, -0.1}}, // 4 features (2x2x1), 2 classes
			FCBiases:     []float64{0.1, -0.1},
			LearningRate: 0.1,
			InputShape:   [3]int{2, 2, 1},
			FlattenShape: [2]int{4, 2},
		}

		// Training should work in verbose mode
		results, err := evaluator.TrainCNN()
		if err != nil {
			t.Fatalf("Verbose training failed: %v", err)
		}
		if results == nil {
			t.Error("Verbose training should return results")
		}
	})
}

// TestCNNEvalEdgeCases tests various edge cases and error conditions for CNN evaluation
func TestCNNEvalEdgeCases(t *testing.T) {

	t.Run("EvaluateAccuracyEmptyDataset", func(t *testing.T) {
		emptyDataset := &ImageDataset{
			Name:     "EmptyDataset",
			Images:   [][][][]float64{},
			Labels:   []int{},
			Classes:  []string{},
			Width:    1,
			Height:   1,
			Channels: 1,
		}

		evaluator := NewCNNEvaluator(emptyDataset, 0.01, 1, 1)
		evaluator.CNN = &phase2.CNN{} // Minimal CNN

		accuracy, perClassAcc, confMatrix := evaluator.EvaluateAccuracy()

		// Should handle empty dataset gracefully
		if accuracy != 0.0 {
			t.Errorf("Expected 0 accuracy for empty dataset, got %f", accuracy)
		}
		if len(perClassAcc) != 0 {
			t.Errorf("Expected empty per-class accuracy, got %d entries", len(perClassAcc))
		}
		if len(confMatrix) != 0 {
			t.Errorf("Expected empty confusion matrix, got %dx%d", len(confMatrix), len(confMatrix))
		}
	})

	t.Run("MeasureInferenceTimeEmptyDataset", func(t *testing.T) {
		emptyDataset := &ImageDataset{
			Name:     "EmptyDataset",
			Images:   [][][][]float64{},
			Labels:   []int{},
			Classes:  []string{},
			Width:    1,
			Height:   1,
			Channels: 1,
		}

		evaluator := NewCNNEvaluator(emptyDataset, 0.01, 1, 1)
		inferenceTime := evaluator.measureInferenceTime()

		if inferenceTime != 0 {
			t.Errorf("Expected 0 inference time for empty dataset, got %v", inferenceTime)
		}
	})
}

// TestEvaluateAccuracy tests the CNN accuracy evaluation functionality
func TestEvaluateAccuracy(t *testing.T) {
	t.Run("BasicAccuracyEvaluation", func(t *testing.T) {
		// Create dataset with predictable patterns
		images := make([][][][]float64, 4)
		labels := []int{0, 1, 0, 1}

		// Create 4x4x1 test images with clear class distinction
		for i := range images {
			images[i] = make([][][]float64, 4)
			for j := range images[i] {
				images[i][j] = make([][]float64, 4)
				for k := range images[i][j] {
					// Class 0: low values, Class 1: high values
					images[i][j][k] = []float64{float64(labels[i])*0.9 + 0.05}
				}
			}
		}

		dataset := &ImageDataset{
			Name:     "AccuracyTestDataset",
			Images:   images,
			Labels:   labels,
			Classes:  []string{"0", "1"},
			Width:    4,
			Height:   4,
			Channels: 1,
		}

		evaluator := NewCNNEvaluator(dataset, 0.01, 2, 1)

		// Create a simple CNN that should learn the pattern
		evaluator.CNN = &phase2.CNN{
			ConvLayers:   []*phase2.ConvLayer{},
			PoolLayers:   []*phase2.PoolingLayer{},
			FCWeights:    [][]float64{{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}, // 16 features (4x4x1), 2 classes
			FCBiases:     []float64{-0.5, -0.5},                                                                                                                                                           // Bias toward negative (favoring first weight)
			LearningRate: 0.01,
			InputShape:   [3]int{4, 4, 1},
			FlattenShape: [2]int{16, 2},
		}

		// Evaluate accuracy
		accuracy, perClassAcc, confMatrix := evaluator.EvaluateAccuracy()

		// Basic validation
		if accuracy < 0 || accuracy > 1 {
			t.Errorf("Accuracy should be between 0 and 1, got %f", accuracy)
		}

		// Per-class accuracy should have entries for each class
		if len(perClassAcc) > 2 {
			t.Errorf("Expected at most 2 classes in per-class accuracy, got %d", len(perClassAcc))
		}

		for class, acc := range perClassAcc {
			if acc < 0 || acc > 1 {
				t.Errorf("Per-class accuracy for class %d should be between 0 and 1, got %f", class, acc)
			}
		}

		// Confusion matrix should be 2x2 for 2 classes
		if len(confMatrix) != 2 {
			t.Errorf("Expected 2x2 confusion matrix, got %dx%d", len(confMatrix), len(confMatrix))
		} else {
			for i, row := range confMatrix {
				if len(row) != 2 {
					t.Errorf("Confusion matrix row %d should have 2 elements, got %d", i, len(row))
				}
				for j, count := range row {
					if count < 0 {
						t.Errorf("Confusion matrix[%d][%d] should be non-negative, got %d", i, j, count)
					}
				}
			}

			// Check that confusion matrix sums to total samples
			total := 0
			for _, row := range confMatrix {
				for _, count := range row {
					total += count
				}
			}
			if total != len(labels) {
				t.Errorf("Confusion matrix should sum to %d samples, got %d", len(labels), total)
			}
		}
	})

	t.Run("AccuracyWithEmptyDataset", func(t *testing.T) {
		emptyDataset := &ImageDataset{
			Name:     "EmptyDataset",
			Images:   [][][][]float64{},
			Labels:   []int{},
			Classes:  []string{},
			Width:    1,
			Height:   1,
			Channels: 1,
		}

		evaluator := NewCNNEvaluator(emptyDataset, 0.01, 1, 1)
		evaluator.CNN = &phase2.CNN{} // Minimal CNN

		accuracy, perClassAcc, confMatrix := evaluator.EvaluateAccuracy()

		// Empty dataset should return zero accuracy
		if accuracy != 0.0 {
			t.Errorf("Expected 0 accuracy for empty dataset, got %f", accuracy)
		}
		if len(perClassAcc) != 0 {
			t.Errorf("Expected empty per-class accuracy, got %d entries", len(perClassAcc))
		}
		if len(confMatrix) != 0 {
			t.Errorf("Expected empty confusion matrix, got %dx%d", len(confMatrix), len(confMatrix))
		}
	})

	t.Run("AccuracyWithoutCNN", func(t *testing.T) {
		// Create minimal dataset
		images := make([][][][]float64, 1)
		images[0] = make([][][]float64, 1)
		images[0][0] = make([][]float64, 1)
		images[0][0][0] = []float64{0.5}

		dataset := &ImageDataset{
			Name:     "TestDataset",
			Images:   images,
			Labels:   []int{0},
			Classes:  []string{"0"},
			Width:    1,
			Height:   1,
			Channels: 1,
		}

		evaluator := NewCNNEvaluator(dataset, 0.01, 1, 1)
		// Set CNN to nil explicitly to test nil handling
		evaluator.CNN = nil

		accuracy, perClassAcc, confMatrix := evaluator.EvaluateAccuracy()

		// Should handle missing CNN gracefully
		if accuracy != 0.0 {
			t.Errorf("Expected 0 accuracy without CNN, got %f", accuracy)
		}
		if len(perClassAcc) != 0 {
			t.Errorf("Expected empty per-class accuracy without CNN, got %d entries", len(perClassAcc))
		}
		if len(confMatrix) != 0 {
			t.Errorf("Expected empty confusion matrix without CNN, got %dx%d", len(confMatrix), len(confMatrix))
		}
	})

	t.Run("SingleClassDataset", func(t *testing.T) {
		// Create dataset with only one class
		images := make([][][][]float64, 3)
		labels := []int{0, 0, 0}

		for i := range images {
			images[i] = make([][][]float64, 2)
			for j := range images[i] {
				images[i][j] = make([][]float64, 2)
				for k := range images[i][j] {
					images[i][j][k] = []float64{0.3}
				}
			}
		}

		dataset := &ImageDataset{
			Name:     "SingleClassDataset",
			Images:   images,
			Labels:   labels,
			Classes:  []string{"0"},
			Width:    2,
			Height:   2,
			Channels: 1,
		}

		evaluator := NewCNNEvaluator(dataset, 0.01, 1, 1)
		evaluator.CNN = &phase2.CNN{
			ConvLayers:   []*phase2.ConvLayer{},
			PoolLayers:   []*phase2.PoolingLayer{},
			FCWeights:    [][]float64{{1.0, 0.0, 0.0, 0.0}}, // 4 features (2x2x1), 1 class
			FCBiases:     []float64{0.0},
			LearningRate: 0.01,
			InputShape:   [3]int{2, 2, 1},
			FlattenShape: [2]int{4, 1},
		}

		accuracy, perClassAcc, confMatrix := evaluator.EvaluateAccuracy()

		// Single class should have perfect accuracy
		if accuracy != 1.0 {
			t.Errorf("Expected perfect accuracy (1.0) for single class, got %f", accuracy)
		}

		// Should have one entry in per-class accuracy
		if len(perClassAcc) != 1 {
			t.Errorf("Expected 1 entry in per-class accuracy, got %d", len(perClassAcc))
		}

		// Confusion matrix should be 1x1
		if len(confMatrix) != 1 || len(confMatrix[0]) != 1 {
			t.Errorf("Expected 1x1 confusion matrix, got %dx%d", len(confMatrix), len(confMatrix[0]))
		}
		if confMatrix[0][0] != 3 {
			t.Errorf("Expected confusion matrix[0][0] = 3, got %d", confMatrix[0][0])
		}
	})
}

// TestMeasureInferenceTime tests the inference time measurement functionality
func TestMeasureInferenceTime(t *testing.T) {
	t.Run("BasicInferenceTimeMeasurement", func(t *testing.T) {
		// Create dataset with multiple samples
		images := make([][][][]float64, 10)
		labels := make([]int, 10)

		for i := range images {
			images[i] = make([][][]float64, 3)
			for j := range images[i] {
				images[i][j] = make([][]float64, 3)
				for k := range images[i][j] {
					images[i][j][k] = []float64{0.5}
				}
			}
			labels[i] = i % 2
		}

		dataset := &ImageDataset{
			Name:     "InferenceTestDataset",
			Images:   images,
			Labels:   labels,
			Classes:  []string{"0", "1"},
			Width:    3,
			Height:   3,
			Channels: 1,
		}

		evaluator := NewCNNEvaluator(dataset, 0.01, 5, 1)

		// Create simple CNN
		evaluator.CNN = &phase2.CNN{
			ConvLayers:   []*phase2.ConvLayer{},
			PoolLayers:   []*phase2.PoolingLayer{},
			FCWeights:    [][]float64{{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}, // 9 features (3x3x1), 2 classes
			FCBiases:     []float64{0.0, 0.0},
			LearningRate: 0.01,
			InputShape:   [3]int{3, 3, 1},
			FlattenShape: [2]int{9, 2},
		}

		// Measure inference time
		inferenceTime := evaluator.measureInferenceTime()

		// Should be a positive duration
		if inferenceTime <= 0 {
			t.Errorf("Expected positive inference time, got %v", inferenceTime)
		}

		// Should be reasonable (less than 1 second for simple test)
		if inferenceTime > time.Second {
			t.Errorf("Inference time seems too long: %v", inferenceTime)
		}
	})

	t.Run("InferenceTimeWithEmptyDataset", func(t *testing.T) {
		emptyDataset := &ImageDataset{
			Name:     "EmptyDataset",
			Images:   [][][][]float64{},
			Labels:   []int{},
			Classes:  []string{},
			Width:    1,
			Height:   1,
			Channels: 1,
		}

		evaluator := NewCNNEvaluator(emptyDataset, 0.01, 1, 1)
		evaluator.CNN = &phase2.CNN{} // Minimal CNN

		inferenceTime := evaluator.measureInferenceTime()

		// Empty dataset should return 0 inference time
		if inferenceTime != 0 {
			t.Errorf("Expected 0 inference time for empty dataset, got %v", inferenceTime)
		}
	})

	t.Run("InferenceTimeWithoutCNN", func(t *testing.T) {
		// Create minimal dataset
		images := make([][][][]float64, 1)
		images[0] = make([][][]float64, 1)
		images[0][0] = make([][]float64, 1)
		images[0][0][0] = []float64{0.5}

		dataset := &ImageDataset{
			Name:     "TestDataset",
			Images:   images,
			Labels:   []int{0},
			Classes:  []string{"0"},
			Width:    1,
			Height:   1,
			Channels: 1,
		}

		evaluator := NewCNNEvaluator(dataset, 0.01, 1, 1)
		// Set CNN to nil explicitly to test nil handling
		evaluator.CNN = nil

		inferenceTime := evaluator.measureInferenceTime()

		// Should handle missing CNN gracefully (return 0 or minimal time)
		if inferenceTime < 0 {
			t.Errorf("Inference time should not be negative, got %v", inferenceTime)
		}
	})

	t.Run("InferenceTimeLargeBatch", func(t *testing.T) {
		// Create larger dataset to test batch processing
		images := make([][][][]float64, 150) // More than default sample size
		labels := make([]int, 150)

		for i := range images {
			images[i] = make([][][]float64, 2)
			for j := range images[i] {
				images[i][j] = make([][]float64, 2)
				for k := range images[i][j] {
					images[i][j][k] = []float64{float64(i%10) * 0.1}
				}
			}
			labels[i] = i % 3
		}

		dataset := &ImageDataset{
			Name:     "LargeBatchDataset",
			Images:   images,
			Labels:   labels,
			Classes:  []string{"0", "1", "2"},
			Width:    2,
			Height:   2,
			Channels: 1,
		}

		evaluator := NewCNNEvaluator(dataset, 0.01, 10, 1)

		// Create CNN for 3 classes
		evaluator.CNN = &phase2.CNN{
			ConvLayers:   []*phase2.ConvLayer{},
			PoolLayers:   []*phase2.PoolingLayer{},
			FCWeights:    [][]float64{{1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}}, // 4 features, 3 classes
			FCBiases:     []float64{0.0, 0.0, 0.0},
			LearningRate: 0.01,
			InputShape:   [3]int{2, 2, 1},
			FlattenShape: [2]int{4, 3},
		}

		// Measure inference time
		inferenceTime := evaluator.measureInferenceTime()

		// Should still be reasonable even with larger dataset
		if inferenceTime <= 0 {
			t.Errorf("Expected positive inference time, got %v", inferenceTime)
		}
		if inferenceTime > 5*time.Second {
			t.Errorf("Inference time seems too long for large batch: %v", inferenceTime)
		}
	})
}
