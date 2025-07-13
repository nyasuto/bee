// Package datasets implements comprehensive tests for sequence evaluation functionality
// Learning Goal: Understanding sequence learning evaluation testing patterns and validation
package datasets

import (
	"math"
	"testing"
	"time"

	"github.com/nyasuto/bee/phase2"
)

// TestSequenceEvaluator tests the sequence evaluator creation and basic functionality
func TestSequenceEvaluator(t *testing.T) {
	// Create a small test dataset
	sequences := make([][][]float64, 4)
	targets := make([][]float64, 4)

	// Create simple 2-timestep sequences
	for i := range sequences {
		sequences[i] = [][]float64{{float64(i)}, {float64(i + 1)}}
		targets[i] = []float64{float64(i + 2)}
	}

	dataset := &TimeSeriesDataset{
		Name:        "TestDataset",
		Type:        "regression",
		Sequences:   sequences,
		Targets:     targets,
		Features:    1,
		OutputSize:  1,
		SeqLength:   2,
		Description: "Test dataset for sequence evaluation",
	}

	t.Run("CreateRNNEvaluator", func(t *testing.T) {
		evaluator := NewSequenceEvaluator(dataset, "RNN", 0.01, 2, 5)

		if evaluator.Dataset != dataset {
			t.Error("Dataset not set correctly")
		}
		if evaluator.ModelType != "RNN" {
			t.Errorf("Expected model type 'RNN', got %s", evaluator.ModelType)
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

	t.Run("CreateLSTMEvaluator", func(t *testing.T) {
		evaluator := NewSequenceEvaluator(dataset, "LSTM", 0.02, 4, 10)

		if evaluator.ModelType != "LSTM" {
			t.Errorf("Expected model type 'LSTM', got %s", evaluator.ModelType)
		}
		if evaluator.LearningRate != 0.02 {
			t.Errorf("Expected learning rate 0.02, got %f", evaluator.LearningRate)
		}
	})

	t.Run("SetVerbose", func(t *testing.T) {
		evaluator := NewSequenceEvaluator(dataset, "RNN", 0.01, 2, 5)
		evaluator.SetVerbose(true)

		if !evaluator.Verbose {
			t.Error("Verbose not set correctly")
		}
	})
}

// TestRNNCreation tests RNN model creation
func TestRNNCreation(t *testing.T) {
	t.Run("ValidRNNCreation", func(t *testing.T) {
		// Create compatible dataset
		sequences := make([][][]float64, 2)
		targets := make([][]float64, 2)

		for i := range sequences {
			sequences[i] = [][]float64{{float64(i)}, {float64(i + 1)}}
			targets[i] = []float64{float64(i + 2)}
		}

		dataset := &TimeSeriesDataset{
			Name:        "TestDataset",
			Type:        "regression",
			Sequences:   sequences,
			Targets:     targets,
			Features:    1,
			OutputSize:  1,
			SeqLength:   2,
			Description: "Test dataset for RNN creation",
		}

		evaluator := NewSequenceEvaluator(dataset, "RNN", 0.01, 1, 1)
		err := evaluator.CreateRNN()
		if err != nil {
			t.Fatalf("Failed to create RNN: %v", err)
		}

		// Verify RNN configuration
		if evaluator.RNN == nil {
			t.Fatal("RNN not created")
		}

		if evaluator.RNN.Cell.InputSize != 1 {
			t.Errorf("Expected input size 1, got %d", evaluator.RNN.Cell.InputSize)
		}
		if evaluator.RNN.OutputSize != 1 {
			t.Errorf("Expected output size 1, got %d", evaluator.RNN.OutputSize)
		}
		if evaluator.RNN.Cell.HiddenSize < 16 {
			t.Errorf("Expected hidden size >= 16, got %d", evaluator.RNN.Cell.HiddenSize)
		}
		if evaluator.ModelType != "RNN" {
			t.Errorf("Expected model type 'RNN', got %s", evaluator.ModelType)
		}
	})

	t.Run("InvalidDatasetDimensions", func(t *testing.T) {
		// Create dataset with invalid dimensions
		dataset := &TimeSeriesDataset{
			Name:       "InvalidDataset",
			Features:   0, // Invalid
			OutputSize: 1,
			SeqLength:  2,
		}

		evaluator := NewSequenceEvaluator(dataset, "RNN", 0.01, 1, 1)
		err := evaluator.CreateRNN()
		if err == nil {
			t.Error("Expected error for invalid dataset dimensions")
		}
	})

	t.Run("LargeSequenceAdaptiveHiddenSize", func(t *testing.T) {
		// Create dataset with longer sequences
		dataset := &TimeSeriesDataset{
			Name:        "LongSequenceDataset",
			Features:    2,
			OutputSize:  1,
			SeqLength:   100, // Long sequence
			Description: "Dataset with long sequences",
		}

		evaluator := NewSequenceEvaluator(dataset, "RNN", 0.01, 1, 1)
		err := evaluator.CreateRNN()
		if err != nil {
			t.Fatalf("Failed to create RNN for long sequences: %v", err)
		}

		// Should have larger hidden size for longer sequences
		expectedMinHiddenSize := dataset.Features * 8 // 2 * 8 = 16, but could be larger
		if evaluator.RNN.Cell.HiddenSize < expectedMinHiddenSize {
			t.Errorf("Expected hidden size >= %d for long sequences, got %d", expectedMinHiddenSize, evaluator.RNN.Cell.HiddenSize)
		}
	})
}

// TestLSTMCreation tests LSTM model creation
func TestLSTMCreation(t *testing.T) {
	t.Run("ValidLSTMCreation", func(t *testing.T) {
		// Create compatible dataset
		sequences := make([][][]float64, 2)
		targets := make([][]float64, 2)

		for i := range sequences {
			sequences[i] = [][]float64{{float64(i)}, {float64(i + 1)}}
			targets[i] = []float64{float64(i + 2)}
		}

		dataset := &TimeSeriesDataset{
			Name:        "TestDataset",
			Type:        "regression",
			Sequences:   sequences,
			Targets:     targets,
			Features:    1,
			OutputSize:  1,
			SeqLength:   2,
			Description: "Test dataset for LSTM creation",
		}

		evaluator := NewSequenceEvaluator(dataset, "LSTM", 0.01, 1, 1)
		err := evaluator.CreateLSTM()
		if err != nil {
			t.Fatalf("Failed to create LSTM: %v", err)
		}

		// Verify LSTM configuration
		if evaluator.LSTM == nil {
			t.Fatal("LSTM not created")
		}

		if evaluator.LSTM.Cell.InputSize != 1 {
			t.Errorf("Expected input size 1, got %d", evaluator.LSTM.Cell.InputSize)
		}
		if evaluator.LSTM.OutputSize != 1 {
			t.Errorf("Expected output size 1, got %d", evaluator.LSTM.OutputSize)
		}
		if evaluator.LSTM.Cell.HiddenSize < 16 {
			t.Errorf("Expected hidden size >= 16, got %d", evaluator.LSTM.Cell.HiddenSize)
		}
		if evaluator.ModelType != "LSTM" {
			t.Errorf("Expected model type 'LSTM', got %s", evaluator.ModelType)
		}
	})

	t.Run("MultiFeatureLSTM", func(t *testing.T) {
		dataset := &TimeSeriesDataset{
			Name:        "MultiFeatureDataset",
			Features:    3, // Multiple features
			OutputSize:  2, // Multiple outputs
			SeqLength:   5,
			Description: "Multi-feature dataset",
		}

		evaluator := NewSequenceEvaluator(dataset, "LSTM", 0.01, 1, 1)
		err := evaluator.CreateLSTM()
		if err != nil {
			t.Fatalf("Failed to create multi-feature LSTM: %v", err)
		}

		if evaluator.LSTM.Cell.InputSize != 3 {
			t.Errorf("Expected input size 3, got %d", evaluator.LSTM.Cell.InputSize)
		}
		if evaluator.LSTM.OutputSize != 2 {
			t.Errorf("Expected output size 2, got %d", evaluator.LSTM.OutputSize)
		}
	})
}

// TestMSELossCalculation tests MSE loss calculation
func TestMSELossCalculation(t *testing.T) {
	evaluator := &SequenceEvaluator{}

	t.Run("PerfectPrediction", func(t *testing.T) {
		prediction := []float64{1.0, 2.0, 3.0}
		target := []float64{1.0, 2.0, 3.0}
		loss := evaluator.calculateMSELoss(prediction, target)

		if loss != 0.0 {
			t.Errorf("Expected MSE loss 0.0 for perfect prediction, got %f", loss)
		}
	})

	t.Run("SimpleLossCalculation", func(t *testing.T) {
		prediction := []float64{1.0, 2.0}
		target := []float64{2.0, 4.0}
		loss := evaluator.calculateMSELoss(prediction, target)

		// Expected: ((1-2)² + (2-4)²) / 2 = (1 + 4) / 2 = 2.5
		expected := 2.5
		if math.Abs(loss-expected) > 1e-9 {
			t.Errorf("Expected MSE loss %f, got %f", expected, loss)
		}
	})

	t.Run("DimensionMismatch", func(t *testing.T) {
		prediction := []float64{1.0, 2.0}
		target := []float64{1.0} // Different length
		loss := evaluator.calculateMSELoss(prediction, target)

		if !math.IsInf(loss, 1) {
			t.Errorf("Expected infinite loss for dimension mismatch, got %f", loss)
		}
	})

	t.Run("EmptySlices", func(t *testing.T) {
		prediction := []float64{}
		target := []float64{}
		loss := evaluator.calculateMSELoss(prediction, target)

		// Empty slices should result in NaN (0/0)
		if !math.IsNaN(loss) {
			t.Errorf("Expected NaN for empty slices, got %f", loss)
		}
	})
}

// TestSequenceTrainBatch tests the batch training functionality
func TestSequenceTrainBatch(t *testing.T) {
	// Create simple test dataset
	sequences := [][][]float64{
		{{1.0}, {2.0}},
		{{3.0}, {4.0}},
	}
	targets := [][]float64{
		{3.0},
		{5.0},
	}

	dataset := &TimeSeriesDataset{
		Name:        "TestDataset",
		Type:        "regression",
		Sequences:   sequences,
		Targets:     targets,
		Features:    1,
		OutputSize:  1,
		SeqLength:   2,
		Description: "Test dataset for batch training",
	}

	t.Run("ValidRNNBatch", func(t *testing.T) {
		evaluator := NewSequenceEvaluator(dataset, "RNN", 0.01, 2, 1)

		// Create simple RNN for testing
		evaluator.RNN = phase2.NewRNN(1, 4, 1, 0.01)

		batchSequences, batchTargets, err := dataset.GetBatch([]int{0, 1})
		if err != nil {
			t.Fatalf("Failed to get batch: %v", err)
		}

		loss := evaluator.trainBatch(batchSequences, batchTargets)

		// Loss should be a reasonable positive value
		if loss < 0 {
			t.Errorf("Expected non-negative loss, got %f", loss)
		}
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			t.Errorf("Loss should be finite, got %f", loss)
		}
	})

	t.Run("ValidLSTMBatch", func(t *testing.T) {
		evaluator := NewSequenceEvaluator(dataset, "LSTM", 0.01, 2, 1)

		// Create simple LSTM for testing
		evaluator.LSTM = phase2.NewLSTM(1, 4, 1, 0.01)

		batchSequences, batchTargets, err := dataset.GetBatch([]int{0, 1})
		if err != nil {
			t.Fatalf("Failed to get batch: %v", err)
		}

		loss := evaluator.trainBatch(batchSequences, batchTargets)

		// Loss should be a reasonable positive value
		if loss < 0 {
			t.Errorf("Expected non-negative loss, got %f", loss)
		}
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			t.Errorf("Loss should be finite, got %f", loss)
		}
	})

	t.Run("EmptyBatch", func(t *testing.T) {
		evaluator := NewSequenceEvaluator(dataset, "RNN", 0.01, 2, 1)

		loss := evaluator.trainBatch([][][]float64{}, [][]float64{})

		// Empty batch should return 0 loss
		if loss != 0.0 {
			t.Errorf("Expected 0 loss for empty batch, got %f", loss)
		}
	})

	t.Run("NoValidModel", func(t *testing.T) {
		evaluator := NewSequenceEvaluator(dataset, "InvalidModel", 0.01, 2, 1)
		// Don't create any model

		batchSequences, batchTargets, err := dataset.GetBatch([]int{0, 1})
		if err != nil {
			t.Fatalf("Failed to get batch: %v", err)
		}

		loss := evaluator.trainBatch(batchSequences, batchTargets)

		// Should return 0 loss when no valid model
		if loss != 0.0 {
			t.Errorf("Expected 0 loss for no valid model, got %f", loss)
		}
	})
}

// TestPredictionMetrics tests prediction accuracy metrics calculation
func TestPredictionMetrics(t *testing.T) {
	// Create simple test scenario with known values
	sequences := [][][]float64{{{1.0}}, {{2.0}}}
	targets := [][]float64{{2.0}, {3.0}}

	dataset := &TimeSeriesDataset{
		Name:       "TestDataset",
		Sequences:  sequences,
		Targets:    targets,
		Features:   1,
		OutputSize: 1,
		SeqLength:  1,
	}

	evaluator := NewSequenceEvaluator(dataset, "RNN", 0.01, 1, 1)

	// Create mock RNN that returns predictable values
	evaluator.RNN = phase2.NewRNN(1, 4, 1, 0.01)

	t.Run("CalculateMetrics", func(t *testing.T) {
		mse, mae, mape, rmse := evaluator.CalculatePredictionMetrics()

		// Metrics should be non-negative
		if mse < 0 {
			t.Errorf("MSE should be non-negative, got %f", mse)
		}
		if mae < 0 {
			t.Errorf("MAE should be non-negative, got %f", mae)
		}
		if mape < 0 {
			t.Errorf("MAPE should be non-negative, got %f", mape)
		}
		if rmse < 0 {
			t.Errorf("RMSE should be non-negative, got %f", rmse)
		}

		// RMSE should be sqrt of MSE
		if math.Abs(rmse-math.Sqrt(mse)) > 1e-9 {
			t.Errorf("RMSE should be sqrt(MSE): %f != sqrt(%f)", rmse, mse)
		}

		// Should not be NaN or Inf
		if math.IsNaN(mse) || math.IsInf(mse, 0) {
			t.Errorf("MSE should be finite, got %f", mse)
		}
	})

	t.Run("EmptyDatasetMetrics", func(t *testing.T) {
		emptyDataset := &TimeSeriesDataset{
			Name:      "Empty",
			Sequences: [][][]float64{},
			Targets:   [][]float64{},
		}

		emptyEvaluator := NewSequenceEvaluator(emptyDataset, "RNN", 0.01, 1, 1)
		mse, mae, mape, rmse := emptyEvaluator.CalculatePredictionMetrics()

		// All metrics should be 0 for empty dataset
		if mse != 0.0 || mae != 0.0 || mape != 0.0 || rmse != 0.0 {
			t.Errorf("Expected all metrics to be 0 for empty dataset, got MSE=%f, MAE=%f, MAPE=%f, RMSE=%f", mse, mae, mape, rmse)
		}
	})
}

// TestLongTermAccuracy tests long-term dependency evaluation
func TestLongTermAccuracy(t *testing.T) {
	// Create dataset with multi-step outputs
	sequences := [][][]float64{{{1.0}}, {{2.0}}}
	targets := [][]float64{{2.0, 3.0, 4.0}, {3.0, 4.0, 5.0}} // 3-step predictions

	dataset := &TimeSeriesDataset{
		Name:       "TestDataset",
		Sequences:  sequences,
		Targets:    targets,
		Features:   1,
		OutputSize: 3, // 3-step prediction
		SeqLength:  1,
	}

	evaluator := NewSequenceEvaluator(dataset, "RNN", 0.01, 1, 1)
	evaluator.RNN = phase2.NewRNN(1, 4, 3, 0.01)

	t.Run("EvaluateLongTermAccuracy", func(t *testing.T) {
		longTermAcc := evaluator.EvaluateLongTermAccuracy()

		// Should have accuracies for horizons 1, 5, 10 (where applicable)
		if len(longTermAcc) == 0 {
			t.Error("Expected non-empty long-term accuracy map")
		}

		// All accuracies should be between 0 and 1
		for horizon, acc := range longTermAcc {
			if acc < 0 || acc > 1 {
				t.Errorf("Long-term accuracy for horizon %d should be between 0 and 1, got %f", horizon, acc)
			}
		}

		// Should have accuracy for horizon 1 (within output size)
		if _, exists := longTermAcc[1]; !exists {
			t.Error("Expected accuracy for 1-step horizon")
		}
	})

	t.Run("SmallOutputSize", func(t *testing.T) {
		// Dataset with small output size
		smallDataset := &TimeSeriesDataset{
			Name:       "SmallDataset",
			Sequences:  [][][]float64{{{1.0}}},
			Targets:    [][]float64{{2.0}}, // 1-step prediction only
			OutputSize: 1,
		}

		smallEvaluator := NewSequenceEvaluator(smallDataset, "RNN", 0.01, 1, 1)
		smallEvaluator.RNN = phase2.NewRNN(1, 4, 1, 0.01)

		longTermAcc := smallEvaluator.EvaluateLongTermAccuracy()

		// Should only have accuracy for horizon 1
		if len(longTermAcc) > 1 {
			t.Errorf("Expected at most 1 horizon for small output size, got %d", len(longTermAcc))
		}
	})
}

// TestPredictionExamples tests prediction example generation
func TestPredictionExamples(t *testing.T) {
	sequences := [][][]float64{{{1.0}}, {{2.0}}, {{3.0}}}
	targets := [][]float64{{2.0}, {3.0}, {4.0}}

	dataset := &TimeSeriesDataset{
		Name:       "TestDataset",
		Sequences:  sequences,
		Targets:    targets,
		Features:   1,
		OutputSize: 1,
		SeqLength:  1,
	}

	evaluator := NewSequenceEvaluator(dataset, "RNN", 0.01, 1, 1)
	evaluator.RNN = phase2.NewRNN(1, 4, 1, 0.01)

	t.Run("GenerateExamples", func(t *testing.T) {
		examples := evaluator.generatePredictionExamples(2)

		if len(examples) > 2 {
			t.Errorf("Expected at most 2 examples, got %d", len(examples))
		}

		for i, example := range examples {
			if len(example.Predicted) == 0 {
				t.Errorf("Example %d should have predictions", i)
			}
			if len(example.Actual) == 0 {
				t.Errorf("Example %d should have actual values", i)
			}
			if example.Error < 0 {
				t.Errorf("Example %d error should be non-negative, got %f", i, example.Error)
			}
		}
	})

	t.Run("MoreExamplesThanData", func(t *testing.T) {
		examples := evaluator.generatePredictionExamples(10) // More than 3 samples

		if len(examples) > 3 {
			t.Errorf("Expected at most 3 examples (dataset size), got %d", len(examples))
		}
	})
}

// TestSequenceMemoryEstimation tests memory usage estimation
func TestSequenceMemoryEstimation(t *testing.T) {
	dataset := &TimeSeriesDataset{
		Features:   2,
		OutputSize: 1,
	}

	t.Run("RNNMemoryEstimation", func(t *testing.T) {
		evaluator := NewSequenceEvaluator(dataset, "RNN", 0.01, 1, 1)
		evaluator.RNN = phase2.NewRNN(2, 10, 1, 0.01)

		memory := evaluator.estimateMemoryUsage()

		// Should be a positive value
		if memory <= 0 {
			t.Errorf("Expected positive memory usage, got %d", memory)
		}

		// Should be reasonable for a small RNN
		// Rough calculation: weights + biases + hidden state
		// Input weights: 2*10, hidden weights: 10*10, output weights: 10*1, biases, hidden state
		expectedMin := (2*10 + 10*10 + 10*1 + 10 + 1 + 10) * 8 // 8 bytes per float64
		if memory < int64(expectedMin/2) {                     // Allow some flexibility
			t.Errorf("Expected at least %d bytes, got %d", expectedMin/2, memory)
		}
	})

	t.Run("LSTMMemoryEstimation", func(t *testing.T) {
		evaluator := NewSequenceEvaluator(dataset, "LSTM", 0.01, 1, 1)
		evaluator.LSTM = phase2.NewLSTM(2, 10, 1, 0.01)

		memory := evaluator.estimateMemoryUsage()

		// Should be a positive value
		if memory <= 0 {
			t.Errorf("Expected positive memory usage, got %d", memory)
		}

		// LSTM should use more memory than RNN due to gates
		// 4 gates * (input + hidden + bias) + output weights + hidden/cell state
		expectedMin := (4*(2+10+1)*10 + (10+1)*1 + 10*2) * 8
		if memory < int64(expectedMin/2) { // Allow some flexibility
			t.Errorf("Expected at least %d bytes for LSTM, got %d", expectedMin/2, memory)
		}
	})

	t.Run("NoModelMemoryEstimation", func(t *testing.T) {
		evaluator := NewSequenceEvaluator(dataset, "None", 0.01, 1, 1)
		// Don't create any model

		memory := evaluator.estimateMemoryUsage()

		// Should return 0 for no model
		if memory != 0 {
			t.Errorf("Expected 0 memory for no model, got %d", memory)
		}
	})
}

// TestInferenceTime tests inference time measurement
func TestInferenceTime(t *testing.T) {
	sequences := [][][]float64{{{1.0}}, {{2.0}}, {{3.0}}}
	targets := [][]float64{{2.0}, {3.0}, {4.0}}

	dataset := &TimeSeriesDataset{
		Name:       "TestDataset",
		Sequences:  sequences,
		Targets:    targets,
		Features:   1,
		OutputSize: 1,
		SeqLength:  1,
	}

	t.Run("RNNInferenceTime", func(t *testing.T) {
		evaluator := NewSequenceEvaluator(dataset, "RNN", 0.01, 1, 1)
		evaluator.RNN = phase2.NewRNN(1, 4, 1, 0.01)

		inferenceTime := evaluator.measureInferenceTime()

		// Should be a positive duration
		if inferenceTime <= 0 {
			t.Errorf("Expected positive inference time, got %v", inferenceTime)
		}

		// Should be reasonable (less than 1 second for small dataset)
		if inferenceTime > time.Second {
			t.Errorf("Inference time seems too large: %v", inferenceTime)
		}
	})

	t.Run("EmptyDatasetInferenceTime", func(t *testing.T) {
		emptyDataset := &TimeSeriesDataset{
			Name:      "Empty",
			Sequences: [][][]float64{},
			Targets:   [][]float64{},
		}

		evaluator := NewSequenceEvaluator(emptyDataset, "RNN", 0.01, 1, 1)
		inferenceTime := evaluator.measureInferenceTime()

		// Should return 0 for empty dataset
		if inferenceTime != 0 {
			t.Errorf("Expected 0 inference time for empty dataset, got %v", inferenceTime)
		}
	})
}

// TestSequenceResults tests the sequence results structure
func TestSequenceResults(t *testing.T) {
	t.Run("CreateResults", func(t *testing.T) {
		results := &SequenceResults{
			ModelName:        "TestRNN",
			DatasetName:      "TestDataset",
			TrainingTime:     time.Second,
			InferenceTime:    time.Millisecond,
			MSE:              0.01,
			MAE:              0.1,
			MAPE:             5.0,
			RMSE:             0.1,
			EpochsCompleted:  10,
			ConvergenceEpoch: 8,
			MemoryUsage:      1024,
			LongTermAccuracy: map[int]float64{1: 0.9, 5: 0.8},
			PredictionExamples: []PredictionPair{
				{Predicted: []float64{1.0}, Actual: []float64{1.1}, Error: 0.01},
			},
			Timestamp:   time.Now(),
			LossHistory: []float64{1.0, 0.5, 0.25, 0.1},
		}

		if results.ModelName != "TestRNN" {
			t.Errorf("Expected model name 'TestRNN', got %s", results.ModelName)
		}
		if results.MSE != 0.01 {
			t.Errorf("Expected MSE 0.01, got %f", results.MSE)
		}
		if len(results.LongTermAccuracy) != 2 {
			t.Errorf("Expected 2 long-term accuracy entries, got %d", len(results.LongTermAccuracy))
		}
		if len(results.PredictionExamples) != 1 {
			t.Errorf("Expected 1 prediction example, got %d", len(results.PredictionExamples))
		}
		if len(results.LossHistory) != 4 {
			t.Errorf("Expected 4 loss history entries, got %d", len(results.LossHistory))
		}
	})
}

// TestPrintSequenceResults tests the results printing function
func TestPrintSequenceResults(t *testing.T) {
	evaluator := &SequenceEvaluator{}

	results := &SequenceResults{
		ModelName:        "TestRNN",
		DatasetName:      "TestDataset",
		TrainingTime:     time.Second,
		InferenceTime:    time.Millisecond,
		MSE:              0.01,
		MAE:              0.1,
		MAPE:             5.0,
		RMSE:             0.1,
		EpochsCompleted:  10,
		ConvergenceEpoch: 8,
		MemoryUsage:      1024,
		LongTermAccuracy: map[int]float64{1: 0.9, 5: 0.8},
		PredictionExamples: []PredictionPair{
			{Predicted: []float64{1.0}, Actual: []float64{1.1}, Error: 0.01},
		},
		Timestamp: time.Now(),
	}

	t.Run("PrintResults", func(t *testing.T) {
		// This test just ensures the function doesn't panic
		// In a real scenario, you might capture stdout to verify formatting
		evaluator.PrintSequenceResults(results)
	})
}

// TestSequenceEdgeCases tests various edge cases and error conditions
func TestSequenceEdgeCases(t *testing.T) {
	t.Run("TrainWithoutModel", func(t *testing.T) {
		dataset := &TimeSeriesDataset{
			Name:       "TestDataset",
			Sequences:  [][][]float64{{{1.0}}},
			Targets:    [][]float64{{2.0}},
			Features:   1,
			OutputSize: 1,
			SeqLength:  1,
		}

		evaluator := NewSequenceEvaluator(dataset, "RNN", 0.01, 1, 1)
		// Don't create any model

		_, err := evaluator.TrainSequenceModel()
		if err == nil {
			t.Error("Expected error when training without model initialization")
		}
	})

	t.Run("TrainEmptyDataset", func(t *testing.T) {
		emptyDataset := &TimeSeriesDataset{
			Name:       "EmptyDataset",
			Sequences:  [][][]float64{},
			Targets:    [][]float64{},
			Features:   1,
			OutputSize: 1,
			SeqLength:  1,
		}

		evaluator := NewSequenceEvaluator(emptyDataset, "RNN", 0.01, 1, 1)
		err := evaluator.CreateRNN()
		if err != nil {
			t.Fatalf("Failed to create RNN: %v", err)
		}

		results, err := evaluator.TrainSequenceModel()
		if err != nil {
			t.Fatalf("Training failed: %v", err)
		}

		// Should handle empty dataset gracefully
		if results.MSE != 0.0 || results.MAE != 0.0 {
			t.Errorf("Expected 0 metrics for empty dataset, got MSE=%f, MAE=%f", results.MSE, results.MAE)
		}
	})
}
