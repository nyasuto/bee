package benchmark

import (
	"testing"
	"time"
)

// TestDatasetCreation tests standard dataset creation
func TestDatasetCreation(t *testing.T) {
	tests := []struct {
		name        string
		createFunc  func() Dataset
		expectedIn  int
		expectedOut int
		minTrain    int
		minTest     int
	}{
		{"XOR", CreateXORDataset, 2, 1, 4, 4},
		{"AND", CreateANDDataset, 2, 1, 4, 4},
		{"OR", CreateORDataset, 2, 1, 4, 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dataset := tt.createFunc()

			// Validate dataset
			err := ValidateDataset(dataset)
			if err != nil {
				t.Fatalf("Dataset validation failed: %v", err)
			}

			// Check dimensions
			if dataset.InputSize != tt.expectedIn {
				t.Errorf("Expected input size %d, got %d", tt.expectedIn, dataset.InputSize)
			}

			if dataset.OutputSize != tt.expectedOut {
				t.Errorf("Expected output size %d, got %d", tt.expectedOut, dataset.OutputSize)
			}

			if dataset.TrainSize < tt.minTrain {
				t.Errorf("Expected at least %d training examples, got %d", tt.minTrain, dataset.TrainSize)
			}

			if dataset.TestSize < tt.minTest {
				t.Errorf("Expected at least %d test examples, got %d", tt.minTest, dataset.TestSize)
			}
		})
	}
}

// TestXORDatasetValues tests XOR dataset values specifically
func TestXORDatasetValues(t *testing.T) {
	dataset := CreateXORDataset()

	// Expected XOR truth table
	expected := map[string]float64{
		"0,0": 0,
		"0,1": 1,
		"1,0": 1,
		"1,1": 0,
	}

	// Check training data
	for i, input := range dataset.TrainInputs {
		key := ""
		for j, val := range input {
			if j > 0 {
				key += ","
			}
			if val >= 0.5 {
				key += "1"
			} else {
				key += "0"
			}
		}

		expectedOutput := expected[key]
		actualOutput := dataset.TrainTargets[i][0]

		if actualOutput != expectedOutput {
			t.Errorf("XOR(%s) = %f, expected %f", key, actualOutput, expectedOutput)
		}
	}
}

// TestPerformanceMetricsCalculation tests metric calculation functions
func TestPerformanceMetricsCalculation(t *testing.T) {
	// Test speedup calculation
	baseline := 100 * time.Millisecond
	comparison := 50 * time.Millisecond
	speedup := CalculateSpeedup(baseline, comparison)
	expectedSpeedup := 2.0

	if speedup != expectedSpeedup {
		t.Errorf("Expected speedup %.2f, got %.2f", expectedSpeedup, speedup)
	}

	// Test memory efficiency
	baselineMem := int64(1000)
	comparisonMem := int64(800)
	efficiency := CalculateMemoryEfficiency(baselineMem, comparisonMem)
	expectedEfficiency := 20.0 // 20% reduction

	if efficiency != expectedEfficiency {
		t.Errorf("Expected memory efficiency %.2f%%, got %.2f%%", expectedEfficiency, efficiency)
	}

	// Test accuracy improvement
	baselineAcc := 0.7
	comparisonAcc := 0.9
	improvement := CalculateAccuracyImprovement(baselineAcc, comparisonAcc)
	expectedImprovement := 20.0 // 20 percentage points

	if improvement != expectedImprovement {
		t.Errorf("Expected accuracy improvement %.2f%%, got %.2f%%", expectedImprovement, improvement)
	}
}

// TestBenchmarkRunner tests the benchmark runner functionality
func TestBenchmarkRunner(t *testing.T) {
	runner := NewBenchmarkRunner().
		SetIterations(10).
		SetWarmupRuns(2).
		SetVerbose(false)

	// Test with XOR dataset (known to work differently for perceptron vs MLP)
	dataset := CreateXORDataset()

	t.Run("PerceptronBenchmark", func(t *testing.T) {
		metrics, err := runner.BenchmarkPerceptron(dataset)
		if err != nil {
			t.Fatalf("Perceptron benchmark failed: %v", err)
		}

		// Validate metrics
		if metrics.ModelType != "perceptron" {
			t.Errorf("Expected model type 'perceptron', got '%s'", metrics.ModelType)
		}

		if metrics.DatasetName != "xor" {
			t.Errorf("Expected dataset name 'xor', got '%s'", metrics.DatasetName)
		}

		if metrics.TrainingTime <= 0 {
			t.Error("Training time should be positive")
		}

		if metrics.InferenceTime <= 0 {
			t.Error("Inference time should be positive")
		}

		if metrics.ConvergenceRate <= 0 {
			t.Error("Convergence rate should be positive")
		}

		// For XOR, perceptron should not achieve high accuracy
		if metrics.Accuracy > 0.8 {
			t.Errorf("Perceptron should not solve XOR with high accuracy, got %.2f", metrics.Accuracy)
		}
	})

	t.Run("MLPBenchmark", func(t *testing.T) {
		hiddenSizes := []int{4}
		metrics, err := runner.BenchmarkMLP(dataset, hiddenSizes)
		if err != nil {
			t.Fatalf("MLP benchmark failed: %v", err)
		}

		// Validate metrics
		if metrics.ModelType != "mlp" {
			t.Errorf("Expected model type 'mlp', got '%s'", metrics.ModelType)
		}

		if metrics.TrainingTime <= 0 {
			t.Error("Training time should be positive")
		}

		if metrics.InferenceTime <= 0 {
			t.Error("Inference time should be positive")
		}

		// MLP should eventually solve XOR (though not guaranteed in test)
		if metrics.Accuracy < 0.25 {
			t.Errorf("MLP should perform better than random on XOR, got %.2f accuracy", metrics.Accuracy)
		}
	})
}

// TestComparisonReport tests comparison report generation
func TestComparisonReport(t *testing.T) {
	// Create mock metrics
	baseline := PerformanceMetrics{
		ModelType:       "perceptron",
		DatasetName:     "xor",
		TrainingTime:    100 * time.Millisecond,
		InferenceTime:   1 * time.Microsecond,
		MemoryUsage:     1000,
		Accuracy:        0.25, // Poor performance as expected for XOR
		ConvergenceRate: 1000,
		FinalLoss:       0.5,
		Timestamp:       time.Now(),
	}

	comparison := PerformanceMetrics{
		ModelType:       "mlp",
		DatasetName:     "xor",
		TrainingTime:    200 * time.Millisecond, // Slower training
		InferenceTime:   2 * time.Microsecond,   // Slower inference
		MemoryUsage:     2000,                   // More memory
		Accuracy:        0.95,                   // Much better accuracy
		ConvergenceRate: 500,                    // Faster convergence
		FinalLoss:       0.05,                   // Lower loss
		Timestamp:       time.Now(),
	}

	report := GenerateComparisonReport(baseline, comparison)

	// Validate report structure
	if report.BaselineModel != "perceptron" {
		t.Errorf("Expected baseline model 'perceptron', got '%s'", report.BaselineModel)
	}

	if report.ComparisonModel != "mlp" {
		t.Errorf("Expected comparison model 'mlp', got '%s'", report.ComparisonModel)
	}

	if report.Dataset != "xor" {
		t.Errorf("Expected dataset 'xor', got '%s'", report.Dataset)
	}

	// Check that accuracy improvement is recorded
	if _, hasAccuracyImprovement := report.Improvements["accuracy"]; !hasAccuracyImprovement {
		t.Error("Expected accuracy improvement to be recorded")
	}

	// Check that convergence improvement is recorded (fewer epochs is better)
	if _, hasConvergenceImprovement := report.Improvements["convergence_rate"]; !hasConvergenceImprovement {
		t.Error("Expected convergence rate improvement to be recorded")
	}

	// Check that training time degradation is recorded (longer time is worse)
	if _, hasTrainingDegradation := report.Degradations["training_time"]; !hasTrainingDegradation {
		t.Error("Expected training time degradation to be recorded")
	}
}

// TestMetricComparisons tests detailed metric comparison functionality
func TestMetricComparisons(t *testing.T) {
	baseline := PerformanceMetrics{
		TrainingTime:    100 * time.Millisecond,
		InferenceTime:   1 * time.Microsecond,
		MemoryUsage:     1000,
		Accuracy:        0.5,
		ConvergenceRate: 100,
	}

	comparison := PerformanceMetrics{
		TrainingTime:    50 * time.Millisecond, // 50% faster
		InferenceTime:   1 * time.Microsecond,  // Same speed
		MemoryUsage:     1200,                  // 20% more memory
		Accuracy:        0.7,                   // 20 percentage points better
		ConvergenceRate: 80,                    // 20% fewer epochs
	}

	comparisons := CompareMetrics(baseline, comparison)

	// Test training time comparison
	if trainingComp, exists := comparisons["training_time"]; exists {
		if trainingComp.ImprovementRatio != 2.0 {
			t.Errorf("Expected training time speedup of 2.0, got %.2f", trainingComp.ImprovementRatio)
		}
	} else {
		t.Error("Training time comparison not found")
	}

	// Test accuracy comparison
	if accuracyComp, exists := comparisons["accuracy"]; exists {
		expectedImprovement := 20.0 // 20 percentage points
		if accuracyComp.PercentChange != expectedImprovement {
			t.Errorf("Expected accuracy improvement of %.2f%%, got %.2f%%",
				expectedImprovement, accuracyComp.PercentChange)
		}
	} else {
		t.Error("Accuracy comparison not found")
	}
}

// BenchmarkBenchmarkRunner benchmarks the benchmark runner itself
func BenchmarkBenchmarkRunner(b *testing.B) {
	runner := NewBenchmarkRunner().
		SetIterations(10).
		SetWarmupRuns(2).
		SetVerbose(false)

	dataset := CreateXORDataset()

	b.Run("PerceptronBenchmark", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := runner.BenchmarkPerceptron(dataset)
			if err != nil {
				b.Fatalf("Benchmark failed: %v", err)
			}
		}
	})

	b.Run("MLPBenchmark", func(b *testing.B) {
		hiddenSizes := []int{4}
		for i := 0; i < b.N; i++ {
			_, err := runner.BenchmarkMLP(dataset, hiddenSizes)
			if err != nil {
				b.Fatalf("Benchmark failed: %v", err)
			}
		}
	})
}
