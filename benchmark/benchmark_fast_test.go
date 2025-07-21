//go:build fast
// +build fast

// Package benchmark fast tests for benchmark functionality
// Learning Goal: Understanding fast unit testing patterns for ML benchmarks
package benchmark

import (
	"testing"
	"time"
)

// TestBenchmarkRunnerFast tests benchmark runner without actual training
func TestBenchmarkRunnerFast(t *testing.T) {
	runner := NewBenchmarkRunner().
		SetIterations(1).
		SetWarmupRuns(0).
		SetVerbose(false)

	// Test runner configuration without slow operations
	if runner.iterations != 1 {
		t.Errorf("Expected 1 iteration, got %d", runner.iterations)
	}

	if runner.warmupRuns != 0 {
		t.Errorf("Expected 0 warmup runs, got %d", runner.warmupRuns)
	}

	if runner.verbose {
		t.Error("Expected verbose to be false")
	}
}

// TestPerformanceMetricsFast tests metrics calculation without training
func TestPerformanceMetricsFast(t *testing.T) {
	// Create mock metrics for testing
	metrics := PerformanceMetrics{
		ModelType:       "perceptron",
		DatasetName:     "xor",
		TrainingTime:    50 * time.Millisecond,
		InferenceTime:   1 * time.Microsecond,
		MemoryUsage:     1024,
		Accuracy:        0.75,
		ConvergenceRate: 100,
		FinalLoss:       0.25,
		Timestamp:       time.Now(),
	}

	// Test metrics validation
	if metrics.ModelType != "perceptron" {
		t.Errorf("Expected model type 'perceptron', got '%s'", metrics.ModelType)
	}

	if metrics.Accuracy < 0 || metrics.Accuracy > 1 {
		t.Errorf("Accuracy should be between 0 and 1, got %.2f", metrics.Accuracy)
	}

	if metrics.TrainingTime <= 0 {
		t.Error("Training time should be positive")
	}

	if metrics.ConvergenceRate <= 0 {
		t.Error("Convergence rate should be positive")
	}
}

// TestDatasetCreationFast tests dataset creation without validation
func TestDatasetCreationFast(t *testing.T) {
	// Test basic dataset properties
	xorDataset := CreateXORDataset()

	if xorDataset.Name != "xor" {
		t.Errorf("Expected name 'xor', got '%s'", xorDataset.Name)
	}

	if xorDataset.InputSize != 2 {
		t.Errorf("Expected input size 2, got %d", xorDataset.InputSize)
	}

	if xorDataset.OutputSize != 1 {
		t.Errorf("Expected output size 1, got %d", xorDataset.OutputSize)
	}

	if xorDataset.TrainSize != 4 {
		t.Errorf("Expected train size 4, got %d", xorDataset.TrainSize)
	}
}

// TestComparisonReportFast tests comparison report generation without benchmarking
func TestComparisonReportFast(t *testing.T) {
	baseline := PerformanceMetrics{
		ModelType:    "perceptron",
		DatasetName:  "xor",
		Accuracy:     0.5,
		TrainingTime: 100 * time.Millisecond,
	}

	comparison := PerformanceMetrics{
		ModelType:    "mlp",
		DatasetName:  "xor",
		Accuracy:     0.9,
		TrainingTime: 200 * time.Millisecond,
	}

	report := GenerateComparisonReport(baseline, comparison)

	// Quick validation without computation
	if report.BaselineModel != "perceptron" {
		t.Errorf("Expected baseline model 'perceptron', got '%s'", report.BaselineModel)
	}

	if report.ComparisonModel != "mlp" {
		t.Errorf("Expected comparison model 'mlp', got '%s'", report.ComparisonModel)
	}

	// Should have some improvements or degradations
	totalChanges := len(report.Improvements) + len(report.Degradations)
	if totalChanges == 0 {
		t.Error("Expected some performance changes in comparison")
	}
}

// BenchmarkFastOperations benchmarks fast operations only
func BenchmarkFastOperations(b *testing.B) {
	b.Run("DatasetCreation", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = CreateXORDataset()
		}
	})

	b.Run("MetricsCalculation", func(b *testing.B) {
		baseline := 100 * time.Millisecond
		comparison := 50 * time.Millisecond

		for i := 0; i < b.N; i++ {
			_ = CalculateSpeedup(baseline, comparison)
		}
	})

	b.Run("RunnerCreation", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = NewBenchmarkRunner()
		}
	})
}
