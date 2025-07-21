//go:build !slow
// +build !slow

// Package benchmark CI-friendly tests with reduced training epochs
// Learning Goal: Understanding CI/CD optimized testing patterns for ML benchmarks
package benchmark

import (
	"testing"
	"time"
)

// TestFastBenchmarkRunner tests the benchmark runner with fast implementations
func TestFastBenchmarkRunner(t *testing.T) {
	runner := NewBenchmarkRunner().
		SetIterations(1).
		SetWarmupRuns(0).
		SetVerbose(false)

	// Test with XOR dataset
	xorDataset := CreateXORDataset()

	// Run perceptron benchmark (fast version)
	metrics, err := runner.FastBenchmarkPerceptron(xorDataset)
	if err != nil {
		t.Fatalf("Fast Perceptron benchmark failed: %v", err)
	}

	// Verify metrics
	if metrics.ModelType != "perceptron" {
		t.Errorf("Expected model type 'perceptron', got '%s'", metrics.ModelType)
	}

	if metrics.DatasetName != "xor" {
		t.Errorf("Expected dataset name 'xor', got '%s'", metrics.DatasetName)
	}

	if metrics.Accuracy < 0 || metrics.Accuracy > 1 {
		t.Errorf("Accuracy should be between 0 and 1, got %.2f", metrics.Accuracy)
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

	// Verify reduced epoch count (should be ≤ 50)
	if metrics.ConvergenceRate > 50 {
		t.Errorf("Fast benchmark should use ≤ 50 epochs, got %d", metrics.ConvergenceRate)
	}
}

// TestFastMLPBenchmark tests MLP with fast implementation
func TestFastMLPBenchmark(t *testing.T) {
	runner := NewBenchmarkRunner().
		SetIterations(1).
		SetWarmupRuns(0).
		SetVerbose(false)

	xorDataset := CreateXORDataset()
	hiddenSizes := []int{4}

	// Run MLP benchmark (fast version)
	metrics, err := runner.FastBenchmarkMLP(xorDataset, hiddenSizes)
	if err != nil {
		t.Fatalf("Fast MLP benchmark failed: %v", err)
	}

	// Verify metrics
	if metrics.ModelType != "mlp" {
		t.Errorf("Expected model type 'mlp', got '%s'", metrics.ModelType)
	}

	if metrics.DatasetName != "xor" {
		t.Errorf("Expected dataset name 'xor', got '%s'", metrics.DatasetName)
	}

	if metrics.Accuracy < 0 || metrics.Accuracy > 1 {
		t.Errorf("Accuracy should be between 0 and 1, got %.2f", metrics.Accuracy)
	}

	if metrics.TrainingTime <= 0 {
		t.Error("Training time should be positive")
	}

	// Verify reduced epoch count (should be ≤ 100)
	if metrics.ConvergenceRate > 100 {
		t.Errorf("Fast MLP benchmark should use ≤ 100 epochs, got %d", metrics.ConvergenceRate)
	}
}

// TestFastRunComparison tests comparative benchmarking with fast implementation
func TestFastRunComparison(t *testing.T) {
	runner := NewBenchmarkRunner().
		SetIterations(1).
		SetWarmupRuns(0).
		SetVerbose(false)

	// Test with XOR dataset
	xorDataset := CreateXORDataset()
	hiddenSizes := []int{4}

	// Run fast comparison
	report, err := runner.FastRunComparison(xorDataset, hiddenSizes)
	if err != nil {
		t.Fatalf("Fast comparison benchmark failed: %v", err)
	}

	// Verify report structure
	if report.BaselineModel != "perceptron" {
		t.Errorf("Expected baseline model 'perceptron', got '%s'", report.BaselineModel)
	}

	if report.ComparisonModel != "mlp" {
		t.Errorf("Expected comparison model 'mlp', got '%s'", report.ComparisonModel)
	}

	if report.Dataset != "xor" {
		t.Errorf("Expected dataset 'xor', got '%s'", report.Dataset)
	}

	// Should have some performance changes
	totalChanges := len(report.Improvements) + len(report.Degradations)
	if totalChanges == 0 {
		t.Error("Expected some performance changes in comparison")
	}
}

// TestFastBenchmarkWithMultipleDatasets tests fast benchmarking across datasets
func TestFastBenchmarkWithMultipleDatasets(t *testing.T) {
	runner := NewBenchmarkRunner().
		SetIterations(1).
		SetWarmupRuns(0).
		SetVerbose(false)

	datasets := []Dataset{
		CreateXORDataset(),
		CreateANDDataset(),
	}

	for _, dataset := range datasets {
		t.Run("Fast_"+dataset.Name, func(t *testing.T) {
			// Run fast perceptron benchmark
			metrics, err := runner.FastBenchmarkPerceptron(dataset)
			if err != nil {
				t.Fatalf("Fast benchmark failed for %s: %v", dataset.Name, err)
			}

			if metrics.DatasetName != dataset.Name {
				t.Errorf("Expected dataset %s, got %s", dataset.Name, metrics.DatasetName)
			}

			if metrics.TrainingTime <= 0 {
				t.Error("Training time should be positive")
			}

			// Verify fast execution (should complete quickly)
			if metrics.TrainingTime > 5*time.Second {
				t.Errorf("Fast benchmark took too long: %v", metrics.TrainingTime)
			}
		})
	}
}

// TestFastBenchmarkPerformanceConstraints tests that fast benchmarks meet performance constraints
func TestFastBenchmarkPerformanceConstraints(t *testing.T) {
	runner := NewBenchmarkRunner().
		SetIterations(1).
		SetWarmupRuns(0).
		SetVerbose(false)

	xorDataset := CreateXORDataset()

	// Test perceptron performance constraints
	start := time.Now()
	_, err := runner.FastBenchmarkPerceptron(xorDataset)
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("Fast perceptron benchmark failed: %v", err)
	}

	// Should complete in under 2 seconds for CI/CD
	if duration > 2*time.Second {
		t.Errorf("Fast perceptron benchmark took %v, expected < 2s for CI/CD", duration)
	}

	// Test MLP performance constraints
	start = time.Now()
	hiddenSizes := []int{4}
	_, err = runner.FastBenchmarkMLP(xorDataset, hiddenSizes)
	duration = time.Since(start)

	if err != nil {
		t.Fatalf("Fast MLP benchmark failed: %v", err)
	}

	// Should complete in under 5 seconds for CI/CD
	if duration > 5*time.Second {
		t.Errorf("Fast MLP benchmark took %v, expected < 5s for CI/CD", duration)
	}
}

// TestFastBenchmarkAccuracy tests that fast benchmarks still provide meaningful accuracy
func TestFastBenchmarkAccuracy(t *testing.T) {
	runner := NewBenchmarkRunner().
		SetIterations(1).
		SetWarmupRuns(0).
		SetVerbose(false)

	// Test with AND dataset (linearly separable, should work well with perceptron)
	andDataset := CreateANDDataset()

	metrics, err := runner.FastBenchmarkPerceptron(andDataset)
	if err != nil {
		t.Fatalf("Fast perceptron benchmark failed: %v", err)
	}

	// Perceptron should achieve reasonable accuracy on AND dataset even with fewer epochs
	if metrics.Accuracy < 0.25 { // At least better than random
		t.Errorf("Fast perceptron should achieve > 25%% accuracy on AND, got %.2f%%", metrics.Accuracy*100)
	}

	// Test with XOR dataset and MLP (should handle non-linearity better)
	xorDataset := CreateXORDataset()
	hiddenSizes := []int{4}

	mlpMetrics, err := runner.FastBenchmarkMLP(xorDataset, hiddenSizes)
	if err != nil {
		t.Fatalf("Fast MLP benchmark failed: %v", err)
	}

	// MLP should achieve reasonable accuracy even with fewer epochs
	if mlpMetrics.Accuracy < 0.25 { // At least better than random
		t.Errorf("Fast MLP should achieve > 25%% accuracy on XOR, got %.2f%%", mlpMetrics.Accuracy*100)
	}
}

// TestFastBenchmarkMemoryUsage tests memory usage tracking in fast benchmarks
func TestFastBenchmarkMemoryUsage(t *testing.T) {
	runner := NewBenchmarkRunner().
		SetIterations(1).
		SetWarmupRuns(0).
		SetVerbose(false)

	xorDataset := CreateXORDataset()

	// Test perceptron memory usage
	metrics, err := runner.FastBenchmarkPerceptron(xorDataset)
	if err != nil {
		t.Fatalf("Fast perceptron benchmark failed: %v", err)
	}

	// Memory usage should be tracked (can be positive or negative due to GC)
	// Just verify it's a reasonable value (not extremely large)
	if metrics.MemoryUsage > 100*1024*1024 { // > 100MB seems unreasonable for simple perceptron
		t.Errorf("Perceptron memory usage seems too high: %d bytes", metrics.MemoryUsage)
	}

	// Test MLP memory usage
	hiddenSizes := []int{4}
	mlpMetrics, err := runner.FastBenchmarkMLP(xorDataset, hiddenSizes)
	if err != nil {
		t.Fatalf("Fast MLP benchmark failed: %v", err)
	}

	// MLP should use more memory than perceptron (in most cases)
	// But still reasonable for a small network
	if mlpMetrics.MemoryUsage > 100*1024*1024 { // > 100MB seems unreasonable
		t.Errorf("MLP memory usage seems too high: %d bytes", mlpMetrics.MemoryUsage)
	}
}

// BenchmarkFastBenchmarkRunner benchmarks the fast benchmark implementations
func BenchmarkFastBenchmarkRunner(b *testing.B) {
	runner := NewBenchmarkRunner().
		SetIterations(1).
		SetWarmupRuns(0).
		SetVerbose(false)

	xorDataset := CreateXORDataset()

	b.Run("FastPerceptron", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := runner.FastBenchmarkPerceptron(xorDataset)
			if err != nil {
				b.Fatalf("Fast perceptron benchmark failed: %v", err)
			}
		}
	})

	b.Run("FastMLP", func(b *testing.B) {
		hiddenSizes := []int{4}
		for i := 0; i < b.N; i++ {
			_, err := runner.FastBenchmarkMLP(xorDataset, hiddenSizes)
			if err != nil {
				b.Fatalf("Fast MLP benchmark failed: %v", err)
			}
		}
	})

	b.Run("FastComparison", func(b *testing.B) {
		hiddenSizes := []int{4}
		for i := 0; i < b.N; i++ {
			_, err := runner.FastRunComparison(xorDataset, hiddenSizes)
			if err != nil {
				b.Fatalf("Fast comparison failed: %v", err)
			}
		}
	})
}

// TestFastBenchmarkInferenceOptimization tests inference optimization in fast mode
func TestFastBenchmarkInferenceOptimization(t *testing.T) {
	// Test with minimal iterations
	runner := NewBenchmarkRunner().
		SetIterations(3). // Very few iterations
		SetWarmupRuns(1). // Minimal warmup
		SetVerbose(false)

	xorDataset := CreateXORDataset()

	metrics, err := runner.FastBenchmarkPerceptron(xorDataset)
	if err != nil {
		t.Fatalf("Fast perceptron benchmark failed: %v", err)
	}

	// Should still produce valid inference time
	if metrics.InferenceTime <= 0 {
		t.Error("Inference time should be positive even with few iterations")
	}

	// Should be very fast due to minimal iterations
	if metrics.InferenceTime > 10*time.Millisecond {
		t.Errorf("Inference time seems too slow for minimal iterations: %v", metrics.InferenceTime)
	}
}
