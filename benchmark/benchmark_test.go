package benchmark

import (
	"math"
	"testing"
	"time"
)

// abs returns the absolute value of a float64
func abs(x float64) float64 {
	return math.Abs(x)
}

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

	if abs(improvement-expectedImprovement) > 0.001 {
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
		if abs(accuracyComp.PercentChange-expectedImprovement) > 0.001 {
			t.Errorf("Expected accuracy improvement of %.2f%%, got %.2f%%",
				expectedImprovement, accuracyComp.PercentChange)
		}
	} else {
		t.Error("Accuracy comparison not found")
	}
}

// TestCreateLinearSeparableDataset tests linear separable dataset creation
func TestCreateLinearSeparableDataset(t *testing.T) {
	dataset := CreateLinearSeparableDataset(100, 42)

	// Validate dataset structure
	if dataset.Name != "linear_separable" {
		t.Errorf("Expected dataset name 'linear_separable', got '%s'", dataset.Name)
	}

	if dataset.InputSize != 2 {
		t.Errorf("Expected input size 2, got %d", dataset.InputSize)
	}

	if dataset.OutputSize != 1 {
		t.Errorf("Expected output size 1, got %d", dataset.OutputSize)
	}

	if dataset.TrainSize != 80 {
		t.Errorf("Expected train size 80, got %d", dataset.TrainSize)
	}

	if dataset.TestSize != 20 {
		t.Errorf("Expected test size 20, got %d", dataset.TestSize)
	}

	// Validate data structure
	err := ValidateDataset(dataset)
	if err != nil {
		t.Fatalf("Dataset validation failed: %v", err)
	}

	// Check that data points are valid
	for i, input := range dataset.TrainInputs {
		if len(input) != 2 {
			t.Errorf("Training input %d has wrong size: expected 2, got %d", i, len(input))
		}
	}

	for i, target := range dataset.TrainTargets {
		if len(target) != 1 {
			t.Errorf("Training target %d has wrong size: expected 1, got %d", i, len(target))
		}
		if target[0] != 0 && target[0] != 1 {
			t.Errorf("Training target %d has invalid value: expected 0 or 1, got %f", i, target[0])
		}
	}
}

// TestCreateNonLinearDataset tests non-linear dataset creation
func TestCreateNonLinearDataset(t *testing.T) {
	dataset := CreateNonLinearDataset(200, 42)

	// Validate dataset structure
	if dataset.Name != "non_linear" {
		t.Errorf("Expected dataset name 'non_linear', got '%s'", dataset.Name)
	}

	if dataset.InputSize != 2 {
		t.Errorf("Expected input size 2, got %d", dataset.InputSize)
	}

	if dataset.OutputSize != 1 {
		t.Errorf("Expected output size 1, got %d", dataset.OutputSize)
	}

	if dataset.TrainSize != 160 {
		t.Errorf("Expected train size 160, got %d", dataset.TrainSize)
	}

	if dataset.TestSize != 40 {
		t.Errorf("Expected test size 40, got %d", dataset.TestSize)
	}

	// Validate data structure
	err := ValidateDataset(dataset)
	if err != nil {
		t.Fatalf("Dataset validation failed: %v", err)
	}

	// Check that data points are valid
	for i, input := range dataset.TrainInputs {
		if len(input) != 2 {
			t.Errorf("Training input %d has wrong size: expected 2, got %d", i, len(input))
		}
	}

	for i, target := range dataset.TrainTargets {
		if len(target) != 1 {
			t.Errorf("Training target %d has wrong size: expected 1, got %d", i, len(target))
		}
		if target[0] != 0 && target[0] != 1 {
			t.Errorf("Training target %d has invalid value: expected 0 or 1, got %f", i, target[0])
		}
	}
}

// TestCreateSinusoidalDataset tests sinusoidal dataset creation
func TestCreateSinusoidalDataset(t *testing.T) {
	dataset := CreateSinusoidalDataset(150, 0.1, 42)

	// Validate dataset structure
	if dataset.Name != "sinusoidal" {
		t.Errorf("Expected dataset name 'sinusoidal', got '%s'", dataset.Name)
	}

	if dataset.InputSize != 1 {
		t.Errorf("Expected input size 1, got %d", dataset.InputSize)
	}

	if dataset.OutputSize != 1 {
		t.Errorf("Expected output size 1, got %d", dataset.OutputSize)
	}

	if dataset.TrainSize != 120 {
		t.Errorf("Expected train size 120, got %d", dataset.TrainSize)
	}

	if dataset.TestSize != 30 {
		t.Errorf("Expected test size 30, got %d", dataset.TestSize)
	}

	// Validate data structure
	err := ValidateDataset(dataset)
	if err != nil {
		t.Fatalf("Dataset validation failed: %v", err)
	}

	// Check that targets are continuous values (not just 0/1)
	hasNonBinaryTarget := false
	for i, target := range dataset.TrainTargets {
		if len(target) != 1 {
			t.Errorf("Training target %d has wrong size: expected 1, got %d", i, len(target))
		}
		if target[0] != 0 && target[0] != 1 {
			hasNonBinaryTarget = true
		}
	}
	if !hasNonBinaryTarget {
		t.Error("Sinusoidal dataset should have continuous target values, not just binary")
	}
}

// TestGetStandardDatasets tests getting all standard datasets
func TestGetStandardDatasets(t *testing.T) {
	datasets := GetStandardDatasets()

	// Should have 5 datasets: XOR, AND, OR, linear_separable, non_linear
	expectedCount := 5
	if len(datasets) != expectedCount {
		t.Errorf("Expected %d datasets, got %d", expectedCount, len(datasets))
	}

	// Check that all datasets are valid
	for i, dataset := range datasets {
		err := ValidateDataset(dataset)
		if err != nil {
			t.Errorf("Dataset %d (%s) validation failed: %v", i, dataset.Name, err)
		}
	}

	// Check that we have the expected dataset names
	expectedNames := map[string]bool{
		"xor":              true,
		"and":              true,
		"or":               true,
		"linear_separable": true,
		"non_linear":       true,
	}

	foundNames := make(map[string]bool)
	for _, dataset := range datasets {
		foundNames[dataset.Name] = true
	}

	for expectedName := range expectedNames {
		if !foundNames[expectedName] {
			t.Errorf("Expected dataset '%s' not found", expectedName)
		}
	}
}

// TestPrintDatasetInfo tests dataset info printing
func TestPrintDatasetInfo(t *testing.T) {
	dataset := CreateXORDataset()

	// This function prints to stdout, so we can't easily test output
	// But we can test that it doesn't panic
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("PrintDatasetInfo panicked: %v", r)
		}
	}()

	PrintDatasetInfo(dataset)
}

// TestGetEnvironmentInfo tests environment info gathering
func TestGetEnvironmentInfo(t *testing.T) {
	envInfo := GetEnvironmentInfo()

	// Check that we get some basic info
	if envInfo.GoVersion == "" {
		t.Error("Expected Go version to be set")
	}

	if envInfo.OS == "" {
		t.Error("Expected OS to be set")
	}

	if envInfo.Architecture == "" {
		t.Error("Expected architecture to be set")
	}

	if envInfo.CPUCores <= 0 {
		t.Error("Expected positive number of CPU cores")
	}
}

// TestFormatDuration tests duration formatting
func TestFormatDuration(t *testing.T) {
	testCases := []struct {
		name     string
		duration time.Duration
		expected string
	}{
		{
			name:     "Nanoseconds",
			duration: 500 * time.Nanosecond,
			expected: "500.00 ns",
		},
		{
			name:     "Microseconds",
			duration: 1500 * time.Nanosecond, // Less than 1 microsecond to test µs format
			expected: "1.50 μs",
		},
		{
			name:     "Milliseconds",
			duration: 2500 * time.Microsecond, // Less than 1 second to test ms format
			expected: "2.50 ms",
		},
		{
			name:     "Seconds",
			duration: 5 * time.Second,
			expected: "5.00 s",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := FormatDuration(tc.duration)
			if result != tc.expected {
				t.Errorf("Expected '%s', got '%s'", tc.expected, result)
			}
		})
	}
}

// TestFormatMemory tests memory formatting
func TestFormatMemory(t *testing.T) {
	testCases := []struct {
		name     string
		bytes    int64
		expected string
	}{
		{
			name:     "Bytes",
			bytes:    512,
			expected: "512 B",
		},
		{
			name:     "Kilobytes",
			bytes:    2048,
			expected: "2.0 KB",
		},
		{
			name:     "Megabytes",
			bytes:    3 * 1024 * 1024,
			expected: "3.0 MB",
		},
		{
			name:     "Gigabytes",
			bytes:    4 * 1024 * 1024 * 1024,
			expected: "4.0 GB",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := FormatMemory(tc.bytes)
			if result != tc.expected {
				t.Errorf("Expected '%s', got '%s'", tc.expected, result)
			}
		})
	}
}

// TestPrintComparisonSummary tests comparison summary printing
func TestPrintComparisonSummary(t *testing.T) {
	// Create a mock comparison report
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

	// This function prints to stdout, test that it doesn't panic
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("PrintComparisonSummary panicked: %v", r)
		}
	}()

	PrintComparisonSummary(report)
}

// TestSaveBenchmarkResult tests saving benchmark results
func TestSaveBenchmarkResult(t *testing.T) {
	result := BenchmarkResult{
		Metrics: PerformanceMetrics{
			ModelType:       "perceptron",
			DatasetName:     "test",
			Accuracy:        0.8,
			TrainingTime:    50 * time.Millisecond,
			InferenceTime:   1 * time.Microsecond,
			MemoryUsage:     1024,
			ConvergenceRate: 100,
			FinalLoss:       0.2,
			Timestamp:       time.Now(),
		},
		Environment: GetEnvironmentInfo(),
	}

	// Test saving (currently returns no error as it's a placeholder)
	filename := "test_benchmark.json"
	err := SaveBenchmarkResult(result, filename)
	if err != nil {
		t.Fatalf("Failed to save benchmark result: %v", err)
	}

	// Note: Since writeFile is a placeholder, we just test that no error occurs
}

// TestLoadBenchmarkResult tests loading benchmark results
func TestLoadBenchmarkResult(t *testing.T) {
	// Test loading (currently readFile is a placeholder that returns empty data)
	filename := "test_load_benchmark.json"

	_, err := LoadBenchmarkResult(filename)
	// Since readFile returns empty data, this should fail with a JSON error
	if err == nil {
		t.Error("Expected error when loading empty file, but got none")
	}

	// This is expected behavior since readFile is a placeholder
	t.Logf("Expected error occurred: %v", err)
}

// TestRunComparison tests the comparison runner
func TestRunComparison(t *testing.T) {
	runner := NewBenchmarkRunner().
		SetIterations(5). // Use fewer iterations for faster tests
		SetVerbose(false)

	dataset := CreateXORDataset()
	hiddenLayers := []int{4}

	report, err := runner.RunComparison(dataset, hiddenLayers)
	if err != nil {
		t.Fatalf("RunComparison failed: %v", err)
	}

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

	// Should have some improvements or degradations
	if len(report.Improvements) == 0 && len(report.Degradations) == 0 {
		t.Error("Expected some improvements or degradations in comparison report")
	}
}

// TestRunMultipleComparisons tests running multiple comparisons
func TestRunMultipleComparisons(t *testing.T) {
	runner := NewBenchmarkRunner().
		SetIterations(3). // Use fewer iterations for faster tests
		SetVerbose(false)

	datasets := []Dataset{
		CreateXORDataset(),
		CreateANDDataset(),
	}
	hiddenLayers := []int{4}

	reports, err := runner.RunMultipleComparisons(datasets, hiddenLayers)
	if err != nil {
		t.Fatalf("RunMultipleComparisons failed: %v", err)
	}

	// Should have reports for both datasets
	if len(reports) != 2 {
		t.Errorf("Expected 2 reports, got %d", len(reports))
	}

	// Validate each report
	expectedDatasets := map[string]bool{"xor": false, "and": false}
	for _, report := range reports {
		if _, exists := expectedDatasets[report.Dataset]; exists {
			expectedDatasets[report.Dataset] = true
		} else {
			t.Errorf("Unexpected dataset in report: %s", report.Dataset)
		}
	}

	// Check that all expected datasets were processed
	for dataset, found := range expectedDatasets {
		if !found {
			t.Errorf("Missing report for dataset: %s", dataset)
		}
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
