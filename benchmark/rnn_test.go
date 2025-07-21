// Package benchmark - RNN/LSTM Performance Benchmarking Tests
// Learning Goal: Understanding performance testing for sequence models

package benchmark

import (
	"testing"
	"time"

	"github.com/nyasuto/bee/phase2"
)

// TestDefaultRNNBenchmarkConfig tests the default configuration
func TestDefaultRNNBenchmarkConfig(t *testing.T) {
	config := DefaultRNNBenchmarkConfig()

	if config.InputSize <= 0 {
		t.Errorf("Expected positive input size, got %d", config.InputSize)
	}
	if config.HiddenSize <= 0 {
		t.Errorf("Expected positive hidden size, got %d", config.HiddenSize)
	}
	if config.OutputSize <= 0 {
		t.Errorf("Expected positive output size, got %d", config.OutputSize)
	}
	if len(config.SequenceLengths) == 0 {
		t.Error("Expected non-empty sequence lengths")
	}
	if config.LearningRate <= 0 {
		t.Errorf("Expected positive learning rate, got %f", config.LearningRate)
	}
}

// TestRNNBenchmarkBasic tests basic RNN benchmarking functionality
func TestRNNBenchmarkBasic(t *testing.T) {
	runner := NewBenchmarkRunner().
		SetIterations(2).  // Minimal iterations for testing
		SetVerbose(false). // Quiet for testing
		SetWarmupRuns(1)   // Minimal warmup

	config := RNNBenchmarkConfig{
		InputSize:       5,
		HiddenSize:      10,
		OutputSize:      3,
		SequenceLengths: []int{3, 5}, // Short sequences for testing
		LearningRate:    0.01,
		MaxEpochs:       10, // Few epochs for testing
		BatchSize:       4,
		GradientClip:    1.0,
	}

	report, err := runner.BenchmarkRNN(config)
	if err != nil {
		t.Fatalf("RNN benchmark failed: %v", err)
	}

	// Validate report structure
	if report.ModelType != "rnn" {
		t.Errorf("Expected model type 'rnn', got '%s'", report.ModelType)
	}
	if len(report.SequenceMetrics) != len(config.SequenceLengths) {
		t.Errorf("Expected %d sequence metrics, got %d",
			len(config.SequenceLengths), len(report.SequenceMetrics))
	}

	// Validate sequence metrics
	for i, metric := range report.SequenceMetrics {
		expectedLength := config.SequenceLengths[i]
		if metric.SequenceLength != expectedLength {
			t.Errorf("Expected sequence length %d, got %d",
				expectedLength, metric.SequenceLength)
		}
		if metric.ForwardTime <= 0 {
			t.Errorf("Expected positive forward time, got %v", metric.ForwardTime)
		}
		if metric.GradientNorm < 0 {
			t.Errorf("Expected non-negative gradient norm, got %f", metric.GradientNorm)
		}
	}

	// Validate overall metrics
	if report.ScalabilityScore < 0 {
		t.Errorf("Expected non-negative scalability score, got %f", report.ScalabilityScore)
	}
	if report.GradientStability < 0 {
		t.Errorf("Expected non-negative gradient stability, got %f", report.GradientStability)
	}
}

// TestLSTMBenchmarkBasic tests basic LSTM benchmarking functionality
func TestLSTMBenchmarkBasic(t *testing.T) {
	runner := NewBenchmarkRunner().
		SetIterations(2).
		SetVerbose(false).
		SetWarmupRuns(1)

	config := RNNBenchmarkConfig{
		InputSize:       5,
		HiddenSize:      10,
		OutputSize:      3,
		SequenceLengths: []int{3, 5},
		LearningRate:    0.01,
		MaxEpochs:       10,
		BatchSize:       4,
		GradientClip:    1.0,
	}

	report, err := runner.BenchmarkLSTM(config)
	if err != nil {
		t.Fatalf("LSTM benchmark failed: %v", err)
	}

	// Validate report structure
	if report.ModelType != "lstm" {
		t.Errorf("Expected model type 'lstm', got '%s'", report.ModelType)
	}
	if len(report.SequenceMetrics) != len(config.SequenceLengths) {
		t.Errorf("Expected %d sequence metrics, got %d",
			len(config.SequenceLengths), len(report.SequenceMetrics))
	}

	// Validate sequence metrics
	for i, metric := range report.SequenceMetrics {
		expectedLength := config.SequenceLengths[i]
		if metric.SequenceLength != expectedLength {
			t.Errorf("Expected sequence length %d, got %d",
				expectedLength, metric.SequenceLength)
		}
		if metric.ForwardTime <= 0 {
			t.Errorf("Expected positive forward time, got %v", metric.ForwardTime)
		}
		if metric.GradientNorm < 0 {
			t.Errorf("Expected non-negative gradient norm, got %f", metric.GradientNorm)
		}
	}
}

// TestRNNLSTMComparison tests RNN vs LSTM comparison functionality
func TestRNNLSTMComparison(t *testing.T) {
	runner := NewBenchmarkRunner().
		SetIterations(2).
		SetVerbose(false).
		SetWarmupRuns(1)

	config := RNNBenchmarkConfig{
		InputSize:       5,
		HiddenSize:      10,
		OutputSize:      3,
		SequenceLengths: []int{3, 5},
		LearningRate:    0.01,
		MaxEpochs:       10,
		BatchSize:       4,
		GradientClip:    1.0,
	}

	comparison, err := runner.CompareRNNvsLSTM(config)
	if err != nil {
		t.Fatalf("RNN vs LSTM comparison failed: %v", err)
	}

	// Validate comparison structure
	if comparison.RNNReport.ModelType != "rnn" {
		t.Errorf("Expected RNN model type 'rnn', got '%s'", comparison.RNNReport.ModelType)
	}
	if comparison.LSTMReport.ModelType != "lstm" {
		t.Errorf("Expected LSTM model type 'lstm', got '%s'", comparison.LSTMReport.ModelType)
	}
	if comparison.Recommendation == "" {
		t.Error("Expected non-empty recommendation")
	}

	// Validate that we have metrics for both models
	if len(comparison.RNNReport.SequenceMetrics) == 0 {
		t.Error("Expected RNN sequence metrics")
	}
	if len(comparison.LSTMReport.SequenceMetrics) == 0 {
		t.Error("Expected LSTM sequence metrics")
	}
}

// TestGenerateSequenceData tests synthetic sequence data generation
func TestGenerateSequenceData(t *testing.T) {
	runner := NewBenchmarkRunner()

	seqLen := 10
	inputSize := 5
	outputSize := 3
	numSequences := 20

	sequences, targets, err := runner.generateSequenceData(seqLen, inputSize, outputSize, numSequences)
	if err != nil {
		t.Fatalf("Failed to generate sequence data: %v", err)
	}

	// Validate data structure
	if len(sequences) != numSequences {
		t.Errorf("Expected %d sequences, got %d", numSequences, len(sequences))
	}
	if len(targets) != numSequences {
		t.Errorf("Expected %d targets, got %d", numSequences, len(targets))
	}

	// Validate sequence structure
	for i, sequence := range sequences {
		if len(sequence) != seqLen {
			t.Errorf("Sequence %d: expected length %d, got %d", i, seqLen, len(sequence))
		}
		for j, timestep := range sequence {
			if len(timestep) != inputSize {
				t.Errorf("Sequence %d, timestep %d: expected input size %d, got %d",
					i, j, inputSize, len(timestep))
			}
		}
	}

	// Validate target structure
	for i, target := range targets {
		if len(target) != seqLen {
			t.Errorf("Target %d: expected length %d, got %d", i, seqLen, len(target))
		}
		for j, timestep := range target {
			if len(timestep) != outputSize {
				t.Errorf("Target %d, timestep %d: expected output size %d, got %d",
					i, j, outputSize, len(timestep))
			}
		}
	}
}

// TestGradientEstimation tests gradient norm estimation functions
func TestGradientEstimation(t *testing.T) {
	runner := NewBenchmarkRunner()

	// Create simple test sequence
	sequence := [][]float64{
		{1.0, 0.5},
		{0.5, 1.0},
		{0.0, 0.5},
	}

	// Test RNN gradient estimation
	rnn := phase2.NewRNN(2, 5, 3, 0.01)
	rnnGradient := runner.estimateRNNGradientNorm(rnn, sequence)
	if rnnGradient < 0 {
		t.Errorf("Expected non-negative RNN gradient norm, got %f", rnnGradient)
	}

	// Test LSTM gradient estimation
	lstm := phase2.NewLSTM(2, 5, 3, 0.01)
	lstmGradient := runner.estimateLSTMGradientNorm(lstm, sequence)
	if lstmGradient < 0 {
		t.Errorf("Expected non-negative LSTM gradient norm, got %f", lstmGradient)
	}
}

// TestScalabilityMetrics tests scalability calculation functions
func TestScalabilityMetrics(t *testing.T) {
	runner := NewBenchmarkRunner()

	// Create sample metrics
	metrics := []SequenceMetrics{
		{
			SequenceLength:   5,
			ForwardTime:      time.Millisecond * 10,
			MemoryUsage:      1000,
			GradientNorm:     0.5,
			FinalAccuracy:    0.8,
			Stability:        0.7,
			ConvergenceEpoch: 10,
		},
		{
			SequenceLength:   10,
			ForwardTime:      time.Millisecond * 20,
			MemoryUsage:      2000,
			GradientNorm:     0.4,
			FinalAccuracy:    0.7,
			Stability:        0.6,
			ConvergenceEpoch: 15,
		},
		{
			SequenceLength:   20,
			ForwardTime:      time.Millisecond * 40,
			MemoryUsage:      4000,
			GradientNorm:     0.3,
			FinalAccuracy:    0.6,
			Stability:        0.5,
			ConvergenceEpoch: 20,
		},
	}

	// Test scalability score calculation
	scalabilityScore := runner.calculateScalabilityScore(metrics)
	if scalabilityScore < 0 {
		t.Errorf("Expected non-negative scalability score, got %f", scalabilityScore)
	}

	// Test memory scaling rate calculation
	memoryScalingRate := runner.calculateMemoryScalingRate(metrics)
	if memoryScalingRate < 0 {
		t.Errorf("Expected non-negative memory scaling rate, got %f", memoryScalingRate)
	}

	// Test max processable sequence length - with sample data, it should have found at least one valid sequence
	maxLength := runner.findMaxProcessableSequenceLength(metrics)
	if maxLength < 0 {
		t.Errorf("Expected non-negative max sequence length, got %d", maxLength)
	}
}

// TestConvergenceEstimation tests convergence estimation functions
func TestConvergenceEstimation(t *testing.T) {
	runner := NewBenchmarkRunner()

	// Create dummy data
	sequences := [][][]float64{
		{{1.0, 0.5}, {0.5, 1.0}},
		{{0.5, 1.0}, {1.0, 0.5}},
	}
	targets := [][][]float64{
		{{1.0, 0.0, 0.5}, {0.5, 1.0, 0.0}},
		{{0.0, 1.0, 0.5}, {1.0, 0.5, 0.0}},
	}

	// Test RNN convergence estimation
	rnn := phase2.NewRNN(2, 5, 3, 0.01)
	epochs, accuracy, stability := runner.testRNNConvergence(rnn, sequences, targets, 100)

	if epochs < 0 {
		t.Errorf("Expected non-negative convergence epochs, got %d", epochs)
	}
	if accuracy < 0 || accuracy > 1 {
		t.Errorf("Expected accuracy in [0,1], got %f", accuracy)
	}
	if stability < 0 || stability > 1 {
		t.Errorf("Expected stability in [0,1], got %f", stability)
	}

	// Test LSTM convergence estimation
	lstm := phase2.NewLSTM(2, 5, 3, 0.01)
	epochs2, accuracy2, stability2 := runner.testLSTMConvergence(lstm, sequences, targets, 100)

	if epochs2 < 0 {
		t.Errorf("Expected non-negative convergence epochs, got %d", epochs2)
	}
	if accuracy2 < 0 || accuracy2 > 1 {
		t.Errorf("Expected accuracy in [0,1], got %f", accuracy2)
	}
	if stability2 < 0 || stability2 > 1 {
		t.Errorf("Expected stability in [0,1], got %f", stability2)
	}

	// LSTM should generally perform better than RNN
	if accuracy2 < accuracy {
		t.Logf("Note: LSTM accuracy (%.3f) < RNN accuracy (%.3f) - this is unusual but not necessarily wrong for synthetic data", accuracy2, accuracy)
	}
}

// TestGradientFlowAnalysis tests gradient flow analysis
func TestGradientFlowAnalysis(t *testing.T) {
	runner := NewBenchmarkRunner().
		SetIterations(2).
		SetVerbose(false).
		SetWarmupRuns(1)

	config := RNNBenchmarkConfig{
		InputSize:       3,
		HiddenSize:      5,
		OutputSize:      2,
		SequenceLengths: []int{3, 5},
		LearningRate:    0.01,
		MaxEpochs:       5,
		BatchSize:       4,
		GradientClip:    1.0,
	}

	gradients, err := runner.RunGradientFlowAnalysis(config)
	if err != nil {
		t.Fatalf("Gradient flow analysis failed: %v", err)
	}

	// Check that we have results for both models
	if len(gradients) == 0 {
		t.Error("Expected gradient flow results")
	}

	for modelType, gradientNorms := range gradients {
		if len(gradientNorms) != len(config.SequenceLengths) {
			t.Errorf("Model %s: expected %d gradient norms, got %d",
				modelType, len(config.SequenceLengths), len(gradientNorms))
		}

		for i, norm := range gradientNorms {
			if norm < 0 {
				t.Errorf("Model %s, sequence %d: expected non-negative gradient norm, got %f",
					modelType, i, norm)
			}
		}
	}
}

// BenchmarkRNNBasic benchmarks basic RNN operations
func BenchmarkRNNBasic(b *testing.B) {
	runner := NewBenchmarkRunner().
		SetIterations(10).
		SetVerbose(false).
		SetWarmupRuns(2)

	config := RNNBenchmarkConfig{
		InputSize:       10,
		HiddenSize:      20,
		OutputSize:      5,
		SequenceLengths: []int{10, 50},
		LearningRate:    0.01,
		MaxEpochs:       20,
		BatchSize:       16,
		GradientClip:    1.0,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := runner.BenchmarkRNN(config)
		if err != nil {
			b.Fatalf("RNN benchmark failed: %v", err)
		}
	}
}

// BenchmarkLSTMBasic benchmarks basic LSTM operations
func BenchmarkLSTMBasic(b *testing.B) {
	runner := NewBenchmarkRunner().
		SetIterations(10).
		SetVerbose(false).
		SetWarmupRuns(2)

	config := RNNBenchmarkConfig{
		InputSize:       10,
		HiddenSize:      20,
		OutputSize:      5,
		SequenceLengths: []int{10, 50},
		LearningRate:    0.01,
		MaxEpochs:       20,
		BatchSize:       16,
		GradientClip:    1.0,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := runner.BenchmarkLSTM(config)
		if err != nil {
			b.Fatalf("LSTM benchmark failed: %v", err)
		}
	}
}
