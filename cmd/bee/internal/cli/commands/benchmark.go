// Package commands implements CLI command handlers
package commands

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/nyasuto/bee/benchmark"
	"github.com/nyasuto/bee/cmd/bee/internal/cli/config"
	"github.com/nyasuto/bee/cmd/bee/internal/output"
	"github.com/nyasuto/bee/datasets"
)

// BenchmarkCommand implements the benchmark command for neural network performance testing
// Learning Goal: Understanding comprehensive performance evaluation across different architectures
type BenchmarkCommand struct {
	outputWriter output.OutputWriter
}

// NewBenchmarkCommand creates a new benchmark command
func NewBenchmarkCommand(outputWriter output.OutputWriter) *BenchmarkCommand {
	return &BenchmarkCommand{
		outputWriter: outputWriter,
	}
}

// Validate checks if the configuration is valid for this command
func (cmd *BenchmarkCommand) Validate(cfg interface{}) error {
	_, ok := cfg.(*config.BenchmarkConfig)
	if !ok {
		return fmt.Errorf("invalid configuration type for benchmark command")
	}
	return nil
}

// Name returns the name of the command
func (cmd *BenchmarkCommand) Name() string {
	return "benchmark"
}

// Description returns a brief description of the command
func (cmd *BenchmarkCommand) Description() string {
	return "Run performance benchmarks for neural network models (Perceptron, MLP, CNN)"
}

// Execute runs the benchmark command
func (cmd *BenchmarkCommand) Execute(ctx context.Context, cfg interface{}) error {
	benchmarkCfg, ok := cfg.(*config.BenchmarkConfig)
	if !ok {
		return fmt.Errorf("invalid configuration type for benchmark command")
	}

	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "🚀 Starting neural network benchmark...")
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Model: %s, Dataset: %s", benchmarkCfg.Model, benchmarkCfg.Dataset)

	// Create benchmark runner
	runner := benchmark.NewBenchmarkRunner().
		SetIterations(benchmarkCfg.Iterations).
		SetVerbose(benchmarkCfg.Verbose)

	// Execute based on model type
	switch strings.ToLower(benchmarkCfg.Model) {
	case "perceptron":
		return cmd.benchmarkPerceptron(runner, benchmarkCfg)
	case "mlp":
		return cmd.benchmarkMLP(runner, benchmarkCfg)
	case "cnn":
		return cmd.benchmarkCNN(runner, benchmarkCfg)
	case "rnn":
		return cmd.benchmarkRNN(runner, benchmarkCfg)
	case "lstm":
		return cmd.benchmarkLSTM(runner, benchmarkCfg)
	case "rnn-compare":
		return cmd.benchmarkRNNComparison(runner, benchmarkCfg)
	case "compare":
		return cmd.benchmarkComparison(runner, benchmarkCfg)
	default:
		return fmt.Errorf("unsupported model type: %s", benchmarkCfg.Model)
	}
}

// benchmarkPerceptron runs perceptron benchmarks
func (cmd *BenchmarkCommand) benchmarkPerceptron(runner *benchmark.BenchmarkRunner, cfg *config.BenchmarkConfig) error {
	// Get dataset
	dataset, err := cmd.getStandardDataset(cfg.Dataset)
	if err != nil {
		return fmt.Errorf("failed to get dataset: %w", err)
	}

	// Run benchmark
	metrics, err := runner.BenchmarkPerceptron(dataset)
	if err != nil {
		return fmt.Errorf("perceptron benchmark failed: %w", err)
	}

	// Display results
	cmd.displayPerformanceMetrics(metrics)

	// Save results if output path specified
	if cfg.OutputPath != "" {
		return cmd.saveMetrics(metrics, cfg.OutputPath)
	}

	return nil
}

// benchmarkMLP runs MLP benchmarks
func (cmd *BenchmarkCommand) benchmarkMLP(runner *benchmark.BenchmarkRunner, cfg *config.BenchmarkConfig) error {
	// Get dataset
	dataset, err := cmd.getStandardDataset(cfg.Dataset)
	if err != nil {
		return fmt.Errorf("failed to get dataset: %w", err)
	}

	// Parse hidden layer configuration
	hiddenSizes, err := cmd.parseMLPHidden(cfg.MLPHidden)
	if err != nil {
		return fmt.Errorf("failed to parse MLP hidden layers: %w", err)
	}

	// Run benchmark
	metrics, err := runner.BenchmarkMLP(dataset, hiddenSizes)
	if err != nil {
		return fmt.Errorf("MLP benchmark failed: %w", err)
	}

	// Display results
	cmd.displayPerformanceMetrics(metrics)

	// Save results if output path specified
	if cfg.OutputPath != "" {
		return cmd.saveMetrics(metrics, cfg.OutputPath)
	}

	return nil
}

// benchmarkCNN runs CNN benchmarks
func (cmd *BenchmarkCommand) benchmarkCNN(runner *benchmark.BenchmarkRunner, cfg *config.BenchmarkConfig) error {
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "🧠 Running CNN benchmark...")

	// Get image dataset
	imageDataset, err := cmd.getImageDataset(cfg.Dataset)
	if err != nil {
		return fmt.Errorf("failed to get image dataset: %w", err)
	}

	// Create CNN configuration
	cnnConfig := benchmark.CNNBenchmarkConfig{
		ImageSize:    [3]int{imageDataset.Height, imageDataset.Width, imageDataset.Channels},
		BatchSize:    cfg.BatchSize,
		Epochs:       cfg.Epochs,
		LearningRate: cfg.LearningRate,
		Architecture: cfg.CNNArch,
	}

	// Set defaults if not specified
	if cnnConfig.BatchSize == 0 {
		cnnConfig.BatchSize = 32
	}
	if cnnConfig.Epochs == 0 {
		cnnConfig.Epochs = 10
	}
	if cnnConfig.LearningRate == 0 {
		cnnConfig.LearningRate = 0.001
	}
	if cnnConfig.Architecture == "" {
		cnnConfig.Architecture = "MNIST"
	}

	// Run CNN benchmark
	metrics, err := runner.BenchmarkCNN(imageDataset, cnnConfig)
	if err != nil {
		return fmt.Errorf("CNN benchmark failed: %w", err)
	}

	// Display results
	cmd.displayCNNMetrics(metrics)

	// Run memory analysis
	err = cmd.runCNNMemoryAnalysis(runner, imageDataset, cnnConfig)
	if err != nil {
		cmd.outputWriter.WriteMessage(output.LogLevelWarn, "Memory analysis failed: %v", err)
	}

	// Run batch processing analysis
	err = cmd.runBatchProcessingAnalysis(runner, imageDataset, cnnConfig)
	if err != nil {
		cmd.outputWriter.WriteMessage(output.LogLevelWarn, "Batch processing analysis failed: %v", err)
	}

	// Save results if output path specified
	if cfg.OutputPath != "" {
		return cmd.saveMetrics(metrics, cfg.OutputPath)
	}

	return nil
}

// benchmarkComparison runs comparative benchmarks
func (cmd *BenchmarkCommand) benchmarkComparison(runner *benchmark.BenchmarkRunner, cfg *config.BenchmarkConfig) error {
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "📊 Running comparative benchmark...")

	// Check if we should compare CNN vs MLP
	if strings.ToLower(cfg.Dataset) == "mnist" || cfg.CNNArch != "" {
		return cmd.runCNNComparison(runner, cfg)
	}

	// Standard dataset comparison (Perceptron vs MLP)
	dataset, err := cmd.getStandardDataset(cfg.Dataset)
	if err != nil {
		return fmt.Errorf("failed to get dataset: %w", err)
	}

	hiddenSizes, err := cmd.parseMLPHidden(cfg.MLPHidden)
	if err != nil {
		return fmt.Errorf("failed to parse MLP hidden layers: %w", err)
	}

	// Run comparison
	report, err := runner.RunComparison(dataset, hiddenSizes)
	if err != nil {
		return fmt.Errorf("comparison benchmark failed: %w", err)
	}

	// Display comparison results
	cmd.displayComparisonReport(report)

	// Save results if output path specified
	if cfg.OutputPath != "" {
		return cmd.saveComparisonReport(report, cfg.OutputPath)
	}

	return nil
}

// runCNNComparison runs CNN vs MLP comparison
func (cmd *BenchmarkCommand) runCNNComparison(runner *benchmark.BenchmarkRunner, cfg *config.BenchmarkConfig) error {
	// Get image dataset
	imageDataset, err := cmd.getImageDataset(cfg.Dataset)
	if err != nil {
		return fmt.Errorf("failed to get image dataset: %w", err)
	}

	// Create CNN configuration
	cnnConfig := benchmark.CNNBenchmarkConfig{
		ImageSize:    [3]int{imageDataset.Height, imageDataset.Width, imageDataset.Channels},
		BatchSize:    cfg.BatchSize,
		Epochs:       cfg.Epochs,
		LearningRate: cfg.LearningRate,
		Architecture: cfg.CNNArch,
	}

	// Set defaults
	if cnnConfig.BatchSize == 0 {
		cnnConfig.BatchSize = 32
	}
	if cnnConfig.Epochs == 0 {
		cnnConfig.Epochs = 10
	}
	if cnnConfig.LearningRate == 0 {
		cnnConfig.LearningRate = 0.001
	}
	if cnnConfig.Architecture == "" {
		cnnConfig.Architecture = "MNIST"
	}

	// Parse MLP hidden layers
	hiddenSizes, err := cmd.parseMLPHidden(cfg.MLPHidden)
	if err != nil {
		// Use default hidden layers for MLP
		hiddenSizes = []int{128, 64}
	}

	// Run CNN vs MLP comparison
	report, err := runner.RunCNNComparison(imageDataset, cnnConfig, hiddenSizes)
	if err != nil {
		return fmt.Errorf("CNN comparison failed: %w", err)
	}

	// Display comparison results
	cmd.displayComparisonReport(report)

	// Save results if output path specified
	if cfg.OutputPath != "" {
		return cmd.saveComparisonReport(report, cfg.OutputPath)
	}

	return nil
}

// runCNNMemoryAnalysis performs detailed memory analysis for CNN
func (cmd *BenchmarkCommand) runCNNMemoryAnalysis(runner *benchmark.BenchmarkRunner, imageDataset *datasets.ImageDataset, config benchmark.CNNBenchmarkConfig) error {
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "🧠 Analyzing CNN memory usage...")

	// Create evaluator for memory analysis
	evaluator := datasets.NewCNNEvaluator(imageDataset, config.LearningRate, config.BatchSize, config.Epochs)
	err := evaluator.CreateMNISTCNN()
	if err != nil {
		return fmt.Errorf("failed to create CNN for memory analysis: %w", err)
	}

	// Run memory analysis
	memAnalysis, err := runner.AnalyzeCNNMemory(evaluator, config)
	if err != nil {
		return fmt.Errorf("memory analysis failed: %w", err)
	}

	// Display memory analysis results
	cmd.displayMemoryAnalysis(memAnalysis)

	return nil
}

// runBatchProcessingAnalysis performs batch processing efficiency analysis
func (cmd *BenchmarkCommand) runBatchProcessingAnalysis(runner *benchmark.BenchmarkRunner, imageDataset *datasets.ImageDataset, config benchmark.CNNBenchmarkConfig) error {
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "📦 Analyzing batch processing efficiency...")

	// Create evaluator for batch analysis
	evaluator := datasets.NewCNNEvaluator(imageDataset, config.LearningRate, config.BatchSize, config.Epochs)
	err := evaluator.CreateMNISTCNN()
	if err != nil {
		return fmt.Errorf("failed to create CNN for batch analysis: %w", err)
	}

	// Run batch processing analysis
	batchMetrics, err := runner.BenchmarkBatchProcessing(evaluator, config)
	if err != nil {
		return fmt.Errorf("batch processing analysis failed: %w", err)
	}

	// Display batch processing results
	cmd.displayBatchProcessingMetrics(batchMetrics)

	return nil
}

// benchmarkRNN runs RNN benchmarks
func (cmd *BenchmarkCommand) benchmarkRNN(runner *benchmark.BenchmarkRunner, cfg *config.BenchmarkConfig) error {
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "🔄 Running RNN benchmark...")

	// Get sequence dataset
	sequenceDataset, err := cmd.getSequenceDataset(cfg.Dataset)
	if err != nil {
		return fmt.Errorf("failed to get sequence dataset: %w", err)
	}

	// Create RNN configuration
	rnnConfig := benchmark.RNNBenchmarkConfig{
		SequenceLength:   cfg.SequenceLength,
		HiddenSize:       cfg.HiddenSize,
		BatchSize:        cfg.BatchSize,
		Epochs:           cfg.Epochs,
		LearningRate:     cfg.LearningRate,
		ModelType:        "RNN",
		MemoryAnalysis:   true,
		GradientAnalysis: false,
	}

	// Set defaults if not specified
	if rnnConfig.SequenceLength == 0 {
		rnnConfig.SequenceLength = 10
	}
	if rnnConfig.HiddenSize == 0 {
		rnnConfig.HiddenSize = 32
	}
	if rnnConfig.BatchSize == 0 {
		rnnConfig.BatchSize = 16
	}
	if rnnConfig.Epochs == 0 {
		rnnConfig.Epochs = 50
	}
	if rnnConfig.LearningRate == 0 {
		rnnConfig.LearningRate = 0.01
	}

	// Run RNN benchmark
	metrics, err := runner.BenchmarkRNN(sequenceDataset, rnnConfig)
	if err != nil {
		return fmt.Errorf("RNN benchmark failed: %w", err)
	}

	// Display results
	cmd.displayRNNMetrics(metrics)

	// Save results if output path specified
	if cfg.OutputPath != "" {
		return cmd.saveMetrics(metrics, cfg.OutputPath)
	}

	return nil
}

// benchmarkLSTM runs LSTM benchmarks
func (cmd *BenchmarkCommand) benchmarkLSTM(runner *benchmark.BenchmarkRunner, cfg *config.BenchmarkConfig) error {
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "🧠 Running LSTM benchmark...")

	// Get sequence dataset
	sequenceDataset, err := cmd.getSequenceDataset(cfg.Dataset)
	if err != nil {
		return fmt.Errorf("failed to get sequence dataset: %w", err)
	}

	// Create LSTM configuration
	lstmConfig := benchmark.RNNBenchmarkConfig{
		SequenceLength:   cfg.SequenceLength,
		HiddenSize:       cfg.HiddenSize,
		BatchSize:        cfg.BatchSize,
		Epochs:           cfg.Epochs,
		LearningRate:     cfg.LearningRate,
		ModelType:        "LSTM",
		MemoryAnalysis:   true,
		GradientAnalysis: false,
	}

	// Set defaults if not specified
	if lstmConfig.SequenceLength == 0 {
		lstmConfig.SequenceLength = 10
	}
	if lstmConfig.HiddenSize == 0 {
		lstmConfig.HiddenSize = 32
	}
	if lstmConfig.BatchSize == 0 {
		lstmConfig.BatchSize = 16
	}
	if lstmConfig.Epochs == 0 {
		lstmConfig.Epochs = 50
	}
	if lstmConfig.LearningRate == 0 {
		lstmConfig.LearningRate = 0.01
	}

	// Run LSTM benchmark
	metrics, err := runner.BenchmarkRNN(sequenceDataset, lstmConfig)
	if err != nil {
		return fmt.Errorf("LSTM benchmark failed: %w", err)
	}

	// Display results
	cmd.displayRNNMetrics(metrics)

	// Save results if output path specified
	if cfg.OutputPath != "" {
		return cmd.saveMetrics(metrics, cfg.OutputPath)
	}

	return nil
}

// benchmarkRNNComparison runs RNN vs LSTM comparative benchmarks
func (cmd *BenchmarkCommand) benchmarkRNNComparison(runner *benchmark.BenchmarkRunner, cfg *config.BenchmarkConfig) error {
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "⚡ Running RNN vs LSTM comparative benchmark...")

	// Get sequence dataset
	sequenceDataset, err := cmd.getSequenceDataset(cfg.Dataset)
	if err != nil {
		return fmt.Errorf("failed to get sequence dataset: %w", err)
	}

	// Create RNN comparison configuration
	comparisonConfig := benchmark.RNNBenchmarkConfig{
		SequenceLength:   cfg.SequenceLength,
		HiddenSize:       cfg.HiddenSize,
		BatchSize:        cfg.BatchSize,
		Epochs:           cfg.Epochs,
		LearningRate:     cfg.LearningRate,
		MemoryAnalysis:   true,
		GradientAnalysis: false,
	}

	// Set defaults if not specified
	if comparisonConfig.SequenceLength == 0 {
		comparisonConfig.SequenceLength = 10
	}
	if comparisonConfig.HiddenSize == 0 {
		comparisonConfig.HiddenSize = 32
	}
	if comparisonConfig.BatchSize == 0 {
		comparisonConfig.BatchSize = 16
	}
	if comparisonConfig.Epochs == 0 {
		comparisonConfig.Epochs = 50
	}
	if comparisonConfig.LearningRate == 0 {
		comparisonConfig.LearningRate = 0.01
	}

	// Run RNN vs LSTM comparison
	report, err := runner.RunRNNComparison(sequenceDataset, comparisonConfig)
	if err != nil {
		return fmt.Errorf("RNN comparison failed: %w", err)
	}

	// Display comparison results
	cmd.displayComparisonReport(report)

	// Save results if output path specified
	if cfg.OutputPath != "" {
		return cmd.saveComparisonReport(report, cfg.OutputPath)
	}

	return nil
}

// getSequenceDataset retrieves a sequence dataset for RNN/LSTM benchmarking
func (cmd *BenchmarkCommand) getSequenceDataset(datasetName string) (*datasets.TimeSeriesDataset, error) {
	switch strings.ToLower(datasetName) {
	case "sine", "sinusoidal":
		return cmd.createSinusoidalDataset(), nil
	case "simple", "sequence":
		return cmd.createSimpleSequenceDataset(), nil
	case "timeseries":
		return cmd.createTimeSeriesDataset(), nil
	default:
		return nil, fmt.Errorf("unsupported sequence dataset: %s", datasetName)
	}
}

// createSinusoidalDataset creates a sinusoidal sequence dataset
func (cmd *BenchmarkCommand) createSinusoidalDataset() *datasets.TimeSeriesDataset {
	numSequences := 100
	sequenceLength := 10
	sequences := make([][][]float64, numSequences)
	targets := make([][]float64, numSequences)

	for i := 0; i < numSequences; i++ {
		// Create sinusoidal sequence
		sequences[i] = make([][]float64, sequenceLength)
		for t := 0; t < sequenceLength; t++ {
			x := float64(i*sequenceLength+t) * 0.1
			sequences[i][t] = []float64{x} // Single input feature
		}

		// Target is the next value in the sine wave
		nextX := float64(i*sequenceLength+sequenceLength) * 0.1
		nextVal := 0.5 + 0.5*math.Sin(nextX) // Normalize to [0,1]
		targets[i] = []float64{nextVal}
	}

	return &datasets.TimeSeriesDataset{
		Name:        "Sinusoidal",
		Type:        "prediction",
		Sequences:   sequences,
		Targets:     targets,
		Features:    1,
		OutputSize:  1,
		SeqLength:   sequenceLength,
		Description: "Sinusoidal wave prediction dataset",
	}
}

// createSimpleSequenceDataset creates a simple pattern recognition dataset
func (cmd *BenchmarkCommand) createSimpleSequenceDataset() *datasets.TimeSeriesDataset {
	numSequences := 80
	sequenceLength := 5
	sequences := make([][][]float64, numSequences)
	targets := make([][]float64, numSequences)

	for i := 0; i < numSequences; i++ {
		// Create sequence with pattern: [1,0,1,0,1] or [0,1,0,1,0]
		sequences[i] = make([][]float64, sequenceLength)
		pattern := i % 2
		for t := 0; t < sequenceLength; t++ {
			val := float64((pattern + t) % 2)
			sequences[i][t] = []float64{val}
		}

		// Target is the pattern type
		targets[i] = []float64{float64(pattern)}
	}

	return &datasets.TimeSeriesDataset{
		Name:        "SimpleSequence",
		Type:        "classification",
		Sequences:   sequences,
		Targets:     targets,
		Features:    1,
		OutputSize:  1,
		SeqLength:   sequenceLength,
		Description: "Simple binary pattern recognition dataset",
	}
}

// createTimeSeriesDataset creates a time series prediction dataset
func (cmd *BenchmarkCommand) createTimeSeriesDataset() *datasets.TimeSeriesDataset {
	numSequences := 120
	sequenceLength := 8
	sequences := make([][][]float64, numSequences)
	targets := make([][]float64, numSequences)

	for i := 0; i < numSequences; i++ {
		// Create time series with trend and noise
		sequences[i] = make([][]float64, sequenceLength)
		base := float64(i) * 0.01
		for t := 0; t < sequenceLength; t++ {
			// Linear trend with some oscillation
			val := base + float64(t)*0.1 + 0.05*math.Sin(float64(t))
			sequences[i][t] = []float64{val}
		}

		// Target is the next value in the series
		nextVal := base + float64(sequenceLength)*0.1 + 0.05*math.Sin(float64(sequenceLength))
		targets[i] = []float64{nextVal}
	}

	return &datasets.TimeSeriesDataset{
		Name:        "TimeSeries",
		Type:        "regression",
		Sequences:   sequences,
		Targets:     targets,
		Features:    1,
		OutputSize:  1,
		SeqLength:   sequenceLength,
		Description: "Time series prediction with trend and oscillation",
	}
}

// displayRNNMetrics shows RNN/LSTM-specific performance metrics
func (cmd *BenchmarkCommand) displayRNNMetrics(metrics benchmark.PerformanceMetrics) {
	cmd.displayPerformanceMetrics(metrics)

	if metrics.SequenceLength > 0 {
		cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Sequence Length: %d", metrics.SequenceLength)
	}
	if metrics.HiddenSize > 0 {
		cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Hidden Size: %d", metrics.HiddenSize)
	}
	if metrics.SequenceProcessingTime > 0 {
		cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Sequence Processing Time: %s", benchmark.FormatDuration(metrics.SequenceProcessingTime))
	}
	if metrics.MemoryScaling > 0 {
		cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Memory Scaling: %s", benchmark.FormatMemory(metrics.MemoryScaling))
	}
}

// getStandardDataset retrieves a standard benchmark dataset
func (cmd *BenchmarkCommand) getStandardDataset(datasetName string) (benchmark.Dataset, error) {
	switch strings.ToLower(datasetName) {
	case "xor":
		return benchmark.CreateXORDataset(), nil
	case "and":
		return benchmark.CreateANDDataset(), nil
	case "or":
		return benchmark.CreateORDataset(), nil
	case "linear":
		return benchmark.CreateLinearSeparableDataset(200, 42), nil
	case "nonlinear":
		return benchmark.CreateNonLinearDataset(300, 42), nil
	default:
		return benchmark.Dataset{}, fmt.Errorf("unsupported dataset: %s", datasetName)
	}
}

// getImageDataset retrieves an image dataset for CNN benchmarking
func (cmd *BenchmarkCommand) getImageDataset(datasetName string) (*datasets.ImageDataset, error) {
	switch strings.ToLower(datasetName) {
	case "mnist":
		// Create a small MNIST-like dataset for testing
		return cmd.createSampleMNISTDataset(), nil
	default:
		return nil, fmt.Errorf("unsupported image dataset: %s", datasetName)
	}
}

// createSampleMNISTDataset creates a small sample dataset for testing
func (cmd *BenchmarkCommand) createSampleMNISTDataset() *datasets.ImageDataset {
	// Create sample 28x28x1 images
	numSamples := 100 // Small dataset for testing
	images := make([][][][]float64, numSamples)
	labels := make([]int, numSamples)

	for i := 0; i < numSamples; i++ {
		// Create 28x28x1 image
		images[i] = make([][][]float64, 28)
		for h := 0; h < 28; h++ {
			images[i][h] = make([][]float64, 28)
			for w := 0; w < 28; w++ {
				images[i][h][w] = []float64{float64(i%2) * 0.5} // Simple pattern
			}
		}
		labels[i] = i % 10 // Labels 0-9
	}

	return &datasets.ImageDataset{
		Name:     "SampleMNIST",
		Images:   images,
		Labels:   labels,
		Classes:  []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"},
		Width:    28,
		Height:   28,
		Channels: 1,
	}
}

// parseMLPHidden parses hidden layer configuration string
func (cmd *BenchmarkCommand) parseMLPHidden(hiddenStr string) ([]int, error) {
	if hiddenStr == "" {
		return []int{10}, nil // Default single hidden layer
	}

	parts := strings.Split(hiddenStr, ",")
	hiddenSizes := make([]int, len(parts))

	for i, part := range parts {
		size, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil {
			return nil, fmt.Errorf("invalid hidden layer size: %s", part)
		}
		if size <= 0 {
			return nil, fmt.Errorf("hidden layer size must be positive: %d", size)
		}
		hiddenSizes[i] = size
	}

	return hiddenSizes, nil
}

// displayPerformanceMetrics shows performance metrics in a formatted way
func (cmd *BenchmarkCommand) displayPerformanceMetrics(metrics benchmark.PerformanceMetrics) {
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "\n📊 Performance Results:")
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "─────────────────────────────────────")
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Model: %s", metrics.ModelType)
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Dataset: %s", metrics.DatasetName)
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Training Time: %s", benchmark.FormatDuration(metrics.TrainingTime))
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Inference Time: %s", benchmark.FormatDuration(metrics.InferenceTime))
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Memory Usage: %s", benchmark.FormatMemory(metrics.MemoryUsage))
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Accuracy: %.2f%%", metrics.Accuracy*100)
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Convergence: %d epochs", metrics.ConvergenceRate)
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Final Loss: %.6f", metrics.FinalLoss)
}

// displayCNNMetrics shows CNN-specific performance metrics
func (cmd *BenchmarkCommand) displayCNNMetrics(metrics benchmark.PerformanceMetrics) {
	cmd.displayPerformanceMetrics(metrics)

	if metrics.ConvolutionTime > 0 {
		cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Convolution Time: %s", benchmark.FormatDuration(metrics.ConvolutionTime))
	}
	if metrics.PoolingTime > 0 {
		cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Pooling Time: %s", benchmark.FormatDuration(metrics.PoolingTime))
	}
	if metrics.BatchSize > 0 {
		cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Batch Size: %d", metrics.BatchSize)
	}
	if metrics.FeatureMapSize > 0 {
		cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Feature Map Memory: %s", benchmark.FormatMemory(metrics.FeatureMapSize))
	}
}

// displayMemoryAnalysis shows detailed memory usage analysis
func (cmd *BenchmarkCommand) displayMemoryAnalysis(analysis *benchmark.CNNMemoryAnalysis) {
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "\n🧠 Memory Analysis:")
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "─────────────────────────────────────")
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Feature Maps: %s", benchmark.FormatMemory(analysis.FeatureMapMemory))
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Kernels: %s", benchmark.FormatMemory(analysis.KernelMemory))
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Activations: %s", benchmark.FormatMemory(analysis.ActivationMemory))
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Pooling: %s", benchmark.FormatMemory(analysis.PoolingMemory))
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "FC Layers: %s", benchmark.FormatMemory(analysis.FCMemory))
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Total Memory: %s", benchmark.FormatMemory(analysis.TotalMemory))
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Batch Scale Factor: %dx", analysis.BatchScaleFactor)
}

// displayBatchProcessingMetrics shows batch processing efficiency metrics
func (cmd *BenchmarkCommand) displayBatchProcessingMetrics(metrics *benchmark.BatchProcessingMetrics) {
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "\n📦 Batch Processing Analysis:")
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "─────────────────────────────────────")
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Batch Size: %d", metrics.BatchSize)
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Throughput: %.2f FPS", metrics.ThroughputFPS)
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Latency per Item: %s", benchmark.FormatDuration(metrics.LatencyPerItem))
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Memory Overhead: %s", benchmark.FormatMemory(metrics.MemoryOverhead))
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "Efficiency: %.2fx", metrics.Efficiency)
}

// displayComparisonReport shows comparison results
func (cmd *BenchmarkCommand) displayComparisonReport(report benchmark.ComparisonReport) {
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "\n🔍 Performance Comparison:")
	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "─────────────────────────────────────")
	benchmark.PrintComparisonSummary(report)
}

// saveMetrics saves performance metrics to a JSON file
func (cmd *BenchmarkCommand) saveMetrics(metrics benchmark.PerformanceMetrics, outputPath string) error {
	result := benchmark.BenchmarkResult{
		Metrics:     metrics,
		Environment: benchmark.GetEnvironmentInfo(),
	}

	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal metrics: %w", err)
	}

	err = os.WriteFile(outputPath, data, 0600)
	if err != nil {
		return fmt.Errorf("failed to write metrics file: %w", err)
	}

	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "📄 Results saved to: %s", outputPath)
	return nil
}

// saveComparisonReport saves comparison report to a JSON file
func (cmd *BenchmarkCommand) saveComparisonReport(report benchmark.ComparisonReport, outputPath string) error {
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal comparison report: %w", err)
	}

	err = os.WriteFile(outputPath, data, 0600)
	if err != nil {
		return fmt.Errorf("failed to write comparison report file: %w", err)
	}

	cmd.outputWriter.WriteMessage(output.LogLevelInfo, "📄 Comparison report saved to: %s", outputPath)
	return nil
}
