// Package commands implements CLI command handlers
package commands

import (
	"context"
	"encoding/json"
	"fmt"
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
