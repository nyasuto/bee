// Package commands - RNN/LSTM Benchmarking CLI Commands
// Learning Goal: Understanding command-line interface for sequence model performance testing

package commands

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/nyasuto/bee/benchmark"
)

// RNNBenchmarkCommand handles RNN/LSTM benchmarking operations
// Learning Goal: Understanding sequence model performance evaluation through CLI
type RNNBenchmarkCommand struct {
	ModelType       string  // "rnn", "lstm", or "compare"
	InputSize       int     // Input vector dimensionality
	HiddenSize      int     // Hidden state dimensionality
	OutputSize      int     // Output vector dimensionality
	SequenceLengths []int   // Sequence lengths to test
	LearningRate    float64 // Learning rate for training
	MaxEpochs       int     // Maximum training epochs
	BatchSize       int     // Batch size for processing
	Iterations      int     // Number of benchmark iterations
	Verbose         bool    // Enable verbose output
	OutputDir       string  // Output directory for results
	SaveResults     bool    // Save results to files
	DatasetType     string  // Type of synthetic dataset ("sine", "fibonacci", "random")
}

// DefaultRNNBenchmarkCommand creates a default RNN benchmark command
func DefaultRNNBenchmarkCommand() *RNNBenchmarkCommand {
	return &RNNBenchmarkCommand{
		ModelType:       "compare",
		InputSize:       10,
		HiddenSize:      20,
		OutputSize:      5,
		SequenceLengths: []int{5, 10, 25, 50, 100},
		LearningRate:    0.01,
		MaxEpochs:       100,
		BatchSize:       32,
		Iterations:      100,
		Verbose:         true,
		OutputDir:       "benchmark_results",
		SaveResults:     true,
		DatasetType:     "sine",
	}
}

// Execute runs the RNN/LSTM benchmark command
// Learning Goal: Understanding end-to-end sequence model benchmarking
func (cmd *RNNBenchmarkCommand) Execute() error {
	fmt.Println("🐝 Bee Neural Network - RNN/LSTM Performance Benchmark")
	fmt.Println("═══════════════════════════════════════════════════════")

	// Create benchmark runner
	runner := benchmark.NewBenchmarkRunner().
		SetIterations(cmd.Iterations).
		SetVerbose(cmd.Verbose)

	// Create benchmark configuration
	config := benchmark.RNNBenchmarkConfig{
		InputSize:       cmd.InputSize,
		HiddenSize:      cmd.HiddenSize,
		OutputSize:      cmd.OutputSize,
		SequenceLengths: cmd.SequenceLengths,
		LearningRate:    cmd.LearningRate,
		MaxEpochs:       cmd.MaxEpochs,
		BatchSize:       cmd.BatchSize,
		GradientClip:    1.0,
	}

	if cmd.Verbose {
		fmt.Printf("📋 Benchmark Configuration:\n")
		fmt.Printf("   Model Type: %s\n", cmd.ModelType)
		fmt.Printf("   Architecture: %d→%d→%d\n", cmd.InputSize, cmd.HiddenSize, cmd.OutputSize)
		fmt.Printf("   Sequence Lengths: %v\n", cmd.SequenceLengths)
		fmt.Printf("   Learning Rate: %.4f\n", cmd.LearningRate)
		fmt.Printf("   Max Epochs: %d\n", cmd.MaxEpochs)
		fmt.Printf("   Benchmark Iterations: %d\n", cmd.Iterations)
		fmt.Printf("   Dataset Type: %s\n", cmd.DatasetType)
		fmt.Println()
	}

	// Create output directory if saving results
	if cmd.SaveResults {
		err := os.MkdirAll(cmd.OutputDir, 0755)
		if err != nil {
			return fmt.Errorf("failed to create output directory: %w", err)
		}
	}

	// Execute benchmark based on model type
	switch cmd.ModelType {
	case "rnn":
		return cmd.benchmarkRNN(runner, config)
	case "lstm":
		return cmd.benchmarkLSTM(runner, config)
	case "compare":
		return cmd.benchmarkComparison(runner, config)
	case "gradient":
		return cmd.benchmarkGradientFlow(runner, config)
	default:
		return fmt.Errorf("unsupported model type: %s (use 'rnn', 'lstm', 'compare', or 'gradient')", cmd.ModelType)
	}
}

// benchmarkRNN performs RNN-only benchmarking
// Learning Goal: Understanding RNN performance characteristics
func (cmd *RNNBenchmarkCommand) benchmarkRNN(runner *benchmark.BenchmarkRunner, config benchmark.RNNBenchmarkConfig) error {
	fmt.Println("🔬 Running RNN Performance Analysis")
	fmt.Println("──────────────────────────────────────")

	report, err := runner.BenchmarkRNN(config)
	if err != nil {
		return fmt.Errorf("RNN benchmark failed: %w", err)
	}

	// Display results
	cmd.displayRNNReport(report)

	// Save results if requested
	if cmd.SaveResults {
		filename := filepath.Join(cmd.OutputDir, fmt.Sprintf("rnn_benchmark_%d.json", time.Now().Unix()))
		err := cmd.saveRNNReport(report, filename)
		if err != nil {
			fmt.Printf("Warning: Failed to save results: %v\n", err)
		} else {
			fmt.Printf("📄 Results saved to: %s\n", filename)
		}
	}

	return nil
}

// benchmarkLSTM performs LSTM-only benchmarking
// Learning Goal: Understanding LSTM performance characteristics
func (cmd *RNNBenchmarkCommand) benchmarkLSTM(runner *benchmark.BenchmarkRunner, config benchmark.RNNBenchmarkConfig) error {
	fmt.Println("🔬 Running LSTM Performance Analysis")
	fmt.Println("───────────────────────────────────────")

	report, err := runner.BenchmarkLSTM(config)
	if err != nil {
		return fmt.Errorf("LSTM benchmark failed: %w", err)
	}

	// Display results
	cmd.displayRNNReport(report)

	// Save results if requested
	if cmd.SaveResults {
		filename := filepath.Join(cmd.OutputDir, fmt.Sprintf("lstm_benchmark_%d.json", time.Now().Unix()))
		err := cmd.saveRNNReport(report, filename)
		if err != nil {
			fmt.Printf("Warning: Failed to save results: %v\n", err)
		} else {
			fmt.Printf("📄 Results saved to: %s\n", filename)
		}
	}

	return nil
}

// benchmarkComparison performs RNN vs LSTM comparison
// Learning Goal: Understanding architectural trade-offs in sequence processing
func (cmd *RNNBenchmarkCommand) benchmarkComparison(runner *benchmark.BenchmarkRunner, config benchmark.RNNBenchmarkConfig) error {
	fmt.Println("⚖️  Running RNN vs LSTM Comparative Analysis")
	fmt.Println("──────────────────────────────────────────────")

	comparison, err := runner.CompareRNNvsLSTM(config)
	if err != nil {
		return fmt.Errorf("RNN vs LSTM comparison failed: %w", err)
	}

	// Display comparison results
	cmd.displayComparison(comparison)

	// Save results if requested
	if cmd.SaveResults {
		filename := filepath.Join(cmd.OutputDir, fmt.Sprintf("rnn_lstm_comparison_%d.json", time.Now().Unix()))
		err := cmd.saveComparison(comparison, filename)
		if err != nil {
			fmt.Printf("Warning: Failed to save comparison results: %v\n", err)
		} else {
			fmt.Printf("📄 Comparison results saved to: %s\n", filename)
		}
	}

	return nil
}

// benchmarkGradientFlow performs gradient flow analysis
// Learning Goal: Understanding gradient vanishing/exploding problems
func (cmd *RNNBenchmarkCommand) benchmarkGradientFlow(runner *benchmark.BenchmarkRunner, config benchmark.RNNBenchmarkConfig) error {
	fmt.Println("🌊 Running Gradient Flow Analysis")
	fmt.Println("─────────────────────────────────────")

	gradients, err := runner.RunGradientFlowAnalysis(config)
	if err != nil {
		return fmt.Errorf("gradient flow analysis failed: %w", err)
	}

	// Display gradient analysis
	cmd.displayGradientAnalysis(gradients, config.SequenceLengths)

	// Save results if requested
	if cmd.SaveResults {
		filename := filepath.Join(cmd.OutputDir, fmt.Sprintf("gradient_analysis_%d.json", time.Now().Unix()))
		err := cmd.saveGradientAnalysis(gradients, filename)
		if err != nil {
			fmt.Printf("Warning: Failed to save gradient analysis: %v\n", err)
		} else {
			fmt.Printf("📄 Gradient analysis saved to: %s\n", filename)
		}
	}

	return nil
}

// displayRNNReport displays RNN/LSTM performance report
func (cmd *RNNBenchmarkCommand) displayRNNReport(report benchmark.RNNPerformanceReport) {
	fmt.Printf("📊 %s Performance Report\n", strings.ToUpper(report.ModelType))
	fmt.Println("══════════════════════════════════════")

	fmt.Printf("🎯 Overall Metrics:\n")
	fmt.Printf("   Scalability Score: %.3f\n", report.ScalabilityScore)
	fmt.Printf("   Gradient Stability: %.4f\n", report.GradientStability)
	fmt.Printf("   Max Sequence Length: %d\n", report.MaxSequenceLength)
	fmt.Printf("   Memory Scaling Rate: %.3f\n", report.MemoryScalingRate)
	fmt.Println()

	fmt.Printf("📏 Per-Sequence-Length Metrics:\n")
	fmt.Println("   Length │ Forward Time │ Memory Usage │ Gradient Norm │ Accuracy │ Convergence")
	fmt.Println("   ───────┼──────────────┼──────────────┼───────────────┼──────────┼────────────")

	for _, metric := range report.SequenceMetrics {
		fmt.Printf("   %6d │ %11s │ %11s │ %12.4f │ %7.2f%% │ %10d\n",
			metric.SequenceLength,
			formatDurationShort(metric.ForwardTime),
			formatMemoryShort(metric.MemoryUsage),
			metric.GradientNorm,
			metric.FinalAccuracy*100,
			metric.ConvergenceEpoch)
	}
	fmt.Println()
}

// displayComparison displays RNN vs LSTM comparison results
func (cmd *RNNBenchmarkCommand) displayComparison(comparison benchmark.RNNComparison) {
	fmt.Println("⚔️  RNN vs LSTM Battle Results")
	fmt.Println("══════════════════════════════════════")

	// Display individual model summaries
	fmt.Printf("🔴 RNN Summary:\n")
	fmt.Printf("   Scalability: %.3f │ Gradient Stability: %.4f │ Max Length: %d\n",
		comparison.RNNReport.ScalabilityScore,
		comparison.RNNReport.GradientStability,
		comparison.RNNReport.MaxSequenceLength)

	fmt.Printf("🔵 LSTM Summary:\n")
	fmt.Printf("   Scalability: %.3f │ Gradient Stability: %.4f │ Max Length: %d\n",
		comparison.LSTMReport.ScalabilityScore,
		comparison.LSTMReport.GradientStability,
		comparison.LSTMReport.MaxSequenceLength)
	fmt.Println()

	// Display improvements
	if len(comparison.Improvements) > 0 {
		fmt.Printf("✅ LSTM Advantages:\n")
		for metric, improvement := range comparison.Improvements {
			fmt.Printf("   • %s: +%.2f%%\n", strings.ReplaceAll(metric, "_", " "), improvement)
		}
		fmt.Println()
	}

	// Display trade-offs
	if len(comparison.TradeOffs) > 0 {
		fmt.Printf("⚖️  RNN Advantages:\n")
		for metric, advantage := range comparison.TradeOffs {
			fmt.Printf("   • %s: +%.2f%%\n", strings.ReplaceAll(metric, "_", " "), advantage)
		}
		fmt.Println()
	}

	// Display recommendation
	fmt.Printf("💡 Recommendation:\n")
	fmt.Printf("   %s\n", comparison.Recommendation)
	fmt.Println()
}

// displayGradientAnalysis displays gradient flow analysis results
func (cmd *RNNBenchmarkCommand) displayGradientAnalysis(gradients map[string][]float64, sequenceLengths []int) {
	fmt.Println("🌊 Gradient Flow Analysis Results")
	fmt.Println("═══════════════════════════════════════")

	for modelType, gradientNorms := range gradients {
		fmt.Printf("📈 %s Gradient Norms by Sequence Length:\n", strings.ToUpper(modelType))
		fmt.Println("   Length │ Gradient Norm │ Flow Quality")
		fmt.Println("   ───────┼───────────────┼─────────────")

		for i, norm := range gradientNorms {
			length := sequenceLengths[i]
			quality := cmd.assessGradientQuality(norm)
			fmt.Printf("   %6d │ %12.6f │ %s\n", length, norm, quality)
		}
		fmt.Println()
	}

	// Gradient flow recommendations
	fmt.Printf("🎯 Gradient Flow Assessment:\n")
	for modelType, gradientNorms := range gradients {
		avgNorm := cmd.calculateAverageGradient(gradientNorms)
		stability := cmd.calculateGradientStability(gradientNorms)

		fmt.Printf("   %s: Avg Norm=%.6f, Stability=%.3f (%s)\n",
			strings.ToUpper(modelType), avgNorm, stability, cmd.assessOverallGradientHealth(avgNorm, stability))
	}
}

// Helper functions for formatting and analysis

func formatDurationShort(d time.Duration) string {
	if d < time.Microsecond {
		return fmt.Sprintf("%.0f ns", float64(d.Nanoseconds()))
	} else if d < time.Millisecond {
		return fmt.Sprintf("%.1f μs", float64(d.Nanoseconds())/1000)
	} else {
		return fmt.Sprintf("%.1f ms", float64(d.Nanoseconds())/1000000)
	}
}

func formatMemoryShort(bytes int64) string {
	if bytes < 1024 {
		return fmt.Sprintf("%d B", bytes)
	} else if bytes < 1024*1024 {
		return fmt.Sprintf("%.1f KB", float64(bytes)/1024)
	} else {
		return fmt.Sprintf("%.1f MB", float64(bytes)/(1024*1024))
	}
}

func (cmd *RNNBenchmarkCommand) assessGradientQuality(norm float64) string {
	if norm < 0.001 {
		return "🔴 Vanishing"
	} else if norm > 10.0 {
		return "🟠 Exploding"
	} else if norm > 1.0 {
		return "🟡 High"
	} else {
		return "🟢 Good"
	}
}

func (cmd *RNNBenchmarkCommand) calculateAverageGradient(gradients []float64) float64 {
	if len(gradients) == 0 {
		return 0
	}
	sum := 0.0
	for _, g := range gradients {
		sum += g
	}
	return sum / float64(len(gradients))
}

func (cmd *RNNBenchmarkCommand) calculateGradientStability(gradients []float64) float64 {
	if len(gradients) <= 1 {
		return 1.0
	}

	mean := cmd.calculateAverageGradient(gradients)
	variance := 0.0
	for _, g := range gradients {
		diff := g - mean
		variance += diff * diff
	}
	variance /= float64(len(gradients))

	// Stability = 1 / (1 + coefficient_of_variation)
	cv := variance / (mean * mean)
	return 1.0 / (1.0 + cv)
}

func (cmd *RNNBenchmarkCommand) assessOverallGradientHealth(avgNorm, stability float64) string {
	if avgNorm < 0.01 || avgNorm > 5.0 {
		return "Poor"
	} else if stability < 0.5 {
		return "Unstable"
	} else if stability > 0.8 && avgNorm > 0.1 && avgNorm < 2.0 {
		return "Excellent"
	} else {
		return "Good"
	}
}

// Save functions

func (cmd *RNNBenchmarkCommand) saveRNNReport(report benchmark.RNNPerformanceReport, filename string) error {
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0600)
}

func (cmd *RNNBenchmarkCommand) saveComparison(comparison benchmark.RNNComparison, filename string) error {
	data, err := json.MarshalIndent(comparison, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0600)
}

func (cmd *RNNBenchmarkCommand) saveGradientAnalysis(gradients map[string][]float64, filename string) error {
	data, err := json.MarshalIndent(gradients, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0600)
}

// ParseRNNBenchmarkArgs parses command line arguments for RNN benchmarking
// Learning Goal: Understanding CLI argument parsing for complex benchmark configurations
func ParseRNNBenchmarkArgs(args []string) (*RNNBenchmarkCommand, error) {
	cmd := DefaultRNNBenchmarkCommand()

	for i := 0; i < len(args); i++ {
		arg := args[i]

		switch arg {
		case "--model", "-m":
			if i+1 < len(args) {
				cmd.ModelType = args[i+1]
				i++
			}
		case "--input-size":
			if i+1 < len(args) {
				size, err := strconv.Atoi(args[i+1])
				if err == nil && size > 0 {
					cmd.InputSize = size
				}
				i++
			}
		case "--hidden-size":
			if i+1 < len(args) {
				size, err := strconv.Atoi(args[i+1])
				if err == nil && size > 0 {
					cmd.HiddenSize = size
				}
				i++
			}
		case "--output-size":
			if i+1 < len(args) {
				size, err := strconv.Atoi(args[i+1])
				if err == nil && size > 0 {
					cmd.OutputSize = size
				}
				i++
			}
		case "--sequence-lengths":
			if i+1 < len(args) {
				lengthsStr := args[i+1]
				lengths, err := parseIntList(lengthsStr)
				if err == nil && len(lengths) > 0 {
					cmd.SequenceLengths = lengths
				}
				i++
			}
		case "--learning-rate":
			if i+1 < len(args) {
				rate, err := strconv.ParseFloat(args[i+1], 64)
				if err == nil && rate > 0 {
					cmd.LearningRate = rate
				}
				i++
			}
		case "--max-epochs":
			if i+1 < len(args) {
				epochs, err := strconv.Atoi(args[i+1])
				if err == nil && epochs > 0 {
					cmd.MaxEpochs = epochs
				}
				i++
			}
		case "--iterations":
			if i+1 < len(args) {
				iters, err := strconv.Atoi(args[i+1])
				if err == nil && iters > 0 {
					cmd.Iterations = iters
				}
				i++
			}
		case "--output-dir":
			if i+1 < len(args) {
				cmd.OutputDir = args[i+1]
				i++
			}
		case "--dataset":
			if i+1 < len(args) {
				cmd.DatasetType = args[i+1]
				i++
			}
		case "--verbose", "-v":
			cmd.Verbose = true
		case "--quiet", "-q":
			cmd.Verbose = false
		case "--no-save":
			cmd.SaveResults = false
		case "--help", "-h":
			fmt.Println(getRNNBenchmarkUsage())
			return nil, fmt.Errorf("help requested")
		}
	}

	return cmd, nil
}

// parseIntList parses comma-separated integer list
func parseIntList(str string) ([]int, error) {
	parts := strings.Split(str, ",")
	result := make([]int, 0, len(parts))

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		val, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("invalid integer: %s", part)
		}
		result = append(result, val)
	}

	return result, nil
}

// getRNNBenchmarkUsage returns usage information for RNN benchmarking
func getRNNBenchmarkUsage() string {
	return `Usage: bee benchmark rnn [OPTIONS]

Benchmark RNN/LSTM performance across different sequence lengths

OPTIONS:
  -m, --model TYPE          Model type: rnn, lstm, compare, gradient (default: compare)
  --input-size SIZE         Input vector size (default: 10)
  --hidden-size SIZE        Hidden state size (default: 20)
  --output-size SIZE        Output vector size (default: 5)
  --sequence-lengths LIST   Comma-separated sequence lengths (default: 5,10,25,50,100)
  --learning-rate RATE      Learning rate (default: 0.01)
  --max-epochs NUM          Maximum training epochs (default: 100)
  --iterations NUM          Benchmark iterations (default: 100)
  --output-dir DIR          Output directory for results (default: benchmark_results)
  --dataset TYPE            Dataset type: sine, fibonacci, random (default: sine)
  -v, --verbose             Enable verbose output
  -q, --quiet               Disable verbose output
  --no-save                 Don't save results to files
  -h, --help                Show this help message

EXAMPLES:
  bee benchmark rnn --model rnn --sequence-lengths 10,50,100
  bee benchmark rnn --model compare --verbose
  bee benchmark rnn --model gradient --sequence-lengths 5,10,25,50
  bee benchmark rnn --model lstm --hidden-size 50 --max-epochs 200`
}
