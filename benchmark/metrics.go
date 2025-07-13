// Package benchmark provides performance measurement and comparison tools
// for neural network implementations
package benchmark

import (
	"encoding/json"
	"fmt"
	"runtime"
	"time"
)

// PerformanceMetrics represents comprehensive performance measurements
// Mathematical Foundation: Quantitative evaluation of neural network efficiency
type PerformanceMetrics struct {
	ModelType       string        `json:"model_type"`       // "perceptron" or "mlp"
	DatasetName     string        `json:"dataset_name"`     // e.g., "xor", "and", "or"
	TrainingTime    time.Duration `json:"training_time"`    // Time to train the model
	InferenceTime   time.Duration `json:"inference_time"`   // Average time per prediction
	MemoryUsage     int64         `json:"memory_usage"`     // Memory usage in bytes
	Accuracy        float64       `json:"accuracy"`         // Final accuracy (0-1)
	ConvergenceRate int           `json:"convergence_rate"` // Epochs to converge
	FinalLoss       float64       `json:"final_loss"`       // Final training loss
	Timestamp       time.Time     `json:"timestamp"`        // When the benchmark was run
}

// BenchmarkResult represents a single benchmark execution result
type BenchmarkResult struct {
	Metrics       PerformanceMetrics `json:"metrics"`
	Configuration map[string]any     `json:"configuration"` // Model hyperparameters
	Environment   EnvironmentInfo    `json:"environment"`   // System information
}

// EnvironmentInfo captures system information for reproducible benchmarks
type EnvironmentInfo struct {
	GoVersion    string `json:"go_version"`
	OS           string `json:"os"`
	Architecture string `json:"architecture"`
	CPUCores     int    `json:"cpu_cores"`
	Hostname     string `json:"hostname,omitempty"`
}

// ComparisonReport represents a comparative analysis between models
type ComparisonReport struct {
	BaselineModel    string                     `json:"baseline_model"`
	ComparisonModel  string                     `json:"comparison_model"`
	Dataset          string                     `json:"dataset"`
	Improvements     map[string]float64         `json:"improvements"` // Percentage improvements
	Degradations     map[string]float64         `json:"degradations"` // Percentage degradations
	StatisticalTests map[string]StatisticalTest `json:"statistical_tests"`
	Timestamp        time.Time                  `json:"timestamp"`
}

// StatisticalTest represents statistical significance testing results
type StatisticalTest struct {
	TestType    string  `json:"test_type"`   // e.g., "t-test", "wilcoxon"
	PValue      float64 `json:"p_value"`     // Statistical p-value
	Significant bool    `json:"significant"` // Is the difference significant?
	EffectSize  float64 `json:"effect_size"` // Magnitude of the difference
}

// MetricComparison provides detailed comparison of a specific metric
type MetricComparison struct {
	MetricName       string  `json:"metric_name"`
	BaselineValue    float64 `json:"baseline_value"`
	ComparisonValue  float64 `json:"comparison_value"`
	PercentChange    float64 `json:"percent_change"`
	AbsoluteChange   float64 `json:"absolute_change"`
	ImprovementRatio float64 `json:"improvement_ratio"`
}

// GetEnvironmentInfo captures current system environment
func GetEnvironmentInfo() EnvironmentInfo {
	return EnvironmentInfo{
		GoVersion:    runtime.Version(),
		OS:           runtime.GOOS,
		Architecture: runtime.GOARCH,
		CPUCores:     runtime.NumCPU(),
	}
}

// CalculateSpeedup computes the speedup ratio between two time measurements
// Mathematical: speedup = baseline_time / comparison_time
func CalculateSpeedup(baselineTime, comparisonTime time.Duration) float64 {
	if comparisonTime == 0 {
		return 0
	}
	return float64(baselineTime) / float64(comparisonTime)
}

// CalculateMemoryEfficiency computes memory efficiency improvement
// Mathematical: efficiency = (baseline_memory - comparison_memory) / baseline_memory * 100
func CalculateMemoryEfficiency(baselineMemory, comparisonMemory int64) float64 {
	if baselineMemory == 0 {
		return 0
	}
	return float64(baselineMemory-comparisonMemory) / float64(baselineMemory) * 100
}

// CalculateAccuracyImprovement computes accuracy improvement
// Mathematical: improvement = (comparison_accuracy - baseline_accuracy) * 100
func CalculateAccuracyImprovement(baselineAccuracy, comparisonAccuracy float64) float64 {
	return (comparisonAccuracy - baselineAccuracy) * 100
}

// CompareMetrics provides detailed comparison between two performance metrics
func CompareMetrics(baseline, comparison PerformanceMetrics) map[string]MetricComparison {
	comparisons := make(map[string]MetricComparison)

	// Training Time Comparison
	comparisons["training_time"] = MetricComparison{
		MetricName:       "Training Time",
		BaselineValue:    float64(baseline.TrainingTime.Milliseconds()),
		ComparisonValue:  float64(comparison.TrainingTime.Milliseconds()),
		PercentChange:    (float64(comparison.TrainingTime-baseline.TrainingTime) / float64(baseline.TrainingTime)) * 100,
		AbsoluteChange:   float64(comparison.TrainingTime - baseline.TrainingTime),
		ImprovementRatio: CalculateSpeedup(baseline.TrainingTime, comparison.TrainingTime),
	}

	// Inference Time Comparison
	comparisons["inference_time"] = MetricComparison{
		MetricName:       "Inference Time",
		BaselineValue:    float64(baseline.InferenceTime.Nanoseconds()),
		ComparisonValue:  float64(comparison.InferenceTime.Nanoseconds()),
		PercentChange:    (float64(comparison.InferenceTime-baseline.InferenceTime) / float64(baseline.InferenceTime)) * 100,
		AbsoluteChange:   float64(comparison.InferenceTime - baseline.InferenceTime),
		ImprovementRatio: CalculateSpeedup(baseline.InferenceTime, comparison.InferenceTime),
	}

	// Memory Usage Comparison
	comparisons["memory_usage"] = MetricComparison{
		MetricName:      "Memory Usage",
		BaselineValue:   float64(baseline.MemoryUsage),
		ComparisonValue: float64(comparison.MemoryUsage),
		PercentChange:   (float64(comparison.MemoryUsage-baseline.MemoryUsage) / float64(baseline.MemoryUsage)) * 100,
		AbsoluteChange:  float64(comparison.MemoryUsage - baseline.MemoryUsage),
	}

	// Accuracy Comparison
	comparisons["accuracy"] = MetricComparison{
		MetricName:      "Accuracy",
		BaselineValue:   baseline.Accuracy * 100, // Convert to percentage
		ComparisonValue: comparison.Accuracy * 100,
		PercentChange:   CalculateAccuracyImprovement(baseline.Accuracy, comparison.Accuracy),
		AbsoluteChange:  (comparison.Accuracy - baseline.Accuracy) * 100,
	}

	// Convergence Rate Comparison
	if baseline.ConvergenceRate > 0 && comparison.ConvergenceRate > 0 {
		comparisons["convergence_rate"] = MetricComparison{
			MetricName:       "Convergence Rate",
			BaselineValue:    float64(baseline.ConvergenceRate),
			ComparisonValue:  float64(comparison.ConvergenceRate),
			PercentChange:    (float64(comparison.ConvergenceRate-baseline.ConvergenceRate) / float64(baseline.ConvergenceRate)) * 100,
			AbsoluteChange:   float64(comparison.ConvergenceRate - baseline.ConvergenceRate),
			ImprovementRatio: float64(baseline.ConvergenceRate) / float64(comparison.ConvergenceRate), // Lower is better
		}
	}

	return comparisons
}

// GenerateComparisonReport creates a comprehensive comparison report
func GenerateComparisonReport(baseline, comparison PerformanceMetrics) ComparisonReport {
	comparisons := CompareMetrics(baseline, comparison)
	improvements := make(map[string]float64)
	degradations := make(map[string]float64)

	for metricName, comp := range comparisons {
		// For time and convergence metrics, negative change is improvement
		if metricName == "training_time" || metricName == "inference_time" || metricName == "convergence_rate" || metricName == "memory_usage" {
			if comp.PercentChange < 0 {
				improvements[metricName] = -comp.PercentChange // Make positive for display
			} else {
				degradations[metricName] = comp.PercentChange
			}
		} else {
			// For accuracy and other metrics, positive change is improvement
			if comp.PercentChange > 0 {
				improvements[metricName] = comp.PercentChange
			} else {
				degradations[metricName] = -comp.PercentChange // Make positive for display
			}
		}
	}

	return ComparisonReport{
		BaselineModel:   baseline.ModelType,
		ComparisonModel: comparison.ModelType,
		Dataset:         baseline.DatasetName,
		Improvements:    improvements,
		Degradations:    degradations,
		Timestamp:       time.Now(),
	}
}

// FormatDuration formats a duration for human-readable display
func FormatDuration(d time.Duration) string {
	if d < time.Microsecond {
		return fmt.Sprintf("%.2f ns", float64(d.Nanoseconds()))
	} else if d < time.Millisecond {
		return fmt.Sprintf("%.2f μs", float64(d.Nanoseconds())/1000)
	} else if d < time.Second {
		return fmt.Sprintf("%.2f ms", float64(d.Nanoseconds())/1000000)
	} else {
		return fmt.Sprintf("%.2f s", d.Seconds())
	}
}

// FormatMemory formats memory usage for human-readable display
func FormatMemory(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

// PrintComparisonSummary displays a formatted comparison summary
func PrintComparisonSummary(report ComparisonReport) {
	fmt.Printf("🔍 Performance Comparison: %s vs %s (Dataset: %s)\n",
		report.BaselineModel, report.ComparisonModel, report.Dataset)
	fmt.Println("─────────────────────────────────────────────────────")

	// Print improvements
	if len(report.Improvements) > 0 {
		fmt.Println("✅ Improvements:")
		for metric, improvement := range report.Improvements {
			fmt.Printf("  • %s: +%.2f%%\n", metric, improvement)
		}
	}

	// Print degradations
	if len(report.Degradations) > 0 {
		fmt.Println("\n❌ Degradations:")
		for metric, degradation := range report.Degradations {
			fmt.Printf("  • %s: -%.2f%%\n", metric, degradation)
		}
	}

	fmt.Printf("\n📅 Generated: %s\n", report.Timestamp.Format("2006-01-02 15:04:05"))
}

// SaveBenchmarkResult saves benchmark results to JSON file
func SaveBenchmarkResult(result BenchmarkResult, filename string) error {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal benchmark result: %w", err)
	}

	return writeFile(filename, data)
}

// LoadBenchmarkResult loads benchmark results from JSON file
func LoadBenchmarkResult(filename string) (BenchmarkResult, error) {
	var result BenchmarkResult

	data, err := readFile(filename)
	if err != nil {
		return result, fmt.Errorf("failed to read benchmark file: %w", err)
	}

	err = json.Unmarshal(data, &result)
	if err != nil {
		return result, fmt.Errorf("failed to unmarshal benchmark result: %w", err)
	}

	return result, nil
}

// writeFile and readFile are placeholder functions for file I/O
// In real implementation, these would use os.WriteFile and os.ReadFile
func writeFile(filename string, data []byte) error {
	// Implementation would use os.WriteFile(filename, data, 0644)
	return nil // Placeholder
}

func readFile(filename string) ([]byte, error) {
	// Implementation would use os.ReadFile(filename)
	return nil, nil // Placeholder
}
