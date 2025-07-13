package benchmark

import (
	"fmt"
	"runtime"
	"time"

	"github.com/nyasuto/bee/phase1"
)

// BenchmarkRunner executes performance benchmarks for neural network models
type BenchmarkRunner struct {
	iterations int
	warmupRuns int
	verbose    bool
}

// NewBenchmarkRunner creates a new benchmark runner with default settings
func NewBenchmarkRunner() *BenchmarkRunner {
	return &BenchmarkRunner{
		iterations: 100,
		warmupRuns: 10,
		verbose:    false,
	}
}

// SetIterations configures the number of benchmark iterations
func (br *BenchmarkRunner) SetIterations(iterations int) *BenchmarkRunner {
	br.iterations = iterations
	return br
}

// SetWarmupRuns configures the number of warmup runs
func (br *BenchmarkRunner) SetWarmupRuns(warmup int) *BenchmarkRunner {
	br.warmupRuns = warmup
	return br
}

// SetVerbose enables or disables verbose output
func (br *BenchmarkRunner) SetVerbose(verbose bool) *BenchmarkRunner {
	br.verbose = verbose
	return br
}

// BenchmarkPerceptron measures perceptron performance on given dataset
func (br *BenchmarkRunner) BenchmarkPerceptron(dataset Dataset) (PerformanceMetrics, error) {
	if br.verbose {
		fmt.Printf("🔍 Benchmarking Perceptron on %s dataset...\n", dataset.Name)
	}

	// Create perceptron
	perceptron := phase1.NewPerceptron(len(dataset.TrainInputs[0]), 0.1)

	// Measure memory before training
	var memBefore runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memBefore)

	// Measure training time
	startTime := time.Now()

	convergenceEpochs := 0
	maxEpochs := 1000

	for epoch := 0; epoch < maxEpochs; epoch++ {
		epochError := 0.0

		// Train on all examples
		for i := range dataset.TrainInputs {
			err := perceptron.Train(dataset.TrainInputs[i], dataset.TrainTargets[i][0])
			if err != nil {
				return PerformanceMetrics{}, fmt.Errorf("training failed: %w", err)
			}
		}

		// Calculate epoch accuracy
		correct := 0
		for i := range dataset.TrainInputs {
			output, _ := perceptron.Forward(dataset.TrainInputs[i])
			if (output >= 0.5 && dataset.TrainTargets[i][0] >= 0.5) ||
				(output < 0.5 && dataset.TrainTargets[i][0] < 0.5) {
				correct++
			} else {
				epochError += 1.0
			}
		}

		accuracy := float64(correct) / float64(len(dataset.TrainInputs))

		// Check convergence (perfect accuracy or no improvement)
		if accuracy >= 0.95 || epochError == 0 {
			convergenceEpochs = epoch + 1
			break
		}

		convergenceEpochs = epoch + 1
	}

	trainingTime := time.Since(startTime)

	// Measure memory after training
	var memAfter runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memAfter)
	memoryUsage := int64(memAfter.Alloc - memBefore.Alloc)

	// Measure inference time (multiple runs for accuracy)
	inferenceRuns := br.iterations
	if inferenceRuns < 100 {
		inferenceRuns = 100
	}

	// Warmup runs
	for i := 0; i < br.warmupRuns; i++ {
		perceptron.Forward(dataset.TestInputs[0])
	}

	startInference := time.Now()
	for i := 0; i < inferenceRuns; i++ {
		for j := range dataset.TestInputs {
			perceptron.Forward(dataset.TestInputs[j])
		}
	}
	totalInferenceTime := time.Since(startInference)
	avgInferenceTime := totalInferenceTime / time.Duration(inferenceRuns*len(dataset.TestInputs))

	// Calculate final accuracy on test set
	correct := 0
	totalLoss := 0.0
	for i := range dataset.TestInputs {
		output, _ := perceptron.Forward(dataset.TestInputs[i])
		if (output >= 0.5 && dataset.TestTargets[i][0] >= 0.5) ||
			(output < 0.5 && dataset.TestTargets[i][0] < 0.5) {
			correct++
		}
		// Calculate squared error loss
		diff := output - dataset.TestTargets[i][0]
		totalLoss += diff * diff
	}

	accuracy := float64(correct) / float64(len(dataset.TestInputs))
	avgLoss := totalLoss / float64(len(dataset.TestInputs))

	if br.verbose {
		fmt.Printf("  ✅ Training completed in %d epochs, %.2f%% accuracy\n",
			convergenceEpochs, accuracy*100)
	}

	return PerformanceMetrics{
		ModelType:       "perceptron",
		DatasetName:     dataset.Name,
		TrainingTime:    trainingTime,
		InferenceTime:   avgInferenceTime,
		MemoryUsage:     memoryUsage,
		Accuracy:        accuracy,
		ConvergenceRate: convergenceEpochs,
		FinalLoss:       avgLoss,
		Timestamp:       time.Now(),
	}, nil
}

// BenchmarkMLP measures MLP performance on given dataset
func (br *BenchmarkRunner) BenchmarkMLP(dataset Dataset, hiddenSizes []int) (PerformanceMetrics, error) {
	if br.verbose {
		fmt.Printf("🔍 Benchmarking MLP %v on %s dataset...\n", hiddenSizes, dataset.Name)
	}

	// Create MLP
	activations := make([]phase1.ActivationFunction, len(hiddenSizes)+1)
	for i := range activations {
		activations[i] = phase1.Sigmoid
	}

	mlp, err := phase1.NewMLP(len(dataset.TrainInputs[0]), hiddenSizes,
		len(dataset.TrainTargets[0]), activations, 0.5)
	if err != nil {
		return PerformanceMetrics{}, fmt.Errorf("failed to create MLP: %w", err)
	}

	// Measure memory before training
	var memBefore runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memBefore)

	// Measure training time
	startTime := time.Now()

	convergenceEpochs := 0
	maxEpochs := 2000

	for epoch := 0; epoch < maxEpochs; epoch++ {
		// Train on all examples
		for i := range dataset.TrainInputs {
			err := mlp.Train(dataset.TrainInputs[i], dataset.TrainTargets[i])
			if err != nil {
				return PerformanceMetrics{}, fmt.Errorf("training failed: %w", err)
			}
		}

		// Check convergence every 50 epochs
		if epoch%50 == 0 || epoch == maxEpochs-1 {
			correct := 0
			for i := range dataset.TrainInputs {
				output, _ := mlp.Predict(dataset.TrainInputs[i])
				predicted := 0.0
				if len(output) == 1 {
					if output[0] > 0.5 {
						predicted = 1.0
					}
				}

				expected := dataset.TrainTargets[i]
				if len(expected) == 1 {
					if (predicted >= 0.5 && expected[0] >= 0.5) ||
						(predicted < 0.5 && expected[0] < 0.5) {
						correct++
					}
				}
			}

			accuracy := float64(correct) / float64(len(dataset.TrainInputs))

			if accuracy >= 0.95 {
				convergenceEpochs = epoch + 1
				break
			}
		}

		convergenceEpochs = epoch + 1
	}

	trainingTime := time.Since(startTime)

	// Measure memory after training
	var memAfter runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memAfter)
	memoryUsage := int64(memAfter.Alloc - memBefore.Alloc)

	// Measure inference time
	inferenceRuns := br.iterations
	if inferenceRuns < 100 {
		inferenceRuns = 100
	}

	// Warmup runs
	for i := 0; i < br.warmupRuns; i++ {
		mlp.Predict(dataset.TestInputs[0])
	}

	startInference := time.Now()
	for i := 0; i < inferenceRuns; i++ {
		for j := range dataset.TestInputs {
			mlp.Predict(dataset.TestInputs[j])
		}
	}
	totalInferenceTime := time.Since(startInference)
	avgInferenceTime := totalInferenceTime / time.Duration(inferenceRuns*len(dataset.TestInputs))

	// Calculate final accuracy on test set
	correct := 0
	totalLoss := 0.0
	for i := range dataset.TestInputs {
		output, _ := mlp.Predict(dataset.TestInputs[i])

		predicted := 0.0
		if len(output) == 1 && output[0] > 0.5 {
			predicted = 1.0
		}

		expected := 0.0
		if len(dataset.TestTargets[i]) == 1 {
			expected = dataset.TestTargets[i][0]
		}

		if (predicted >= 0.5 && expected >= 0.5) ||
			(predicted < 0.5 && expected < 0.5) {
			correct++
		}

		// Calculate squared error loss
		if len(output) == 1 && len(dataset.TestTargets[i]) == 1 {
			diff := output[0] - dataset.TestTargets[i][0]
			totalLoss += diff * diff
		}
	}

	accuracy := float64(correct) / float64(len(dataset.TestInputs))
	avgLoss := totalLoss / float64(len(dataset.TestInputs))

	if br.verbose {
		fmt.Printf("  ✅ Training completed in %d epochs, %.2f%% accuracy\n",
			convergenceEpochs, accuracy*100)
	}

	return PerformanceMetrics{
		ModelType:       "mlp",
		DatasetName:     dataset.Name,
		TrainingTime:    trainingTime,
		InferenceTime:   avgInferenceTime,
		MemoryUsage:     memoryUsage,
		Accuracy:        accuracy,
		ConvergenceRate: convergenceEpochs,
		FinalLoss:       avgLoss,
		Timestamp:       time.Now(),
	}, nil
}

// RunComparison executes comparative benchmark between perceptron and MLP
func (br *BenchmarkRunner) RunComparison(dataset Dataset, mlpHidden []int) (ComparisonReport, error) {
	if br.verbose {
		fmt.Printf("🚀 Running comparative benchmark on %s dataset\n", dataset.Name)
		fmt.Println("─────────────────────────────────────────────────────")
	}

	// Benchmark Perceptron
	perceptronMetrics, err := br.BenchmarkPerceptron(dataset)
	if err != nil {
		return ComparisonReport{}, fmt.Errorf("perceptron benchmark failed: %w", err)
	}

	// Benchmark MLP
	mlpMetrics, err := br.BenchmarkMLP(dataset, mlpHidden)
	if err != nil {
		return ComparisonReport{}, fmt.Errorf("MLP benchmark failed: %w", err)
	}

	// Generate comparison report
	report := GenerateComparisonReport(perceptronMetrics, mlpMetrics)

	if br.verbose {
		fmt.Println("\n📊 Benchmark Results:")
		fmt.Printf("Perceptron: %.2f%% accuracy, %s training, %s inference\n",
			perceptronMetrics.Accuracy*100,
			FormatDuration(perceptronMetrics.TrainingTime),
			FormatDuration(perceptronMetrics.InferenceTime))
		fmt.Printf("MLP:        %.2f%% accuracy, %s training, %s inference\n",
			mlpMetrics.Accuracy*100,
			FormatDuration(mlpMetrics.TrainingTime),
			FormatDuration(mlpMetrics.InferenceTime))

		fmt.Println("\n" + "─────────────────────────────────────────────────────")
		PrintComparisonSummary(report)
	}

	return report, nil
}

// RunMultipleComparisons runs benchmarks across multiple datasets
func (br *BenchmarkRunner) RunMultipleComparisons(datasets []Dataset, mlpHidden []int) ([]ComparisonReport, error) {
	reports := make([]ComparisonReport, 0, len(datasets))

	for _, dataset := range datasets {
		report, err := br.RunComparison(dataset, mlpHidden)
		if err != nil {
			return nil, fmt.Errorf("comparison failed for dataset %s: %w", dataset.Name, err)
		}
		reports = append(reports, report)
	}

	return reports, nil
}
