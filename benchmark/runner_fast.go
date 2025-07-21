//go:build !slow
// +build !slow

package benchmark

import (
	"fmt"
	"runtime"
	"time"

	"github.com/nyasuto/bee/phase1"
)

// FastBenchmarkPerceptron is an optimized version for CI/CD with reduced epochs
func (br *BenchmarkRunner) FastBenchmarkPerceptron(dataset Dataset) (PerformanceMetrics, error) {
	if br.verbose {
		fmt.Printf("🔍 Fast Benchmarking Perceptron on %s dataset...\n", dataset.Name)
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
	maxEpochs := 50 // Reduced from 1000 for fast testing

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
			output, err := perceptron.Forward(dataset.TrainInputs[i])
			if err != nil {
				continue // Skip failed predictions in benchmark
			}
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
	// Safe conversion with overflow check
	memoryUsage := int64(0)
	if memAfter.Alloc >= memBefore.Alloc {
		diff := memAfter.Alloc - memBefore.Alloc
		if diff <= uint64(^uint64(0)>>1) { // Check if fits in int64
			memoryUsage = int64(diff)
		}
	} else {
		// Negative memory usage (memory was freed)
		diff := memBefore.Alloc - memAfter.Alloc
		if diff <= uint64(^uint64(0)>>1) {
			memoryUsage = -int64(diff)
		}
	}

	// Measure inference time (fewer runs for speed)
	inferenceRuns := 10 // Reduced from 100+
	if br.iterations > 0 && br.iterations < 10 {
		inferenceRuns = br.iterations
	}

	// Minimal warmup runs
	warmupRuns := 1
	if br.warmupRuns > 0 && br.warmupRuns < 3 {
		warmupRuns = br.warmupRuns
	}

	// Warmup runs
	for i := 0; i < warmupRuns; i++ {
		_, _ = perceptron.Forward(dataset.TestInputs[0]) //nolint:errcheck // Ignore errors in warmup
	}

	startInference := time.Now()
	for i := 0; i < inferenceRuns; i++ {
		for j := range dataset.TestInputs {
			_, _ = perceptron.Forward(dataset.TestInputs[j]) //nolint:errcheck // Ignore errors in benchmark timing
		}
	}
	totalInferenceTime := time.Since(startInference)
	avgInferenceTime := totalInferenceTime / time.Duration(inferenceRuns*len(dataset.TestInputs))

	// Calculate final accuracy on test set
	correct := 0
	totalLoss := 0.0
	for i := range dataset.TestInputs {
		output, err := perceptron.Forward(dataset.TestInputs[i])
		if err != nil {
			continue // Skip failed predictions
		}
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
		fmt.Printf("  ✅ Fast training completed in %d epochs, %.2f%% accuracy\n",
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

// FastBenchmarkMLP is an optimized version for CI/CD with reduced epochs
func (br *BenchmarkRunner) FastBenchmarkMLP(dataset Dataset, hiddenSizes []int) (PerformanceMetrics, error) {
	if br.verbose {
		fmt.Printf("🔍 Fast Benchmarking MLP %v on %s dataset...\n", hiddenSizes, dataset.Name)
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
	maxEpochs := 100 // Reduced from 2000 for fast testing

	for epoch := 0; epoch < maxEpochs; epoch++ {
		// Train on all examples
		for i := range dataset.TrainInputs {
			err := mlp.Train(dataset.TrainInputs[i], dataset.TrainTargets[i])
			if err != nil {
				return PerformanceMetrics{}, fmt.Errorf("training failed: %w", err)
			}
		}

		// Check convergence every 10 epochs (reduced from 50)
		if epoch%10 == 0 || epoch == maxEpochs-1 {
			correct := 0
			for i := range dataset.TrainInputs {
				output, err := mlp.Predict(dataset.TrainInputs[i])
				if err != nil {
					continue // Skip failed predictions
				}
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
	// Safe conversion with overflow check
	memoryUsage := int64(0)
	if memAfter.Alloc >= memBefore.Alloc {
		diff := memAfter.Alloc - memBefore.Alloc
		if diff <= uint64(^uint64(0)>>1) { // Check if fits in int64
			memoryUsage = int64(diff)
		}
	} else {
		// Negative memory usage (memory was freed)
		diff := memBefore.Alloc - memAfter.Alloc
		if diff <= uint64(^uint64(0)>>1) {
			memoryUsage = -int64(diff)
		}
	}

	// Measure inference time (fewer runs)
	inferenceRuns := 10 // Reduced from 100+
	if br.iterations > 0 && br.iterations < 10 {
		inferenceRuns = br.iterations
	}

	// Minimal warmup runs
	warmupRuns := 1
	if br.warmupRuns > 0 && br.warmupRuns < 3 {
		warmupRuns = br.warmupRuns
	}

	// Warmup runs
	for i := 0; i < warmupRuns; i++ {
		_, _ = mlp.Predict(dataset.TestInputs[0]) //nolint:errcheck // Ignore errors in warmup
	}

	startInference := time.Now()
	for i := 0; i < inferenceRuns; i++ {
		for j := range dataset.TestInputs {
			_, _ = mlp.Predict(dataset.TestInputs[j]) //nolint:errcheck // Ignore errors in benchmark timing
		}
	}
	totalInferenceTime := time.Since(startInference)
	avgInferenceTime := totalInferenceTime / time.Duration(inferenceRuns*len(dataset.TestInputs))

	// Calculate final accuracy on test set
	correct := 0
	totalLoss := 0.0
	for i := range dataset.TestInputs {
		output, err := mlp.Predict(dataset.TestInputs[i])
		if err != nil {
			continue // Skip failed predictions
		}

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
		fmt.Printf("  ✅ Fast training completed in %d epochs, %.2f%% accuracy\n",
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

// FastRunComparison executes comparative benchmark with reduced training time
func (br *BenchmarkRunner) FastRunComparison(dataset Dataset, mlpHidden []int) (ComparisonReport, error) {
	if br.verbose {
		fmt.Printf("🚀 Fast Running comparative benchmark on %s dataset\n", dataset.Name)
		fmt.Println("─────────────────────────────────────────────────────")
	}

	// Benchmark Perceptron (fast version)
	perceptronMetrics, err := br.FastBenchmarkPerceptron(dataset)
	if err != nil {
		return ComparisonReport{}, fmt.Errorf("perceptron benchmark failed: %w", err)
	}

	// Benchmark MLP (fast version)
	mlpMetrics, err := br.FastBenchmarkMLP(dataset, mlpHidden)
	if err != nil {
		return ComparisonReport{}, fmt.Errorf("MLP benchmark failed: %w", err)
	}

	// Generate comparison report
	report := GenerateComparisonReport(perceptronMetrics, mlpMetrics)

	if br.verbose {
		fmt.Println("\n📊 Fast Benchmark Results:")
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
