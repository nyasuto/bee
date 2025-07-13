package benchmark

import (
	"fmt"
	"runtime"
	"time"

	"github.com/nyasuto/bee/datasets"
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

	// Measure inference time (multiple runs for accuracy)
	inferenceRuns := br.iterations
	if inferenceRuns < 100 {
		inferenceRuns = 100
	}

	// Warmup runs
	for i := 0; i < br.warmupRuns; i++ {
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

	// Measure inference time
	inferenceRuns := br.iterations
	if inferenceRuns < 100 {
		inferenceRuns = 100
	}

	// Warmup runs
	for i := 0; i < br.warmupRuns; i++ {
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

// CNNBenchmarkConfig holds configuration for CNN benchmarking
// Learning Goal: Understanding CNN-specific performance configuration
type CNNBenchmarkConfig struct {
	ImageSize    [3]int  // [height, width, channels]
	BatchSize    int     // Batch size for processing
	Epochs       int     // Number of training epochs
	LearningRate float64 // Learning rate for training
	Architecture string  // Architecture type (MNIST, CIFAR-10, Custom)
}

// BenchmarkCNN measures CNN performance on image datasets
// Learning Goal: Understanding CNN-specific performance measurement patterns
func (br *BenchmarkRunner) BenchmarkCNN(imageDataset *datasets.ImageDataset, config CNNBenchmarkConfig) (PerformanceMetrics, error) {
	if br.verbose {
		fmt.Printf("🔍 Benchmarking CNN on %s dataset...\n", imageDataset.Name)
		fmt.Printf("   Architecture: %s\n", config.Architecture)
		fmt.Printf("   Image size: %dx%dx%d\n", config.ImageSize[0], config.ImageSize[1], config.ImageSize[2])
		fmt.Printf("   Batch size: %d\n", config.BatchSize)
	}

	// Create CNN evaluator
	evaluator := datasets.NewCNNEvaluator(imageDataset, config.LearningRate, config.BatchSize, config.Epochs)
	evaluator.SetVerbose(br.verbose)

	// Create appropriate CNN architecture
	var err error
	switch config.Architecture {
	case "MNIST":
		err = evaluator.CreateMNISTCNN()
	default:
		return PerformanceMetrics{}, fmt.Errorf("unsupported CNN architecture: %s", config.Architecture)
	}

	if err != nil {
		return PerformanceMetrics{}, fmt.Errorf("failed to create CNN: %w", err)
	}

	// Measure memory before training
	var memBefore runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memBefore)

	// Measure training time
	startTime := time.Now()

	// Train CNN
	evalResults, err := evaluator.TrainCNN()
	if err != nil {
		return PerformanceMetrics{}, fmt.Errorf("CNN training failed: %w", err)
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

	// Measure convolution operation performance
	convMetrics, err := br.benchmarkConvolutionOps(evaluator, config)
	if err != nil {
		return PerformanceMetrics{}, fmt.Errorf("convolution benchmark failed: %w", err)
	}

	// Measure pooling operation performance
	poolMetrics, err := br.benchmarkPoolingOps(evaluator, config)
	if err != nil {
		return PerformanceMetrics{}, fmt.Errorf("pooling benchmark failed: %w", err)
	}

	if br.verbose {
		fmt.Printf("  ✅ Training completed in %d epochs, %.2f%% accuracy\n",
			evalResults.EpochsCompleted, evalResults.Accuracy*100)
		fmt.Printf("  📊 Convolution ops: %v avg time\n", convMetrics)
		fmt.Printf("  📊 Pooling ops: %v avg time\n", poolMetrics)
	}

	return PerformanceMetrics{
		ModelType:       "cnn",
		DatasetName:     imageDataset.Name,
		TrainingTime:    trainingTime,
		InferenceTime:   evalResults.InferenceTime,
		MemoryUsage:     memoryUsage,
		Accuracy:        evalResults.Accuracy,
		ConvergenceRate: evalResults.EpochsCompleted,
		FinalLoss:       evalResults.Loss,
		Timestamp:       time.Now(),
		// CNN-specific metrics
		ConvolutionTime: convMetrics,
		PoolingTime:     poolMetrics,
		BatchSize:       config.BatchSize,
		FeatureMapSize:  evalResults.MemoryUsage,
	}, nil
}

// benchmarkConvolutionOps measures convolution operation performance
// Learning Goal: Understanding convolution operation efficiency
func (br *BenchmarkRunner) benchmarkConvolutionOps(evaluator *datasets.CNNEvaluator, config CNNBenchmarkConfig) (time.Duration, error) {
	if evaluator.CNN == nil {
		return 0, fmt.Errorf("CNN not initialized")
	}

	// Create test input for convolution benchmarking
	testInput := make([][][]float64, config.ImageSize[0])
	for h := 0; h < config.ImageSize[0]; h++ {
		testInput[h] = make([][]float64, config.ImageSize[1])
		for w := 0; w < config.ImageSize[1]; w++ {
			testInput[h][w] = make([]float64, config.ImageSize[2])
			for c := 0; c < config.ImageSize[2]; c++ {
				testInput[h][w][c] = 0.5 // Neutral test value
			}
		}
	}

	// Warmup runs
	for i := 0; i < br.warmupRuns; i++ {
		if len(evaluator.CNN.ConvLayers) > 0 {
			_, _ = evaluator.CNN.ConvLayers[0].Forward(testInput) //nolint:errcheck // Ignore errors in warmup
		}
	}

	// Benchmark convolution operations
	var totalTime time.Duration
	validOps := 0

	for _, convLayer := range evaluator.CNN.ConvLayers {
		start := time.Now()
		for i := 0; i < br.iterations; i++ {
			_, err := convLayer.Forward(testInput)
			if err == nil {
				validOps++
			}
		}
		totalTime += time.Since(start)
	}

	if validOps == 0 {
		return 0, fmt.Errorf("no valid convolution operations")
	}

	return totalTime / time.Duration(validOps), nil
}

// benchmarkPoolingOps measures pooling operation performance
// Learning Goal: Understanding pooling operation efficiency
func (br *BenchmarkRunner) benchmarkPoolingOps(evaluator *datasets.CNNEvaluator, config CNNBenchmarkConfig) (time.Duration, error) {
	if evaluator.CNN == nil {
		return 0, fmt.Errorf("CNN not initialized")
	}

	// Create test input for pooling benchmarking
	// Use output size from first conv layer if available
	var testInput [][][]float64
	if len(evaluator.CNN.ConvLayers) > 0 {
		outputShape := evaluator.CNN.ConvLayers[0].OutputShape
		testInput = make([][][]float64, outputShape[0])
		for h := 0; h < outputShape[0]; h++ {
			testInput[h] = make([][]float64, outputShape[1])
			for w := 0; w < outputShape[1]; w++ {
				testInput[h][w] = make([]float64, outputShape[2])
				for c := 0; c < outputShape[2]; c++ {
					testInput[h][w][c] = 0.5 // Neutral test value
				}
			}
		}
	} else {
		// Fallback to original input size
		testInput = make([][][]float64, config.ImageSize[0])
		for h := 0; h < config.ImageSize[0]; h++ {
			testInput[h] = make([][]float64, config.ImageSize[1])
			for w := 0; w < config.ImageSize[1]; w++ {
				testInput[h][w] = make([]float64, config.ImageSize[2])
				for c := 0; c < config.ImageSize[2]; c++ {
					testInput[h][w][c] = 0.5
				}
			}
		}
	}

	// Warmup runs
	for i := 0; i < br.warmupRuns; i++ {
		if len(evaluator.CNN.PoolLayers) > 0 {
			_, _ = evaluator.CNN.PoolLayers[0].Forward(testInput) //nolint:errcheck // Ignore errors in warmup
		}
	}

	// Benchmark pooling operations
	var totalTime time.Duration
	validOps := 0

	for _, poolLayer := range evaluator.CNN.PoolLayers {
		start := time.Now()
		for i := 0; i < br.iterations; i++ {
			_, err := poolLayer.Forward(testInput)
			if err == nil {
				validOps++
			}
		}
		totalTime += time.Since(start)
	}

	if validOps == 0 {
		return 0, fmt.Errorf("no valid pooling operations")
	}

	return totalTime / time.Duration(validOps), nil
}

// RunCNNComparison executes comparative benchmark between MLP and CNN
// Learning Goal: Understanding architectural performance trade-offs
func (br *BenchmarkRunner) RunCNNComparison(imageDataset *datasets.ImageDataset, cnnConfig CNNBenchmarkConfig, mlpHidden []int) (ComparisonReport, error) {
	if br.verbose {
		fmt.Printf("🚀 Running CNN vs MLP comparative benchmark on %s dataset\n", imageDataset.Name)
		fmt.Println("─────────────────────────────────────────────────────")
	}

	// Create flat dataset from image dataset for MLP comparison
	flatDataset, err := br.convertImageDatasetToFlat(imageDataset)
	if err != nil {
		return ComparisonReport{}, fmt.Errorf("failed to convert image dataset: %w", err)
	}

	// Benchmark MLP (baseline)
	mlpMetrics, err := br.BenchmarkMLP(flatDataset, mlpHidden)
	if err != nil {
		return ComparisonReport{}, fmt.Errorf("MLP benchmark failed: %w", err)
	}

	// Benchmark CNN
	cnnMetrics, err := br.BenchmarkCNN(imageDataset, cnnConfig)
	if err != nil {
		return ComparisonReport{}, fmt.Errorf("CNN benchmark failed: %w", err)
	}

	// Generate comparison report
	report := GenerateComparisonReport(mlpMetrics, cnnMetrics)

	if br.verbose {
		fmt.Println("\n📊 Benchmark Results:")
		fmt.Printf("MLP: %.2f%% accuracy, %s training, %s inference\n",
			mlpMetrics.Accuracy*100,
			FormatDuration(mlpMetrics.TrainingTime),
			FormatDuration(mlpMetrics.InferenceTime))
		fmt.Printf("CNN: %.2f%% accuracy, %s training, %s inference\n",
			cnnMetrics.Accuracy*100,
			FormatDuration(cnnMetrics.TrainingTime),
			FormatDuration(cnnMetrics.InferenceTime))

		fmt.Println("\n" + "─────────────────────────────────────────────────────")
		PrintComparisonSummary(report)
	}

	return report, nil
}

// convertImageDatasetToFlat converts image dataset to flat dataset for MLP comparison
// Learning Goal: Understanding data representation differences between CNN and MLP
func (br *BenchmarkRunner) convertImageDatasetToFlat(imageDataset *datasets.ImageDataset) (Dataset, error) {
	if len(imageDataset.Images) == 0 {
		return Dataset{}, fmt.Errorf("empty image dataset")
	}

	// Calculate flattened input size
	sampleImage := imageDataset.Images[0]
	inputSize := len(sampleImage) * len(sampleImage[0]) * len(sampleImage[0][0])

	builder := NewDatasetBuilder(imageDataset.Name + "_flat").
		WithDescription("Flattened version of " + imageDataset.Name + " for MLP comparison")

	// Convert 80% to training, 20% to testing
	trainSize := int(float64(len(imageDataset.Images)) * 0.8)

	for i, image := range imageDataset.Images {
		// Flatten image
		flatInput := make([]float64, inputSize)
		idx := 0
		for h := 0; h < len(image); h++ {
			for w := 0; w < len(image[h]); w++ {
				for c := 0; c < len(image[h][w]); c++ {
					flatInput[idx] = image[h][w][c]
					idx++
				}
			}
		}

		// Convert label to one-hot encoding
		label := imageDataset.Labels[i]
		target := make([]float64, len(imageDataset.Classes))
		if label >= 0 && label < len(target) {
			target[label] = 1.0
		}

		// Add to training or test set
		if i < trainSize {
			builder.AddTrainExample(flatInput, target)
		} else {
			builder.AddTestExample(flatInput, target)
		}
	}

	return builder.Build(), nil
}

// CNNMemoryAnalysis represents detailed memory usage analysis for CNN
// Learning Goal: Understanding CNN memory consumption patterns
type CNNMemoryAnalysis struct {
	FeatureMapMemory int64 `json:"feature_map_memory"` // Memory used by feature maps
	KernelMemory     int64 `json:"kernel_memory"`      // Memory used by convolution kernels
	ActivationMemory int64 `json:"activation_memory"`  // Memory used by activations
	GradientMemory   int64 `json:"gradient_memory"`    // Memory used by gradients (if available)
	PoolingMemory    int64 `json:"pooling_memory"`     // Memory used by pooling layers
	FCMemory         int64 `json:"fc_memory"`          // Memory used by fully connected layers
	TotalMemory      int64 `json:"total_memory"`       // Total estimated memory usage
	BatchScaleFactor int   `json:"batch_scale_factor"` // Memory scaling factor for batch processing
}

// BatchProcessingMetrics represents batch processing efficiency metrics
// Learning Goal: Understanding batch processing impact on CNN performance
type BatchProcessingMetrics struct {
	BatchSize      int           `json:"batch_size"`
	ThroughputFPS  float64       `json:"throughput_fps"`   // Frames per second
	LatencyPerItem time.Duration `json:"latency_per_item"` // Average latency per item
	MemoryOverhead int64         `json:"memory_overhead"`  // Memory overhead for batching
	Efficiency     float64       `json:"efficiency"`       // Efficiency compared to single item processing
}

// AnalyzeCNNMemory performs detailed memory usage analysis for CNN
// Learning Goal: Understanding memory allocation patterns in CNN architectures
func (br *BenchmarkRunner) AnalyzeCNNMemory(evaluator *datasets.CNNEvaluator, config CNNBenchmarkConfig) (*CNNMemoryAnalysis, error) {
	if evaluator.CNN == nil {
		return nil, fmt.Errorf("CNN not initialized")
	}

	analysis := &CNNMemoryAnalysis{
		BatchScaleFactor: config.BatchSize,
	}

	// Analyze convolution layer memory
	for _, convLayer := range evaluator.CNN.ConvLayers {
		// Kernel memory: [outputChannels][inputChannels][kernelHeight][kernelWidth]
		for _, outputChannel := range convLayer.Kernels {
			for _, inputChannel := range outputChannel {
				for _, row := range inputChannel {
					analysis.KernelMemory += int64(len(row) * 8) // 8 bytes per float64
				}
			}
		}

		// Bias memory
		analysis.KernelMemory += int64(len(convLayer.Biases) * 8)

		// Feature map memory (output cache)
		if convLayer.OutputCache != nil {
			for _, row := range convLayer.OutputCache {
				for _, col := range row {
					analysis.FeatureMapMemory += int64(len(col) * 8)
				}
			}
		}

		// Activation memory (input cache)
		if convLayer.InputCache != nil {
			for _, row := range convLayer.InputCache {
				for _, col := range row {
					analysis.ActivationMemory += int64(len(col) * 8)
				}
			}
		}
	}

	// Analyze pooling layer memory
	for _, poolLayer := range evaluator.CNN.PoolLayers {
		// Input cache memory
		if poolLayer.InputCache != nil {
			for _, row := range poolLayer.InputCache {
				for _, col := range row {
					analysis.PoolingMemory += int64(len(col) * 8)
				}
			}
		}

		// Max indices memory (for max pooling)
		if poolLayer.MaxIndices != nil {
			for _, row := range poolLayer.MaxIndices {
				for _, col := range row {
					analysis.PoolingMemory += int64(len(col) * 4) // 4 bytes per int
				}
			}
		}
	}

	// Analyze fully connected layer memory
	for _, weights := range evaluator.CNN.FCWeights {
		analysis.FCMemory += int64(len(weights) * 8)
	}
	analysis.FCMemory += int64(len(evaluator.CNN.FCBiases) * 8)

	// Scale by batch size for training memory requirements
	analysis.FeatureMapMemory *= int64(config.BatchSize)
	analysis.ActivationMemory *= int64(config.BatchSize)
	analysis.PoolingMemory *= int64(config.BatchSize)

	// Calculate total memory
	analysis.TotalMemory = analysis.FeatureMapMemory + analysis.KernelMemory +
		analysis.ActivationMemory + analysis.GradientMemory +
		analysis.PoolingMemory + analysis.FCMemory

	return analysis, nil
}

// BenchmarkBatchProcessing analyzes batch processing efficiency
// Learning Goal: Understanding the performance characteristics of batch processing in CNNs
func (br *BenchmarkRunner) BenchmarkBatchProcessing(evaluator *datasets.CNNEvaluator, config CNNBenchmarkConfig) (*BatchProcessingMetrics, error) {
	if evaluator.CNN == nil {
		return nil, fmt.Errorf("CNN not initialized")
	}

	if len(evaluator.Dataset.Images) == 0 {
		return nil, fmt.Errorf("empty dataset")
	}

	// Test with single item first
	singleItemStart := time.Now()
	for i := 0; i < 10; i++ { // Test with 10 samples
		imageIndex := i % len(evaluator.Dataset.Images)
		_, err := evaluator.CNN.Forward(evaluator.Dataset.Images[imageIndex])
		if err != nil {
			continue // Skip failed inferences
		}
	}
	singleItemTime := time.Since(singleItemStart)
	singleItemLatency := singleItemTime / 10

	// Measure memory before batch processing
	var memBefore runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memBefore)

	// Create batch for testing
	batchSize := config.BatchSize
	if batchSize > len(evaluator.Dataset.Images) {
		batchSize = len(evaluator.Dataset.Images)
	}

	batchImages := make([][][][]float64, batchSize)
	batchLabels := make([]int, batchSize)
	for i := 0; i < batchSize; i++ {
		imageIndex := i % len(evaluator.Dataset.Images)
		batchImages[i] = evaluator.Dataset.Images[imageIndex]
		batchLabels[i] = evaluator.Dataset.Labels[imageIndex]
	}

	// Warmup for batch processing
	for i := 0; i < br.warmupRuns; i++ {
		for j := 0; j < batchSize; j++ {
			_, _ = evaluator.CNN.Forward(batchImages[j]) //nolint:errcheck // Ignore errors in warmup
		}
	}

	// Benchmark batch processing
	batchStart := time.Now()
	successfulInferences := 0

	for iter := 0; iter < br.iterations; iter++ {
		for i := 0; i < batchSize; i++ {
			_, err := evaluator.CNN.Forward(batchImages[i])
			if err == nil {
				successfulInferences++
			}
		}
	}

	batchTime := time.Since(batchStart)

	// Measure memory after batch processing
	var memAfter runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memAfter)

	// Calculate memory overhead
	memoryOverhead := int64(0)
	if memAfter.Alloc >= memBefore.Alloc {
		diff := memAfter.Alloc - memBefore.Alloc
		if diff <= uint64(^uint64(0)>>1) {
			memoryOverhead = int64(diff)
		}
	}

	// Calculate metrics
	totalItems := successfulInferences
	if totalItems == 0 {
		return nil, fmt.Errorf("no successful inferences")
	}

	avgBatchLatency := batchTime / time.Duration(totalItems)
	throughputFPS := float64(totalItems) / batchTime.Seconds()

	// Calculate efficiency (batch vs single item)
	efficiency := 1.0
	if singleItemLatency > 0 {
		efficiency = float64(singleItemLatency) / float64(avgBatchLatency)
	}

	return &BatchProcessingMetrics{
		BatchSize:      batchSize,
		ThroughputFPS:  throughputFPS,
		LatencyPerItem: avgBatchLatency,
		MemoryOverhead: memoryOverhead,
		Efficiency:     efficiency,
	}, nil
}
