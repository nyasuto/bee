// Package datasets implements CNN evaluation utilities for image datasets
// Learning Goal: Understanding CNN performance evaluation on real image data
package datasets

import (
	"fmt"
	"math"
	"time"

	"github.com/nyasuto/bee/phase2"
)

// CNNEvaluator handles CNN training and evaluation on image datasets
// Learning Goal: Understanding end-to-end CNN training pipeline
type CNNEvaluator struct {
	CNN          *phase2.CNN
	Dataset      *ImageDataset
	LearningRate float64
	BatchSize    int
	Epochs       int
	Verbose      bool
}

// EvaluationResults stores comprehensive evaluation metrics
// Mathematical Foundation: Standard machine learning evaluation metrics
type EvaluationResults struct {
	ModelName        string          // CNN architecture name
	DatasetName      string          // Dataset name (MNIST, CIFAR-10)
	TrainingTime     time.Duration   // Total training time
	InferenceTime    time.Duration   // Average inference time per sample
	Accuracy         float64         // Classification accuracy [0,1]
	Loss             float64         // Final training loss
	EpochsCompleted  int             // Number of training epochs completed
	MemoryUsage      int64           // Estimated memory usage in bytes
	PerClassAccuracy map[int]float64 // Accuracy per class
	ConfusionMatrix  [][]int         // Confusion matrix
	Timestamp        time.Time       // Evaluation timestamp
}

// NewCNNEvaluator creates a new CNN evaluator
// Learning Goal: Understanding CNN architecture configuration for different tasks
func NewCNNEvaluator(dataset *ImageDataset, learningRate float64, batchSize int, epochs int) *CNNEvaluator {
	return &CNNEvaluator{
		Dataset:      dataset,
		LearningRate: learningRate,
		BatchSize:    batchSize,
		Epochs:       epochs,
		Verbose:      false,
	}
}

// SetVerbose enables/disables verbose output
func (eval *CNNEvaluator) SetVerbose(verbose bool) *CNNEvaluator {
	eval.Verbose = verbose
	return eval
}

// CreateMNISTCNN creates a CNN architecture optimized for MNIST
// Learning Goal: Understanding CNN design patterns for grayscale image classification
func (eval *CNNEvaluator) CreateMNISTCNN() error {
	if eval.Dataset.Width != 28 || eval.Dataset.Height != 28 || eval.Dataset.Channels != 1 {
		return fmt.Errorf("dataset not compatible with MNIST CNN: expected 28x28x1, got %dx%dx%d",
			eval.Dataset.Height, eval.Dataset.Width, eval.Dataset.Channels)
	}

	// Create CNN for MNIST: 28x28x1 -> 10 classes
	// Architecture: Conv(32,5x5) -> Pool(2x2) -> Conv(64,3x3) -> Pool(2x2) -> FC(128) -> FC(10)

	cnn := &phase2.CNN{
		ConvLayers:   []*phase2.ConvLayer{},
		PoolLayers:   []*phase2.PoolingLayer{},
		LearningRate: eval.LearningRate,
		InputShape:   [3]int{28, 28, 1},
	}

	// First convolutional layer: 1 -> 16 channels, 5x5 kernel
	conv1 := phase2.NewConvLayer(1, 16, 5, 1, 2, phase2.ReLU) // padding=2 to maintain size
	conv1.InputShape = [3]int{28, 28, 1}
	conv1.OutputShape = [3]int{28, 28, 16} // Same size due to padding
	cnn.ConvLayers = append(cnn.ConvLayers, conv1)

	// First pooling layer: 28x28 -> 14x14
	pool1 := &phase2.PoolingLayer{
		PoolSize:    2,
		Stride:      2,
		PoolType:    phase2.MaxPooling,
		InputShape:  [3]int{28, 28, 16},
		OutputShape: [3]int{14, 14, 16},
	}
	cnn.PoolLayers = append(cnn.PoolLayers, pool1)

	// Second convolutional layer: 16 -> 32 channels, 3x3 kernel
	conv2 := phase2.NewConvLayer(16, 32, 3, 1, 1, phase2.ReLU) // padding=1 to maintain size
	conv2.InputShape = [3]int{14, 14, 16}
	conv2.OutputShape = [3]int{14, 14, 32}
	cnn.ConvLayers = append(cnn.ConvLayers, conv2)

	// Second pooling layer: 14x14 -> 7x7
	pool2 := &phase2.PoolingLayer{
		PoolSize:    2,
		Stride:      2,
		PoolType:    phase2.MaxPooling,
		InputShape:  [3]int{14, 14, 32},
		OutputShape: [3]int{7, 7, 32},
	}
	cnn.PoolLayers = append(cnn.PoolLayers, pool2)

	// Calculate flattened size: 7 * 7 * 32 = 1568
	flattenedSize := 7 * 7 * 32
	outputSize := 10 // MNIST has 10 classes

	// Fully connected layers
	cnn.FlattenShape = [2]int{flattenedSize, outputSize}
	cnn.FCWeights = make([][]float64, outputSize)
	cnn.FCBiases = make([]float64, outputSize)

	// Xavier initialization for FC weights
	fanIn := flattenedSize
	fanOut := outputSize
	variance := 2.0 / float64(fanIn+fanOut)
	stddev := math.Sqrt(variance)

	for i := 0; i < outputSize; i++ {
		cnn.FCWeights[i] = make([]float64, flattenedSize)
		for j := 0; j < flattenedSize; j++ {
			cnn.FCWeights[i][j] = phase2.RandNormal() * stddev
		}
		cnn.FCBiases[i] = 0.0 // Initialize bias to zero
	}

	eval.CNN = cnn

	if eval.Verbose {
		fmt.Printf("🧠 Created MNIST CNN Architecture:\n")
		fmt.Printf("   Conv1: 1->16 channels, 5x5 kernel, ReLU\n")
		fmt.Printf("   Pool1: 2x2 max pooling\n")
		fmt.Printf("   Conv2: 16->32 channels, 3x3 kernel, ReLU\n")
		fmt.Printf("   Pool2: 2x2 max pooling\n")
		fmt.Printf("   FC: %d -> %d (fully connected)\n", flattenedSize, outputSize)
	}

	return nil
}

// TrainCNN trains the CNN on the dataset
// Learning Goal: Understanding CNN training loop and batch processing
func (eval *CNNEvaluator) TrainCNN() (*EvaluationResults, error) {
	if eval.CNN == nil {
		return nil, fmt.Errorf("CNN not initialized - call CreateMNISTCNN() first")
	}

	if eval.Verbose {
		fmt.Printf("🚀 Starting CNN training...\n")
		fmt.Printf("   Dataset: %s (%d samples)\n", eval.Dataset.Name, len(eval.Dataset.Images))
		fmt.Printf("   Learning Rate: %.4f\n", eval.LearningRate)
		fmt.Printf("   Batch Size: %d\n", eval.BatchSize)
		fmt.Printf("   Max Epochs: %d\n", eval.Epochs)
	}

	startTime := time.Now()
	var finalLoss float64

	for epoch := 0; epoch < eval.Epochs; epoch++ {
		epochStart := time.Now()
		epochLoss := 0.0
		batchCount := 0

		// Shuffle dataset for each epoch
		indices := eval.Dataset.Shuffle()

		// Process batches
		for i := 0; i < len(indices); i += eval.BatchSize {
			end := i + eval.BatchSize
			if end > len(indices) {
				end = len(indices)
			}

			batchIndices := indices[i:end]
			batchImages, batchLabels, err := eval.Dataset.GetBatch(batchIndices)
			if err != nil {
				return nil, fmt.Errorf("failed to get batch: %w", err)
			}

			// Train on batch (simplified - forward pass only for now)
			batchLoss := eval.trainBatch(batchImages, batchLabels)
			epochLoss += batchLoss
			batchCount++
		}

		finalLoss = epochLoss / float64(batchCount)

		if eval.Verbose && (epoch+1)%10 == 0 {
			epochTime := time.Since(epochStart)
			fmt.Printf("   Epoch %d/%d: Loss=%.4f, Time=%v\n",
				epoch+1, eval.Epochs, finalLoss, epochTime)
		}
	}

	trainingTime := time.Since(startTime)

	// Evaluate trained model
	accuracy, perClassAcc, confMatrix := eval.EvaluateAccuracy()

	if eval.Verbose {
		fmt.Printf("✅ Training completed in %v\n", trainingTime)
		fmt.Printf("📊 Final accuracy: %.2f%%\n", accuracy*100)
	}

	return &EvaluationResults{
		ModelName:        "MNIST-CNN",
		DatasetName:      eval.Dataset.Name,
		TrainingTime:     trainingTime,
		InferenceTime:    eval.measureInferenceTime(),
		Accuracy:         accuracy,
		Loss:             finalLoss,
		EpochsCompleted:  eval.Epochs,
		MemoryUsage:      eval.estimateMemoryUsage(),
		PerClassAccuracy: perClassAcc,
		ConfusionMatrix:  confMatrix,
		Timestamp:        time.Now(),
	}, nil
}

// trainBatch performs forward pass on a batch (simplified training)
// Learning Goal: Understanding batch processing and loss calculation
func (eval *CNNEvaluator) trainBatch(images [][][][]float64, labels []int) float64 {
	if len(images) == 0 {
		return 0.0 // Return 0 for empty batch
	}

	totalLoss := 0.0
	validSamples := 0

	for i, image := range images {
		// Forward pass
		output, err := eval.CNN.Forward(image)
		if err != nil {
			continue // Skip problematic samples
		}

		// Calculate loss (cross-entropy)
		target := labels[i]
		loss := eval.calculateCrossEntropyLoss(output, target)
		if !math.IsNaN(loss) && !math.IsInf(loss, 0) {
			totalLoss += loss
			validSamples++
		}

		// Note: Backward pass would be implemented here in a complete training loop
		// For educational purposes, we're focusing on the evaluation framework
	}

	if validSamples == 0 {
		return 0.0
	}
	return totalLoss / float64(validSamples)
}

// calculateCrossEntropyLoss computes cross-entropy loss for classification
// Mathematical Foundation: L = -log(p_target) where p_target is predicted probability of correct class
func (eval *CNNEvaluator) calculateCrossEntropyLoss(output []float64, target int) float64 {
	// Apply softmax to get probabilities
	probabilities := eval.softmax(output)

	// Cross-entropy loss: -log(p_target)
	if target >= 0 && target < len(probabilities) {
		return -math.Log(math.Max(probabilities[target], 1e-15)) // Avoid log(0)
	}
	return math.Inf(1) // Invalid target
}

// softmax applies softmax activation function
// Mathematical Foundation: softmax(x_i) = exp(x_i) / Σ(exp(x_j))
func (eval *CNNEvaluator) softmax(input []float64) []float64 {
	// Find max for numerical stability
	maxVal := input[0]
	for _, val := range input {
		if val > maxVal {
			maxVal = val
		}
	}

	// Calculate exp(x_i - max) and sum
	exp := make([]float64, len(input))
	sum := 0.0
	for i, val := range input {
		exp[i] = math.Exp(val - maxVal)
		sum += exp[i]
	}

	// Normalize to get probabilities
	probabilities := make([]float64, len(input))
	for i := range exp {
		probabilities[i] = exp[i] / sum
	}

	return probabilities
}

// EvaluateAccuracy calculates classification accuracy and detailed metrics
// Learning Goal: Understanding classification evaluation metrics
func (eval *CNNEvaluator) EvaluateAccuracy() (float64, map[int]float64, [][]int) {
	total := len(eval.Dataset.Images)
	if total == 0 {
		// Return empty results for empty dataset
		return 0.0, make(map[int]float64), [][]int{}
	}

	// Check if CNN is available
	if eval.CNN == nil {
		// Return empty results if no CNN
		return 0.0, make(map[int]float64), [][]int{}
	}

	correct := 0

	// Per-class accuracy tracking
	classCorrect := make(map[int]int)
	classTotal := make(map[int]int)

	// Confusion matrix (actual x predicted)
	numClasses := len(eval.Dataset.Classes)
	confusionMatrix := make([][]int, numClasses)
	for i := range confusionMatrix {
		confusionMatrix[i] = make([]int, numClasses)
	}

	// Evaluate each sample
	for i, image := range eval.Dataset.Images {
		actualLabel := eval.Dataset.Labels[i]

		// Forward pass
		output, err := eval.CNN.Forward(image)
		if err != nil {
			continue // Skip problematic samples
		}

		// Get predicted class (argmax)
		predictedLabel := eval.argmax(output)

		// Update counters
		classTotal[actualLabel]++
		if actualLabel < numClasses && predictedLabel < numClasses {
			confusionMatrix[actualLabel][predictedLabel]++
		}

		if predictedLabel == actualLabel {
			correct++
			classCorrect[actualLabel]++
		}
	}

	// Calculate overall accuracy
	accuracy := float64(correct) / float64(total)

	// Calculate per-class accuracy
	perClassAccuracy := make(map[int]float64)
	for class, totalSamples := range classTotal {
		if totalSamples > 0 {
			perClassAccuracy[class] = float64(classCorrect[class]) / float64(totalSamples)
		}
	}

	return accuracy, perClassAccuracy, confusionMatrix
}

// argmax returns the index of the maximum value in a slice
func (eval *CNNEvaluator) argmax(values []float64) int {
	if len(values) == 0 {
		return -1
	}

	maxIndex := 0
	maxValue := values[0]
	for i, val := range values {
		if val > maxValue {
			maxValue = val
			maxIndex = i
		}
	}
	return maxIndex
}

// measureInferenceTime measures average inference time per sample
// Learning Goal: Understanding performance profiling for neural networks
func (eval *CNNEvaluator) measureInferenceTime() time.Duration {
	if len(eval.Dataset.Images) == 0 {
		return 0
	}

	// Check if CNN is available
	if eval.CNN == nil {
		return 0
	}

	// Measure inference time on a subset of samples
	sampleSize := 100
	if len(eval.Dataset.Images) < sampleSize {
		sampleSize = len(eval.Dataset.Images)
	}

	start := time.Now()
	for i := 0; i < sampleSize; i++ {
		_, err := eval.CNN.Forward(eval.Dataset.Images[i])
		if err != nil {
			continue // Skip problematic samples
		}
	}
	totalTime := time.Since(start)

	return totalTime / time.Duration(sampleSize)
}

// estimateMemoryUsage estimates memory usage of the CNN model
// Learning Goal: Understanding memory requirements for neural networks
func (eval *CNNEvaluator) estimateMemoryUsage() int64 {
	var totalMemory int64

	// Check if CNN is available
	if eval.CNN == nil {
		return 0
	}

	// Estimate convolution layer memory
	for _, conv := range eval.CNN.ConvLayers {
		// Kernels memory
		for _, outputChannel := range conv.Kernels {
			for _, inputChannel := range outputChannel {
				for _, row := range inputChannel {
					totalMemory += int64(len(row) * 8) // 8 bytes per float64
				}
			}
		}
		// Biases memory
		totalMemory += int64(len(conv.Biases) * 8)

		// Cache memory (input and output)
		if conv.InputCache != nil {
			for _, row := range conv.InputCache {
				for _, col := range row {
					totalMemory += int64(len(col) * 8)
				}
			}
		}
		if conv.OutputCache != nil {
			for _, row := range conv.OutputCache {
				for _, col := range row {
					totalMemory += int64(len(col) * 8)
				}
			}
		}
	}

	// Estimate pooling layer memory
	for _, pool := range eval.CNN.PoolLayers {
		if pool.InputCache != nil {
			for _, row := range pool.InputCache {
				for _, col := range row {
					totalMemory += int64(len(col) * 8)
				}
			}
		}
		if pool.MaxIndices != nil {
			for _, row := range pool.MaxIndices {
				for _, col := range row {
					totalMemory += int64(len(col) * 4) // 4 bytes per int
				}
			}
		}
	}

	// Estimate fully connected layer memory
	for _, weights := range eval.CNN.FCWeights {
		totalMemory += int64(len(weights) * 8)
	}
	totalMemory += int64(len(eval.CNN.FCBiases) * 8)

	return totalMemory
}

// PrintEvaluationResults displays comprehensive evaluation results
// Learning Goal: Understanding model evaluation reporting
func (eval *CNNEvaluator) PrintEvaluationResults(results *EvaluationResults) {
	fmt.Printf("\n🧠 CNN Evaluation Results\n")
	fmt.Printf("═══════════════════════════════════════════════\n")
	fmt.Printf("📊 Model: %s\n", results.ModelName)
	fmt.Printf("📊 Dataset: %s\n", results.DatasetName)
	fmt.Printf("⏱️  Training Time: %v\n", results.TrainingTime)
	fmt.Printf("⚡ Inference Time: %v (per sample)\n", results.InferenceTime)
	fmt.Printf("🎯 Accuracy: %.2f%%\n", results.Accuracy*100)
	fmt.Printf("📉 Final Loss: %.4f\n", results.Loss)
	fmt.Printf("🔄 Epochs: %d\n", results.EpochsCompleted)
	fmt.Printf("💾 Memory Usage: %.2f MB\n", float64(results.MemoryUsage)/(1024*1024))
	fmt.Printf("🕐 Timestamp: %s\n", results.Timestamp.Format("2006-01-02 15:04:05"))

	// Per-class accuracy
	if len(results.PerClassAccuracy) > 0 {
		fmt.Printf("\n📈 Per-Class Accuracy:\n")
		for class, acc := range results.PerClassAccuracy {
			fmt.Printf("   Class %d: %.2f%%\n", class, acc*100)
		}
	}

	// Confusion matrix (simplified view for readability)
	if len(results.ConfusionMatrix) > 0 && len(results.ConfusionMatrix) <= 10 {
		fmt.Printf("\n🔲 Confusion Matrix:\n")
		fmt.Printf("   Actual \\ Predicted: ")
		for i := 0; i < len(results.ConfusionMatrix); i++ {
			fmt.Printf("%4d", i)
		}
		fmt.Println()

		for i, row := range results.ConfusionMatrix {
			fmt.Printf("   Class %d:          ", i)
			for _, val := range row {
				fmt.Printf("%4d", val)
			}
			fmt.Println()
		}
	}
}
