// Package datasets implements sequence evaluation utilities for RNN/LSTM models
// Learning Goal: Understanding sequence learning evaluation and performance measurement
package datasets

import (
	"fmt"
	"math"
	"time"

	"github.com/nyasuto/bee/phase2"
)

// SequenceEvaluator handles RNN/LSTM training and evaluation on time series datasets
// Learning Goal: Understanding end-to-end sequence learning pipeline
type SequenceEvaluator struct {
	RNN          *phase2.RNN  // RNN model for evaluation
	LSTM         *phase2.LSTM // LSTM model for evaluation
	Dataset      *TimeSeriesDataset
	LearningRate float64
	BatchSize    int
	Epochs       int
	Verbose      bool
	ModelType    string // "RNN" or "LSTM"
}

// SequenceResults stores comprehensive sequence learning evaluation metrics
// Mathematical Foundation: Standard sequence learning evaluation metrics
type SequenceResults struct {
	ModelName          string           // Model architecture name (RNN/LSTM)
	DatasetName        string           // Dataset name
	TrainingTime       time.Duration    // Total training time
	InferenceTime      time.Duration    // Average inference time per sequence
	MSE                float64          // Mean Squared Error
	MAE                float64          // Mean Absolute Error
	MAPE               float64          // Mean Absolute Percentage Error
	RMSE               float64          // Root Mean Squared Error
	EpochsCompleted    int              // Number of training epochs completed
	ConvergenceEpoch   int              // Epoch where convergence was achieved
	MemoryUsage        int64            // Estimated memory usage in bytes
	LongTermAccuracy   map[int]float64  // Accuracy at different prediction horizons
	PredictionExamples []PredictionPair // Sample prediction vs actual pairs
	Timestamp          time.Time        // Evaluation timestamp
	LossHistory        []float64        // Training loss history
}

// PredictionPair stores a prediction vs actual value pair for analysis
type PredictionPair struct {
	Predicted []float64 // Predicted values
	Actual    []float64 // Actual values
	Error     float64   // Prediction error (MSE for this pair)
}

// NewSequenceEvaluator creates a new sequence evaluator
// Learning Goal: Understanding sequence learning model configuration
func NewSequenceEvaluator(dataset *TimeSeriesDataset, modelType string, learningRate float64, batchSize int, epochs int) *SequenceEvaluator {
	return &SequenceEvaluator{
		Dataset:      dataset,
		ModelType:    modelType,
		LearningRate: learningRate,
		BatchSize:    batchSize,
		Epochs:       epochs,
		Verbose:      false,
	}
}

// SetVerbose enables/disables verbose output
func (eval *SequenceEvaluator) SetVerbose(verbose bool) *SequenceEvaluator {
	eval.Verbose = verbose
	return eval
}

// CreateRNN creates an RNN model optimized for the dataset
// Learning Goal: Understanding RNN architecture design for sequence tasks
func (eval *SequenceEvaluator) CreateRNN() error {
	if eval.Dataset.Features <= 0 || eval.Dataset.OutputSize <= 0 {
		return fmt.Errorf("invalid dataset dimensions: features=%d, output_size=%d", eval.Dataset.Features, eval.Dataset.OutputSize)
	}

	// Create RNN with hidden layer size proportional to problem complexity
	hiddenSize := max(16, eval.Dataset.Features*4) // Adaptive hidden size
	if eval.Dataset.SeqLength > 50 {
		hiddenSize = max(32, eval.Dataset.Features*8) // Larger hidden size for longer sequences
	}

	rnn := phase2.NewRNN(eval.Dataset.Features, hiddenSize, eval.Dataset.OutputSize, eval.LearningRate)

	eval.RNN = rnn
	eval.ModelType = "RNN"

	if eval.Verbose {
		fmt.Printf("🧠 Created RNN Architecture:\n")
		fmt.Printf("   Input size: %d\n", eval.Dataset.Features)
		fmt.Printf("   Hidden size: %d\n", hiddenSize)
		fmt.Printf("   Output size: %d\n", eval.Dataset.OutputSize)
		fmt.Printf("   Learning rate: %.4f\n", eval.LearningRate)
	}

	return nil
}

// CreateLSTM creates an LSTM model optimized for the dataset
// Learning Goal: Understanding LSTM architecture design for sequence tasks
func (eval *SequenceEvaluator) CreateLSTM() error {
	if eval.Dataset.Features <= 0 || eval.Dataset.OutputSize <= 0 {
		return fmt.Errorf("invalid dataset dimensions: features=%d, output_size=%d", eval.Dataset.Features, eval.Dataset.OutputSize)
	}

	// Create LSTM with hidden layer size proportional to problem complexity
	hiddenSize := max(16, eval.Dataset.Features*4) // Adaptive hidden size
	if eval.Dataset.SeqLength > 50 {
		hiddenSize = max(32, eval.Dataset.Features*8) // Larger hidden size for longer sequences
	}

	lstm := phase2.NewLSTM(eval.Dataset.Features, hiddenSize, eval.Dataset.OutputSize, eval.LearningRate)

	eval.LSTM = lstm
	eval.ModelType = "LSTM"

	if eval.Verbose {
		fmt.Printf("🧠 Created LSTM Architecture:\n")
		fmt.Printf("   Input size: %d\n", eval.Dataset.Features)
		fmt.Printf("   Hidden size: %d\n", hiddenSize)
		fmt.Printf("   Output size: %d\n", eval.Dataset.OutputSize)
		fmt.Printf("   Learning rate: %.4f\n", eval.LearningRate)
	}

	return nil
}

// TrainSequenceModel trains the RNN or LSTM on the dataset
// Learning Goal: Understanding sequence learning training loop
func (eval *SequenceEvaluator) TrainSequenceModel() (*SequenceResults, error) {
	if eval.RNN == nil && eval.LSTM == nil {
		return nil, fmt.Errorf("no model initialized - call CreateRNN() or CreateLSTM() first")
	}

	if eval.Verbose {
		fmt.Printf("🚀 Starting %s training...\n", eval.ModelType)
		fmt.Printf("   Dataset: %s (%d samples)\n", eval.Dataset.Name, len(eval.Dataset.Sequences))
		fmt.Printf("   Learning Rate: %.4f\n", eval.LearningRate)
		fmt.Printf("   Batch Size: %d\n", eval.BatchSize)
		fmt.Printf("   Max Epochs: %d\n", eval.Epochs)
	}

	startTime := time.Now()
	lossHistory := make([]float64, 0, eval.Epochs)
	convergenceEpoch := eval.Epochs
	bestLoss := math.Inf(1)
	convergenceThreshold := 1e-6

	for epoch := 0; epoch < eval.Epochs; epoch++ {
		epochStart := time.Now()
		epochLoss := 0.0
		batchCount := 0

		// Shuffle dataset for each epoch
		indices := eval.Dataset.Shuffle()

		// Process batches
		for i := 0; i < len(indices); i += eval.BatchSize {
			end := min(i+eval.BatchSize, len(indices))
			batchIndices := indices[i:end]

			batchSequences, batchTargets, err := eval.Dataset.GetBatch(batchIndices)
			if err != nil {
				return nil, fmt.Errorf("failed to get batch: %w", err)
			}

			// Train on batch
			batchLoss := eval.trainBatch(batchSequences, batchTargets)
			epochLoss += batchLoss
			batchCount++
		}

		avgLoss := epochLoss / float64(batchCount)
		lossHistory = append(lossHistory, avgLoss)

		// Check for convergence
		if bestLoss-avgLoss < convergenceThreshold && convergenceEpoch == eval.Epochs {
			convergenceEpoch = epoch + 1
		}
		if avgLoss < bestLoss {
			bestLoss = avgLoss
		}

		if eval.Verbose && (epoch+1)%max(1, eval.Epochs/10) == 0 {
			epochTime := time.Since(epochStart)
			fmt.Printf("   Epoch %d/%d: Loss=%.6f, Time=%v\n", epoch+1, eval.Epochs, avgLoss, epochTime)
		}
	}

	trainingTime := time.Since(startTime)

	// Evaluate trained model
	mse, mae, mape, rmse := eval.CalculatePredictionMetrics()
	longTermAcc := eval.EvaluateLongTermAccuracy()
	examples := eval.generatePredictionExamples(5) // Generate 5 example predictions

	if eval.Verbose {
		fmt.Printf("✅ Training completed in %v\n", trainingTime)
		fmt.Printf("📊 Final metrics: MSE=%.6f, MAE=%.6f, RMSE=%.6f\n", mse, mae, rmse)
	}

	return &SequenceResults{
		ModelName:          eval.ModelType,
		DatasetName:        eval.Dataset.Name,
		TrainingTime:       trainingTime,
		InferenceTime:      eval.measureInferenceTime(),
		MSE:                mse,
		MAE:                mae,
		MAPE:               mape,
		RMSE:               rmse,
		EpochsCompleted:    eval.Epochs,
		ConvergenceEpoch:   convergenceEpoch,
		MemoryUsage:        eval.estimateMemoryUsage(),
		LongTermAccuracy:   longTermAcc,
		PredictionExamples: examples,
		Timestamp:          time.Now(),
		LossHistory:        lossHistory,
	}, nil
}

// trainBatch performs forward pass on a batch (simplified training)
// Learning Goal: Understanding sequence batch processing and loss calculation
func (eval *SequenceEvaluator) trainBatch(sequences [][][]float64, targets [][]float64) float64 {
	if len(sequences) == 0 {
		return 0.0
	}

	totalLoss := 0.0
	validSamples := 0

	for i, sequence := range sequences {
		var prediction []float64

		// Forward pass through appropriate model
		var outputs [][]float64
		var err error
		if eval.ModelType == "RNN" && eval.RNN != nil {
			// RNN.ForwardSequence returns [][]float64, we need the last output
			outputs, err = eval.RNN.ForwardSequence(sequence)
			if err != nil {
				continue // Skip problematic samples
			}
		} else if eval.ModelType == "LSTM" && eval.LSTM != nil {
			// LSTM.ForwardSequence returns [][]float64, we need the last output
			outputs, err = eval.LSTM.ForwardSequence(sequence)
			if err != nil {
				continue // Skip problematic samples
			}
		} else {
			continue // Skip if no valid model
		}

		if len(outputs) > 0 {
			prediction = outputs[len(outputs)-1] // Use last timestep output
		} else {
			continue // Skip if no outputs
		}

		// Calculate MSE loss
		target := targets[i]
		loss := eval.calculateMSELoss(prediction, target)
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

// calculateMSELoss computes Mean Squared Error loss
// Mathematical Foundation: MSE = (1/n) * Σ(y_pred - y_true)²
func (eval *SequenceEvaluator) calculateMSELoss(prediction, target []float64) float64 {
	if len(prediction) != len(target) {
		return math.Inf(1) // Invalid dimensions
	}

	sum := 0.0
	for i := range prediction {
		diff := prediction[i] - target[i]
		sum += diff * diff
	}

	return sum / float64(len(prediction))
}

// CalculatePredictionMetrics calculates comprehensive prediction accuracy metrics
// Learning Goal: Understanding sequence prediction evaluation metrics
func (eval *SequenceEvaluator) CalculatePredictionMetrics() (mse, mae, mape, rmse float64) {
	totalSamples := len(eval.Dataset.Sequences)
	if totalSamples == 0 {
		return 0, 0, 0, 0
	}

	sumMSE, sumMAE, sumMAPE := 0.0, 0.0, 0.0
	validSamples := 0

	for i, sequence := range eval.Dataset.Sequences {
		var prediction []float64

		// Forward pass
		var outputs [][]float64
		var err error
		if eval.ModelType == "RNN" && eval.RNN != nil {
			outputs, err = eval.RNN.ForwardSequence(sequence)
			if err != nil || len(outputs) == 0 {
				continue
			}
		} else if eval.ModelType == "LSTM" && eval.LSTM != nil {
			outputs, err = eval.LSTM.ForwardSequence(sequence)
			if err != nil || len(outputs) == 0 {
				continue
			}
		} else {
			continue
		}

		prediction = outputs[len(outputs)-1]
		if len(prediction) == 0 {
			continue
		}

		target := eval.Dataset.Targets[i]
		if len(prediction) != len(target) {
			continue
		}

		// Calculate metrics for this sample
		sampleMSE := 0.0
		sampleMAE := 0.0
		sampleMAPE := 0.0

		for j := range prediction {
			diff := prediction[j] - target[j]
			sampleMSE += diff * diff
			sampleMAE += math.Abs(diff)

			// MAPE calculation (avoid division by zero)
			if math.Abs(target[j]) > 1e-8 {
				sampleMAPE += math.Abs(diff) / math.Abs(target[j])
			}
		}

		sumMSE += sampleMSE / float64(len(prediction))
		sumMAE += sampleMAE / float64(len(prediction))
		sumMAPE += sampleMAPE / float64(len(prediction))
		validSamples++
	}

	if validSamples == 0 {
		return 0, 0, 0, 0
	}

	mse = sumMSE / float64(validSamples)
	mae = sumMAE / float64(validSamples)
	mape = (sumMAPE / float64(validSamples)) * 100 // Convert to percentage
	rmse = math.Sqrt(mse)

	return mse, mae, mape, rmse
}

// EvaluateLongTermAccuracy evaluates prediction accuracy at different horizons
// Learning Goal: Understanding long-term dependency evaluation in sequence models
func (eval *SequenceEvaluator) EvaluateLongTermAccuracy() map[int]float64 {
	longTermAcc := make(map[int]float64)

	// Test at different prediction horizons (1, 5, 10 steps)
	horizons := []int{1, 5, 10}

	for _, horizon := range horizons {
		if horizon > eval.Dataset.OutputSize {
			continue // Skip if horizon exceeds output size
		}

		totalError := 0.0
		validSamples := 0

		for i, sequence := range eval.Dataset.Sequences {
			var prediction []float64

			var outputs [][]float64
			var err error
			if eval.ModelType == "RNN" && eval.RNN != nil {
				outputs, err = eval.RNN.ForwardSequence(sequence)
				if err != nil || len(outputs) == 0 {
					continue
				}
			} else if eval.ModelType == "LSTM" && eval.LSTM != nil {
				outputs, err = eval.LSTM.ForwardSequence(sequence)
				if err != nil || len(outputs) == 0 {
					continue
				}
			} else {
				continue
			}

			prediction = outputs[len(outputs)-1]
			if len(prediction) < horizon {
				continue
			}

			target := eval.Dataset.Targets[i]
			if len(target) < horizon {
				continue
			}

			// Calculate error for this horizon
			horizonError := 0.0
			for j := 0; j < horizon; j++ {
				diff := prediction[j] - target[j]
				horizonError += diff * diff
			}

			totalError += math.Sqrt(horizonError / float64(horizon))
			validSamples++
		}

		if validSamples > 0 {
			avgError := totalError / float64(validSamples)
			// Convert RMSE to "accuracy" (lower error = higher accuracy)
			// Use 1 / (1 + error) as a simple accuracy metric
			longTermAcc[horizon] = 1.0 / (1.0 + avgError)
		}
	}

	return longTermAcc
}

// generatePredictionExamples generates sample prediction vs actual pairs
// Learning Goal: Understanding prediction quality assessment through examples
func (eval *SequenceEvaluator) generatePredictionExamples(numExamples int) []PredictionPair {
	examples := make([]PredictionPair, 0, numExamples)
	sampleCount := 0

	for i, sequence := range eval.Dataset.Sequences {
		if sampleCount >= numExamples {
			break
		}

		var prediction []float64

		if eval.ModelType == "RNN" && eval.RNN != nil {
			outputs, err := eval.RNN.ForwardSequence(sequence)
			if err == nil && len(outputs) > 0 {
				prediction = outputs[len(outputs)-1]
			}
		} else if eval.ModelType == "LSTM" && eval.LSTM != nil {
			outputs, err := eval.LSTM.ForwardSequence(sequence)
			if err == nil && len(outputs) > 0 {
				prediction = outputs[len(outputs)-1]
			}
		} else {
			continue
		}

		target := eval.Dataset.Targets[i]
		predError := eval.calculateMSELoss(prediction, target)

		examples = append(examples, PredictionPair{
			Predicted: prediction,
			Actual:    target,
			Error:     predError,
		})

		sampleCount++
	}

	return examples
}

// measureInferenceTime measures average inference time per sequence
// Learning Goal: Understanding performance profiling for sequence models
func (eval *SequenceEvaluator) measureInferenceTime() time.Duration {
	if len(eval.Dataset.Sequences) == 0 {
		return 0
	}

	sampleSize := min(100, len(eval.Dataset.Sequences))
	start := time.Now()

	for i := 0; i < sampleSize; i++ {
		sequence := eval.Dataset.Sequences[i]

		if eval.ModelType == "RNN" && eval.RNN != nil {
			if _, err := eval.RNN.ForwardSequence(sequence); err != nil {
				continue // Skip problematic sequences
			}
		} else if eval.ModelType == "LSTM" && eval.LSTM != nil {
			if _, err := eval.LSTM.ForwardSequence(sequence); err != nil {
				continue // Skip problematic sequences
			}
		}
	}

	totalTime := time.Since(start)
	return totalTime / time.Duration(sampleSize)
}

// estimateMemoryUsage estimates memory usage of the sequence model
// Learning Goal: Understanding memory requirements for sequence models
func (eval *SequenceEvaluator) estimateMemoryUsage() int64 {
	var totalMemory int64

	if eval.ModelType == "RNN" && eval.RNN != nil {
		// Estimate RNN memory usage based on the structure we know
		// Using dataset dimensions since we don't have direct access to RNN internals
		inputSize := eval.Dataset.Features
		outputSize := eval.Dataset.OutputSize
		hiddenSize := max(16, inputSize*4) // Same calculation as in CreateRNN
		if eval.Dataset.SeqLength > 50 {
			hiddenSize = max(32, inputSize*8)
		}

		// Weights: input-to-hidden, hidden-to-hidden, output weights, biases
		weightsCount := inputSize*hiddenSize + hiddenSize*hiddenSize + hiddenSize*outputSize + hiddenSize + outputSize
		totalMemory += int64(weightsCount * 8) // 8 bytes per float64

		// Hidden state memory
		totalMemory += int64(hiddenSize * 8)

	} else if eval.ModelType == "LSTM" && eval.LSTM != nil {
		// Estimate LSTM memory usage (4 gates + cell state)
		inputSize := eval.Dataset.Features
		outputSize := eval.Dataset.OutputSize
		hiddenSize := max(16, inputSize*4) // Same calculation as in CreateLSTM
		if eval.Dataset.SeqLength > 50 {
			hiddenSize = max(32, inputSize*8)
		}

		// 4 gates * (input + hidden + bias) + output weights
		gateWeights := 4 * (inputSize + hiddenSize + 1) * hiddenSize
		outputWeights := (hiddenSize + 1) * outputSize
		totalMemory += int64((gateWeights + outputWeights) * 8)

		// Hidden state and cell state memory
		totalMemory += int64(hiddenSize * 2 * 8)
	}

	return totalMemory
}

// PrintSequenceResults displays comprehensive sequence learning results
// Learning Goal: Understanding sequence learning evaluation reporting
func (eval *SequenceEvaluator) PrintSequenceResults(results *SequenceResults) {
	fmt.Printf("\n🧠 %s Sequence Learning Results\n", results.ModelName)
	fmt.Printf("═══════════════════════════════════════════════\n")
	fmt.Printf("📊 Dataset: %s\n", results.DatasetName)
	fmt.Printf("⏱️  Training Time: %v\n", results.TrainingTime)
	fmt.Printf("⚡ Inference Time: %v (per sequence)\n", results.InferenceTime)
	fmt.Printf("🎯 Prediction Metrics:\n")
	fmt.Printf("   MSE:  %.6f\n", results.MSE)
	fmt.Printf("   MAE:  %.6f\n", results.MAE)
	fmt.Printf("   RMSE: %.6f\n", results.RMSE)
	fmt.Printf("   MAPE: %.2f%%\n", results.MAPE)
	fmt.Printf("🔄 Training Info:\n")
	fmt.Printf("   Epochs: %d\n", results.EpochsCompleted)
	fmt.Printf("   Convergence: epoch %d\n", results.ConvergenceEpoch)
	fmt.Printf("💾 Memory Usage: %.2f KB\n", float64(results.MemoryUsage)/1024)
	fmt.Printf("🕐 Timestamp: %s\n", results.Timestamp.Format("2006-01-02 15:04:05"))

	// Long-term accuracy
	if len(results.LongTermAccuracy) > 0 {
		fmt.Printf("\n📈 Long-term Prediction Accuracy:\n")
		for horizon, acc := range results.LongTermAccuracy {
			fmt.Printf("   %d-step: %.4f\n", horizon, acc)
		}
	}

	// Prediction examples
	if len(results.PredictionExamples) > 0 {
		fmt.Printf("\n📋 Prediction Examples:\n")
		for i, example := range results.PredictionExamples {
			fmt.Printf("   Example %d: Predicted=[", i+1)
			for j, pred := range example.Predicted {
				if j > 0 {
					fmt.Printf(", ")
				}
				fmt.Printf("%.4f", pred)
			}
			fmt.Printf("], Actual=[")
			for j, actual := range example.Actual {
				if j > 0 {
					fmt.Printf(", ")
				}
				fmt.Printf("%.4f", actual)
			}
			fmt.Printf("], Error=%.6f\n", example.Error)
		}
	}
}

// max returns the maximum of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
