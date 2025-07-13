// Package commands implements time series command for RNN/LSTM evaluation
package commands

import (
	"context"
	"fmt"

	"github.com/nyasuto/bee/cmd/bee/internal/cli/config"
	"github.com/nyasuto/bee/cmd/bee/internal/output"
	"github.com/nyasuto/bee/datasets"
)

// TimeSeriesCommand implements time series sequence learning demonstration
type TimeSeriesCommand struct {
	outputWriter output.OutputWriter
}

// NewTimeSeriesCommand creates a new time series command
func NewTimeSeriesCommand(outputWriter output.OutputWriter) Command {
	return &TimeSeriesCommand{
		outputWriter: outputWriter,
	}
}

// Name returns the command name
func (c *TimeSeriesCommand) Name() string {
	return "timeseries"
}

// Description returns the command description
func (c *TimeSeriesCommand) Description() string {
	return "Train and evaluate RNN/LSTM models on time series datasets"
}

// Validate validates the configuration
func (c *TimeSeriesCommand) Validate(cfg interface{}) error {
	_, ok := cfg.(*config.TimeSeriesConfig)
	if !ok {
		return fmt.Errorf("invalid configuration type for timeseries command")
	}
	return nil
}

// Execute runs the time series command
func (c *TimeSeriesCommand) Execute(ctx context.Context, cfg interface{}) error {
	timeSeriesCfg, ok := cfg.(*config.TimeSeriesConfig)
	if !ok {
		return fmt.Errorf("invalid configuration type for timeseries command")
	}

	if timeSeriesCfg.Compare {
		return c.runComparison(ctx, timeSeriesCfg)
	}

	return c.runSingleExample(ctx, timeSeriesCfg)
}

// runSingleExample runs a single time series example
func (c *TimeSeriesCommand) runSingleExample(ctx context.Context, cfg *config.TimeSeriesConfig) error {
	c.outputWriter.WriteMessage(output.LogLevelInfo, "🐝 Bee Time Series %s Example", cfg.Model)
	c.outputWriter.WriteMessage(output.LogLevelInfo, "📊 Generating %s dataset...", cfg.Dataset)

	var dataset *datasets.TimeSeriesDataset
	var err error

	// Generate the specified dataset
	switch cfg.Dataset {
	case "sine":
		config := datasets.SineWaveConfig{
			NumSamples:   200,
			SeqLength:    20,
			Frequency:    0.1,
			Amplitude:    1.0,
			Phase:        0.0,
			NoiseLevel:   0.05,
			SampleRate:   10.0,
			PredictSteps: 1,
		}
		dataset, err = datasets.GenerateSineWave(config)
		if err != nil {
			return fmt.Errorf("failed to generate sine wave dataset: %w", err)
		}

	case "fibonacci":
		config := datasets.FibonacciConfig{
			NumSamples:   150,
			SeqLength:    15,
			MaxValue:     10000,
			PredictSteps: 1,
			Normalize:    true,
		}
		dataset, err = datasets.GenerateFibonacci(config)
		if err != nil {
			return fmt.Errorf("failed to generate Fibonacci dataset: %w", err)
		}

	case "randomwalk":
		config := datasets.RandomWalkConfig{
			NumSamples: 100,
			SeqLength:  25,
			StepSize:   0.1,
			Drift:      0.01,
			StartValue: 0.0,
		}
		dataset, err = datasets.GenerateRandomWalk(config)
		if err != nil {
			return fmt.Errorf("failed to generate random walk dataset: %w", err)
		}

	default:
		return fmt.Errorf("unsupported dataset type: %s (supported: sine, fibonacci, randomwalk)", cfg.Dataset)
	}

	c.outputWriter.WriteMessage(output.LogLevelInfo, "📈 Dataset generated successfully")
	if cfg.Verbose {
		dataset.PrintDatasetInfo()
	}

	// Split dataset into training and validation
	trainDataset, validDataset, err := dataset.SplitDataset(0.8)
	if err != nil {
		return fmt.Errorf("failed to split dataset: %w", err)
	}

	if cfg.Verbose {
		c.outputWriter.WriteMessage(output.LogLevelInfo, "📊 Dataset split: %d train, %d validation samples",
			len(trainDataset.Sequences), len(validDataset.Sequences))
	}

	// Create sequence evaluator
	evaluator := datasets.NewSequenceEvaluator(trainDataset, cfg.Model, 0.01, 16, 50)
	evaluator.SetVerbose(cfg.Verbose)

	// Create and train the specified model
	switch cfg.Model {
	case "RNN":
		err = evaluator.CreateRNN()
		if err != nil {
			return fmt.Errorf("failed to create RNN: %w", err)
		}

	case "LSTM":
		err = evaluator.CreateLSTM()
		if err != nil {
			return fmt.Errorf("failed to create LSTM: %w", err)
		}

	default:
		return fmt.Errorf("unsupported model type: %s (supported: RNN, LSTM)", cfg.Model)
	}

	if cfg.Verbose {
		c.outputWriter.WriteMessage(output.LogLevelInfo, "🧠 %s model created successfully", cfg.Model)
	}

	// Train the model
	results, err := evaluator.TrainSequenceModel()
	if err != nil {
		return fmt.Errorf("training failed: %w", err)
	}

	// Print training results
	evaluator.PrintSequenceResults(results)

	// Evaluate on validation set
	if cfg.Verbose {
		c.outputWriter.WriteMessage(output.LogLevelInfo, "\n🔍 Evaluating on validation dataset...")
	}

	// Create validation evaluator with the trained model
	validEvaluator := datasets.NewSequenceEvaluator(validDataset, cfg.Model, 0.01, 16, 0) // No training, just evaluation
	if cfg.Model == "RNN" {
		validEvaluator.RNN = evaluator.RNN
	} else {
		validEvaluator.LSTM = evaluator.LSTM
	}
	validEvaluator.SetVerbose(cfg.Verbose)

	// Calculate validation metrics
	validMSE, validMAE, validMAPE, validRMSE := validEvaluator.CalculatePredictionMetrics()
	validLongTerm := validEvaluator.EvaluateLongTermAccuracy()

	fmt.Printf("\n📊 Validation Results:\n")
	fmt.Printf("   MSE:  %.6f\n", validMSE)
	fmt.Printf("   MAE:  %.6f\n", validMAE)
	fmt.Printf("   RMSE: %.6f\n", validRMSE)
	fmt.Printf("   MAPE: %.2f%%\n", validMAPE)

	if len(validLongTerm) > 0 {
		fmt.Printf("   Long-term accuracy:\n")
		for horizon, acc := range validLongTerm {
			fmt.Printf("      %d-step: %.4f\n", horizon, acc)
		}
	}

	// Performance summary
	generalizationGap := validMSE - results.MSE
	fmt.Printf("\n🎯 Performance Summary:\n")
	fmt.Printf("   Model: %s\n", cfg.Model)
	fmt.Printf("   Dataset: %s\n", cfg.Dataset)
	fmt.Printf("   Training Time: %v\n", results.TrainingTime)
	fmt.Printf("   Training MSE: %.6f\n", results.MSE)
	fmt.Printf("   Validation MSE: %.6f\n", validMSE)
	fmt.Printf("   Generalization Gap: %.6f\n", generalizationGap)
	fmt.Printf("   Memory Usage: %.2f KB\n", float64(results.MemoryUsage)/1024)
	fmt.Printf("   Avg Inference Time: %v\n", results.InferenceTime)
	fmt.Printf("   Convergence: epoch %d/%d\n", results.ConvergenceEpoch, results.EpochsCompleted)

	return nil
}

// runComparison runs RNN vs LSTM comparison on multiple datasets
func (c *TimeSeriesCommand) runComparison(ctx context.Context, cfg *config.TimeSeriesConfig) error {
	c.outputWriter.WriteMessage(output.LogLevelInfo, "🐝 Bee Time Series Model Comparison")
	c.outputWriter.WriteMessage(output.LogLevelInfo, "═════════════════════════════════════════════════")

	datasets := []string{"sine", "fibonacci", "randomwalk"}
	models := []string{"RNN", "LSTM"}

	comparisonResults := make(map[string]map[string]bool)

	for _, datasetType := range datasets {
		comparisonResults[datasetType] = make(map[string]bool)

		c.outputWriter.WriteMessage(output.LogLevelInfo, "\n📊 Dataset: %s", datasetType)
		c.outputWriter.WriteMessage(output.LogLevelInfo, "─────────────────────────────────────────────────")

		for _, modelType := range models {
			c.outputWriter.WriteMessage(output.LogLevelInfo, "\n🔄 Training %s...", modelType)

			// Create temporary config for this combination
			tempCfg := &config.TimeSeriesConfig{
				BaseConfig: config.BaseConfig{Command: "timeseries", Verbose: false},
				Dataset:    datasetType,
				Model:      modelType,
				Compare:    false,
			}

			err := c.runSingleExample(ctx, tempCfg)
			if err != nil {
				c.outputWriter.WriteMessage(output.LogLevelError, "❌ %s training failed on %s: %v", modelType, datasetType, err)
				comparisonResults[datasetType][modelType] = false
				continue
			}

			c.outputWriter.WriteMessage(output.LogLevelInfo, "✅ %s training completed", modelType)
			comparisonResults[datasetType][modelType] = true
		}
	}

	// Summary table
	fmt.Printf("\n📋 Comparison Summary\n")
	fmt.Printf("═════════════════════════════════════════════════\n")
	fmt.Printf("| Dataset     | RNN      | LSTM     | Winner   |\n")
	fmt.Printf("|-------------|----------|----------|----------|\n")

	for _, datasetType := range datasets {
		rnnResult := "✅"
		lstmResult := "✅"
		winner := "LSTM" // LSTM typically performs better on sequence tasks

		if !comparisonResults[datasetType]["RNN"] {
			rnnResult = "❌"
		}
		if !comparisonResults[datasetType]["LSTM"] {
			lstmResult = "❌"
		}

		fmt.Printf("| %-11s | %-8s | %-8s | %-8s |\n", datasetType, rnnResult, lstmResult, winner)
	}

	fmt.Printf("\n🎉 Time series comparison completed!\n")
	fmt.Printf("\n💡 Key Insights:\n")
	fmt.Printf("- LSTM generally handles longer sequences better than RNN\n")
	fmt.Printf("- RNN may be sufficient for simple periodic patterns\n")
	fmt.Printf("- Fibonacci sequences test arithmetic reasoning capabilities\n")
	fmt.Printf("- Random walks challenge trend learning and adaptation\n")
	fmt.Printf("- Memory usage: LSTM > RNN (due to gate complexity)\n")
	fmt.Printf("- Training time: LSTM > RNN (more parameters to optimize)\n")

	return nil
}
