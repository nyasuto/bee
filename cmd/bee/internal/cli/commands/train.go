// Package commands provides command implementations
package commands

import (
	"context"
	"fmt"

	"github.com/nyasuto/bee/cmd/bee/internal/cli/config"
	"github.com/nyasuto/bee/cmd/bee/internal/data"
	"github.com/nyasuto/bee/cmd/bee/internal/model"
	"github.com/nyasuto/bee/cmd/bee/internal/output"
)

// TrainCommand implements the train command
type TrainCommand struct {
	dataLoader    data.DataLoader
	dataValidator data.DataValidator
	modelManager  model.ModelManager
	outputWriter  output.OutputWriter
}

// NewTrainCommand creates a new train command
func NewTrainCommand(
	dataLoader data.DataLoader,
	dataValidator data.DataValidator,
	modelManager model.ModelManager,
	outputWriter output.OutputWriter,
) *TrainCommand {
	return &TrainCommand{
		dataLoader:    dataLoader,
		dataValidator: dataValidator,
		modelManager:  modelManager,
		outputWriter:  outputWriter,
	}
}

// Execute runs the train command
func (t *TrainCommand) Execute(ctx context.Context, cfg interface{}) error {
	config, ok := cfg.(*config.TrainConfig)
	if !ok {
		return fmt.Errorf("invalid configuration type for train command")
	}

	if err := t.Validate(config); err != nil {
		return err
	}

	if config.Verbose {
		t.outputWriter.WriteMessage(output.LogLevelInfo, "🐝 Bee Training - %s Model", config.Model)
		t.outputWriter.WriteMessage(output.LogLevelInfo, "📊 Data: %s", config.DataPath)
		t.outputWriter.WriteMessage(output.LogLevelInfo, "⚙️  Learning Rate: %.3f", config.LearningRate)
		t.outputWriter.WriteMessage(output.LogLevelInfo, "🔄 Max Epochs: %d", config.Epochs)
	}

	// Load training data
	inputs, targets, err := t.dataLoader.LoadTrainingData(config.DataPath)
	if err != nil {
		return fmt.Errorf("failed to load data: %w", err)
	}

	// Validate training data
	if err := t.dataValidator.ValidateTrainingData(inputs, targets); err != nil {
		return fmt.Errorf("invalid training data: %w", err)
	}

	if config.Verbose {
		t.outputWriter.WriteMessage(output.LogLevelInfo, "📈 Loaded %d training samples with %d features",
			len(inputs), len(inputs[0]))
	}

	// Create model
	model, err := t.modelManager.CreateModel(config.Model, len(inputs[0]), config.LearningRate)
	if err != nil {
		return fmt.Errorf("failed to create model: %w", err)
	}

	if config.Verbose {
		t.outputWriter.WriteMessage(output.LogLevelInfo, "🧠 Training %s...", config.Model)
	}

	// Train model
	epochs, err := model.Train(inputs, targets, config.Epochs)
	if err != nil {
		return fmt.Errorf("training failed: %w", err)
	}

	// Calculate final accuracy
	accuracy, err := model.Accuracy(inputs, targets)
	if err != nil {
		return fmt.Errorf("accuracy calculation failed: %w", err)
	}

	// Write training results
	t.outputWriter.WriteTrainingResult(epochs, accuracy, config.Verbose)

	if config.Verbose {
		t.outputWriter.WriteMessage(output.LogLevelInfo, "🔧 Final weights: %v", model.GetWeights())
		t.outputWriter.WriteMessage(output.LogLevelInfo, "🔧 Final bias: %.4f", model.GetBias())
	}

	// Save model
	if err := t.modelManager.SaveModel(model, config.ModelPath); err != nil {
		return fmt.Errorf("failed to save model: %w", err)
	}

	t.outputWriter.WriteMessage(output.LogLevelInfo, "💾 Model saved to: %s", config.ModelPath)

	return nil
}

// Validate checks if the configuration is valid for this command
func (t *TrainCommand) Validate(cfg interface{}) error {
	config, ok := cfg.(*config.TrainConfig)
	if !ok {
		return fmt.Errorf("invalid configuration type for train command")
	}

	if config.DataPath == "" {
		return fmt.Errorf("data path is required for training")
	}

	if config.LearningRate <= 0 {
		return fmt.Errorf("learning rate must be positive")
	}

	if config.Epochs <= 0 {
		return fmt.Errorf("epochs must be positive")
	}

	return nil
}

// Name returns the name of the command
func (t *TrainCommand) Name() string {
	return "train"
}

// Description returns a brief description of the command
func (t *TrainCommand) Description() string {
	return "Train a neural network model"
}
