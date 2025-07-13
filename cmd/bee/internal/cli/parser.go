// Package cli provides command line interface parsing and handling
package cli

import (
	"flag"
	"fmt"
	"os"

	"github.com/nyasuto/bee/cmd/bee/internal/cli/config"
)

// Parser handles command line argument parsing
type Parser struct {
	args []string
}

// NewParser creates a new parser with the given arguments
func NewParser(args []string) *Parser {
	return &Parser{args: args}
}

// NewParserFromOS creates a parser using os.Args
func NewParserFromOS() *Parser {
	return &Parser{args: os.Args}
}

// Parse parses the command line arguments and returns appropriate config
func (p *Parser) Parse() (interface{}, error) {
	if len(p.args) < 2 {
		return nil, fmt.Errorf("no command specified")
	}

	command := p.args[1]

	switch command {
	case "train":
		return p.parseTrainCommand()
	case "infer":
		return p.parseInferCommand()
	case "test":
		return p.parseTestCommand()
	case "benchmark":
		return p.parseBenchmarkCommand()
	case "compare":
		return p.parseCompareCommand()
	case "mnist":
		return p.parseMnistCommand()
	case "timeseries":
		return p.parseTimeSeriesCommand()
	case "help", "":
		return &config.BaseConfig{Command: "help"}, nil
	default:
		return nil, fmt.Errorf("unknown command: %s", command)
	}
}

func (p *Parser) parseTrainCommand() (*config.TrainConfig, error) {
	flagSet := flag.NewFlagSet("train", flag.ContinueOnError)

	cfg := &config.TrainConfig{
		BaseConfig: config.BaseConfig{Command: "train"},
	}

	flagSet.StringVar(&cfg.Model, "model", "perceptron", "Model type (perceptron)")
	flagSet.StringVar(&cfg.DataPath, "data", "", "Path to training data (CSV)")
	flagSet.StringVar(&cfg.ModelPath, "output", "model.json", "Output model path")
	flagSet.Float64Var(&cfg.LearningRate, "lr", 0.1, "Learning rate")
	flagSet.IntVar(&cfg.Epochs, "epochs", 1000, "Maximum training epochs")
	flagSet.BoolVar(&cfg.Verbose, "verbose", false, "Verbose output")

	if err := flagSet.Parse(p.args[2:]); err != nil {
		return nil, fmt.Errorf("failed to parse train flags: %w", err)
	}

	return cfg, nil
}

func (p *Parser) parseInferCommand() (*config.InferConfig, error) {
	flagSet := flag.NewFlagSet("infer", flag.ContinueOnError)

	cfg := &config.InferConfig{
		BaseConfig: config.BaseConfig{Command: "infer"},
	}

	flagSet.StringVar(&cfg.ModelPath, "model", "model.json", "Path to trained model")
	flagSet.StringVar(&cfg.InputData, "input", "", "Comma-separated input values")
	flagSet.BoolVar(&cfg.Verbose, "verbose", false, "Verbose output")

	if err := flagSet.Parse(p.args[2:]); err != nil {
		return nil, fmt.Errorf("failed to parse infer flags: %w", err)
	}

	return cfg, nil
}

func (p *Parser) parseTestCommand() (*config.TestConfig, error) {
	flagSet := flag.NewFlagSet("test", flag.ContinueOnError)

	cfg := &config.TestConfig{
		BaseConfig: config.BaseConfig{Command: "test"},
	}

	flagSet.StringVar(&cfg.Model, "model", "perceptron", "Model type (perceptron)")
	flagSet.StringVar(&cfg.DataPath, "data", "", "Path to test data (CSV)")
	flagSet.StringVar(&cfg.ModelPath, "model-path", "model.json", "Path to trained model")
	flagSet.BoolVar(&cfg.Verbose, "verbose", false, "Verbose output")

	if err := flagSet.Parse(p.args[2:]); err != nil {
		return nil, fmt.Errorf("failed to parse test flags: %w", err)
	}

	return cfg, nil
}

func (p *Parser) parseBenchmarkCommand() (*config.BenchmarkConfig, error) {
	flagSet := flag.NewFlagSet("benchmark", flag.ContinueOnError)

	cfg := &config.BenchmarkConfig{
		BaseConfig: config.BaseConfig{Command: "benchmark"},
	}

	flagSet.StringVar(&cfg.Model, "model", "perceptron", "Model type (perceptron, mlp, cnn, compare)")
	flagSet.StringVar(&cfg.Dataset, "dataset", "xor", "Dataset type (xor, and, or, mnist, all)")
	flagSet.IntVar(&cfg.Iterations, "iterations", 100, "Number of benchmark iterations")
	flagSet.StringVar(&cfg.OutputPath, "output", "", "Output file for benchmark results (JSON)")
	flagSet.StringVar(&cfg.MLPHidden, "mlp-hidden", "4", "Hidden layer sizes for MLP (comma-separated)")
	// CNN-specific flags
	flagSet.StringVar(&cfg.CNNArch, "cnn-arch", "MNIST", "CNN architecture (MNIST, CIFAR-10, Custom)")
	flagSet.IntVar(&cfg.BatchSize, "batch-size", 32, "Batch size for CNN training")
	flagSet.Float64Var(&cfg.LearningRate, "learning-rate", 0.001, "Learning rate for CNN training")
	flagSet.IntVar(&cfg.Epochs, "epochs", 10, "Number of training epochs for CNN")
	flagSet.BoolVar(&cfg.Verbose, "verbose", false, "Verbose output")

	if err := flagSet.Parse(p.args[2:]); err != nil {
		return nil, fmt.Errorf("failed to parse benchmark flags: %w", err)
	}

	return cfg, nil
}

func (p *Parser) parseCompareCommand() (*config.CompareConfig, error) {
	flagSet := flag.NewFlagSet("compare", flag.ContinueOnError)

	cfg := &config.CompareConfig{
		BaseConfig: config.BaseConfig{Command: "compare"},
	}

	flagSet.StringVar(&cfg.Dataset, "dataset", "xor", "Dataset type (xor, and, or, all)")
	flagSet.IntVar(&cfg.Iterations, "iterations", 100, "Number of benchmark iterations")
	flagSet.StringVar(&cfg.OutputPath, "output", "", "Output file for comparison results (JSON)")
	flagSet.StringVar(&cfg.MLPHidden, "mlp-hidden", "4", "Hidden layer sizes for MLP (comma-separated)")
	flagSet.BoolVar(&cfg.Verbose, "verbose", false, "Verbose output")

	if err := flagSet.Parse(p.args[2:]); err != nil {
		return nil, fmt.Errorf("failed to parse compare flags: %w", err)
	}

	return cfg, nil
}

func (p *Parser) parseMnistCommand() (*config.MnistConfig, error) {
	flagSet := flag.NewFlagSet("mnist", flag.ContinueOnError)

	cfg := &config.MnistConfig{
		BaseConfig: config.BaseConfig{Command: "mnist"},
	}

	flagSet.StringVar(&cfg.DataPath, "data-dir", "datasets/mnist", "Directory for MNIST data")
	flagSet.BoolVar(&cfg.Verbose, "verbose", false, "Verbose output")

	if err := flagSet.Parse(p.args[2:]); err != nil {
		return nil, fmt.Errorf("failed to parse mnist flags: %w", err)
	}

	return cfg, nil
}

func (p *Parser) parseTimeSeriesCommand() (*config.TimeSeriesConfig, error) {
	flagSet := flag.NewFlagSet("timeseries", flag.ContinueOnError)

	cfg := &config.TimeSeriesConfig{
		BaseConfig: config.BaseConfig{Command: "timeseries"},
	}

	flagSet.StringVar(&cfg.Dataset, "dataset", "sine", "Dataset type (sine, fibonacci, randomwalk)")
	flagSet.StringVar(&cfg.Model, "model", "RNN", "Model type (RNN, LSTM)")
	flagSet.BoolVar(&cfg.Compare, "compare", false, "Run RNN vs LSTM comparison on all datasets")
	flagSet.BoolVar(&cfg.Verbose, "verbose", false, "Verbose output")

	if err := flagSet.Parse(p.args[2:]); err != nil {
		return nil, fmt.Errorf("failed to parse timeseries flags: %w", err)
	}

	return cfg, nil
}
