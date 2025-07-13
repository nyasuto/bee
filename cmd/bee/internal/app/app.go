// Package app provides the main application setup and dependency injection
package app

import (
	"context"
	"fmt"

	"github.com/nyasuto/bee/cmd/bee/internal/cli"
	"github.com/nyasuto/bee/cmd/bee/internal/cli/commands"
	"github.com/nyasuto/bee/cmd/bee/internal/cli/config"
	"github.com/nyasuto/bee/cmd/bee/internal/data"
	"github.com/nyasuto/bee/cmd/bee/internal/model"
	"github.com/nyasuto/bee/cmd/bee/internal/output"
)

// App represents the main application
type App struct {
	parser       *cli.Parser
	outputWriter output.OutputWriter
	commandMap   map[string]commands.Command
}

// NewApp creates a new application with dependencies injected
func NewApp(args []string) *App {
	// Create dependencies
	dataLoader := data.NewCSVDataLoader()
	dataValidator := data.NewDefaultDataValidator()
	modelManager := model.NewDefaultModelManager()
	outputWriter := output.NewConsoleOutputWriter()

	// Create parser
	var parser *cli.Parser
	if args != nil {
		parser = cli.NewParser(args)
	} else {
		parser = cli.NewParserFromOS()
	}

	// Create commands with dependency injection
	trainCommand := commands.NewTrainCommand(dataLoader, dataValidator, modelManager, outputWriter)
	timeSeriesCommand := commands.NewTimeSeriesCommand(outputWriter)
	benchmarkCommand := commands.NewBenchmarkCommand(outputWriter)

	// Create command map
	commandMap := map[string]commands.Command{
		"train":      trainCommand,
		"timeseries": timeSeriesCommand,
		"benchmark":  benchmarkCommand,
		// TODO: Add other commands as they are implemented
	}

	return &App{
		parser:       parser,
		outputWriter: outputWriter,
		commandMap:   commandMap,
	}
}

// NewAppFromOS creates a new application using os.Args
func NewAppFromOS() *App {
	return NewApp(nil) // Will use os.Args internally
}

// Run executes the application
func (a *App) Run(ctx context.Context) error {
	// Parse command line arguments
	cfg, err := a.parser.Parse()
	if err != nil {
		a.outputWriter.WriteMessage(output.LogLevelError, "Failed to parse arguments: %v", err)
		a.outputWriter.WriteUsage()
		return err
	}

	// Handle help command
	if baseCfg, ok := cfg.(*config.BaseConfig); ok && baseCfg.Command == "help" {
		a.outputWriter.WriteUsage()
		return nil
	}

	// Get command name
	commandName := a.getCommandName(cfg)
	if commandName == "" {
		a.outputWriter.WriteMessage(output.LogLevelError, "Unable to determine command")
		a.outputWriter.WriteUsage()
		return fmt.Errorf("unable to determine command")
	}

	// Get command implementation
	command, exists := a.commandMap[commandName]
	if !exists {
		a.outputWriter.WriteMessage(output.LogLevelError, "Unknown command: %s", commandName)
		a.outputWriter.WriteUsage()
		return fmt.Errorf("unknown command: %s", commandName)
	}

	// Execute command
	if err := command.Execute(ctx, cfg); err != nil {
		a.outputWriter.WriteMessage(output.LogLevelError, "%s failed: %v", commandName, err)
		return err
	}

	return nil
}

// getCommandName extracts the command name from the configuration
func (a *App) getCommandName(cfg interface{}) string {
	switch c := cfg.(type) {
	case *config.TrainConfig:
		return c.Command
	case *config.InferConfig:
		return c.Command
	case *config.TestConfig:
		return c.Command
	case *config.BenchmarkConfig:
		return c.Command
	case *config.CompareConfig:
		return c.Command
	case *config.MnistConfig:
		return c.Command
	case *config.TimeSeriesConfig:
		return c.Command
	case *config.BaseConfig:
		return c.Command
	default:
		return ""
	}
}
